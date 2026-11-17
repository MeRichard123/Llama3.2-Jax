"""Repeat paired BF16/INT4 evaluations in fresh processes and summarize variation.

.env/bin/python compare_evaluations.py --repeats 5 --limit 300
Add --component-carbon for instrumented component comparisons.
Other arguments are forwarded to evaluate.py, e.g. --context-size 512.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def statistics_for(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                stddev=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def summarize(runs):
    metrics = ['whole_script_emissions_kg', 'component_emissions_kg', 'accuracy',
               'mean_latency_seconds', 'generated_tokens', 'whole_script_kg_per_generated_token',
               'component_kg_per_generated_token']
    variants = {}
    for variant in ('bf16', 'int4'):
        selected = [r for r in runs if r['variant'] == variant]
        variants[variant] = {name: statistics_for([r.get(name) for r in selected]) for name in metrics}
        keys = sorted({k for r in selected for k in r.get('components', {})})
        variants[variant]['components'] = {
            key: statistics_for([r.get('components', {}).get(key) for r in selected]) for key in keys}
    pairs = []
    for repeat in sorted({r['repeat'] for r in runs}):
        pair = {r['variant']: r for r in runs if r['repeat'] == repeat}
        if len(pair) == 2:
            pairs.append(pair['int4']['whole_script_emissions_kg'] - pair['bf16']['whole_script_emissions_kg'])
    return dict(variants=variants, paired_int4_minus_bf16_emissions_kg=statistics_for(pairs))


def read_run(directory, variant, repeat):
    carbon = json.loads((directory / 'carbon.json').read_text())
    metrics = json.loads((directory / 'metrics.json').read_text())
    whole = carbon.get('whole_script_emissions_kg')
    if whole is None or not math.isfinite(whole) or whole < 0:
        raise ValueError(f'Missing or invalid whole-script emissions in {directory}')
    predictions = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines() if line]
    tokens = (sum(p['generated_tokens'] for p in predictions)
              if predictions and all('generated_tokens' in p for p in predictions) else None)
    component = carbon.get('accumulated_emissions_kg')
    return dict(variant=variant, repeat=repeat, output=str(directory),
                whole_script_emissions_kg=whole, component_emissions_kg=component,
                generated_tokens=tokens,
                whole_script_kg_per_generated_token=whole / tokens if tokens else None,
                component_kg_per_generated_token=component / tokens if tokens and component is not None else None,
                accuracy=metrics['accuracy'], mean_latency_seconds=metrics['latency_seconds']['mean'],
                components={k: v['emissions_kg'] for k, v in carbon.get('components', {}).items()})


def save_summary(output, runs, plan, complete):
    summary = dict(complete=complete, planned_runs=len(plan), completed_runs=len(runs),
                   scope='Fresh-process whole-script emissions include model loading, quantisation, compilation and warmup. '
                         'Component emissions include synchronization and tracker overhead. '
                         'Error bars are sample standard deviations, not confidence intervals. '
                         'Generated-token normalization includes emitted EOS tokens; workloads may differ between models.',
                   plan=plan, runs=runs, **summarize(runs))
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants = summary['variants']
    fields = [('whole_script_emissions_kg', 'Whole script')]
    if all(variants[v]['component_emissions_kg']['n'] for v in variants):
        fields.append(('component_emissions_kg', 'Measured components'))
    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5), squeeze=False)
    for ax, (field, label) in zip(axes[0], fields):
        stats = [variants[v][field] for v in ('bf16', 'int4')]
        means = [s['mean'] * 1000 for s in stats]
        ax.bar(['BF16', 'INT4'], means, yerr=[(s['stddev'] or 0) * 1000 for s in stats],
               capsize=7, color=['#2673a5', '#e28c32'])
        for i, (mean, s) in enumerate(zip(means, stats)):
            ax.annotate(f'{mean:.4f} g\nn={s["n"]}', (i, mean), xytext=(0, 12),
                        textcoords='offset points', ha='center')
        ax.set_ylabel('Emissions per evaluation (g CO₂e)')
        ax.set_title(label)
        ax.margins(y=.35)
    fig.suptitle('BF16 vs INT4 · mean ± sample standard deviation')
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(output / f'comparison.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--repeats', type=int, default=5, help='Paired repeats (default: 5, producing 10 evaluations)')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation_results' / 'comparisons')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without evaluating')
    args, forwarded = parser.parse_known_args()
    if args.repeats < 2:
        parser.error('--repeats must be at least 2 to estimate variation')
    managed = {'--quantisation', '--suffix', '--profile-dir', '--profile-generations'}
    if any(arg.split('=')[0] in managed for arg in forwarded):
        parser.error('Quantisation, suffix and profiling paths are managed by this comparison script')
    output = args.output.resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    plan = []
    for repeat in range(1, args.repeats + 1):
        # Reverse order each pair to reduce systematic first/second-run effects.
        for variant in (('bf16', 'int4') if repeat % 2 else ('int4', 'bf16')):
            directory = output / f'pair_{repeat:02d}' / variant
            command = [sys.executable, str(ROOT / 'evaluate.py'), '--quantisation',
                       'bf16' if variant == 'bf16' else '4', '--output', str(directory), *forwarded]
            plan.append(dict(repeat=repeat, variant=variant, output=str(directory), command=command))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    save_summary(output, runs, plan, False)
    for job in plan:
        directory = Path(job['output'])
        directory.mkdir(parents=True)
        print(f'Pair {job["repeat"]}/{args.repeats}: {job["variant"]} → {directory}', flush=True)
        with (directory / 'evaluation.log').open('w') as log:
            result = subprocess.run(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Evaluation failed (exit {result.returncode}); see {directory / "evaluation.log"}. '
                             f'Partial results: {output / "comparison.json"}')
        runs.append(read_run(directory, job['variant'], job['repeat']))
        summary = save_summary(output, runs, plan, len(runs) == len(plan))
    plot_summary(summary, output)
    for variant, entries in summary['variants'].items():
        stats = entries['whole_script_emissions_kg']
        print(f'{variant}: {stats["mean"] * 1000:.6f} ± {stats["stddev"] * 1000:.6f} g CO₂e (n={stats["n"]})')
    print(f'Summary and charts: {output}')


if __name__ == '__main__':
    main()
