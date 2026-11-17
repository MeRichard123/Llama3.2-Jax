"""Plot a component carbon budget from evaluate.py output.

Run: .env/bin/python utils/data_vis.py
Or:  .env/bin/python utils/data_vis.py evaluation_results/carbon_int4.json
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

LABELS = {
    'attention': 'Attention', 'mlp': 'MLP', 'embedding': 'Embedding',
    'attention_norm': 'Attention norm', 'mlp_norm': 'MLP norm',
    'final_norm': 'Final norm', 'residual': 'Residual additions',
    'output_head': 'Output head',
}


def load_budget(path):
    """Accept carbon.json or the component_carbon section of metrics.json."""
    data = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    data = data.get('component_carbon', data)
    if not data.get('components'):
        raise ValueError('No component measurements found. Run evaluate.py --component-carbon first.')
    totals, stages = {}, {}
    for name, entry in data['components'].items():
        stage, separator, component = name.partition('.')
        if not separator:
            raise ValueError(f'Expected stage.component key, got {name!r}')
        value = entry['emissions_kg']
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid emissions_kg for {name}: {value!r}')
        # Convert kilograms to grams for readable labels.
        grams = value * 1000
        totals[component] = totals.get(component, 0.0) + grams
        stages.setdefault(stage, {})[component] = grams
    if sum(totals.values()) <= 0:
        raise ValueError('Component emissions are all zero; there is no pie-chart budget to plot.')
    return data, totals, stages


def plot_budget(source, output, suffix=None, average_note=None):
    if suffix is None:
        stem = Path(source).stem
        suffix = next((stem[len(prefix):] for prefix in ('carbon_', 'metrics_')
                       if stem.startswith(prefix)), '')
    if suffix and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', suffix):
        raise ValueError('Suffix must start with a letter or digit and contain only letters, digits, _, . or -')
    filename_suffix = f'_{suffix}' if suffix else ''
    title_suffix = f' ({suffix})' if suffix else ''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data, totals, stages = load_budget(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    components = sorted(totals, key=totals.get, reverse=True)
    labels = [LABELS.get(c, c.replace('_', ' ').title()) for c in components]
    values = [totals[c] for c in components]
    total = sum(values)
    colors = list(plt.get_cmap('tab20').colors[:len(components)])
    note = ('Measured inference components only; warmup and compilation excluded.\n'
            'CodeCarbon estimates include idle power and measurement overhead; short tasks may be noisy.')
    if average_note:
        note = average_note + '\n' + note
    missing = data.get('missing_measurement_calls', 0)
    if missing:
        note += f'  Missing task readings: {missing}.'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})

    # Use a legend to keep tiny slices readable without overlapping labels.
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    wedges, _, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                          autopct=lambda pct: f'{pct:.1f}%' if pct >= 3 else '',
                          pctdistance=.77, wedgeprops={'width': .42, 'edgecolor': 'white', 'linewidth': 2})
    ax.text(0, .06, f'{total:.3f} g', ha='center', va='center', fontsize=25, weight='bold')
    ax.text(0, -.14, 'CO₂e measured', ha='center', va='center', fontsize=11, color='#555555')
    legend = [f'{label}  ·  {value:.4f} g  ({value / total:.1%})'
              for label, value in zip(labels, values)]
    ax.legend(wedges, legend, loc='center left', bbox_to_anchor=(1, .5), frameon=False)
    ax.set_title('Carbon budget by model component' + title_suffix, fontsize=19, weight='bold', pad=18)
    fig.text(.05, .035, note, fontsize=9, color='#555555')
    fig.subplots_adjust(left=.04, right=.62, bottom=.14, top=.88)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'component_carbon_pie{filename_suffix}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor='white')
    left = [0.0] * len(components)
    for i, (stage, entries) in enumerate(stages.items()):
        values = [entries.get(c, 0.0) for c in components]
        ax.barh(labels, values, left=left, label=stage.replace('_', ' ').title(),
                color=plt.get_cmap('tab10')(i % 10), height=.66)
        left = [a + b for a, b in zip(left, values)]
    for y, value in enumerate(left):
        ax.text(value + max(left) * .015, y, f'{value:.4f}', va='center', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.2)
    ax.set_xlabel('Carbon emissions (g CO₂e)')
    ax.set_title('Component emissions by inference stage' + title_suffix, fontsize=18, weight='bold', pad=16)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.15)
    ax.legend(frameon=False)
    fig.text(.03, .025, note, fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .12, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'component_carbon_stages{filename_suffix}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

    with (output / f'component_carbon_budget{filename_suffix}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['component', 'emissions_g_co2e', 'share_percent',
                         *[f'{stage}_g_co2e' for stage in stages]])
        for component in components:
            writer.writerow([component, totals[component], totals[component] / total * 100,
                             *[entries.get(component, 0.0) for entries in stages.values()]])
    print(f'Measured component total: {total:.6f} g CO₂e')
    whole = data.get('whole_script_emissions_kg')
    if whole is not None:
        print(f'Whole-script total: {whole * 1000:.6f} g CO₂e (shown separately from component shares)')
    print(f'Charts (PNG/SVG) and budget CSV: {output.resolve()}')
    return output


def plot_comparison(source, output, suffix=None):
    """Plot actual repeated-run statistics; component pies require component data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads(Path(source).read_text())
    variants = data['variants']
    if not variants:
        raise ValueError('Comparison contains no variants.')
    if suffix and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', suffix):
        raise ValueError('Invalid filename suffix')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tag = f'_{suffix}' if suffix else ''
    fields = [('whole_script_emissions_kg', 'Whole-script emissions', 'g CO₂e / evaluation', 1000),
              ('mean_latency_seconds', 'Generation latency', 'seconds / example', 1),
              ('accuracy', 'Accuracy', 'percent', 100)]
    fields = [field for field in fields if all(v.get(field[0], {}).get('n', 0) for v in variants.values())]
    if not fields:
        raise ValueError('Comparison contains no completed measurements to plot.')
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 5), squeeze=False)
    rows = []
    for ax, (key, title, unit, scale) in zip(axes[0], fields):
        for i, (variant, values) in enumerate(variants.items()):
            stats = values[key]
            mean, stddev = stats['mean'] * scale, (stats['stddev'] or 0) * scale
            ax.bar(i, mean, yerr=stddev, capsize=6, color=plt.get_cmap('tab10')(i))
            ax.annotate(f'{mean:.4g}\nn={stats["n"]}', (i, mean + stddev),
                        xytext=(0, 8), textcoords='offset points', ha='center')
            rows.append([variant, key, stats['mean'], stats['stddev'], stats['n']])
        ax.set_xticks(range(len(variants)), [v.upper() for v in variants])
        ax.set_title(title)
        ax.set_ylabel(unit)
        upper = max((v[key]['mean'] + (v[key]['stddev'] or 0)) * scale for v in variants.values())
        ax.set_ylim(0, upper * 1.3 if upper > 0 else 1)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('BF16 vs INT4 · mean ± sample standard deviation' + (' (partial)' if not data.get('complete') else ''))
    fig.text(.025, .015, 'Whole-script carbon includes loading, quantisation, compilation and warmup. Error bars show run-to-run variation.', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'carbon_comparison{tag}.{extension}', dpi=200)
    plt.close(fig)
    with (output / f'carbon_comparison{tag}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'metric', 'mean_original_units', 'sample_stddev_original_units', 'runs'])
        writer.writerows(rows)

    component_variants = []
    for variant, values in variants.items():
        entries = values.get('components', {})
        if not entries:
            continue
        # Avoid mixing means calculated from differing subsets of runs.
        counts = {entry['n'] for entry in entries.values()}
        if len(counts) != 1 or not next(iter(counts)):
            raise ValueError(f'{variant}: inconsistent component sample counts; cannot plot a mean budget.')
        label = f'{suffix}_{variant}_mean' if suffix else f'{variant}_mean'
        budget = {'components': {name: {'emissions_kg': entry['mean']} for name, entry in entries.items()}}
        plot_budget(budget, output, suffix=label,
                    average_note=f'Mean component emissions per evaluation across {next(iter(counts))} runs; slices show shares of mean emissions.')
        component_variants.append(variant)
    if not component_variants:
        print('No component measurements in this comparison: attention/MLP pie charts cannot be reconstructed. '
              'To collect them, run compare_evaluations.py with --component-carbon.')
    print(f'Comparison charts and CSV: {output.resolve()}')
    return output


def main():
    parent_path = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, default=Path(parent_path / 'evaluation_results/metrics.json'))
    parser.add_argument('--output', type=Path, default=None,
                        help='Destination folder (default: SOURCE_DIRECTORY/figures)')
    parser.add_argument('--suffix', default=None,
                        help='Figure filename/title label (default: inferred from carbon_LABEL.json or metrics_LABEL.json)')
    args = parser.parse_args()
    try:
        print(args.source)
        data = json.loads(args.source.read_text())
        plot = plot_comparison if 'variants' in data else plot_budget
        plot(args.source, args.output or args.source.parent / 'figures', suffix=args.suffix)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Cannot plot carbon budget: {error}\n')


if __name__ == '__main__':
    main()
