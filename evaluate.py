"""Evaluate a local Llama checkpoint on warehouse pallet pick ordering."""
import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from codecarbon import EmissionsTracker

def load_sst2():
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2")
    names = ds['validation'].features['label'].names
    return [dict(row=int(item['idx']), text=item['sentence'], label=names[item['label']])
            for item in ds['validation']]

WAREHOUSE_DATASET = 'RichardMe123/Dynamic_FEFO_Cold-Zone_Dispatch'

WAREHOUSE_INSTRUCTION = (
    'You are a warehouse dispatch assistant. Choose the pallet pick order.\n'
    'Process orders in the listed order. Within each order, pick available,\n'
    'unreserved pallets of the matching SKU with the shortest remaining life\n'
    'first. Use enough stock to satisfy the order. Remaining life is in hours;\n'
    'quantities are in kg.\n\n'
    'Return only a JSON object with a "pick_priorities" array. Each entry\n'
    'must contain "pallet_id" and "priority" (integer starting at 1).\n'
    'Use only the supplied pallet IDs.\n\n'
    'Warehouse data:\n'
)


def canonical_pick_priorities(value):
    """Validate the response and compare by priority, preserving pallet ID case."""
    if not isinstance(value, dict) or set(value) != {'pick_priorities'}:
        raise ValueError('Expected a JSON object containing only pick_priorities')
    entries = value['pick_priorities']
    if not isinstance(entries, list):
        raise ValueError('pick_priorities must be an array')
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {'pallet_id', 'priority'}
                or not isinstance(entry['pallet_id'], str)
                or type(entry['priority']) is not int):
            raise ValueError('Each pick needs a pallet_id string and integer priority')
    if (sorted(e['priority'] for e in entries) != list(range(1, len(entries) + 1))
            or len({e['pallet_id'] for e in entries}) != len(entries)):
        raise ValueError('Priorities must be unique and consecutive; pallet IDs must be unique')
    return json.dumps({'pick_priorities': sorted(entries, key=lambda e: e['priority'])},
                      sort_keys=True, separators=(',', ':'))


def parse_warehouse_prediction(raw):
    try:
        return canonical_pick_priorities(json.loads(raw))
    except (ValueError, TypeError):
        return '<INVALID>'


def load_warehouse(split='validation'):
    from datasets import load_dataset

    ds = load_dataset(WAREHOUSE_DATASET, split=split)
    rows = []
    for index, item in enumerate(ds):
        try:
            user_input = next(m['content'] for m in item['messages'] if m['role'] == 'user')
            state, _ = json.JSONDecoder().raw_decode(user_input[user_input.index('{'):])
            answer = json.loads(next(m['content'] for m in item['messages']
                                     if m['role'] == 'assistant'))
            inputs = {
                'orders': [{key: wave[key] for key in ('at_minute', 'sku', 'quantity_kg')}
                           for wave in state['waves']],
                'pallets': [{key: pallet[key] for key in
                            ('id', 'sku', 'quantity_kg', 'remaining_life_hours', 'reserved')}
                            for pallet in state['initial_pallets']],
            }
            label = canonical_pick_priorities(
                {'pick_priorities': answer['dispatch_plan']['pick_priorities']})
        except (KeyError, ValueError, TypeError, StopIteration) as error:
            raise ValueError(f'Invalid warehouse row {index}: {error}') from error
        rows.append(dict(row=index, text=json.dumps(inputs, ensure_ascii=False), label=label))
    if not rows:
        raise ValueError(f'Warehouse split {split!r} is empty')
    return rows


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def parse_prediction(raw):
    """Accept a bare label or a label preceded by the prompt's Category marker."""
    prediction = raw.strip().casefold()
    if prediction.startswith('category:'):
        prediction = prediction[len('category:'):].strip()
    return prediction


def summarize(results, labels):
    truth = Counter(r['label'] for r in results)
    predicted = Counter(r['prediction'] if r['prediction'] in labels else '<INVALID>' for r in results)
    support = labels + ['<INVALID>']
    # Fixed additive smoothing keeps KL finite even for absent predicted classes.
    alpha = 1e-6
    denominator = len(results) + alpha * len(support)
    p = {k: (truth[k] + alpha) / denominator for k in support}
    q = {k: (predicted[k] + alpha) / denominator for k in support}
    per_class = {}
    for label in labels:
        tp = sum(r['label'] == label and r['prediction'] == label for r in results)
        precision = tp / predicted[label] if predicted[label] else 0.0
        recall = tp / truth[label] if truth[label] else 0.0
        per_class[label] = dict(support=truth[label], precision=precision, recall=recall,
                               f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    times = [r['latency_seconds'] for r in results]
    return dict(examples=len(results), accuracy=sum(r['correct'] for r in results) / len(results),
                invalid_responses=predicted['<INVALID>'],
                macro_f1=sum(v['f1'] for v in per_class.values()) / len(labels),
                per_category=per_class,
                confusion_matrix={label: dict(Counter(r['prediction'] if r['prediction'] in labels else '<INVALID>'
                                                     for r in results if r['label'] == label)) for label in labels},
                category_kl_true_to_predicted_nats=sum(p[k] * math.log(p[k] / q[k]) for k in support),
                kl_smoothing_alpha=alpha, true_category_distribution=p, predicted_category_distribution=q,
                latency_seconds=dict(mean=sum(times)/len(times), p50=percentile(times, .5),
                                     p95=percentile(times, .95), minimum=min(times), maximum=max(times)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-reference', default=None)
    parser.add_argument('--split', default='validation', help='Warehouse dataset split (default: validation)')
    parser.add_argument('--quantisation', choices = ('4', '8', 'bf16', '1.58'), default = None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--limit', type=int, default=300, help='Number of examples to evaluate (default: 300)')
    selection.add_argument('--all', action='store_true', help='Evaluate all examples')
    parser.add_argument('--weights', type=Path, default=Path(__file__).with_name('model.safetensors'))
    parser.add_argument('--tokenizer', type=Path, default=Path(__file__).with_name('tokenizer.model'))
    parser.add_argument('--context-size', type=int, default=4096)
    parser.add_argument('--kv-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='Use fixed-shape prefill and cached decoding (default: enabled)')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--chat', action=argparse.BooleanOptionalAction, default=True,
                        help='Use the Llama instruction chat template (default: enabled; --no-chat for base checkpoints)')
    parser.add_argument('--output', type=Path, default=Path('evaluation_results'))
    parser.add_argument('--suffix', default='',
                        help='Run label appended to carbon, metrics and predictions filenames, e.g. int4 or bf16')
    parser.add_argument('--component-carbon', action='store_true',
                        help='Measure synchronized attention, MLP, norms, embedding, residual and head tasks; changes execution timing')
    parser.add_argument('--profile-dir', type=Path,
                        help='Write a JAX/Perfetto trace of post-warmup generations, then exit')
    parser.add_argument('--profile-generations', type=int, default=3,
                        help='Number of generations to trace with --profile-dir (default: 3)')
    parser.add_argument('--warmup-each', action='store_true',
                        help='Run each input once before timing it to exclude shape-specific compilation')
    args = parser.parse_args()
    if args.suffix and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.suffix):
        parser.error('--suffix must start with a letter or digit and contain only letters, digits, _, . or -')
    return parser, args


def main(tracker=None, parsed_args=None):
    parser, args = parsed_args if parsed_args is not None else parse_args()
    filename_suffix = f'_{args.suffix}' if args.suffix else ''
    if args.component_carbon and tracker is None:
        parser.error('--component-carbon requires a running emissions tracker')
    carbon_path = (args.profile_dir or args.output) / f'carbon{filename_suffix}.json'
    component_profiler = None
    if args.component_carbon:
        from utils.component_carbon import ComponentCarbonProfiler
        component_profiler = ComponentCarbonProfiler(tracker)
        print('Component carbon profiling splits compiled forwards and synchronizes each component. '
              'Emissions include task overhead; short tasks may be below sensor resolution. '
              'Latency reflects this profiling mode.', flush=True)

    def save_carbon():
        carbon_path.parent.mkdir(parents=True, exist_ok=True)
        report = component_profiler.report() if component_profiler is not None else {}
        carbon_path.write_text(json.dumps(report, indent=2) + '\n')
        return carbon_path
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if args.profile_generations < 1:
        parser.error('--profile-generations must be positive')
    try:
        all_rows = load_warehouse(args.split)
    except ValueError as error:
        parser.error(str(error))
    labels = sorted({row['label'] for row in all_rows})
    if '<INVALID>' in labels:
        parser.error('The label <INVALID> is reserved')
    rows = all_rows if args.all else all_rows[:args.limit]
    instruction = WAREHOUSE_INSTRUCTION
    def prompt_for(row):
        return instruction + row['text']

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from utils.utils import load_model
    from model import generate as generate_tokens
    from tokenizer import ChatFormat, Llama3Tokenizer

    tokenizer = Llama3Tokenizer(str(args.tokenizer))
    chat_format = ChatFormat(tokenizer)
    encoded = [(chat_format.encode(prompt_for(row)) if args.chat
                else tokenizer.encode(prompt_for(row), bos=True)) for row in rows]
    if any(len(ids) + args.max_new_tokens > args.context_size for ids in encoded):
        parser.error('An input plus max-new-tokens exceeds context-size; increase it or shorten the text')
    started = time.perf_counter()
    model = load_model(args.weights, args.context_size, quantisation=args.quantisation)
    jax.block_until_ready(nnx.state(model))
    load_seconds = time.perf_counter() - started
    stop = tokenizer.special['<|eot_id|>' if args.chat else '<|end_of_text|>']

    last_generated_tokens = 0

    def predict(ids, *, measure=False):
        nonlocal last_generated_tokens
        if component_profiler is not None:
            component_profiler.enabled = measure
        with jax.profiler.TraceAnnotation('generate_tokens'):
            output = generate_tokens(model, jnp.array([ids], dtype=jnp.int32),
                                     args.max_new_tokens, args.context_size, temperature=0.0,
                                     eos_id=stop, use_cache=args.kv_cache,
                                     component_profiler=component_profiler)
        with jax.profiler.TraceAnnotation('output_transfer'):
            tokens = jax.device_get(output[0, len(ids):]).tolist()
        last_generated_tokens = len(tokens)
        if tokens and tokens[-1] == stop:
            tokens.pop()
        with jax.profiler.TraceAnnotation('tokenizer_decode'):
            return tokenizer.decode(tokens)

    started = time.perf_counter()
    predict(encoded[0])
    warmup_seconds = time.perf_counter() - started
    if args.profile_dir is not None:
        profile_inputs = encoded[:args.profile_generations]
        if args.warmup_each:
            for ids in profile_inputs:
                predict(ids)
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        with jax.profiler.trace(str(args.profile_dir), create_perfetto_trace=True, profiler_options=options):
            for i, ids in enumerate(profile_inputs):
                with jax.profiler.TraceAnnotation('generation', generation=i,
                                                  prompt_tokens=len(ids)):
                    predict(ids, measure=True)
        print(f'Profiled {len(profile_inputs)} generations after warmup. '
              f'Open perfetto_trace.json.gz under {args.profile_dir.resolve()} '
              'at https://ui.perfetto.dev. Evaluation metrics were not written.')
        return save_carbon()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output / f'predictions{filename_suffix}.jsonl').open('w', encoding='utf-8') as handle:
        for row, ids in zip(rows, encoded):
            if args.warmup_each:
                predict(ids)
            started = time.perf_counter()
            raw = predict(ids, measure=True)
            elapsed = time.perf_counter() - started
            prediction = parse_warehouse_prediction(raw)
            result = dict(**row, raw_output=raw, prediction=prediction,
                          correct=prediction == row['label'], latency_seconds=elapsed,
                          prompt_tokens=len(ids), generated_tokens=last_generated_tokens)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            print(f'{len(results)}/{len(rows)} correct={result["correct"]} latency={elapsed:.3f}s', flush=True)
    times = [r['latency_seconds'] for r in results]
    report = dict(
        examples=len(results), accuracy=sum(r['correct'] for r in results) / len(results),
        invalid_responses=sum(r['prediction'] == '<INVALID>' for r in results),
        scoring='Exact match of pallet IDs and priorities, ignoring JSON key and array order',
        reference_scope='Simulation-generated pick priorities; simplified inputs may omit constraints used by the simulation',
        latency_seconds=dict(mean=sum(times) / len(times), p50=percentile(times, .5),
                             p95=percentile(times, .95), minimum=min(times), maximum=max(times)))
    if component_profiler is not None:
        report['component_carbon'] = component_profiler.report()
    report['metadata'] = dict(
                              dataset=WAREHOUSE_DATASET,
                              quantisation=args.quantisation or 'bf16', warmup_each=args.warmup_each,
                              quantisation_scheme=('grouped symmetric INT4' if args.quantisation == '4'
                                                   else args.quantisation or 'bf16'),
                              quantisation_group_size=(model.trf_blocks[0].att.W_query.group_size
                                                       if args.quantisation == '4' else None),
                              split=args.split, task='simplified_pick_order',
                              weights=str(args.weights), tokenizer=str(args.tokenizer),
                              prompt_instruction=instruction, chat=args.chat, temperature=0.0,
                              context_size=args.context_size, max_new_tokens=args.max_new_tokens,
                              kv_cache=args.kv_cache,
                              component_carbon=args.component_carbon,
                              run_suffix=args.suffix,
                              input_shapes=('fixed padded prefill, single-token decode'
                                            if args.kv_cache else 'variable sequence lengths'),
                              model_load_seconds=load_seconds, warmup_seconds=warmup_seconds,
                              device=str(jax.devices()[0]), 
                              source_reference=(args.source_reference or f'https://huggingface.co/datasets/{WAREHOUSE_DATASET}'),
                              latency_scope='Post-warmup generation, device synchronization and decoding; excludes tokenization and model load. Shape-specific compilation may remain.')
    (args.output / f'metrics{filename_suffix}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return save_carbon()


if __name__ == '__main__':
    parsed_args = parse_args()
    args = parsed_args[1]
    args.output.mkdir(parents=True, exist_ok=True)
    filename_suffix = f'_{args.suffix}' if args.suffix else ''
    tracker = EmissionsTracker(output_dir=str(args.output),
                               output_file=f'emissions{filename_suffix}.csv')
    tracker.start()
    carbon_path = None
    try:
        carbon_path = main(tracker, parsed_args)
    finally:
        emissions = tracker.stop()
        print(f"Whole-script emissions: {emissions} kg CO₂e")
        if carbon_path is not None:
            carbon = json.loads(carbon_path.read_text())
            carbon['whole_script_emissions_kg'] = emissions
            carbon['whole_script_scope'] = 'Includes loading, warmup, compilation, evaluation and profiling overhead. Component totals cover measured inference tasks only.'
            carbon_path.write_text(json.dumps(carbon, indent=2) + '\n')
            if 'accumulated_emissions_kg' in carbon:
                print(f"Accumulated component emissions: {carbon['accumulated_emissions_kg']:.9g} kg CO₂e")
            print(f'Carbon report: {carbon_path}')
