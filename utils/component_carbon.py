"""Synchronized component profiling using CodeCarbon task energy estimates."""
from copy import deepcopy
import jax
import jax.numpy as jnp
from flax import nnx


@nnx.jit(static_argnames=('method',))
def _invoke(module, *args, method='__call__'):
    return getattr(module, method)(*args)


@jax.jit
def _add(x, y):
    return x + y


class ComponentCarbonProfiler:
    """Split inference into compiled components; this changes fusion and latency.

    Task readings estimate whole-machine CPU/GPU/RAM emissions during each
    synchronized interval, including measurement overhead and idle power.
    Short intervals may be below sensor resolution. Unmeasured first calls
    warm new component shapes before starting a task. No FLOP apportionment.
    """

    def __init__(self, tracker):
        self.tracker = tracker
        self.enabled = False
        self.components = {}
        self.warmed = set()
        self.serial = 0

    def call(self, stage, component, layer, fn, *args, **kwargs):
        jax.block_until_ready((args, kwargs))
        key = (stage, component, layer, str(jax.tree.structure((args, kwargs))),
               tuple((x.shape, str(x.dtype)) for x in jax.tree.leaves((args, kwargs))
                     if hasattr(x, 'shape')))
        # Pass modules as explicit arguments to jit, never captured weight constants.
        if key not in self.warmed:
            jax.block_until_ready(fn(*args, **kwargs))
            self.warmed.add(key)
        if not self.enabled:
            return jax.block_until_ready(fn(*args, **kwargs))
        self.serial += 1
        self.tracker.start_task(f'{stage}.{component}.{layer}.{self.serial}')
        try:
            output = jax.block_until_ready(fn(*args, **kwargs))
        finally:
            data = self.tracker.stop_task()
        entry = self.components.setdefault(f'{stage}.{component}', dict(
            calls=0, measured_calls=0, emissions_kg=0.0, energy_kwh=0.0, duration_seconds=0.0))
        entry['calls'] += 1
        if data is not None:
            entry['measured_calls'] += 1
            entry['emissions_kg'] += data.emissions
            entry['energy_kwh'] += data.energy_consumed
            entry['duration_seconds'] += data.duration
        return output

    def forward(self, model, ids, *, stage, prompt_length=None, caches=None, position=None):
        def run(component, layer, module, *args, method='__call__'):
            return self.call(stage, component, layer, _invoke, module, *args, method=method)

        x = run('embedding', None, model.tok_emb, ids)
        length = ids.shape[1]
        positions = jnp.arange(length)
        mask = positions[None, :] > positions[:, None]
        if prompt_length is not None:
            mask = mask | (positions[None, :] >= prompt_length)
        updated = []
        for i, block in enumerate(model.trf_blocks):
            normalized = run('attention_norm', i, block.norm1, x)
            if caches is not None:
                attended, cache = run('attention', i, block.att, normalized, caches[i],
                                      position, model.cos, model.sin, method='decode')
            elif prompt_length is not None:
                attended, cache = run('attention', i, block.att, normalized, mask,
                                      model.cos, model.sin, method='prefill')
            else:
                attended = run('attention', i, block.att, normalized, mask, model.cos, model.sin)
            if caches is not None or prompt_length is not None:
                updated.append(cache)
            x = self.call(stage, 'residual', i, _add, x, attended)
            normalized = run('mlp_norm', i, block.norm2, x)
            mlp = run('mlp', i, block.ff, normalized)
            x = self.call(stage, 'residual', i, _add, x, mlp)
        if prompt_length is not None:
            x = jax.lax.dynamic_slice(x, (0, prompt_length - 1, 0), (x.shape[0], 1, x.shape[2]))
        else:
            x = x[:, -1:, :]
        x = run('final_norm', None, model.final_norm, x)
        logits = run('output_head', None, model.out_head, x)[:, 0].astype(jnp.float32)
        return (logits, tuple(updated)) if prompt_length is not None or caches is not None else logits

    def prefill(self, model, ids, prompt_length, *, cache_capacity=None):
        from model import pad_kv_cache
        logits, caches = self.forward(model, ids, stage='prefill', prompt_length=prompt_length)
        if cache_capacity is not None:
            caches = jax.block_until_ready(pad_kv_cache(caches, cache_capacity))
        return logits, caches

    def decode(self, model, ids, caches, position):
        return self.forward(model, ids, stage='decode', caches=caches, position=position)

    def uncached(self, model, ids):
        return self.forward(model, ids, stage='uncached')

    def report(self):
        return dict(method='CodeCarbon synchronized component tasks',
                    scope=self.__class__.__doc__.strip(),
                    components=deepcopy(self.components),
                    accumulated_emissions_kg=sum(x['emissions_kg'] for x in self.components.values()),
                    accumulated_energy_kwh=sum(x['energy_kwh'] for x in self.components.values()),
                    missing_measurement_calls=sum(x['calls'] - x['measured_calls'] for x in self.components.values()))
