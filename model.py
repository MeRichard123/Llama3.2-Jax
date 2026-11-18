"""Llama 3.2 in JAX/Flax NNX.

Architecture reference: https://huggingface.co/rasbt/llama-3.2-from-scratch
(Sebastian Raschka, Apache-2.0). NNX usage follows the JAX AI Stack tutorial:
https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html

Construct with ``Llama3Model(cfg, rngs=nnx.Rngs(0))`` and call with integer
IDs of shape (batch, sequence). Weights are randomly initialized, not pretrained.
"""

import struct, json
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from kernels.FlashAttention import flash_attention 
from quantisation import QUANTISATIONS, resolve_quantisation

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": jnp.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}
LLAMA32_CONFIG_3B = {
    **LLAMA32_CONFIG_1B,
    "emb_dim": 3072,
    "n_heads": 24,
    "n_layers": 28,
    "rope_freq": dict(LLAMA32_CONFIG_1B["rope_freq"]),
}


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096,
                        freq_config=None, dtype=jnp.float32):
    """Return cosine/sine tables, including optional Llama 3 frequency scaling."""
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be positive and even")
    if context_length <= 0 or theta_base <= 0:
        raise ValueError("context_length and theta_base must be positive")

    frequencies = theta_base ** (-jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    
    if freq_config is not None:
        factor = freq_config["factor"]
        low = freq_config["low_freq_factor"]
        high = freq_config["high_freq_factor"]
        original = freq_config["original_context_length"]
        
        if factor <= 0 or not 0 < low < high or original <= 0:
            raise ValueError("Invalid RoPE frequency scaling configuration")

        wavelength = 2 * jnp.pi / frequencies
        blend = jnp.clip((original / wavelength - low) / (high - low), 0, 1)
        frequencies = frequencies * (blend + (1 - blend) / factor)
    
    phase = jnp.arange(context_length, dtype=jnp.float32)[:, None] * frequencies
    phase = jnp.concatenate((phase, phase), axis=-1)
    return jnp.cos(phase).astype(dtype), jnp.sin(phase).astype(dtype)


def apply_rope(x, cos, sin):
    """Rotate (batch, heads, sequence, head_dim) using split-half RoPE."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires an even head dimension")
    if x.shape[-2] > cos.shape[0] or cos.shape != sin.shape or cos.shape[-1] != x.shape[-1]:
        raise ValueError("RoPE tables do not match the input shape")

    first, second = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    
    rotated = jnp.concatenate((-second, first), axis=-1)
    length = x.shape[-2]
    
    return (x * cos[:length] + rotated * sin[:length]).astype(x.dtype)


def _linear(inputs, outputs, dtype, rngs):
    return nnx.Linear(inputs, outputs, use_bias=False, dtype=dtype,
                      param_dtype=dtype, rngs=rngs)


def _norm(features, dtype, rngs):
    return nnx.RMSNorm(features, epsilon=1e-5, dtype=dtype,
                       param_dtype=dtype, rngs=rngs)


class FeedForward(nnx.Module):
    """Bias-free SwiLU feed-forward network."""

    def __init__(self, cfg, *, rngs):
        width, hidden, dtype = cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"]
        self.fc1 = _linear(width, hidden, dtype, rngs)
        self.fc2 = _linear(width, hidden, dtype, rngs)
        self.fc3 = _linear(hidden, width, dtype, rngs)

    def __call__(self, x):
        return self.fc3(jax.nn.silu(self.fc1(x)) * self.fc2(x))


class GroupedQueryAttention(nnx.Module):
    """Causal attention; each KV head serves a contiguous group of query heads."""

    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=jnp.float32, *, rngs):
        if min(d_in, d_out, num_heads, num_kv_groups) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if d_out % num_heads or num_heads % num_kv_groups:
            raise ValueError("d_out must divide into heads, and heads into KV groups")
        self.head_dim = d_out // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.d_out = d_out

        self.W_query = _linear(d_in, d_out, dtype, rngs)
        self.W_key = _linear(d_in, num_kv_groups * self.head_dim, dtype, rngs)
        self.W_value = _linear(d_in, num_kv_groups * self.head_dim, dtype, rngs)
        self.out_proj = _linear(d_out, d_out, dtype, rngs)

    def _project(self, x, cos, sin):
        batch, length, _ = x.shape

        def heads(projection, count):
            return projection(x).reshape(batch, length, count, self.head_dim).transpose(0, 2, 1, 3)

        return (apply_rope(heads(self.W_query, self.num_heads), cos, sin),
                apply_rope(heads(self.W_key, self.num_kv_groups), cos, sin),
                heads(self.W_value, self.num_kv_groups))

    def _attend(self, q, k, v, mask, *, key_length=None):
        batch, _, length, _ = q.shape
        if False:
            attended = flash_attention(
                    q, k, v, mask=mask, causal=False, key_length=key_length
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(
                batch, length, self.d_out
            )
            return self.out_proj(attended)
        q = q.reshape(batch, self.num_kv_groups, self.group_size, length, self.head_dim)
        scores = jnp.einsum("bgrtd,bgsd->bgrts", q.astype(jnp.float32), k.astype(jnp.float32))
        scores = scores / self.head_dim**0.5
        if mask is not None:
            scores = jnp.where(mask, -jnp.inf, scores)
        if key_length is not None:
            unused_keys = jnp.arange(k.shape[2]) >= key_length
            scores = jnp.where(unused_keys, -jnp.inf, scores)
        weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        attended = jnp.einsum("bgrts,bgsd->bgrtd", weights, v)
        attended = attended.reshape(batch, self.num_heads, length, self.head_dim)
        return self.out_proj(attended.transpose(0, 2, 1, 3).reshape(batch, length, self.d_out))

    def __call__(self, x, mask, cos, sin):
        q, k, v = self._project(x, cos, sin)
        length = x.shape[1]
        return self._attend(q, k, v, mask[:length, :length])

    def prefill(self, x, mask, cos, sin):
        q, k, v = self._project(x, cos, sin)
        return self._attend(q, k, v, mask), (k, v)

    def decode(self, x, cache, position, cos, sin):
        # RoPE must use the token's absolute cache position, rather than zero.
        cos = jax.lax.dynamic_slice(cos, (position, 0), (1, self.head_dim))
        sin = jax.lax.dynamic_slice(sin, (position, 0), (1, self.head_dim))
        q, k, v = self._project(x, cos, sin)
        old_k, old_v = cache
        k = jax.lax.dynamic_update_slice(old_k, k, (0, 0, position, 0))
        v = jax.lax.dynamic_update_slice(old_v, v, (0, 0, position, 0))
        return self._attend(q, k, v, None, key_length=position + 1), (k, v)


class TransformerBlock(nnx.Module):
    def __init__(self, cfg, *, rngs):
        width, dtype = cfg["emb_dim"], cfg["dtype"]
        self.att = GroupedQueryAttention(width, width, cfg["n_heads"],
                                         cfg["n_kv_groups"], dtype, rngs=rngs)
        self.ff = FeedForward(cfg, rngs=rngs)
        self.norm1 = _norm(width, dtype, rngs)
        self.norm2 = _norm(width, dtype, rngs)

    def __call__(self, x, mask, cos, sin):
        x = x + self.att(self.norm1(x), mask, cos, sin)
        return x + self.ff(self.norm2(x))


class Llama3Model(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.cfg = {**cfg, "rope_freq": dict(cfg["rope_freq"]) if cfg.get("rope_freq") else None}

        for name in ("vocab_size", "context_length", "emb_dim", "n_heads",
                     "n_layers", "hidden_dim", "n_kv_groups"):
            if cfg[name] <= 0:
                raise ValueError(f"{name} must be positive")
        
        width, dtype = cfg["emb_dim"], cfg["dtype"]
        
        # Validate/build blocks before allocating the large vocabulary matrices.
        self.trf_blocks = nnx.List([TransformerBlock(cfg, rngs=rngs) for _ in range(cfg["n_layers"])])
        self.tok_emb = nnx.Embed(cfg["vocab_size"], width, dtype=dtype,
                                 param_dtype=dtype, rngs=rngs)
        self.final_norm = _norm(width, dtype, rngs)
        self.out_head = _linear(width, cfg["vocab_size"], dtype, rngs)
        self.cos, self.sin = compute_rope_params(
            width // cfg["n_heads"], cfg["rope_base"], cfg["context_length"], cfg.get("rope_freq"))

    def __call__(self, in_idx, *, last_token_only=False):
        """Return [batch, sequence, vocab] logits, optionally for just the last token."""
        if in_idx.ndim != 2 or not jnp.issubdtype(in_idx.dtype, jnp.integer):
            raise ValueError("in_idx must be a rank-2 array of integer token IDs")
        length = in_idx.shape[1]
        if not 0 < length <= self.cfg["context_length"]:
            raise ValueError("Sequence length must be within the configured context length")
        mask = jnp.triu(jnp.ones((length, length), dtype=jnp.bool_), k=1)
        x = self.tok_emb(in_idx)
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        if last_token_only:
            x = x[:, -1:, :]
        return self.out_head(self.final_norm(x))


    def prefill(self, padded_ids, prompt_length, *, cache_capacity=None):
        """Build fixed-capacity per-layer KV caches from a right-padded prompt.

        Every row has the same valid prompt length. Padded keys are masked;
        only the final valid prompt position is projected to vocabulary logits.
        """
        capacity = padded_ids.shape[1]
        positions = jnp.arange(capacity)
        mask = ((positions[None, :] > positions[:, None]) |
                (positions[None, :] >= prompt_length))
        x = self.tok_emb(padded_ids)
        caches = []
        for block in self.trf_blocks:
            attended, cache = block.att.prefill(block.norm1(x), mask, self.cos, self.sin)
            x = x + attended
            x = x + block.ff(block.norm2(x))
            caches.append(cache)
        last = jax.lax.dynamic_slice(x, (0, prompt_length - 1, 0),
                                     (x.shape[0], 1, x.shape[2]))
        logits = self.out_head(self.final_norm(last))[:, 0].astype(jnp.float32)
        caches = tuple(caches)
        if cache_capacity is not None:
            caches = pad_kv_cache(caches, cache_capacity)
        return logits, caches

    def decode(self, token_ids, caches, position):
        """Append [batch, 1] tokens at a dynamic position in fixed-size caches."""
        x = self.tok_emb(token_ids)
        updated = []
        for block, cache in zip(self.trf_blocks, caches):
            attended, cache = block.att.decode(block.norm1(x), cache, position, self.cos, self.sin)
            x = x + attended
            x = x + block.ff(block.norm2(x))
            updated.append(cache)
        logits = self.out_head(self.final_norm(x))[:, 0].astype(jnp.float32)
        return logits, tuple(updated)

 
    @classmethod
    def from_pretrained(cls, weights_path, context_length=512, quantisation=None):
        """
        Load a single HF safetensors file, transposing PyTorch linear kernels.
        Read tensors through a memory map so the checkpoint is not copied into RAM
        all at once. BF16 is read as raw uint16 then bitcast, preserving its bits.
        quantisation accepts "4", "8", "bf16", "1.58", or a kernel callable.
        Callables return a same-shaped float array or a callable NNX module.
        Embeddings and norms retain the configured dtype.
        """
        quantise = resolve_quantisation(quantisation)
        with open(weights_path, 'rb') as file:
            header_size = struct.unpack('<Q', file.read(8))[0]
            header = json.loads(file.read(header_size))
        width = header['model.embed_tokens.weight']['shape'][1]
        configs = {2048: LLAMA32_CONFIG_1B, 3072: LLAMA32_CONFIG_3B}
        if width not in configs:
            raise ValueError(f'Unsupported Llama 3.2 embedding width: {width}')
        cfg = {**configs[width], 'context_length': context_length}
        if not 0 < context_length <= configs[width]['context_length']:
            raise ValueError('context_length must be within the model context limit')

        # Trace initialization to obtain the graph without allocating random weights.
        model = nnx.eval_shape(lambda: cls(cfg, rngs=nnx.Rngs(0)))
        storage = np.memmap(weights_path, mode='r', dtype=np.uint8)
        used = set()

        def assign(parameter, name, transpose=False, owner=None, attribute=None):
            entry = header[name]
            dtypes = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32}
            if entry['dtype'] not in dtypes:
                raise ValueError(f'Unsupported tensor dtype: {entry["dtype"]}')
            start, end = entry['data_offsets']
            dtype = np.dtype(dtypes[entry['dtype']]).newbyteorder('<')
            if end - start != int(np.prod(entry['shape'])) * dtype.itemsize:
                raise ValueError(f'Invalid tensor size: {name}')
            array = np.ndarray(entry['shape'], dtype=dtype, buffer=storage,
                               offset=8 + header_size + start)
            value = jnp.asarray(array)
            if entry['dtype'] == 'BF16':
                value = jax.lax.bitcast_convert_type(value, jnp.bfloat16)
            if transpose:
                value = value.T
            if value.shape != parameter.shape:
                raise ValueError(f'{name}: expected {parameter.shape}, got {value.shape}')
            if quantise is not None and transpose:
                result = quantise(value)
                if isinstance(result, nnx.Module):
                    if not callable(result):
                        raise ValueError('Quantisation modules must be callable')
                    setattr(owner, attribute, result)
                    jax.block_until_ready(nnx.state(result))
                    used.add(name)
                    return
                value = jnp.asarray(result)
                if value.shape != parameter.shape or not jnp.issubdtype(value.dtype, jnp.floating):
                    raise ValueError('Custom quantisation must return a same-shaped floating array or callable NNX module')
            parameter.value = value.astype(cfg['dtype'])
            jax.block_until_ready(parameter.value)
            used.add(name)

        assign(model.tok_emb.embedding, 'model.embed_tokens.weight')
        for i, block in enumerate(model.trf_blocks):
            prefix = f'model.layers.{i}.'
            for attribute, source in (('W_query', 'q_proj'), ('W_key', 'k_proj'),
                                      ('W_value', 'v_proj'), ('out_proj', 'o_proj')):
                assign(getattr(block.att, attribute).kernel,
                       prefix + 'self_attn.' + source + '.weight', True, block.att, attribute)
            for attribute, source in (('fc1', 'gate_proj'), ('fc2', 'up_proj'),
                                      ('fc3', 'down_proj')):
                assign(getattr(block.ff, attribute).kernel,
                       prefix + 'mlp.' + source + '.weight', True, block.ff, attribute)
            assign(block.norm1.scale, prefix + 'input_layernorm.weight')
            assign(block.norm2.scale, prefix + 'post_attention_layernorm.weight')
        assign(model.final_norm.scale, 'model.norm.weight')
        assign(model.out_head.kernel,
               'lm_head.weight' if 'lm_head.weight' in header else 'model.embed_tokens.weight', True, model, 'out_head')
        unexpected = set(header) - used - {'__metadata__'}
        if unexpected:
            raise ValueError(f'Unexpected checkpoint tensors: {sorted(unexpected)}')
        model.cos, model.sin = compute_rope_params(
            width // cfg['n_heads'], cfg['rope_base'], context_length, cfg['rope_freq'])
        parameters = jax.tree.leaves(nnx.state(model, nnx.Param))
        parameter_bytes = sum(value.nbytes for value in parameters)
        quantisation_name = (quantisation if isinstance(quantisation, str)
                             else getattr(quantisation, '__name__', 'bf16'))
        print(f'Loaded model: quantisation={quantisation_name}, '
              f'parameter storage={parameter_bytes / 1024**2:,.2f} MiB '
              f'({parameter_bytes / 1024**3:.3f} GiB). '
              'Excludes runtime buffers and JAX allocator overhead.', flush=True)
        return model


def pad_kv_cache(caches, capacity):
    """Expand bucket-sized caches without running model layers on the padding."""
    if any(x.shape[2] > capacity for x in jax.tree.leaves(caches)):
        raise ValueError('Cache capacity must cover the prefill bucket')
    return jax.tree.map(lambda x: jnp.pad(
        x, ((0, 0), (0, 0), (0, capacity - x.shape[2]), (0, 0))), caches)


def prefill_bucket_size(length, capacity):
    """Power-of-two prompt buckets, capped at the requested cache capacity."""
    if not 0 < length <= capacity:
        raise ValueError('Prompt length must be within cache capacity')
    return min(capacity, 1 << (length - 1).bit_length())


@nnx.jit(static_argnames=('cache_capacity',))
def prefill_cache(model, padded_ids, prompt_length, *, cache_capacity=None):
    """Compile per prompt bucket; expand only the resulting K/V to capacity."""
    return model.prefill(padded_ids, prompt_length, cache_capacity=cache_capacity)


@nnx.jit(donate_argnames=('caches',))
def decode_cached(model, token_ids, caches, position):
    """Consume caches and return their replacements; never reuse input caches.

    Donation allows XLA to update K/V in place without copying whole buffers.
    Model parameters and token inputs are not donated.
    """
    return model.decode(token_ids, caches, position)


@nnx.jit
def forward_last_token(model, in_idx):
    """Compiled inference forward pass returning FP32 [batch, vocab] logits.

    Kept at module scope so generation calls reuse the compiled function.
    Different input shapes can still trigger separate compilations.
    """
    return model(in_idx, last_token_only=True)[:, -1, :].astype(jnp.float32)


def text_to_token_ids(text, tokenizer):
    return jnp.asarray(tokenizer.encode(text), dtype=jnp.int32)[None, :]

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(jax.device_get(token_ids).reshape(-1).tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0,
             top_k=None, eos_id=None, *, key=None, use_cache=True, flop_counter=None,
             component_profiler=None):
    """Generate using fixed-shape prefill and single-token KV-cached decoding.

    Cache capacity is context_size; use_cache=False retains the uncached path.
    At capacity, rebuild from the last context_size tokens to preserve the
    original sliding-context/RoPE semantics. Cache state is local to each call.
    Batch rows share the input sequence length. Sampling requires an explicit
    key; completed rows emit EOS until every row has completed.
    Pass a flops.FlopCounter to accumulate stage estimates for executed forwards.
    component_profiler splits forwards into synchronized components for energy profiling.
    """
    if not 0 < context_size <= model.cfg["context_length"]:
        raise ValueError("context_size must be within the model context length")
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("max_new_tokens and temperature must be nonnegative")
    if top_k is not None and not 1 <= top_k <= model.cfg["vocab_size"]:
        raise ValueError("top_k must be within the vocabulary size")
    if temperature > 0 and key is None:
        raise ValueError("Provide a JAX PRNG key for sampling")

    if idx.ndim != 2 or not jnp.issubdtype(idx.dtype, jnp.integer) or idx.shape[1] == 0:
        raise ValueError('idx must be a nonempty rank-2 array of integer token IDs')
    if max_new_tokens == 0:
        return idx

    prefill_fn = prefill_cache if component_profiler is None else component_profiler.prefill
    decode_fn = decode_cached if component_profiler is None else component_profiler.decode
    forward_fn = forward_last_token if component_profiler is None else component_profiler.uncached

    def count(stage, query_length, key_length):
        if flop_counter is not None:
            flop_counter.record(stage, model.cfg, idx.shape[0], query_length, key_length)

    def prefill(ids):
        ids = ids[:, -context_size:]
        length = ids.shape[1]
        bucket = prefill_bucket_size(length, context_size)
        padded = jnp.pad(ids, ((0, 0), (0, bucket - length)))
        logits, caches = prefill_fn(model, padded, jnp.asarray(length, jnp.int32),
                                   cache_capacity=context_size)
        count('prefill', bucket, bucket)
        return logits, caches, length

    finished = jnp.zeros((idx.shape[0],), dtype=jnp.bool_)
    if use_cache:
        with jax.profiler.TraceAnnotation('prefill'):
            logits, caches, position = prefill(idx)
        original_length = idx.shape[1]
        # Keep token bookkeeping fixed-size too, rather than compiling a new
        # concatenate for every generated sequence length.
        history = jnp.zeros((idx.shape[0], max(context_size, original_length) + max_new_tokens),
                            dtype=idx.dtype)
        history = jax.lax.dynamic_update_slice(history, idx, (0, 0))

    for step in range(max_new_tokens):
        if not use_cache:
            with jax.profiler.TraceAnnotation('uncached_forward', step=step):
                logits = forward_fn(model, idx[:, -context_size:])
                length = min(idx.shape[1], context_size)
                count('uncached_forward', length, length)
        if top_k is not None:
            cutoff = jax.lax.top_k(logits, top_k)[0][:, -1:]
            logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
        if temperature > 0:
            key, sample_key = jax.random.split(key)
            next_id = jax.random.categorical(sample_key, logits / temperature)
        else:
            next_id = jnp.argmax(logits, axis=-1)
        if eos_id is not None:
            next_id = jnp.where(finished, eos_id, next_id)
            finished = finished | (next_id == eos_id)
        token = next_id.astype(idx.dtype)[:, None]
        if use_cache:
            history_length = original_length + step + 1
            history = jax.lax.dynamic_update_slice(
                history, token, (0, jnp.asarray(history_length - 1, jnp.int32)))
        else:
            idx = jnp.concatenate((idx, token), axis=1)
        if eos_id is not None:
            with jax.profiler.TraceAnnotation('eos_check', step=step):
                if bool(jnp.all(finished)):
                    break
        if use_cache and step + 1 < max_new_tokens:
            if position == context_size:
                window = jax.lax.dynamic_slice(
                    history, (0, jnp.asarray(history_length - context_size, jnp.int32)),
                    (idx.shape[0], context_size))
                with jax.profiler.TraceAnnotation('prefill_window', step=step):
                    logits, caches = prefill_fn(model, window, jnp.asarray(context_size, jnp.int32))
                    count('prefill_window', context_size, context_size)
            else:
                with jax.profiler.TraceAnnotation('decode', step=step + 1):
                    logits, caches = decode_fn(
                        model, token, caches,
                        jnp.asarray(position, jnp.int32))
                    count('decode', 1, context_size)
                position += 1
    return history[:, :history_length] if use_cache else idx
