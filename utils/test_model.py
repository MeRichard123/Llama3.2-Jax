"""Small CPU regression tests; run with JAX_PLATFORMS=cpu python -m unittest."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model import (LLAMA32_CONFIG_1B, Llama3Model, GroupedQueryAttention,
                   apply_rope, compute_rope_params, forward_last_token,
                   prefill_cache, decode_cached, generate, prefill_bucket_size)


def reference_attention(q, k, v, mask=None, *, causal=True, scale=None,
                        query_offset=0, key_length=None):
    """CPU oracle for model tests; Triton itself is checked in GPU tests."""
    repeats = q.shape[1] // k.shape[1]
    k, v = (jnp.repeat(x, repeats, axis=1) for x in (k, v))
    scores = q.astype(jnp.float32) @ k.astype(jnp.float32).swapaxes(-1, -2)
    scores *= q.shape[-1] ** -0.5 if scale is None else scale
    blocked = jnp.zeros((q.shape[2], k.shape[2]), dtype=bool)
    if mask is not None:
        blocked |= mask
    if causal:
        blocked |= jnp.arange(k.shape[2])[None, :] > (jnp.arange(q.shape[2])[:, None] + query_offset)
    if key_length is not None:
        blocked |= jnp.arange(k.shape[2])[None, :] >= key_length
    scores = jnp.where(blocked, -jnp.inf, scores)
    weights = jnp.nan_to_num(jax.nn.softmax(scores, axis=-1)).astype(v.dtype)
    return (weights @ v).astype(q.dtype)


class ModelTests(unittest.TestCase):
    def setUp(self):
        if jax.devices()[0].platform == 'cpu':
            attention_patch = patch('model.flash_attention', reference_attention)
            attention_patch.start()
            self.addCleanup(attention_patch.stop)
        self.cfg = {**LLAMA32_CONFIG_1B, "vocab_size": 32, "context_length": 16,
                    "emb_dim": 16, "n_heads": 4, "n_kv_groups": 2,
                    "n_layers": 2, "hidden_dim": 32, "dtype": jnp.float32}
        self.model = Llama3Model(self.cfg, rngs=nnx.Rngs(0))
        self.ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])

    def test_forward_causality_jit_and_gradients(self):
        output = self.model(self.ids)
        self.assertEqual(output.shape, (2, 4, 32))
        self.assertTrue(np.isfinite(output).all())
        altered = self.model(self.ids.at[:, 2:].set(7))
        np.testing.assert_array_equal(output[:, :2], altered[:, :2])
        compiled = nnx.jit(lambda model, ids: model(ids))
        np.testing.assert_allclose(compiled(self.model, self.ids), output, atol=2e-6)
        gradients = nnx.grad(lambda model: jnp.mean(model(self.ids)**2))(self.model)
        leaves = jax.tree.leaves(gradients)
        self.assertTrue(all(np.isfinite(leaf).all() for leaf in leaves))
        self.assertGreater(float(jnp.linalg.norm(gradients.tok_emb.embedding.value)), 0)

    def test_grouped_attention_matches_repeated_heads(self):
        attention = GroupedQueryAttention(16, 16, 4, 2, rngs=nnx.Rngs(3))
        x = jax.random.normal(jax.random.key(1), (2, 5, 16))
        cos, sin = compute_rope_params(4, context_length=5)
        mask = jnp.triu(jnp.ones((5, 5), dtype=bool), 1)

        def project(layer, heads):
            return layer(x).reshape(2, 5, heads, 4).transpose(0, 2, 1, 3)

        q = apply_rope(project(attention.W_query, 4), cos, sin)
        k = jnp.repeat(apply_rope(project(attention.W_key, 2), cos, sin), 2, axis=1)
        v = jnp.repeat(project(attention.W_value, 2), 2, axis=1)
        scores = jnp.where(mask, -jnp.inf, q @ k.swapaxes(-1, -2) / 2)
        result = (jax.nn.softmax(scores, -1) @ v).transpose(0, 2, 1, 3).reshape(2, 5, 16)
        np.testing.assert_allclose(attention(x, mask, cos, sin), attention.out_proj(result), atol=1e-6)

    def test_final_token_logits_match_full_forward(self):
        for dtype in (jnp.float32, jnp.bfloat16):
            model = Llama3Model({**self.cfg, 'dtype': dtype}, rngs=nnx.Rngs(0))
            full = model(self.ids)[:, -1:, :]
            final = model(self.ids, last_token_only=True)
            self.assertEqual(final.shape, (2, 1, 32))
            np.testing.assert_allclose(final.astype(jnp.float32), full.astype(jnp.float32),
                                       rtol=0.02 if dtype == jnp.bfloat16 else 1e-5,
                                       atol=0.02 if dtype == jnp.bfloat16 else 2e-6)
            np.testing.assert_allclose(forward_last_token(model, self.ids),
                                       full[:, 0].astype(jnp.float32),
                                       rtol=0.02 if dtype == jnp.bfloat16 else 1e-5,
                                       atol=0.02 if dtype == jnp.bfloat16 else 2e-6)

    def test_generation_matches_full_logits_reference(self):
        expected = self.ids
        # Includes changing sequence lengths followed by a cropped context.
        for _ in range(3):
            logits = self.model(expected[:, -5:])[:, -1]
            next_id = jnp.argmax(logits, axis=-1).astype(expected.dtype)
            expected = jnp.concatenate((expected, next_id[:, None]), axis=1)
        actual = generate(self.model, self.ids, 3, 5)
        np.testing.assert_array_equal(actual, expected)

    def test_inference_head_receives_only_final_hidden_state(self):
        class FinalPositionHead(nnx.Module):
            def __init__(self, projection):
                self.projection = projection

            def __call__(self, hidden):
                if hidden.shape[1] != 1:
                    raise AssertionError('Vocabulary projection received multiple positions')
                return self.projection(hidden)

        self.model.out_head = FinalPositionHead(self.model.out_head)
        self.assertEqual(forward_last_token(self.model, self.ids).shape, (2, 32))

    def test_cached_logits_padding_and_shapes(self):
        for dtype in (jnp.float32, jnp.bfloat16):
            model = Llama3Model({**self.cfg, 'dtype': dtype}, rngs=nnx.Rngs(5))
            for length in (2, 4):
                ids = self.ids[:, :length]
                padded = jnp.pad(ids, ((0, 0), (0, 8 - length)))
                logits, caches = prefill_cache(model, padded, jnp.asarray(length, jnp.int32))
                reference = model(ids)[:, -1].astype(jnp.float32)
                np.testing.assert_allclose(logits, reference, rtol=0.03, atol=0.03)
                expected_shape = (2, self.cfg['n_kv_groups'], 8,
                                  self.cfg['emb_dim'] // self.cfg['n_heads'])
                self.assertTrue(all(k.shape == v.shape == expected_shape for k, v in caches))
                # Arbitrary padding contents must not change valid-position logits.
                poisoned = padded.at[:, length:].set(7)
                other, _ = prefill_cache(model, poisoned, jnp.asarray(length, jnp.int32))
                np.testing.assert_array_equal(logits, other)
                for position in range(length, length + 2):
                    token = jnp.array([[5], [6]], dtype=ids.dtype)
                    ids = jnp.concatenate((ids, token), axis=1)
                    logits, caches = decode_cached(model, token, caches,
                                                   jnp.asarray(position, jnp.int32))
                    np.testing.assert_allclose(logits, model(ids)[:, -1].astype(jnp.float32),
                                               rtol=0.03, atol=0.03)
                    self.assertTrue(all(k.shape == v.shape == expected_shape for k, v in caches))

    def test_cached_generation_reset_sampling_and_context_overflow(self):
        for context in (3, 5, 12):
            cached = generate(self.model, self.ids, 4, context)
            uncached = generate(self.model, self.ids, 4, context, use_cache=False)
            np.testing.assert_array_equal(cached, uncached)
            np.testing.assert_array_equal(generate(self.model, self.ids, 4, context), cached)
        for cached in (False, True):
            a = generate(self.model, self.ids, 3, 12, temperature=0.8,
                         key=jax.random.key(9), use_cache=cached)
            b = generate(self.model, self.ids, 3, 12, temperature=0.8,
                         key=jax.random.key(9), use_cache=cached)
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(generate(self.model, self.ids, 0, 12), self.ids)

    def test_cached_steps_reuse_compilations_across_prompt_lengths(self):
        compile_counts = []
        for length in (2, 4):
            padded = jnp.pad(self.ids[:, :length], ((0, 0), (0, 8 - length)))
            _, cache = prefill_cache(self.model, padded, jnp.asarray(length, jnp.int32))
            _, cache = decode_cached(self.model, jnp.array([[5], [6]], dtype=self.ids.dtype),
                                     cache, jnp.asarray(length, jnp.int32))
            compile_counts.append((prefill_cache.jitted_fn._cache_size(),
                                   decode_cached.jitted_fn._cache_size()))
        self.assertEqual(compile_counts[0], compile_counts[1])

    def test_prefill_buckets_and_expanded_cache_match_full_prefill(self):
        self.assertEqual([prefill_bucket_size(n, 1000) for n in (1, 332, 512, 513, 999)],
                         [1, 512, 512, 1000, 1000])
        for length in (1, 3, 4):
            ids = self.ids[:, :length]
            bucket = prefill_bucket_size(length, 12)
            padded = jnp.pad(ids, ((0, 0), (0, bucket - length)))
            logits, caches = prefill_cache(self.model, padded, jnp.int32(length),
                                           cache_capacity=12)
            full = jnp.pad(ids, ((0, 0), (0, 12 - length)))
            reference, full_caches = prefill_cache(self.model, full, jnp.int32(length))
            np.testing.assert_allclose(logits, reference, atol=2e-6)
            for (k, v), (rk, rv) in zip(caches, full_caches):
                self.assertEqual(k.shape[2], 12)
                np.testing.assert_allclose(k[:, :, :length], rk[:, :, :length], atol=2e-6)
                np.testing.assert_allclose(v[:, :, :length], rv[:, :, :length], atol=2e-6)
            token = jnp.array([[5], [6]], dtype=ids.dtype)
            decoded, _ = decode_cached(self.model, token, caches, jnp.int32(length))
            expected = self.model(jnp.concatenate((ids, token), axis=1))[:, -1]
            np.testing.assert_allclose(decoded, expected, atol=2e-6)

    def test_decode_reuses_cache_buffers(self):
        _, caches = prefill_cache(self.model, self.ids, jnp.int32(4), cache_capacity=12)
        jax.block_until_ready(caches)
        old_leaves = jax.tree.leaves(caches)
        pointers = [x.unsafe_buffer_pointer() for x in old_leaves]
        _, updated = decode_cached(self.model, jnp.array([[5], [6]], dtype=self.ids.dtype),
                                    caches, jnp.int32(4))
        jax.block_until_ready(updated)
        self.assertTrue(all(x.is_deleted() for x in old_leaves))
        self.assertEqual(pointers, [x.unsafe_buffer_pointer() for x in jax.tree.leaves(updated)])

    def test_generation_uses_bucket_and_component_profiler_matches(self):
        from utils.component_carbon import ComponentCarbonProfiler

        ids = self.ids[:, :3]
        with patch('model.prefill_cache', wraps=prefill_cache) as prefill:
            expected = generate(self.model, ids, 3, 12)
        self.assertEqual(prefill.call_args.args[1].shape[1], 4)
        self.assertEqual(prefill.call_args.kwargs['cache_capacity'], 12)
        profiler = ComponentCarbonProfiler(None)  # Measurements disabled; no tracker needed.
        actual = generate(self.model, ids, 3, 12, component_profiler=profiler)
        np.testing.assert_array_equal(actual, expected)

    def test_cached_batch_eos_handling(self):
        class FixedHead(nnx.Module):
            def __call__(self, hidden):
                logits = jnp.full((hidden.shape[0], hidden.shape[1], 32), -100.0)
                labels = jnp.where(jnp.arange(hidden.shape[0]) == 0, 3, 4)
                return logits.at[jnp.arange(hidden.shape[0]), :, labels].set(100)

        self.model.out_head = FixedHead()
        result = generate(self.model, self.ids, 3, 12, eos_id=3)
        np.testing.assert_array_equal(result[:, 4:], [[3, 3, 3], [4, 4, 4]])
        stopped = generate(self.model, self.ids[:1], 3, 12, eos_id=3)
        self.assertEqual(stopped.shape[1], 5)

    def test_rope_scaling_and_rotation(self):
        cos, sin = compute_rope_params(64, context_length=4,
                                       freq_config=self.cfg["rope_freq"])
        inverse = 10_000.0 ** (-np.arange(0, 64, 2) / 64)
        scaled = []
        for frequency in inverse:
            wavelength = 2 * np.pi / frequency
            if wavelength > 8192:
                frequency /= 32
            elif wavelength >= 2048:
                blend = (8192 / wavelength - 1) / 3
                frequency *= blend + (1 - blend) / 32
            scaled.append(frequency)
        np.testing.assert_allclose(cos[1, :32], np.cos(scaled), atol=1e-7)
        np.testing.assert_allclose(sin[1, :32], np.sin(scaled), atol=1e-7)
        x = jax.random.normal(jax.random.key(2), (2, 3, 4, 64))
        rotated = apply_rope(x, cos, sin)
        np.testing.assert_allclose(jnp.linalg.norm(x, axis=-1),
                                   jnp.linalg.norm(rotated, axis=-1), rtol=2e-7)
        np.testing.assert_array_equal(x[:, :, 0], rotated[:, :, 0])

    def test_bfloat16_and_generation(self):
        model = Llama3Model({**self.cfg, "dtype": jnp.bfloat16}, rngs=nnx.Rngs(0))
        self.assertEqual(model(self.ids).dtype, jnp.bfloat16)
        self.assertTrue(all(x.dtype == jnp.bfloat16 for x in jax.tree.leaves(nnx.state(model, nnx.Param))))
        greedy = generate(self.model, self.ids, 2, 4, top_k=1)
        sampled = generate(self.model, self.ids, 2, 4, top_k=1,
                           temperature=0.8, key=jax.random.key(7))
        self.assertEqual(greedy.shape, (2, 6))
        np.testing.assert_array_equal(greedy, sampled)
        with self.assertRaises(ValueError):
            generate(self.model, self.ids, 2, 4, temperature=1)

    def test_invalid_inputs(self):
        for ids in (jnp.ones((2, 3)), jnp.ones((2, 17), dtype=int),
                    jnp.ones((2, 0), dtype=int)):
            with self.assertRaises(ValueError):
                self.model(ids)
        with self.assertRaises(ValueError):
            Llama3Model({**self.cfg, "n_kv_groups": 3}, rngs=nnx.Rngs(0))


if __name__ == "__main__":
    unittest.main()
