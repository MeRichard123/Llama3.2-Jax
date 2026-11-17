"""Tiled forward Flash Attention for JAX, without a quadratic score buffer."""

import math

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def _flash_attention_kernel(
    Q, K, V, Mask, KeyLength, Out,
    HQ: tl.constexpr, HK: tl.constexpr, # number of query heads and kv heads
    NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr, # number of queries and keys
    SCALE: tl.constexpr, CAUSAL: tl.constexpr, OFFSET: tl.constexpr, # scale = 1/sqrt(D)
    HAS_MASK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, # tile sizes
):
    # 2D launch grid, axis 0 indexes the block of BM
    tile = tl.program_id(0)
    # Axis 1 indexes the batch head pair
    bh = tl.program_id(1) 

    # bh = batch * HQ + head so we recover them:
    batch = bh // HQ
    head = bh % HQ

    # Compute the kv head
    kv_head = head // (HQ // HK)

    # compute tile indices 
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    cols = tl.arange(0, BN)

    # Q is stored as [B*HQ, NQ, D] so we index using (bh*NQ + row)*D + dim
    # rows[:, None] is [BM, 1] and dims[None, :] is [1, BD] 
    # broadcasting them gives [BM, BD] grid of addresses 
    q = tl.load(Q + (bh * NQ + rows[:, None]) * D + dims[None, :],
                (rows[:, None] < NQ) & (dims[None, :] < D), other=0)
    # q is [BM, BD] and stays in the registers 
    kv_base = (batch * HK + kv_head) * NK * D

    # running state: storing the max
    # denominator Σexp(score − maximum) so far (ℓ).
    # accumulator Σexp(score − maximum)·v so far, unnormalised (O)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, BD), tl.float32)

    # ensure causal masking 
    valid_length = tl.minimum(NK, tl.maximum(0, tl.load(KeyLength)))
    end = valid_length
    if CAUSAL:
        end = tl.minimum(valid_length, (tile + 1) * BM + OFFSET)
    # loop over key blocks 
    for start in range(0, end, BN):
        keys = start + cols # [BN] vector
        # Transposed Load 
        k = tl.load(K + kv_base + keys[None, :] * D + dims[:, None],
                    (keys[None, :] < valid_length) & (dims[:, None] < D), other=0)
        # [BM, BD] @ [BD, BN] → [BM, BN]. 
        # Entry (i, j) is q_i · k_j, the raw attention score for query i and key j. 
        # 'ieee' turns off TF32 so fp32 inputs get true fp32 products.
        # The scale is applied afterwards.
        scores = tl.dot(q, k, input_precision='ieee') * SCALE
        # apply the mask 
        valid = (rows[:, None] < NQ) & (keys[None, :] < valid_length)
        if CAUSAL:
            valid = valid & (keys[None, :] <= rows[:, None] + OFFSET)
        if HAS_MASK:
            blocked = tl.load(Mask + rows[:, None] * NK + keys[None, :],
                              (rows[:, None] < NQ) & (keys[None, :] < NK), other=1)
            valid = valid & (blocked == 0)
        scores = tl.where(valid, scores, -float('inf'))
        # online softmax 
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # Fully masked rows must remain zero, including before their first key.
        safe_maximum = tl.where(new_maximum == -float('inf'), 0.0, new_maximum)
        # correction factor alpha
        alpha = tl.exp(maximum - safe_maximum)
        # uncompute unnormalised attention weights 
        probabilities = tl.exp(scores - safe_maximum[:, None]) 
        # rescale the old sum to the new max 
        denominator = denominator * alpha + tl.sum(probabilities, axis=1)
        # load the values 
        v = tl.load(V + kv_base + keys[:, None] * D + dims[None, :],
                    (keys[:, None] < valid_length) & (dims[None, :] < D), other=0)
        # accumulate the scores and rescale 
        accumulator = accumulator * alpha[:, None]
        accumulator = tl.dot(probabilities.to(q.dtype), v, accumulator,
                             input_precision='ieee')
        maximum = new_maximum
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(Out + (bh * NQ + rows[:, None]) * D + dims[None, :], result,
             (rows[:, None] < NQ) & (dims[None, :] < D))


def flash_attention(q, k, v, mask=None, *, causal=True, scale=None, query_offset=0,
                    key_length=None):
    """Return scaled dot-product attention in q's shape and dtype.

    Inputs are [batch, heads, sequence, head_dim] FP16, BF16 or FP32 arrays.
    K and V share a shape; query heads must be a multiple of KV heads.
    ``mask`` is an optional [query_length, key_length] boolean array where
    True blocks attention (the convention used by model.py).
    Causality permits key indices <= query index + query_offset. For cached
    decoding, pass the absolute position of the first query as query_offset.
    Fully masked rows return zero. Scale defaults to 1/sqrt(head_dim).
    ``key_length`` is an optional integer scalar (dynamic under jax.jit).
    Only that prefix of K/V is read; unused cache tiles are skipped entirely.

    Requires a Triton-compatible GPU. This is a forward-only inference
    operation; causal, scale and query_offset must be static under jax.jit.
    """
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError('q, k and v must have rank 4: [batch, heads, sequence, head_dim]')
    batch, heads, nq, dim = q.shape
    if min(q.shape) <= 0 or min(k.shape) <= 0:
        raise ValueError('Attention dimensions must be positive')
    if k.shape != v.shape or k.shape[0] != batch or k.shape[3] != dim:
        raise ValueError('k and v must share a shape and match q batch and head_dim')
    if heads % k.shape[1]:
        raise ValueError('Query heads must be a multiple of KV heads')
    if q.dtype not in (jnp.float16, jnp.bfloat16, jnp.float32) or any(
        x.dtype != q.dtype for x in (k, v)
    ):
        raise ValueError('q, k and v must share an FP16, BF16 or FP32 dtype')
    if dim > 256:
        raise ValueError('head_dim must be at most 256')
    if not isinstance(causal, bool):
        raise ValueError('causal must be a static boolean')
    if not isinstance(query_offset, int) or isinstance(query_offset, bool) or query_offset < 0:
        raise ValueError('query_offset must be a nonnegative static integer')
    nk = k.shape[2]
    key_length = jnp.asarray(nk if key_length is None else key_length)
    if key_length.ndim != 0 or not jnp.issubdtype(key_length.dtype, jnp.integer):
        raise ValueError('key_length must be an integer scalar')
    key_length = key_length.astype(jnp.int32)
    if mask is not None and (mask.shape != (nq, nk) or mask.dtype != jnp.bool_):
        raise ValueError('mask must be boolean with shape [query_length, key_length]')
    scale = dim ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError('scale must be finite')
    return jt.triton_call(
        q, k, v, mask if mask is not None else q, key_length,
        kernel=_flash_attention_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(triton.cdiv(nq, 64), batch * heads),
        HQ=heads, HK=k.shape[1], NQ=nq, NK=nk, D=dim,
        SCALE=scale, CAUSAL=causal, OFFSET=query_offset, HAS_MASK=mask is not None,
        BM=64, BN=64, BD=max(16, triton.next_power_of_2(dim)), num_warps=4,
    )
