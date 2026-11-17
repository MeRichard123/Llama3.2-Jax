"""Weight-only quantisers: callable(kernel) -> array or callable NNX module.

Kernels have shape (input_features, output_features). Four-bit weights use
grouped symmetric INT4 with 128 input weights per output-column group by
default. Reference inference dequantises for BF16
multiplication; 
Ternary uses five base-3 digits per byte (1.6 stored bits/weight, plus scales),
not BitNet training or a guarantee of useful post-training ternary accuracy.

Usage::

    from generate import load_model
    model = load_model('model.safetensors', quantisation='4')
    model = load_model('model.safetensors', quantisation=quantise_ternary)

    from functools import partial
    model = load_model('model.safetensors',
                       quantisation=partial(quantise_4bit, group_size=64))

Custom functions receive each transposed linear kernel and return either a
same-shaped floating array (stored in the model dtype), or a callable NNX module
that owns its compressed state. Add a name to QUANTISATIONS to expose it in the
CLI. Embeddings and normalization parameters are not quantised.
"""
import jax.numpy as jnp
from flax import nnx
from kernels.MatMulKernels import quantised_matmul

class QuantisedLinear(nnx.Module):
    """Packed weights with grouped INT4 or per-column INT8/ternary scales.

    INT4 scale[g, n] covers kernel[g*group_size:(g+1)*group_size, n].
    Storage remains flattened [K, N], low nibble first. Codes are q + 7,
    with q in [-7, 7]. 
    """

    def __init__(self, kernel, bits, *, group_size=128):
        if bits not in (4, 8, '1.58'):
            raise ValueError('bits must be 4, 8, or \'1.58\'')
        if kernel.ndim != 2 or min(kernel.shape) <= 0:
            raise ValueError('kernel must be a nonempty rank-2 array')
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
            raise ValueError('group_size must be a positive integer')
        self.shape = tuple(kernel.shape)
        self.bits = bits
        self.group_size = group_size
        # Quantisation statistics are computed in FP32.
        weight = kernel.astype(jnp.float32)
        if bits == 4:
            # grouped symmetric INT4 because NF4 uses a table and I dont understand it 
            k, n = self.shape
            padding = (-k) % group_size
            grouped = jnp.pad(weight, ((0, padding), (0, 0))).reshape(-1, group_size, n)
            scale = jnp.max(jnp.abs(grouped), axis=1) / 7.0
            scale = jnp.where(scale > 0, scale, 1.0)
            codes = (jnp.clip(jnp.round(grouped / scale[:, None, :]), -7, 7) + 7)
            # Only the original K rows are stored; a partial group needs no extra weights.
            codes = codes.reshape(-1, n)[:k].astype(jnp.uint8)
            group = 2
        elif bits == '1.58':
            # ternary range [-1, 1] 
            scale = jnp.mean(jnp.abs(weight), axis=0, keepdims=True)
            scale = jnp.where(scale > 0, scale, 1.0)
            codes = (jnp.clip(jnp.round(weight / scale), -1, 1) + 1).astype(jnp.uint8)
            group = 5
        else:
            # scaled to a symmetric range of [-limit, limit]
            limit = 2 ** (bits - 1) - 1
            # dequant scales 
            scale = jnp.max(jnp.abs(weight), axis=0, keepdims=True) / limit
            scale = jnp.where(scale > 0, scale, 1.0)
            # integer quantised values, one per weight 
            # + limit shifts [-limit, limit] into [0, 2*limit] for unsigned packing
            codes = (jnp.clip(jnp.round(weight / scale), -limit, limit) + limit).astype(jnp.uint8)
            group = 8 // bits # how many codes per quantisation e.g 2 codes for 4 bits 
        # flatten in row major order and pad for multiple groups
        flat = codes.reshape(-1)
        flat = jnp.pad(flat, (0, (-flat.size) % group)).reshape(-1, group)
        if bits == '1.58':
            # stored weights are bit-packed into  
            packed = jnp.sum(flat.astype(jnp.uint16) * jnp.array([1, 3, 9, 27, 81], jnp.uint16), axis=1)
        else:
            packed = jnp.sum(flat.astype(jnp.uint16) << (jnp.arange(group, dtype=jnp.uint16) * bits), axis=1)
        self.packed = nnx.Param(packed.astype(jnp.uint8))
        self.scale = nnx.Param(scale)

    def dequantise(self):
        data = self.packed.value.astype(jnp.uint16)[:, None]
        if self.bits == '1.58':
            codes = (data // jnp.array([1, 3, 9, 27, 81], jnp.uint16)) % 3
            offset = 1
        else:
            codes = (data >> (jnp.arange(8 // self.bits, dtype=jnp.uint16) * self.bits)) & (2 ** self.bits - 1)
            offset = 2 ** (self.bits - 1) - 1
        count = self.shape[0] * self.shape[1]
        weight = codes.reshape(-1)[:count].reshape(self.shape).astype(jnp.float32) - offset
        if self.bits == 4:
            scales = self.scale.value[jnp.arange(self.shape[0]) // self.group_size]
        else:
            scales = self.scale.value
        return (weight * scales).astype(jnp.bfloat16)

    def __call__(self, x):
        if self.bits == 4:
            return quantised_matmul(x, self.packed.value, self.scale.value,
                                    weight_shape=self.shape, bits=4,
                                    group_size=self.group_size)
        return jnp.matmul(x.astype(jnp.bfloat16), self.dequantise())


def quantise_4bit(kernel, *, group_size=64):
    return QuantisedLinear(kernel, 4, group_size=group_size)
def quantise_8bit(kernel):
    return QuantisedLinear(kernel, 8)
def quantise_bf16(kernel):
    return kernel.astype(jnp.bfloat16)
def quantise_ternary(kernel):
    return QuantisedLinear(kernel, '1.58')


QUANTISATIONS = {'4': quantise_4bit, '8': quantise_8bit,
                 'bf16': quantise_bf16, '1.58': quantise_ternary}


def resolve_quantisation(quantisation):
    """Accept None, a registered name, or a custom kernel callable."""
    if quantisation is None or callable(quantisation):
        return quantisation
    if isinstance(quantisation, str) and quantisation in QUANTISATIONS:
        return QUANTISATIONS[quantisation]
    raise ValueError(f'Unknown quantisation {quantisation!r}; choose {tuple(QUANTISATIONS)} or a callable')
