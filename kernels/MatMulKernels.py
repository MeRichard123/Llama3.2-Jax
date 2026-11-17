"""JAX launch wrapper and fused W4A16 matrix multiplication.

Weights use quantisation.py's flat, low-nibble-first packing and FP32
scales of shape [ceil(K / group_size), N]. The kernel reconstructs BF16 tiles, never a full
floating-point weight matrix.
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

"""
--------------
M   | Token rows being processed
K   | Input features per token
N   | Output Features per token
-------------
Out[M x N] = x[M, K] @ W[K, N]
- We split into tiles to split the matmul accross GPU Programs 
- So an M x N matrix is tiled into:
    - 128 (BLOCK_M) x 64 (BLOCK_N) activation tile
    - 64 x 64 (BLOCK_K) weight tile

Note that W is not stored, it is quantised so it is reconstructed from:
    - Packed: 4-bit codes, two per byte (K*N/2 bytes total)
    - scale: one float per (K-group, column), shape [K/GROUP_SIZE, N]
- Weights are rebuilt in the K-lop one tile at a time

byte = [ high nibble | low nibble ]
            4 bits        4 bits
- One byte = 2 4 bit codes 
byte = low + high * 16 # x 16 slides a number up by 4 bits
"""
@triton.jit
def _matmul_4bit_kernel(
    x_ptr, packed_ptr, scale_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # the encoder turns [K, N] float weights into: 
    #   a packed uint8 array each byte holding 4-bit codes
    #   a scale per K-group and output column 
    # the kernel needs to reconstruct on the fly without materialising the full [K, N]
    
    # compute 2D tiling indices using the number of tiles and the current pid 
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, BLOCK_N) # N // BLOCK_N, each program does one tile
    rows = (pid // tiles_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = (pid % tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulate the mat mul results into FP32 so we do not lose precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # accumulate over the reduction dim (K) in chunks of BLOCK_K 
    for block in range(tl.cdiv(K, BLOCK_K)):
        ks = block * BLOCK_K + tl.arange(0, BLOCK_K)
        # load the activations - these are not quantised (W4A16)
        # row-major indexing, other = 0 ingores out of bounds values 
        # eahc iteration handles one slab of K, with ks being the global K indices
        activations = tl.load(
            x_ptr + rows[:, None] * K + ks[None, :],
            mask=(rows[:, None] < M) & (ks[None, :] < K), other=0,
        ).to(tl.bfloat16) #(A16)
        # mask for the in-bound values for the weight tile
        valid = (ks[:, None] < K) & (cols[None, :] < N)
        if N % 8 == 0: # Fast branch, because the row stride has to be a whole number of words 
            # Model projection widths have complete 32-bit words (8 weights).
            # CUDA allocations are aligned; each row and tile also starts on a
            # word boundary. Explicit word loads avoid relying on byte-load 
            # vectorization across the unpacking layout conversion.
            
            # the byte array is reinterpreted as uint32 
            # each 32 bit word holds 8 weights, so a row of N weights is N/8 words
            # We load a [BLOCK_K, BLOCK_N/8] tile which is 8x fewer elements than weights
            # We ensure the loads are wide aligned
            # this means the address is aligned to n bytes lets say 8 
            word_cols = ((pid % tiles_n) * (BLOCK_N // 8)
                         + tl.arange(0, BLOCK_N // 8))
            # casting to uint32* means we now move in 4-byte units 
            words = tl.load(
                packed_ptr.to(tl.pointer_type(tl.uint32))
                + ks[:, None] * (N // 8) + word_cols[None, :],
                mask=(ks[:, None] < K) & (word_cols[None, :] < N // 8),
                other=0,
            ) # [BK, BN/8, 1]
            shifts = tl.arange(0, 8) * 4 
            # >> shifts: broadcast to [BK, BN/8, 8], where nibble j is shifed down by 4j bits
            #                                    ^
            #                                    j
            # & 15: masks the low 4 bits, giving codes 0 to 15
            #       15 is 0000 1111 so the mask removes the high nibble
            # and reshape to [BK, BN] merges the last two dims word W and nibble j 
            # this becomes 8w + j 
            codes = tl.reshape(
                (words[:, :, None] >> shifts[None, None, :]) & 15,
                (BLOCK_K, BLOCK_N),
            )
        elif N % 2 == 0:
            # Each row starts on a byte boundary. Load each packed byte once
            # along a contiguous half-width tile, then restore column order.
            # Unlike per-weight byte indexing, this exposes contiguous accesses
            # to Triton's load vectorizer without changing the stored format.

            # Load BLOCK_N/2 bytes and then split into low nibbles (even columns) and high nibbles (odd columns) 
            byte_cols = ((pid % tiles_n) * (BLOCK_N // 2)
                         + tl.arange(0, BLOCK_N // 2))
            packed = tl.load(
                packed_ptr + ks[:, None] * (N // 2) + byte_cols[None, :],
                mask=(ks[:, None] < K) & (byte_cols[None, :] < N // 2),
                other=0,
            )
            # interleave the odd and even columns to restore their order
            #   (byte >> 4) & 15 slide it all right 4 bits so the top nibble drops to the bottom
            #       and then & 15 to mask out the high nibble
            codes = tl.interleave(packed & 15, (packed >> 4) & 15)
        else:
            # Odd-width rows share boundary bytes, and alternate their starting
            # nibble. Preserve flat indexing for these uncommon shapes.
            indices = ks[:, None] * N + cols[None, :]
            packed = tl.load(packed_ptr + indices // 2, mask=valid, other=0)
            codes = (packed.to(tl.int32) >> (4 * (indices % 2))) & 15
        # Each K row selects its input group, independently of tile boundaries.
        # this is group wise quantisation, we share scales per column
        # ks // GRPO_SIZE mapes each K row to its group and scale is laid out [K/GROUP_SIZE, N]
        scales = tl.load(scale_ptr + (ks[:, None] // GROUP_SIZE) * N + cols[None, :],
                         mask=valid, other=0)
        # dequantisation: shift codes by a zero point of 7, giving a range of [-7, +8] 
        # this is the inverse of the encoder: code = round(w/scale) + 7
        weights = (codes.to(tl.float32) - 7.0) * scales
        # A masked byte of zero decodes to -7, so mask after decoding too.
        # Out-of-bounds positions contribute 0 to we make sure those are filtered out just in case
        weights = tl.where(valid, weights, 0.0).to(tl.bfloat16)
        # and finally accumulate ACC @ W 
        acc = tl.dot(activations, weights, acc)

    # store in row major order 
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def quantised_matmul(x, packed, scale, weight_shape, bits, *, group_size=128):
    """Multiply [..., K] by packed [K, N] weights, returning BF16 [..., N].

    Requires a GPU supported by Triton with BF16 dot support. Only 4-bit
    weights are implemented here; other formats retain the reference path
    in QuantisedLinear.
    """
    if bits != 4:
        raise NotImplementedError('The Triton kernel currently supports only 4-bit weights')
    if len(weight_shape) != 2 or min(weight_shape) <= 0:
        raise ValueError('weight_shape must contain two positive dimensions')
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError('group_size must be a positive integer')
    K, N = weight_shape
    if x.ndim < 1 or x.shape[-1] != K:
        raise ValueError(f'Expected activations with last dimension {K}, got {x.shape}')
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise ValueError('Activations must have a floating-point dtype')
    if packed.shape != ((K * N + 1) // 2,) or packed.dtype != jnp.uint8:
        raise ValueError('Expected flat uint8 weights with ceil(K * N / 2) bytes')
    scale_shape = ((K + group_size - 1) // group_size, N)
    if scale.shape != scale_shape or scale.dtype != jnp.float32:
        raise ValueError(f'Expected FP32 grouped scales with shape {scale_shape}')

    output_shape = x.shape[:-1] + (N,)
    M = x.size // K
    if M == 0:
        return jnp.empty(output_shape, dtype=jnp.bfloat16)
    block_m, block_n, block_k = 32, 64, 32
    num_stages = 3
    # RTX 3060 prefill sweep (M=512, group_size=64). Use the larger
    # row tile for full prefill tiles, preserving the decode configuration.
    if M >= 128 and group_size == 64:
        if (K, N) == (2048, 8192):
            block_m, block_n, block_k = 128, 64, 64
            num_stages = 2
        elif (K, N) == (8192, 2048):
            block_m, block_n, block_k = 128, 64, 32
    out = jt.triton_call(
        x.astype(jnp.bfloat16).reshape(M, K), packed, scale,
        kernel=_matmul_4bit_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(triton.cdiv(M, block_m) * triton.cdiv(N, block_n),),
        M=M, N=N, K=K, GROUP_SIZE=group_size,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
        num_stages=num_stages,
    )
    return out.reshape(output_shape)
