# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton kernel: ``out[s] = x[s] @ dequantize(w[rhs[s]]).T`` for affine-quantized expert banks (CUDA).

The CUDA counterpart of MLX's ``gather_qmm`` on the `MetalExperts` layout (codes LSB-first in uint32 words, per-group
scales and biases), so one quantized checkpoint runs on both. Fused gather + dequantize + matrix-vector product: each
program handles one row of ``x`` and a block of output features and reads only the packed bytes of its expert.
From tarekziade/moe-expert-streaming (Apache-2.0).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_qmv_kernel(
    x_ptr,  # [S, K] activations
    w_ptr,  # [E, N, KW] int32 (uint32 bits)
    s_ptr,  # [E, N, G] scales
    b_ptr,  # [E, N, G] biases
    rhs_ptr,  # [S] expert of each row
    out_ptr,  # [S, N] float32
    N,
    K,
    KW,
    w_se,  # strides (elements) of the expert and row dimensions; the last dimension is contiguous
    w_sn,
    s_se,
    s_sn,
    b_se,
    b_sn,
    BITS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < N
    expert = tl.load(rhs_ptr + row).to(tl.int64)
    rows64 = n[:, None].to(tl.int64)
    w_rows = w_ptr + expert * w_se + rows64 * w_sn
    s_rows = s_ptr + expert * s_se + rows64 * s_sn
    b_rows = b_ptr + expert * b_se + rows64 * b_sn
    code_mask = (1 << BITS) - 1
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        mask = n_mask[:, None] & k_mask[None, :]
        offset = k * BITS
        word = offset // 32
        shift = (offset % 32).to(tl.uint32)
        lo = tl.load(w_rows + word[None, :], mask=mask, other=0).to(tl.uint32, bitcast=True)
        straddles = (shift + BITS) > 32
        next_word = tl.minimum(word + 1, KW - 1)
        hi = tl.load(w_rows + next_word[None, :], mask=mask & straddles[None, :], other=0).to(tl.uint32, bitcast=True)
        # A shift by 32 is undefined: shift by (32 - shift) & 31 and keep it only where the code straddles two words.
        hi = tl.where(straddles[None, :], hi << ((32 - shift) & 31)[None, :], 0)
        code = ((lo >> shift[None, :]) | hi) & code_mask
        group = k // GROUP
        scale = tl.load(s_rows + group[None, :], mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(b_rows + group[None, :], mask=mask, other=0.0).to(tl.float32)
        xk = tl.load(x_ptr + row.to(tl.int64) * K + k, mask=k_mask, other=0.0).to(tl.float32)
        acc += tl.sum((code.to(tl.float32) * scale + bias) * xk[None, :], axis=1)
    tl.store(out_ptr + row.to(tl.int64) * N + n, acc, mask=n_mask)


def gather_qmv(x, w, scales, biases, rhs, group_size, bits, block_n=32, block_k=128):
    """``x [S, K]`` (any float dtype), ``w [E, N, K*bits/32]`` uint32/int32, ``scales``/``biases [E, N, K/group]``,
    ``rhs [S]`` -> ``[S, N]`` in ``x``'s dtype."""
    rows, k = x.shape
    _, n, kw = w.shape
    # The banks are usually strided views into the expert cache's slot buffer: never copy them.
    w = w.view(torch.int32)
    for bank in (w, scales, biases):
        if bank.stride(-1) != 1:
            raise ValueError("gather_qmv: the last dimension of the banks must be contiguous")
    x = x.contiguous()
    rhs = rhs.reshape(-1).contiguous()
    out = torch.empty(rows, n, dtype=torch.float32, device=x.device)
    grid = (rows, triton.cdiv(n, block_n))
    _gather_qmv_kernel[grid](
        x, w, scales, biases, rhs, out, n, k, kw,
        w.stride(0), w.stride(1), scales.stride(0), scales.stride(1), biases.stride(0), biases.stride(1),
        BITS=bits, GROUP=group_size, BLOCK_N=block_n, BLOCK_K=block_k,
    )  # fmt: skip
    return out.to(x.dtype)
