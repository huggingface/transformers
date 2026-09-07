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
"""A grouped matmul for the MoE experts, tuned for the row counts decoding produces.

At decode the grouped GEMM gets a few rows per expert, so it reads every expert weight once to do almost no
arithmetic: it is a bandwidth problem wearing a GEMM's clothes, and it is 40% of a Qwen3-30B-A3B decode step at
tp4. `torch._grouped_mm` reaches 2.19 TB/s of 2.79 achievable on the gate/up projection and only 1.12 on the
down projection, which has half the bytes and takes the same time. Putting one program on each
(expert, N tile) and letting it walk that expert's rows gets 2.34 and 2.02 TB/s, 1.30x on the pair, because no
program is launched for a row block no expert fills and each weight tile is read by exactly one program.

The advantage is only there while the rows per expert stay few. Measured on both projections at 8, 16, 64 and
256 rows per expert, this kernel is 1.07x/1.82x, 1.05x/1.77x, 0.70x/1.35x and 0.39x/0.79x of the native op on
gate/up and down, so it is used below `_MAX_ROWS_PER_EXPERT` rows and the native op takes everything above,
which is every prefill and every training step. The input gradient is one native grouped matmul; the weight
gradient reduces over the rows the offsets group, which the native op cannot express, so it has a kernel of its
own. Training reaches neither at its own shapes, but a small-batch step reaches both.
"""

import os

import torch

from ..utils import is_triton_available


if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _grouped_gemm_kernel(a_ptr, b_ptr, c_ptr, offs_ptr, scatter_ptr, K, N, se, sk, sn,
                             HAS_SCATTER: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """`c[m] = a[m] @ b[e]` for every row `m` the offsets give to expert `e`, one program per (e, N tile).

        With `scatter_ptr`, row `m` is stored at `scatter_rows[m]` instead of at `m`, which is how the caller
        gets the expert outputs back in token order without a second pass over them.

        Rows past the last offset belong to expert-parallel sentinels. No program covers them, so they keep
        whatever the output was allocated with: what the native op leaves there when the caller masks it, and
        zero when the caller scatters.
        """
        e = tl.program_id(0)
        start = tl.where(e == 0, 0, tl.load(offs_ptr + e - 1))
        end = tl.load(offs_ptr + e)
        off_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        live_n = off_n < N
        b_base = b_ptr + e * se + off_n[None, :] * sn
        for m0 in range(start, end, BLOCK_M):
            off_m = m0 + tl.arange(0, BLOCK_M)
            live_m = off_m < end
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            a_row = a_ptr + off_m[:, None] * K
            for k0 in range(0, K, BLOCK_K):
                off_k = k0 + tl.arange(0, BLOCK_K)
                live_k = off_k < K
                a = tl.load(a_row + off_k[None, :], mask=live_m[:, None] & live_k[None, :], other=0.0)
                b = tl.load(b_base + off_k[:, None] * sk, mask=live_k[:, None] & live_n[None, :], other=0.0)
                acc = tl.dot(a, b, acc)
            row = tl.load(scatter_ptr + off_m, mask=live_m, other=0) if HAS_SCATTER else off_m
            tl.store(c_ptr + row[:, None] * N + off_n[None, :], acc.to(c_ptr.dtype.element_ty),
                     mask=live_m[:, None] & live_n[None, :])


    @triton.jit
    def _grouped_dw_kernel(a_ptr, dy_ptr, dw_ptr, offs_ptr, K, N, se, sk, sn,
                           BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
        """`dw[e] = a[rows of e].T @ dy[rows of e]`, one program per (expert, K tile, N tile).

        The native grouped matmul groups along the rows, so it cannot express a reduction over them: its 2-D
        form, which does, requires every group to be a multiple of 16 bytes wide and routing hands out whatever
        row counts it hands out.
        """
        e = tl.program_id(0)
        start = tl.where(e == 0, 0, tl.load(offs_ptr + e - 1))
        end = tl.load(offs_ptr + e)
        off_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        off_n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
        live_k, live_n = off_k < K, off_n < N
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        for m0 in range(start, end, BLOCK_M):
            off_m = m0 + tl.arange(0, BLOCK_M)
            live_m = off_m < end
            a = tl.load(a_ptr + off_m[:, None] * K + off_k[None, :],
                        mask=live_m[:, None] & live_k[None, :], other=0.0)
            dy = tl.load(dy_ptr + off_m[:, None] * N + off_n[None, :],
                         mask=live_m[:, None] & live_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a), dy, acc)
        tl.store(dw_ptr + e * se + off_k[:, None] * sk + off_n[None, :] * sn, acc.to(dw_ptr.dtype.element_ty),
                 mask=live_k[:, None] & live_n[None, :])


# above this many rows per expert on average the native op is faster, by a lot: it is a real GEMM once there
# is arithmetic to do, and one program per (expert, N tile) then serialises too much of the work
_MAX_ROWS_PER_EXPERT = 32

# rows per expert stay in the tens while decoding, and the N tile is what fills the machine: 128 wide, 8 warps
# and 4 pipeline stages are within 4% of the best of a six-config sweep on both projections at 1k and 2k rows
_BLOCK_M, _BLOCK_N, _BLOCK_K, _STAGES, _WARPS = 32, 128, 64, 4, 8


def _launch_dw(a: torch.Tensor, grad_out: torch.Tensor, offs: torch.Tensor, num_experts: int) -> torch.Tensor:
    k_in, n_out = a.shape[1], grad_out.shape[1]
    dw = torch.empty(num_experts, k_in, n_out, device=a.device, dtype=a.dtype)
    block_k, block_n = min(64, triton.next_power_of_2(k_in)), min(64, triton.next_power_of_2(n_out))
    _grouped_dw_kernel[(num_experts, triton.cdiv(k_in, block_k), triton.cdiv(n_out, block_n))](
        a, grad_out, dw, offs, k_in, n_out, dw.stride(0), dw.stride(1), dw.stride(2),
        BLOCK_M=32, BLOCK_K=block_k, BLOCK_N=block_n, num_stages=3, num_warps=4,
    )
    return dw


def _native_grouped_mm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, "grouped_mm"):
        return torch.nn.functional.grouped_mm(a, b, offs=offs)
    return torch._grouped_mm(a, b, offs=offs)


def _launch(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor, scatter_rows: torch.Tensor | None):
    num_experts, k_in, n_out = b.shape
    # scattering leaves the sentinel rows unwritten, and the caller reads them, so they start at zero
    new = torch.zeros if scatter_rows is not None else torch.empty
    out = new(a.shape[0], n_out, device=a.device, dtype=a.dtype)
    block_n = min(_BLOCK_N, triton.next_power_of_2(n_out))
    _grouped_gemm_kernel[(num_experts, triton.cdiv(n_out, block_n))](
        a, b, out, offs, scatter_rows, k_in, n_out, b.stride(0), b.stride(1), b.stride(2),
        HAS_SCATTER=scatter_rows is not None,
        BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=_BLOCK_K, num_stages=_STAGES, num_warps=_WARPS,
    )
    return out


class _TritonGroupedMM(torch.autograd.Function):
    """The Triton kernel forward, the native grouped matmul backward."""

    @staticmethod
    def forward(ctx, a, b, offs, scatter_rows):
        ctx.save_for_backward(a, b, offs, scatter_rows)
        return _launch(a, b, offs, scatter_rows)

    @staticmethod
    def backward(ctx, grad_out):
        a, b, offs, scatter_rows = ctx.saved_tensors
        if scatter_rows is not None:  # undo the scatter: row m took its gradient from where it was stored
            grad_out = grad_out[scatter_rows.long()]
        grad_out = grad_out.contiguous()
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            # (S, N) x (E, N, K) grouped along S -> (S, K)
            grad_a = _native_grouped_mm(grad_out, b.transpose(-2, -1), offs=offs)
        if ctx.needs_input_grad[1]:
            grad_b = _launch_dw(a.contiguous(), grad_out, offs, b.shape[0])
        return grad_a, grad_b, None, None


def triton_grouped_mm_available(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether to route this grouped matmul through the Triton kernel.

    Opt-in, and only for the shapes and layout the kernel is written for: few rows per expert, a 2-D
    activation, 3-D weights whose reduction dimension is contiguous, and no fp8.
    """
    return (
        os.environ.get("HF_MOE_TRITON_GROUPED_MM") == "1"
        and is_triton_available()
        and a.is_cuda
        and a.dim() == 2
        and b.dim() == 3
        and a.dtype == b.dtype
        and a.dtype in (torch.bfloat16, torch.float16)
        and a.stride(1) == 1
        and 1 in (b.stride(1), b.stride(2))
        and a.shape[0] <= _MAX_ROWS_PER_EXPERT * b.shape[0]
    )


def triton_grouped_mm(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor, scatter_rows: torch.Tensor | None = None
) -> torch.Tensor:
    """What `torch._grouped_mm(a, b, offs=offs)` returns, on the Triton kernel.

    With `scatter_rows`, row `m` of the result lands at row `scatter_rows[m]` and every row no expert owns
    stays zero, so a caller that would otherwise permute the result afterwards does not have to.
    """
    return _TritonGroupedMM.apply(a, b, offs, scatter_rows)
