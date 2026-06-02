"""Fused row-shared-dropout + residual-add kernel for the pair stream.

For the pairformer pattern:

    pair_new = pair + Dropout(r, batch_dim=1)(delta)

where `delta` is the output of e.g. a triangle multiplication, the row-shared
dropout multiplies `delta` by a Bernoulli mask of shape `[B, 1, N_col, D]`
(broadcast over the row dim) scaled by `1/(1-r)`. Naively this materializes
two intermediate `[B, N, N, D]` tensors (one for `delta * mask`, one for
`pair + ...`) — three full HBM round-trips of the pair tensor.

This kernel reads `pair`, `delta`, and the small `[N_col, D]` shared mask
(via modulo-`N_col` indexing — no broadcast materialization) and writes the
combined `pair + delta * mask` once.

Backward is also a single Triton kernel: it mutates the saved mask buffer in
place to produce `ddelta = dout * mask`, and the residual gradient passes
through unchanged.

Convention: the mask is expected to already be scaled (i.e. it's the output
of `nn.Dropout(r)(ones_like(...))` so that retained entries have value
`1/(1-r)`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_dropout_residual_fwd_kernel(
    pair_ptr,  # [M, D]            M = B*N_row*N_col
    delta_ptr,  # [M, D]
    mask_ptr,  # [B*N_col, D]      per-batch row-shared mask
    out_ptr,  # [M, D]
    M,
    D: tl.constexpr,
    N_COL: tl.constexpr,
    STRIDE_B: tl.constexpr,  # = N_row * N_col, used to recover batch index
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mm_m = offs_m < M
    mm_d = offs_d < D

    # m = b*N_row*N_col + i*N_col + j  →  b = m // (N_row*N_col), j = m % N_col
    # mask is laid out as [B*N_col, D] so per-batch index is b*N_col + j.
    b = offs_m // STRIDE_B
    j = offs_m % N_COL
    mask_row = b * N_COL + j

    pair_ptrs = pair_ptr + offs_m[:, None] * D + offs_d[None, :]
    delta_ptrs = delta_ptr + offs_m[:, None] * D + offs_d[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * D + offs_d[None, :]
    mask_ptrs = mask_ptr + mask_row[:, None] * D + offs_d[None, :]

    p = tl.load(pair_ptrs, mask=mm_m[:, None] & mm_d[None, :], other=0.0)
    de = tl.load(delta_ptrs, mask=mm_m[:, None] & mm_d[None, :], other=0.0)
    mk = tl.load(mask_ptrs, mask=mm_m[:, None] & mm_d[None, :], other=0.0)

    o = p + de * mk
    tl.store(
        out_ptrs, o.to(out_ptr.type.element_ty), mask=mm_m[:, None] & mm_d[None, :]
    )


@triton.jit
def _fused_dropout_residual_bwd_kernel(
    dout_ptr,  # [M, D]
    mask_ptr,  # [B*N_col, D]
    ddelta_ptr,  # [M, D]
    M,
    D: tl.constexpr,
    N_COL: tl.constexpr,
    STRIDE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mm_m = offs_m < M
    mm_d = offs_d < D

    b = offs_m // STRIDE_B
    j = offs_m % N_COL
    mask_row = b * N_COL + j
    mask_ptrs = mask_ptr + mask_row[:, None] * D + offs_d[None, :]

    do = tl.load(
        dout_ptr + offs_m[:, None] * D + offs_d[None, :],
        mask=mm_m[:, None] & mm_d[None, :],
        other=0.0,
    )
    mk = tl.load(mask_ptrs, mask=mm_m[:, None] & mm_d[None, :], other=0.0)
    tl.store(
        ddelta_ptr + offs_m[:, None] * D + offs_d[None, :],
        (do * mk).to(ddelta_ptr.type.element_ty),
        mask=mm_m[:, None] & mm_d[None, :],
    )


def _fused_dropout_residual_fwd(
    pair_2d: torch.Tensor,
    delta_2d: torch.Tensor,
    mask_2d: torch.Tensor,
    n_row: int,
    n_col: int,
) -> torch.Tensor:
    M, D = pair_2d.shape
    out = torch.empty_like(pair_2d)
    BLOCK_M = 64
    BLOCK_D = min(128, _next_pow2(D))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
    _fused_dropout_residual_fwd_kernel[grid](
        pair_2d,
        delta_2d,
        mask_2d,
        out,
        M,
        D,
        n_col,
        n_row * n_col,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return out


def _fused_dropout_residual_bwd(
    dout_2d: torch.Tensor, mask_2d: torch.Tensor, n_row: int, n_col: int
) -> torch.Tensor:
    M, D = dout_2d.shape
    ddelta = torch.empty_like(dout_2d)
    BLOCK_M = 64
    BLOCK_D = min(128, _next_pow2(D))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
    _fused_dropout_residual_bwd_kernel[grid](
        dout_2d,
        mask_2d,
        ddelta,
        M,
        D,
        n_col,
        n_row * n_col,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return ddelta


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class FusedDropoutResidualFn(torch.autograd.Function):
    """`pair_out = pair + delta * mask` fused into one kernel.

    Args:
      pair:  [B, N_row, N_col, D]
      delta: [B, N_row, N_col, D]
      mask:  [B, 1, N_col, D]  or  [1, 1, N_col, D]   — already scaled to 1/(1-r) for kept entries

    The mask is row-shared (size 1 along `N_row`) — we read it directly with
    modular indexing rather than broadcasting and materializing a full [B, N, N, D]
    copy.
    """

    @staticmethod
    def forward(ctx, pair, delta, mask):
        in_shape = pair.shape  # [B, N_row, N_col, D]
        N_row = in_shape[-3]
        N_col = in_shape[-2]
        D = in_shape[-1]
        pair_2d = pair.contiguous().view(-1, D)
        delta_2d = delta.contiguous().view(-1, D)
        # [B, 1, N_col, D] → [B*N_col, D]; kernel indexes b*N_col + (m % N_col)
        # so each batch sample uses its own draw.
        mask_2d = mask.contiguous().view(-1, D)
        out_2d = _fused_dropout_residual_fwd(pair_2d, delta_2d, mask_2d, N_row, N_col)
        ctx.save_for_backward(mask_2d)
        ctx.in_shape = in_shape
        ctx.n_row = N_row
        ctx.n_col = N_col
        return out_2d.view(in_shape)

    @staticmethod
    def backward(ctx, dout):
        (mask_2d,) = ctx.saved_tensors
        D = ctx.in_shape[-1]
        dout_2d = dout.contiguous().view(-1, D)
        ddelta_2d = _fused_dropout_residual_bwd(dout_2d, mask_2d, ctx.n_row, ctx.n_col)
        # Residual: gradient passes through. Mask: not differentiable.
        return dout, ddelta_2d.view(ctx.in_shape), None


class FusedDropoutResidual(nn.Module):
    """Fused module for the pattern `pair + Dropout(r, batch_dim=1)(delta)`.

    The kernel only supports row-shared dropout — i.e. a `[B, 1, N_col, D]`
    mask broadcast over the row axis (dim 1) — so this module hardcodes that
    layout. The `j = m % N_col` indexing in the Triton kernel would read out
    of bounds with any other sharing pattern, so we don't expose `batch_dim`
    as a parameter. Use plain `Dropout(r, batch_dim=2)` for col-shared.

    Usage in a pairformer block:

        # Before
        pair = pair + self.row_drop(self.tri_mul_out(pair, mask=...))

        # After
        pair = self.row_drop(pair, self.tri_mul_out(pair, mask=...))

    where `self.row_drop` is `FusedDropoutResidual(r)`.
    """

    def __init__(self, r: float):
        super().__init__()
        self.r = r

    def forward(self, pair: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if not self.training or self.r == 0.0:
            return pair + delta
        # Use delta.dtype: F.dropout's cuRAND draw depends on input dtype, so
        # building from pair (fp32) vs delta (bf16 under autocast) breaks
        # same-seed parity with the unfused `pair + Dropout(delta)` path.
        shape = list(pair.shape)
        shape[1] = 1  # row-shared mask: [B, 1, N_col, D]
        ones = delta.new_ones(shape)
        mask = torch.nn.functional.dropout(ones, p=self.r, training=True)
        return FusedDropoutResidualFn.apply(pair, delta, mask)  # type: ignore[return-value]
