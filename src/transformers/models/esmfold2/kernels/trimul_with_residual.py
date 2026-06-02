"""TriMul with output residual + dropout-mask epilogue fused into the final GEMM.

Inspired by cuequivariance's ``triangle_multiplicative_update``
(https://docs.nvidia.com/cuda/cuequivariance/index.html); independently
re-implemented in Triton with the output residual and dropout mask folded
into the final gated GEMM so the ``delta = TriMul(pair)`` intermediate is
never written to HBM.

Replaces the standard pattern

    pair_new = pair + dropout_mask * triangle_multiplicative_update(pair)

— where the two ops materialize the full ``delta = TriMul(pair)`` tensor
between them — with a single fused path. The final gated GEMM fuses three
operations end-to-end:

    1. Computes ``delta = sigmoid(x_in @ Wg) * (x_out @ Wp)`` per output tile.
    2. Multiplies by the row-shared dropout mask in-register
       (``mask[b, j, d]`` indexed by decoded ``b, j`` from the flat ``m`` index).
    3. Adds the prior pair residual loaded from HBM.
    4. Writes the new pair tensor.

Saves: one full pair-tensor write (``delta``) + one full pair-tensor read
(``delta`` re-read by FusedDropoutResidual). Roughly ~250 MB per call at
L=768, B=5 in bf16; compounds across loops.

Forward + backward both implemented for the bf16 path. fp8 IO paths and
``transpose_out`` remain inference-only. The backward kernel recomputes
``gate = sigmoid(x1 @ w1.T)`` and ``val = x2 @ w2.T`` (the two dual GEMMs)
to keep saved-context cost down, and emits per-element
``d_gate_logits``, ``d_val_acc``, ``d_drop_mask_partials``; outer cuBLAS
calls handle the weight/input GEMM reductions. The residual add is
identity in the backward: ``d_residual = d_out``.
"""
# ruff: noqa: E402

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

import torch
import triton
import triton.language as tl

from .fused_dual_gemm import fused_gated_dual_gemm_split
from .fused_ln_residual import fused_ln_transpose, fused_ln_with_residual_link
from .trimul_einsum_triton import trimul_batched_einsum

# Static config — runtime autotune cold-start is unshippable for inference.
_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"TILE_M": 64, "TILE_N": 64, "TILE_K": 64, "GROUP_M": 8},
        num_stages=3,
        num_warps=4,
    )
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K", "HAS_DROP_MASK"])
@triton.jit
def _gated_gemm_with_residual_kernel(
    x1_ptr,  # [M, K]  — gate input (pre-residual pair)
    x2_ptr,  # [M, K]  — value input (post-LN_out)
    w1_ptr,  # [N, K]  — gate weight
    w2_ptr,  # [N, K]  — value weight
    residual_ptr,  # [M, N]   pair tensor pre-TriMul
    drop_mask_ptr,  # [B*N_COL, N]  row-shared dropout mask
    o_ptr,  # [M, N]  output = residual + drop_mask * sigmoid(x1@w1) * (x2@w2)
    M,
    N,
    K,
    STRIDE_B: tl.constexpr,  # = N_ROW * N_COL  (for decoding b from m)
    N_COL: tl.constexpr,  # = L_col  (for decoding j from m)
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    PRECISION: tl.constexpr,
    HAS_DROP_MASK: tl.constexpr,
    NEEDS_INT64: tl.constexpr = True,  # type: ignore[assignment]
):
    pid_m_raw = tl.program_id(axis=0)
    pid_n_raw = tl.program_id(axis=1)

    # GROUP_M swizzle for L2 reuse of x rows across consecutive CTAs.
    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    pid = pid_n_raw * num_pid_m + pid_m_raw
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if NEEDS_INT64:
        pid_m = tl.cast(pid_m, tl.int64)
        pid_n = tl.cast(pid_n, tl.int64)
        M = tl.cast(M, tl.int64)
        N = tl.cast(N, tl.int64)
        K = tl.cast(K, tl.int64)

    start_m = pid_m * TILE_M
    start_n = pid_n * TILE_N

    offs_xm = start_m + tl.arange(0, TILE_M)
    offs_wn = start_n + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if NEEDS_INT64:
        offs_xm = tl.cast(offs_xm, tl.int64)
        offs_wn = tl.cast(offs_wn, tl.int64)
        offs_k = tl.cast(offs_k, tl.int64)

    x1_ptrs = x1_ptr + (offs_xm[:, None] * K + offs_k[None, :])
    x2_ptrs = x2_ptr + (offs_xm[:, None] * K + offs_k[None, :])
    w_tile_offs = offs_wn[None, :] * K + offs_k[:, None]

    acc_1 = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    acc_2 = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    mask_m = offs_xm < M

    for _ in range(0, tl.cdiv(K, TILE_K)):
        x1 = tl.load(x1_ptrs, mask=mask_m[:, None], other=0.0).to(
            w1_ptr.type.element_ty
        )
        w1_ptrs = w1_ptr + w_tile_offs
        w1 = tl.load(w1_ptrs)
        if PRECISION == 0:
            acc_1 = tl.dot(x1, w1, acc_1)
        elif PRECISION == 2:
            acc_1 = tl.dot(x1, w1, acc_1, input_precision="tf32x3")
        else:
            acc_1 = tl.dot(x1, w1, acc_1, input_precision="ieee")
        x1_ptrs += TILE_K
        w1_ptr += TILE_K

    for _ in range(0, tl.cdiv(K, TILE_K)):
        x2 = tl.load(x2_ptrs, mask=mask_m[:, None], other=0.0).to(
            w2_ptr.type.element_ty
        )
        w2_ptrs = w2_ptr + w_tile_offs
        w2 = tl.load(w2_ptrs)
        if PRECISION == 0:
            acc_2 = tl.dot(x2, w2, acc_2)
        elif PRECISION == 2:
            acc_2 = tl.dot(x2, w2, acc_2, input_precision="tf32x3")
        else:
            acc_2 = tl.dot(x2, w2, acc_2, input_precision="ieee")
        x2_ptrs += TILE_K
        w2_ptr += TILE_K

    acc_1 = 1.0 / (1.0 + tl.exp(-acc_1))
    delta = acc_1 * acc_2

    offs_om = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_on = pid_n * TILE_N + tl.arange(0, TILE_N)
    if NEEDS_INT64:
        offs_om = tl.cast(offs_om, tl.int64)
        offs_on = tl.cast(offs_on, tl.int64)

    # Row-shared dropout: mask is [B*N_COL, N]; decode (b, j) from m.
    if HAS_DROP_MASK:
        b = offs_om // STRIDE_B
        j = offs_om % N_COL
        mask_row = b * N_COL + j
        drop_ptrs = drop_mask_ptr + mask_row[:, None] * N + offs_on[None, :]
        drop_tile = tl.load(drop_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        delta = delta * drop_tile

    resid_ptrs = residual_ptr + offs_om[:, None] * N + offs_on[None, :]
    resid_tile = tl.load(resid_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    out_val = delta + resid_tile

    o_ptrs = o_ptr + offs_om[:, None] * N + offs_on[None, :]
    o_mask = mask_m[:, None]
    tl.store(o_ptrs, out_val.to(o_ptr.type.element_ty), mask=o_mask)


# Backward recomputes the two fwd GEMMs, emits per-element grads for the
# outer cuBLAS GEMMs (d_x1, d_x2, d_w1, d_w2). d_residual is identity through
# the residual add. d_drop_mask is row-shared → per-pid_n partial + host reduce.


# Narrow TILE_N=32 in bwd: register pressure is dominated by the two output
# gradient tensors (vs forward's single output).
_BWD_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"TILE_M": 64, "TILE_N": 32, "TILE_K": 64, "GROUP_M": 8},
        num_stages=3,
        num_warps=4,
    )
]


@triton.autotune(configs=_BWD_AUTOTUNE_CONFIGS, key=["M", "N", "K", "HAS_DROP_MASK"])
@triton.jit
def _gated_gemm_with_residual_backward_kernel(
    grad_out_ptr,  # [M, N] bf16 — incoming grad
    x1_ptr,  # [M, K] bf16 — saved gate input
    x2_ptr,  # [M, K] bf16 — saved value input
    w1_ptr,  # [N, K] bf16
    w2_ptr,  # [N, K] bf16
    drop_mask_ptr,  # [B*N_COL, N] bf16 — row-shared (unused if HAS_DROP_MASK=0)
    grad_gate_logits_ptr,  # [M, N] bf16 — out
    grad_val_acc_ptr,  # [M, N] bf16 — out
    grad_drop_mask_buf_ptr,  # [B*N_COL, N] fp32 — drop_mask grad accumulator (atomic_add)
    M,
    N,
    K,
    STRIDE_B: tl.constexpr,
    N_COL: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_DROP_MASK: tl.constexpr,
    NEEDS_INT64: tl.constexpr = True,  # type: ignore[assignment]
):
    pid_m_raw = tl.program_id(axis=0)
    pid_n_raw = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    pid = pid_n_raw * num_pid_m + pid_m_raw
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if NEEDS_INT64:
        pid_m = tl.cast(pid_m, tl.int64)
        pid_n = tl.cast(pid_n, tl.int64)
        M = tl.cast(M, tl.int64)
        N = tl.cast(N, tl.int64)
        K = tl.cast(K, tl.int64)

    start_m = pid_m * TILE_M
    start_n = pid_n * TILE_N

    offs_xm = start_m + tl.arange(0, TILE_M)
    offs_wn = start_n + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    if NEEDS_INT64:
        offs_xm = tl.cast(offs_xm, tl.int64)
        offs_wn = tl.cast(offs_wn, tl.int64)
        offs_k = tl.cast(offs_k, tl.int64)

    x1_ptrs = x1_ptr + (offs_xm[:, None] * K + offs_k[None, :])
    x2_ptrs = x2_ptr + (offs_xm[:, None] * K + offs_k[None, :])
    w_tile_offs = offs_wn[None, :] * K + offs_k[:, None]

    acc_1 = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    acc_2 = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    mask_m = offs_xm < M

    w1p = w1_ptr
    for _ in range(0, tl.cdiv(K, TILE_K)):
        x1 = tl.load(x1_ptrs, mask=mask_m[:, None], other=0.0).to(
            w1_ptr.type.element_ty
        )
        w1 = tl.load(w1p + w_tile_offs)
        acc_1 = tl.dot(x1, w1, acc_1)
        x1_ptrs += TILE_K
        w1p += TILE_K

    w2p = w2_ptr
    for _ in range(0, tl.cdiv(K, TILE_K)):
        x2 = tl.load(x2_ptrs, mask=mask_m[:, None], other=0.0).to(
            w2_ptr.type.element_ty
        )
        w2 = tl.load(w2p + w_tile_offs)
        acc_2 = tl.dot(x2, w2, acc_2)
        x2_ptrs += TILE_K
        w2p += TILE_K

    g = tl.sigmoid(acc_1)
    delta = g * acc_2  # pre-mask delta (forward post-mask = delta * drop_mask)

    offs_om = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_on = pid_n * TILE_N + tl.arange(0, TILE_N)
    if NEEDS_INT64:
        offs_om = tl.cast(offs_om, tl.int64)
        offs_on = tl.cast(offs_on, tl.int64)

    grad_o_ptrs = grad_out_ptr + offs_om[:, None] * N + offs_on[None, :]
    grad_o = tl.load(grad_o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    if HAS_DROP_MASK:
        # Multiple m's collide on the same mask_row, so use fp32 atomic_add to a
        # (B*N_COL, N) accumulator (handles intra- and inter-tile collisions).
        b = offs_om // STRIDE_B
        j = offs_om % N_COL
        mask_row = b * N_COL + j
        drop_ptrs = drop_mask_ptr + mask_row[:, None] * N + offs_on[None, :]
        drop_tile = tl.load(drop_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        d_drop_per_elem = grad_o * delta
        acc_ptrs = grad_drop_mask_buf_ptr + mask_row[:, None] * N + offs_on[None, :]
        tl.atomic_add(acc_ptrs, d_drop_per_elem, mask=mask_m[:, None])

        grad_o = grad_o * drop_tile

    d_val_acc = (grad_o * g).to(grad_val_acc_ptr.type.element_ty)
    d_val_acc_ptrs = grad_val_acc_ptr + offs_om[:, None] * N + offs_on[None, :]
    tl.store(d_val_acc_ptrs, d_val_acc, mask=mask_m[:, None])

    d_gate_logits = (grad_o * acc_2 * g * (1.0 - g)).to(
        grad_gate_logits_ptr.type.element_ty
    )
    d_gate_logits_ptrs = grad_gate_logits_ptr + offs_om[:, None] * N + offs_on[None, :]
    tl.store(d_gate_logits_ptrs, d_gate_logits, mask=mask_m[:, None])


def _gated_gemm_with_residual_bwd(
    grad_out: torch.Tensor,  # [M, N] bf16
    x1: torch.Tensor,  # [M, K] bf16
    x2: torch.Tensor,  # [M, K] bf16
    w1: torch.Tensor,
    w2: torch.Tensor,
    drop_mask: torch.Tensor | None,
    n_row: int,
    n_col: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Returns (d_x1, d_x2, d_w1, d_w2, d_residual, d_drop_mask)."""
    M, K = x1.shape
    N = w1.shape[0]
    assert x2.shape == x1.shape
    assert (
        w1.dtype == w2.dtype == torch.bfloat16
    ), f"weights must be bf16; got {w1.dtype}/{w2.dtype}"
    assert x1.dtype == torch.bfloat16 and x2.dtype == torch.bfloat16

    # Don't call .contiguous() unconditionally (avoids a full-tensor clone).
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    grad_out_2d = grad_out.view(M, N)
    x1_c = x1 if x1.is_contiguous() else x1.contiguous()
    x2_c = x2 if x2.is_contiguous() else x2.contiguous()
    w1_c = w1 if w1.is_contiguous() else w1.contiguous()
    w2_c = w2 if w2.is_contiguous() else w2.contiguous()

    grad_gate_logits = torch.empty((M, N), device=x1.device, dtype=torch.bfloat16)
    grad_val_acc = torch.empty((M, N), device=x1.device, dtype=torch.bfloat16)

    # fp32 accumulator for row-shared mask_row collisions across m.
    if drop_mask is not None:
        drop_mask = drop_mask.contiguous().view(-1, N)
        n_mask_rows = drop_mask.shape[0]
        grad_drop_buf = torch.zeros(
            (n_mask_rows, N), device=x1.device, dtype=torch.float32
        )
    else:
        grad_drop_buf = torch.empty((1,), device=x1.device, dtype=torch.float32)

    _dummy_mask = torch.zeros((), device=x1.device, dtype=torch.bfloat16)

    NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * N >= 2**31 - 1)

    def grid(meta):
        return (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(N, meta["TILE_N"]))

    _gated_gemm_with_residual_backward_kernel[grid](
        grad_out_2d,
        x1_c,
        x2_c,
        w1_c,
        w2_c,
        drop_mask if drop_mask is not None else _dummy_mask,
        grad_gate_logits,
        grad_val_acc,
        grad_drop_buf,
        M,
        N,
        K,
        STRIDE_B=n_row * n_col,
        N_COL=n_col,
        HAS_DROP_MASK=drop_mask is not None,
        NEEDS_INT64=NEEDS_INT64,
    )

    d_w1 = grad_gate_logits.t() @ x1_c  # [N, K]
    d_w2 = grad_val_acc.t() @ x2_c  # [N, K]
    d_x1 = grad_gate_logits @ w1_c  # [M, K]
    d_x2 = grad_val_acc @ w2_c  # [M, K]

    d_residual = grad_out_2d

    if drop_mask is not None:
        d_drop_mask = grad_drop_buf.to(drop_mask.dtype)
    else:
        d_drop_mask = None

    return d_x1, d_x2, d_w1, d_w2, d_residual, d_drop_mask


class GatedGEMMWithResidual(torch.autograd.Function):
    """Autograd wrapper for ``_gated_gemm_with_residual`` (bf16 path)."""

    @staticmethod
    def forward(
        ctx,
        x1: torch.Tensor,
        x2: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        residual: torch.Tensor,
        drop_mask: torch.Tensor | None,
        n_row: int,
        n_col: int,
        precision: int,
    ) -> torch.Tensor:
        out = _gated_gemm_with_residual_fwd(
            x1, x2, w1, w2, residual, drop_mask, n_row, n_col, precision=precision
        )
        if drop_mask is None:
            ctx.save_for_backward(x1, x2, w1, w2)
            ctx.has_drop_mask = False
        else:
            ctx.save_for_backward(x1, x2, w1, w2, drop_mask)
            ctx.has_drop_mask = True
        ctx.n_row = n_row
        ctx.n_col = n_col
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx.has_drop_mask:
            x1, x2, w1, w2, drop_mask = ctx.saved_tensors
        else:
            x1, x2, w1, w2 = ctx.saved_tensors
            drop_mask = None
        d_x1, d_x2, d_w1, d_w2, d_res, d_drop = _gated_gemm_with_residual_bwd(
            grad_out.contiguous(), x1, x2, w1, w2, drop_mask, ctx.n_row, ctx.n_col
        )
        return d_x1, d_x2, d_w1, d_w2, d_res, d_drop, None, None, None


def _gated_gemm_with_residual_fwd(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    residual: torch.Tensor,
    drop_mask: torch.Tensor | None,
    n_row: int,
    n_col: int,
    precision: int = 0,
) -> torch.Tensor:
    """Run the residual+dropout-aware final GEMM.

    Shapes: x1, x2: (M, K). w1, w2: (N, K). residual: (M, N). drop_mask: (B*n_col, N) or None.
    Output: (M, N).
    """
    M, K = x1.shape
    N = w1.shape[0]
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    residual = residual.contiguous().view(M, N)
    if drop_mask is not None:
        drop_mask = drop_mask.contiguous().view(-1, N)
    out = residual.new_empty((M, N))

    NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * N >= 2**31 - 1)

    def grid(meta):
        return (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(N, meta["TILE_N"]))

    _gated_gemm_with_residual_kernel[grid](
        x1,
        x2,
        w1,
        w2,
        residual,
        drop_mask if drop_mask is not None else residual,  # dummy pass-through
        out,
        M,
        N,
        K,
        STRIDE_B=n_row * n_col,
        N_COL=n_col,
        PRECISION=precision,
        HAS_DROP_MASK=drop_mask is not None,
        NEEDS_INT64=NEEDS_INT64,
    )
    return out


def _gated_gemm_with_residual(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    residual: torch.Tensor,
    drop_mask: torch.Tensor | None,
    n_row: int,
    n_col: int,
    precision: int = 0,
) -> torch.Tensor:
    """Public wrapper: dispatches to autograd Function if grad is enabled."""
    if torch.is_grad_enabled() and (
        x1.requires_grad
        or x2.requires_grad
        or w1.requires_grad
        or w2.requires_grad
        or residual.requires_grad
    ):
        return GatedGEMMWithResidual.apply(  # type: ignore[return-value]
            x1, x2, w1, w2, residual, drop_mask, n_row, n_col, precision
        )
    return _gated_gemm_with_residual_fwd(
        x1, x2, w1, w2, residual, drop_mask, n_row, n_col, precision=precision
    )


@torch._dynamo.disable
def triangle_multiplicative_update_with_residual(
    pair: torch.Tensor,
    direction: str,
    residual: torch.Tensor,
    drop_mask: torch.Tensor | None,
    *,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-5,
    precision: int = 0,
) -> torch.Tensor:
    """Fused TriMul-and-residual: pair_new = residual + drop_mask * TriMul(pair).

    The intermediate ``delta = TriMul(pair)`` is never materialized in HBM —
    the dropout-mask multiply and residual-add happen in-kernel at the final
    GEMM. Saves one pair-tensor write + read vs the
    ``TriMul → FusedDropoutResidual`` baseline.

    Shapes:
        pair, residual: (B, L, L, c_z)
        drop_mask:      (B, 1, L, c_z) or (B*L, c_z) — row-shared
        mask:           (B, L, L) or None
        weights:        (N, K) bf16 — see arg list for each stage.
    """
    assert pair.shape == residual.shape
    B, L_row, L_col, c_z = pair.shape

    # Stage 1: input LayerNorm. When ``residual is pair`` (common case for trimul),
    # use the fused LN-fwd + LN-bwd-with-residual variant so the bwd pass adds
    # ``grad_residual`` into ``grad_x`` in one kernel (saves one pair-tensor
    # HBM round-trip).
    if residual is pair:
        x, residual_alias = fused_ln_with_residual_link(
            pair, norm_in_weight, norm_in_bias, pair, eps=eps, layout="bijd->bijd"
        )
    else:
        x = fused_ln_transpose(
            pair, norm_in_weight, norm_in_bias, eps=eps, layout="bijd->bijd"
        )
        residual_alias = residual
    x_in = x

    # Stage 2: gated dual GEMM → ab (transposed for the einsum's dbij layout).
    # Train path runs transpose_out=False (no bwd for in-kernel transpose),
    # then a view/permute reaches the dbij layout downstream.
    a, b_t = fused_gated_dual_gemm_split(x, g_in_weight, p_in_weight, mask=mask)

    # Stage 3: triangular einsum.
    # Training: native (D, B, L, L) Triton kernel — its bwd reads the layout
    # natively, eliminating the contiguous copy torch.einsum's autograd does.
    # Inference: torch.einsum (cuBLAS bgemm) is faster fwd-only.
    if torch.is_grad_enabled():
        x = trimul_batched_einsum(a, b_t, direction)
    elif direction == "outgoing":
        x = torch.einsum("dbik,dbjk->dbij", a, b_t)
    else:
        x = torch.einsum("dbki,dbkj->dbij", a, b_t)

    # Stage 4: output LayerNorm (back to bijd).
    x_out = fused_ln_transpose(
        x, norm_out_weight, norm_out_bias, eps=eps, layout="dbij->bijd"
    )

    # Stage 5: fused output gated GEMM + dropout mask + residual add.
    x_in_2d = x_in.reshape(-1, c_z)
    x_out_2d = x_out.reshape(-1, c_z)
    # Stage-5 residual MUST be the LN1 alias when LN1-residual-fusion is on,
    # otherwise the grad routing through the fused LN bwd won't fire.
    residual_2d = residual_alias.reshape(-1, c_z)
    out_2d = _gated_gemm_with_residual(
        x_in_2d,
        x_out_2d,
        g_out_weight,
        p_out_weight,
        residual_2d,
        drop_mask,
        n_row=L_row,
        n_col=L_col,
        precision=precision,
    )
    return out_2d.view(B, L_row, L_col, c_z)
