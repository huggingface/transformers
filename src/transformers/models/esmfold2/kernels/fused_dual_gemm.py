"""Native Triton ``fused_sigmoid_gated_dual_gemm`` for TriMul stage 2.

Computes ``sigmoid(x @ w1.T) * (x @ w2.T)`` with optional row-shared mask,
bias, and transposed output. Forward + backward implemented in bf16.

Inspired by cuequivariance's ``fused_sigmoid_gated_dual_gemm``
(https://docs.nvidia.com/cuda/cuequivariance/index.html); independently
re-implemented in Triton.
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

# Static config — runtime autotune cold-start is unshippable for inference.
_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"TILE_M": 128, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 8},
        num_stages=4,
        num_warps=4,
    )
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K", "HAS_MASK", "TRANSPOSE_OUT"]
)
@triton.jit
def _gated_dual_gemm_kernel(
    x_ptr,  # [M, K] bf16
    w1_ptr,  # [N, K] bf16 — gate weight
    w2_ptr,  # [N, K] bf16 — value weight
    mask_ptr,  # [M] bf16 — row-shared mask broadcast to (M, N)
    out_ptr,  # [M, N] or [N, M] bf16 — sigmoid(x@w1) * (x@w2)
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_MASK: tl.constexpr,
    TRANSPOSE_OUT: tl.constexpr,  # store (N, M) instead of (M, N)
    NEEDS_INT64: tl.constexpr = True,  # type: ignore[assignment]
):
    """Per (TILE_M, TILE_N) output tile:
    gate_acc = Σ_K (x[:, k] @ w1[:, k]) over k
    val_acc  = Σ_K (x[:, k] @ w2[:, k]) over k
    delta    = sigmoid(gate_acc) * val_acc
    if mask: delta *= mask[m_tile]   (broadcast over N)
    store delta (transposed if TRANSPOSE_OUT)
    """
    pid_m_raw = tl.program_id(0)
    pid_n_raw = tl.program_id(1)

    # GROUP_M swizzle for L2 reuse of ``x`` across consecutive CTAs.
    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    pid = pid_n_raw * num_pid_m + pid_m_raw  # row-major program id
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

    offs_m = start_m + tl.arange(0, TILE_M)
    offs_n = start_n + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    if NEEDS_INT64:
        offs_m = tl.cast(offs_m, tl.int64)
        offs_n = tl.cast(offs_n, tl.int64)
        offs_k = tl.cast(offs_k, tl.int64)

    x_ptrs = x_ptr + (offs_m[:, None] * K + offs_k[None, :])
    w1_base = w1_ptr + (offs_n[None, :] * K + offs_k[:, None])
    w2_base = w2_ptr + (offs_n[None, :] * K + offs_k[:, None])

    gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    mask_m = offs_m < M

    for _ in range(0, tl.cdiv(K, TILE_K)):
        x_raw = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        x_op = x_raw.to(w1_ptr.type.element_ty)
        w1_tile = tl.load(w1_base)
        w2_tile = tl.load(w2_base)
        gate_acc = tl.dot(x_op, w1_tile, gate_acc)
        val_acc = tl.dot(x_op, w2_tile, val_acc)
        x_ptrs += TILE_K
        w1_base += TILE_K
        w2_base += TILE_K

    delta = tl.sigmoid(gate_acc) * val_acc  # fp32

    if HAS_MASK:
        mask_tile = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        delta = delta * mask_tile[:, None]

    if TRANSPOSE_OUT:
        out_ptrs = out_ptr + (offs_n[:, None] * M + offs_m[None, :])
        tl.store(
            out_ptrs, tl.trans(delta).to(out_ptr.type.element_ty), mask=mask_m[None, :]
        )
    else:
        out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
        tl.store(out_ptrs, delta.to(out_ptr.type.element_ty), mask=mask_m[:, None])


# Backward emits per-element grad_gate_logits and grad_val_acc by recomputing
# the two forward GEMMs. Weight grads (d_w1, d_w2, d_x) are done in cuBLAS in
# the autograd Function — cuBLAS beats Triton bf16 at these reduction shapes.
# Narrower TILE_N=64 vs forward's 128 (bwd writes two M*N tensors, doubling
# register pressure).
_BWD_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"TILE_M": 64, "TILE_N": 64, "TILE_K": 64, "GROUP_M": 8},
        num_stages=3,
        num_warps=4,
    )
]


@triton.autotune(
    configs=_BWD_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "HAS_MASK", "GRAD_OUT_TRANSPOSED", "GRAD_OUT_SPLIT"],
)
@triton.jit
def _gated_dual_gemm_backward_kernel(
    grad_out_ptr,  # [M, N] (or [N, M] if GRAD_OUT_TRANSPOSED, or [N/2, M] if GRAD_OUT_SPLIT)
    grad_out2_ptr,  # GRAD_OUT_SPLIT only: second half of (N, M); else dummy
    x_ptr,  # [M, K] bf16 — saved input
    w1_ptr,  # [N, K] bf16
    w2_ptr,  # [N, K] bf16
    mask_ptr,  # [M] bf16 — row-shared (unused if HAS_MASK=0)
    grad_gate_logits_ptr,  # [M, N] bf16 — out: d (x @ w1.T)
    grad_val_acc_ptr,  # [M, N] bf16 — out: d (x @ w2.T)
    grad_mask_partials_ptr,  # [num_pid_n, M] fp32 — partial mask grads
    M,
    N,
    K,
    HALF_N: tl.constexpr,  # = N // 2, only used when GRAD_OUT_SPLIT=1
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_MASK: tl.constexpr,
    GRAD_OUT_TRANSPOSED: tl.constexpr,  # 1: load grad from (N, M) layout
    GRAD_OUT_SPLIT: tl.constexpr,  # 1: read from two (N/2, M) tensors (chunk-free path)
    NEEDS_INT64: tl.constexpr = True,  # type: ignore[assignment]
):
    """Per (TILE_M, TILE_N) output tile:
    gate_acc  = Σ_K x[:, k] @ w1[:, k]
    val_acc   = Σ_K x[:, k] @ w2[:, k]
    g         = sigmoid(gate_acc)
    grad_o    = grad_out_tile (post-mask: multiply by mask if HAS_MASK)
    d_gate_logits = grad_o * val_acc * g * (1 - g)
    d_val_acc     = grad_o * g
    d_mask_partial[pid_n, m] = sum_n (grad_out_tile * g * val_acc)   [if HAS_MASK]
    """
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

    offs_m = start_m + tl.arange(0, TILE_M)
    offs_n = start_n + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    if NEEDS_INT64:
        offs_m = tl.cast(offs_m, tl.int64)
        offs_n = tl.cast(offs_n, tl.int64)
        offs_k = tl.cast(offs_k, tl.int64)

    x_ptrs = x_ptr + (offs_m[:, None] * K + offs_k[None, :])
    w1_base = w1_ptr + (offs_n[None, :] * K + offs_k[:, None])
    w2_base = w2_ptr + (offs_n[None, :] * K + offs_k[:, None])

    gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    mask_m = offs_m < M

    # Recompute fwd GEMMs (cheaper than saving N*M activations).
    for _ in range(0, tl.cdiv(K, TILE_K)):
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        x_op = x_tile.to(w1_ptr.type.element_ty)
        w1_tile = tl.load(w1_base)
        w2_tile = tl.load(w2_base)
        gate_acc = tl.dot(x_op, w1_tile, gate_acc)
        val_acc = tl.dot(x_op, w2_tile, val_acc)
        x_ptrs += TILE_K
        w1_base += TILE_K
        w2_base += TILE_K

    g = tl.sigmoid(gate_acc)

    if GRAD_OUT_SPLIT:
        # Two (N/2, M) grad tensors; caller guarantees TILE_N divides HALF_N.
        if start_n < HALF_N:
            offs_n_local = offs_n
            grad_o_ptrs = grad_out_ptr + (offs_n_local[None, :] * M + offs_m[:, None])
        else:
            offs_n_local = offs_n - HALF_N
            grad_o_ptrs = grad_out2_ptr + (offs_n_local[None, :] * M + offs_m[:, None])
    elif GRAD_OUT_TRANSPOSED:
        grad_o_ptrs = grad_out_ptr + (offs_n[None, :] * M + offs_m[:, None])
    else:
        grad_o_ptrs = grad_out_ptr + (offs_m[:, None] * N + offs_n[None, :])
    grad_o = tl.load(grad_o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    if HAS_MASK:
        # d_mask = row-sum BEFORE mask multiply; per-pid_n partial, host reduces.
        d_mask_partial = tl.sum(grad_o * g * val_acc, axis=1)
        d_mask_partials_ptrs = grad_mask_partials_ptr + pid_n * M + offs_m
        tl.store(d_mask_partials_ptrs, d_mask_partial, mask=mask_m)
        mask_tile = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        grad_o = grad_o * mask_tile[:, None]

    # Both grad ptrs index into one (M, 2N) buffer (val_acc cols [0:N],
    # gate_logits cols [N:2N]) — lets downstream d_x / d_w fold into single GEMMs.
    d_val_acc = (grad_o * g).to(grad_val_acc_ptr.type.element_ty)
    d_val_acc_ptrs = grad_val_acc_ptr + (offs_m[:, None] * (2 * N) + offs_n[None, :])
    tl.store(d_val_acc_ptrs, d_val_acc, mask=mask_m[:, None])

    d_gate_logits = (grad_o * val_acc * g * (1.0 - g)).to(
        grad_gate_logits_ptr.type.element_ty
    )
    d_gate_logits_ptrs = grad_gate_logits_ptr + (
        offs_m[:, None] * (2 * N) + offs_n[None, :]
    )
    tl.store(d_gate_logits_ptrs, d_gate_logits, mask=mask_m[:, None])


# Backward TILE_N must match the kernel's autotuned tile in ``_BWD_AUTOTUNE_CONFIGS``;
# host code uses it to size the per-tile mask-grad partials buffer statically.
_BWD_TILE_N = 64


def _fused_gated_dual_gemm_bwd(
    grad_out: torch.Tensor,  # (M, N) or (N, M) layout (see grad_out_transposed)
    x: torch.Tensor,  # [..., K]  (un-flattened, original)
    w1: torch.Tensor,
    w2: torch.Tensor,
    mask: torch.Tensor | None,
    grad_out_transposed: bool = False,
    grad_out_split: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute (d_x, d_w1, d_w2, d_mask) for ``fused_gated_dual_gemm``.

    Three grad_out layouts supported:
      * default: ``grad_out`` of shape ``(..., N)`` flattened to (M, N).
      * ``grad_out_transposed=True``: shape ``(N, ...)`` flattened to (N, M).
      * ``grad_out_split=(g1, g2)``: two tensors, each of shape ``(N/2, ...)``
        flattened to ``(N/2, M)``; the kernel reads from the appropriate
        pointer per tile. Avoids the autograd-introduced concat that
        ``torch.chunk(dim=0)``'s backward inserts (saves 3-4 ms at prod
        shape, B=5 L=768 c_z=128).
    """
    in_shape = x.shape
    K = in_shape[-1]
    x_c = x if x.is_contiguous() else x.contiguous()
    x_2d = x_c.view(-1, K)
    M = x_2d.shape[0]
    N = w1.shape[0]

    assert (
        w1.dtype == w2.dtype == torch.bfloat16
    ), f"weights must be bf16; got {w1.dtype}/{w2.dtype}"
    assert x_2d.dtype == torch.bfloat16, "bwd only supports bf16 x"

    if grad_out_split is not None:
        g1, g2 = grad_out_split
        half_n = N // 2
        # Materialize only if non-contig (avoids a full-tensor copy).
        if not g1.is_contiguous():
            g1 = g1.contiguous()
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        assert g1.numel() == half_n * M and g2.numel() == half_n * M, (
            f"split grad sizes mismatch: g1={g1.numel()} g2={g2.numel()} "
            f"vs half_n*M={half_n * M}"
        )
        grad_out_2d_a = g1.view(half_n, M)
        grad_out_2d_b = g2.view(half_n, M)
    elif grad_out_transposed:
        grad_out_2d_a = grad_out.contiguous().view(N, M)
        grad_out_2d_b = grad_out_2d_a  # unused (dummy)
    else:
        grad_out_2d_a = grad_out.contiguous().view(M, N)
        grad_out_2d_b = grad_out_2d_a  # unused (dummy)

    mask_flat = mask.contiguous().view(-1) if mask is not None else None
    if mask_flat is not None:
        assert mask_flat.shape[0] == M

    # (M, 2N) combined-grad buffer; matched stacked_w order (w2 then w1)
    # lets d_w and d_x each collapse to a single cuBLAS GEMM.
    grad_combined = torch.empty((M, 2 * N), device=x.device, dtype=torch.bfloat16)
    grad_val_acc = grad_combined[:, :N]
    grad_gate_logits = grad_combined[:, N:]

    tiles_n_max = triton.cdiv(N, _BWD_TILE_N)
    if mask is not None:
        grad_mask_partials = torch.empty(
            (tiles_n_max, M), device=x.device, dtype=torch.float32
        )
    else:
        grad_mask_partials = torch.empty((1,), device=x.device, dtype=torch.float32)

    _dummy_mask = torch.zeros((), device=x.device, dtype=torch.bfloat16)

    # (M, 2N) layout → use stride 2N for the int32-overflow gate.
    NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * (2 * N) >= 2**31 - 1)

    def grid(meta):
        assert N % meta["TILE_N"] == 0
        return (triton.cdiv(M, meta["TILE_M"]), N // meta["TILE_N"])

    half_n = N // 2 if grad_out_split is not None else 1

    _gated_dual_gemm_backward_kernel[grid](
        grad_out_2d_a,
        grad_out_2d_b,
        x_2d,
        w1.contiguous(),
        w2.contiguous(),
        mask_flat if mask_flat is not None else _dummy_mask,
        grad_gate_logits,
        grad_val_acc,
        grad_mask_partials,
        M,
        N,
        K,
        HALF_N=half_n,
        HAS_MASK=mask is not None,
        GRAD_OUT_TRANSPOSED=grad_out_transposed,
        GRAD_OUT_SPLIT=grad_out_split is not None,
        NEEDS_INT64=NEEDS_INT64,
    )

    # Stacked weights for the single-GEMM d_x path.
    stacked_w = torch.cat([w2, w1], dim=0)  # (2N, K), matches val|gate layout

    d_x_2d = grad_combined @ stacked_w
    d_x = d_x_2d.view(in_shape)

    d_w_combined = grad_combined.t() @ x_2d  # (2N, K)
    d_w2 = d_w_combined[:N]
    d_w1 = d_w_combined[N:]

    if mask is not None:
        actual_tiles = triton.cdiv(N, _BWD_TILE_N)
        d_mask_flat = grad_mask_partials[:actual_tiles].sum(dim=0).to(mask.dtype)
        d_mask = d_mask_flat.view(mask.shape)
    else:
        d_mask = None

    return d_x, d_w1, d_w2, d_mask


class FusedGatedDualGEMM(torch.autograd.Function):
    """Autograd wrapper for ``fused_gated_dual_gemm`` (bf16 path,
    ``transpose_out=False``).

    Saves ``x``, ``w1``, ``w2``, ``mask`` for the backward kernel; the
    backward kernel recomputes the two dual-GEMM activations (cheap vs
    materializing them in fwd context).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        out = _fused_gated_dual_gemm_fwd(x, w1, w2, mask=mask, transpose_out=False)
        if mask is None:
            ctx.save_for_backward(x, w1, w2)
            ctx.has_mask = False
        else:
            ctx.save_for_backward(x, w1, w2, mask)
            ctx.has_mask = True
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx.has_mask:
            x, w1, w2, mask = ctx.saved_tensors
        else:
            x, w1, w2 = ctx.saved_tensors
            mask = None
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        d_x, d_w1, d_w2, d_mask = _fused_gated_dual_gemm_bwd(grad_out, x, w1, w2, mask)
        return d_x, d_w1, d_w2, d_mask


class FusedGatedDualGEMMSplit(torch.autograd.Function):
    """Autograd wrapper that returns the dual-GEMM output as two split halves.

    Forward: runs the kernel with ``transpose_out=True``, producing a single
    ``(N=2*c_z, *trailing)`` buffer where the gate half (first c_z) and the
    value half (next c_z) live contiguously along dim 0. Returns the two
    views ``(a, b_t)`` so downstream einsums consume them in
    ``(c_z, B, L, L)`` layout — no chunk in the autograd graph.

    Backward: receives ``(grad_a, grad_b_t)`` (each ``(c_z, *trailing)``)
    and passes them as the ``grad_out_split`` pair to the backward kernel.
    Eliminates the ~3.5 ms ``torch.chunk(dim=0)`` backward concat at the
    prod shape (B=5, L=768, c_z=128) by reading the two halves from
    separate pointers inside the kernel.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # (B, L, L, c_z) bf16
        w1: torch.Tensor,  # (N=2*c_z, c_z) bf16 — gate
        w2: torch.Tensor,  # (N=2*c_z, c_z) bf16 — value
        mask: torch.Tensor | None,  # (B, L, L) bf16 or None
        trailing_shape: tuple,  # (B, L, L) for output view
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = _fused_gated_dual_gemm_fwd(x, w1, w2, mask=mask, transpose_out=True)
        # Slicing on a leading dim of a contiguous tensor stays contiguous (no copy).
        N = w1.shape[0]
        half_n = N // 2
        out_view = out.view((N,) + trailing_shape)
        a = out_view[:half_n]
        b_t = out_view[half_n:]
        if mask is None:
            ctx.save_for_backward(x, w1, w2)
            ctx.has_mask = False
        else:
            ctx.save_for_backward(x, w1, w2, mask)
            ctx.has_mask = True
        return a, b_t

    @staticmethod
    def backward(ctx, grad_a: torch.Tensor, grad_b_t: torch.Tensor):  # type: ignore[override]
        if ctx.has_mask:
            x, w1, w2, mask = ctx.saved_tensors
        else:
            x, w1, w2 = ctx.saved_tensors
            mask = None
        # Don't call .contiguous() — _fused_gated_dual_gemm_bwd validates instead
        # (avoids a full-tensor copy on the common contig path).
        d_x, d_w1, d_w2, d_mask = _fused_gated_dual_gemm_bwd(
            None,  # type: ignore[arg-type]
            x,
            w1,
            w2,
            mask,
            grad_out_split=(grad_a, grad_b_t),
        )
        return d_x, d_w1, d_w2, d_mask, None


def fused_gated_dual_gemm_split(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-output variant of ``fused_gated_dual_gemm``: returns ``(a, b_t)``
    with layout ``(c_z, B, L, L)`` each, avoiding chunk in the autograd graph.

    Inference fallback (no grad) just calls the regular fwd with
    ``transpose_out=True`` and chunks the result — same layout, no kernel
    change required.
    """
    trailing_shape = tuple(x.shape[:-1])
    if torch.is_grad_enabled() and (
        x.requires_grad or w1.requires_grad or w2.requires_grad
    ):
        return FusedGatedDualGEMMSplit.apply(x, w1, w2, mask, trailing_shape)  # type: ignore[return-value]
    out = _fused_gated_dual_gemm_fwd(x, w1, w2, mask=mask, transpose_out=True)
    N = w1.shape[0]
    out_view = out.view((N,) + trailing_shape)
    half_n = N // 2
    return out_view[:half_n], out_view[half_n:]


def _fused_gated_dual_gemm_fwd(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
) -> torch.Tensor:
    """Native Triton implementation of sigmoid-gated dual GEMM.

    Computes ``sigmoid(x @ w1.T) * (x @ w2.T)`` with optional row-shared mask.

    Shapes:
        x:    (..., K)   bf16
        w1:   (N, K)     bf16
        w2:   (N, K)     bf16
        mask: (...,)     bf16 — flattened to a per-row scalar
        out:  (..., N) or (N, ...) when ``transpose_out=True``
    """
    in_shape = x.shape
    K = in_shape[-1]
    x_2d = x.contiguous().view(-1, K)
    M = x_2d.shape[0]
    N = w1.shape[0]

    assert w1.shape == w2.shape, f"w1 {w1.shape} ≠ w2 {w2.shape}"
    assert w1.shape[1] == K
    assert (
        w1.dtype == w2.dtype == torch.bfloat16
    ), f"weights must be bf16; got {w1.dtype}/{w2.dtype}"

    out_dtype = torch.bfloat16
    if transpose_out:
        out = torch.empty((N, M), device=x.device, dtype=out_dtype)
        out_shape = (N,) + in_shape[:-1]
    else:
        out = torch.empty((M, N), device=x.device, dtype=out_dtype)
        out_shape = in_shape[:-1] + (N,)

    mask_flat = mask.contiguous().view(-1) if mask is not None else None
    if mask_flat is not None:
        assert mask_flat.shape[0] == M, f"mask len {mask_flat.shape[0]} ≠ M {M}"

    _dummy = torch.zeros((), device=x.device, dtype=torch.float32)

    NEEDS_INT64 = (M * K >= 2**31 - 1) or (M * N >= 2**31 - 1)

    def grid(meta):
        assert N % meta["TILE_N"] == 0
        return (triton.cdiv(M, meta["TILE_M"]), N // meta["TILE_N"])

    _gated_dual_gemm_kernel[grid](
        x_2d,
        w1.contiguous(),
        w2.contiguous(),
        mask_flat if mask_flat is not None else _dummy,
        out,
        M,
        N,
        K,
        HAS_MASK=mask is not None,
        TRANSPOSE_OUT=transpose_out,
        NEEDS_INT64=NEEDS_INT64,
    )
    return out.view(out_shape)


def fused_gated_dual_gemm(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
) -> torch.Tensor:
    """Public wrapper: dispatches to autograd Function if grad is enabled.

    ``transpose_out=True`` is inference-only (bypasses autograd).
    """
    if torch.is_grad_enabled() and (
        x.requires_grad or w1.requires_grad or w2.requires_grad
    ):
        assert not transpose_out, (
            "transpose_out=True is inference-only; train path must use the "
            "non-transposed output (post-stage-3 einsum already handles layout)"
        )
        return FusedGatedDualGEMM.apply(x, w1, w2, mask)  # type: ignore[return-value]
    return _fused_gated_dual_gemm_fwd(x, w1, w2, mask=mask, transpose_out=transpose_out)
