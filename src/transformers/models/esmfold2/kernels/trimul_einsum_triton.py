"""Triton kernel for trimul stage-3 batched einsum in native (D, B, L, L) layout.

    outgoing: ``out[d,b,i,j] = sum_k a[d,b,i,k] * b[d,b,j,k]``    (TRANSPOSE_B=True)
    incoming: ``out[d,b,i,j] = sum_k a[d,b,k,i] * b[d,b,k,j]``    (TRANSPOSE_A=True)

All three tensors share the dense ``(D, B, L, L)`` row-major layout produced
upstream by ``fused_gated_dual_gemm_split`` and consumed downstream by
``layer_norm_transpose(..., layout="dbij->bijd")``. Operating natively in
this layout avoids the contiguous copy that ``torch.einsum``'s autograd
inserts in its backward when saved tensors don't match its expected layout.

A single kernel covers all 4 transpose patterns (fwd + 4 bwd grad patterns)
via ``TRANSPOSE_A`` / ``TRANSPOSE_B`` constexpr flags.
"""

import torch
import triton
import triton.language as tl

# Static config — runtime autotune cold-start is unshippable for inference.
# Triton fwd loses to cuBLAS bgemm; bwd wins by avoiding torch.einsum's
# autograd-internal contiguous copy on (D, B, L, L) saved tensors.
_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"TILE_M": 128, "TILE_N": 128, "TILE_K": 64, "GROUP_M": 8},
        num_stages=3,
        num_warps=8,
    )
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["L", "TRANSPOSE_A", "TRANSPOSE_B"])
@triton.jit
def _batched_einsum_kernel(
    a_ptr,  # (D, B, L, L) row-major
    b_ptr,  # (D, B, L, L) row-major
    out_ptr,  # (D, B, L, L) row-major
    D,
    B,
    L,
    stride_a_d,  # = B*L*L
    stride_a_b,  # = L*L
    stride_b_d,
    stride_b_b,
    stride_out_d,
    stride_out_b,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
):
    # Grid: (num_m * num_n, D * B); (m, n) swizzled for L2 reuse, (d, b) on axis-1.
    pid_mn = tl.program_id(axis=0)
    pid_db = tl.program_id(axis=1)

    pid_d = pid_db // B
    pid_b = pid_db % B

    num_pid_m = tl.cdiv(L, TILE_M)
    num_pid_n = tl.cdiv(L, TILE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    # int64 indexing — (D,B,L,L) at L=1024 D=128 is well over 2**31.
    pid_d = tl.cast(pid_d, tl.int64)
    pid_b = tl.cast(pid_b, tl.int64)
    pid_m_i = tl.cast(pid_m, tl.int64)
    pid_n_i = tl.cast(pid_n, tl.int64)
    L_i = tl.cast(L, tl.int64)

    a_plane = a_ptr + pid_d * stride_a_d + pid_b * stride_a_b
    b_plane = b_ptr + pid_d * stride_b_d + pid_b * stride_b_b
    out_plane = out_ptr + pid_d * stride_out_d + pid_b * stride_out_b

    offs_m = pid_m_i * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    offs_n = pid_n_i * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
    offs_k = tl.arange(0, TILE_K).to(tl.int64)

    mask_m = offs_m < L_i
    mask_n = offs_n < L_i

    if TRANSPOSE_A:
        a_ptrs = a_plane + (offs_k[:, None] * L_i + offs_m[None, :])
        a_step = TILE_K * L_i
    else:
        a_ptrs = a_plane + (offs_m[:, None] * L_i + offs_k[None, :])
        a_step = TILE_K

    if TRANSPOSE_B:
        b_ptrs = b_plane + (offs_n[None, :] * L_i + offs_k[:, None])
        b_step = TILE_K
    else:
        b_ptrs = b_plane + (offs_k[:, None] * L_i + offs_n[None, :])
        b_step = TILE_K * L_i

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    n_k_tiles = tl.cdiv(L, TILE_K)
    for kk in range(0, n_k_tiles):
        k_remaining = L_i - kk * TILE_K
        mask_k = offs_k < k_remaining

        if TRANSPOSE_A:
            a_mask = mask_k[:, None] & mask_m[None, :]
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            a_tile = tl.trans(a_tile)  # tl.dot wants (M, K)
        else:
            a_mask = mask_m[:, None] & mask_k[None, :]
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        if TRANSPOSE_B:
            b_mask = mask_k[:, None] & mask_n[None, :]
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            b_mask = mask_k[:, None] & mask_n[None, :]
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a_tile, b_tile, acc)

        a_ptrs += a_step
        b_ptrs += b_step

    out_ptrs = out_plane + (offs_m[:, None] * L_i + offs_n[None, :])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.type.element_ty), mask=out_mask)


def _batched_einsum(
    a: torch.Tensor, b: torch.Tensor, transpose_a: bool, transpose_b: bool
) -> torch.Tensor:
    """Compute ``(A op_a) @ (B op_b)`` per (d, b) plane.

    Parameters
    ----------
    a, b : torch.Tensor
        Shape (D, B, L, L), bf16, contiguous.
    transpose_a : bool
        If True, contract over the row dim of A (k = leading), else over col dim.
    transpose_b : bool
        If True, B is read as (n, k) (i.e. B^T enters the matmul: out += A·B^T),
        else as (k, n) (out += A·B).

    Returns
    -------
    out : torch.Tensor
        Shape (D, B, L, L), bf16.
    """
    assert a.shape == b.shape, f"a {a.shape} ≠ b {b.shape}"
    assert a.ndim == 4
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert (
        a.is_contiguous() and b.is_contiguous()
    ), "trimul einsum kernel requires contiguous (D,B,L,L) inputs"

    D, B, L_row, L_col = a.shape
    assert L_row == L_col, "L_row must equal L_col for stage-3 einsum"
    L = L_row

    out = torch.empty_like(a)

    stride_a_d, stride_a_b = a.stride(0), a.stride(1)
    stride_b_d, stride_b_b = b.stride(0), b.stride(1)
    stride_out_d, stride_out_b = out.stride(0), out.stride(1)

    def grid(meta):
        num_m = triton.cdiv(L, meta["TILE_M"])
        num_n = triton.cdiv(L, meta["TILE_N"])
        return (num_m * num_n, D * B)

    _batched_einsum_kernel[grid](
        a,
        b,
        out,
        D,
        B,
        L,
        stride_a_d,
        stride_a_b,
        stride_b_d,
        stride_b_b,
        stride_out_d,
        stride_out_b,
        TRANSPOSE_A=transpose_a,
        TRANSPOSE_B=transpose_b,
    )
    return out


class TrimulBatchedEinsum(torch.autograd.Function):
    """Autograd-aware wrapper for trimul stage-3 batched einsum.

    Forward direction is one of:
        "outgoing": out[d,b,i,j] = sum_k a[d,b,i,k] * b[d,b,j,k]    (A · B^T)
        "incoming": out[d,b,i,j] = sum_k a[d,b,k,i] * b[d,b,k,j]    (A^T · B)
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, a: torch.Tensor, b: torch.Tensor, direction: str
    ) -> torch.Tensor:
        if direction == "outgoing":
            out = _batched_einsum(a, b, transpose_a=False, transpose_b=True)
        elif direction == "incoming":
            out = _batched_einsum(a, b, transpose_a=True, transpose_b=False)
        else:
            raise ValueError(f"unknown direction {direction!r}")
        ctx.save_for_backward(a, b)
        ctx.direction = direction
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        a, b = ctx.saved_tensors
        direction = ctx.direction
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        if direction == "outgoing":
            grad_a = _batched_einsum(grad_out, b, transpose_a=False, transpose_b=False)
            grad_b = _batched_einsum(grad_out, a, transpose_a=True, transpose_b=False)
        else:  # incoming
            grad_a = _batched_einsum(b, grad_out, transpose_a=False, transpose_b=True)
            grad_b = _batched_einsum(a, grad_out, transpose_a=False, transpose_b=False)
        return grad_a, grad_b, None


def trimul_batched_einsum(
    a: torch.Tensor, b: torch.Tensor, direction: str
) -> torch.Tensor:
    """Public entry: dispatches to autograd Function if grad is enabled."""
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return TrimulBatchedEinsum.apply(a, b, direction)  # type: ignore[return-value]
    if direction == "outgoing":
        return _batched_einsum(a, b, transpose_a=False, transpose_b=True)
    elif direction == "incoming":
        return _batched_einsum(a, b, transpose_a=True, transpose_b=False)
    raise ValueError(f"unknown direction {direction!r}")
