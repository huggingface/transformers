# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op


@triton.jit
def _compute_expert_block(
    E_idx,
    E_mask,
    M_in_idx,
    N_block,
    N_mask,
    X_ptr,
    stride_xm,
    stride_xk,
    W_ptr,
    stride_we,
    stride_wk,
    stride_wn,
    K,
    acc,
    no_k_mask,
    BLOCK_K,
):
    K_block = tl.arange(0, BLOCK_K)
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we
    iters = tl.cdiv(K, BLOCK_K)

    for K_block_id in range(iters):
        if no_k_mask:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])

        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc = tl.dot(x, w, acc, allow_tf32=True)

    return acc


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4)],
    key=["N", "K"],
)
@triton.jit
def scatter2scatter_triton_kernel(
    X_ptr,
    stride_xm,
    stride_xk,
    W_ptr,
    stride_we,
    stride_wk,
    stride_wn,
    Y_ptr,
    stride_ym,
    stride_yn,
    grouped_idx_ptr,
    expert_idxs_ptr,
    FAN_OUT,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    x_grouped,
    y_grouped,
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT

    M_block = M_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    M_boundary_mask = M_block < (FAN_OUT * M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)

    no_k_mask = K % BLOCK_K == 0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask).to(tl.int32)

    for E_idx in range(E_first_idx, E_last_idx + 1):
        E_mask = E_idxs == E_idx
        E_M_idx = M_idx
        if x_grouped:
            M_in_idx = M_block
        else:
            M_in_idx = E_M_idx // FAN_OUT

        acc = _compute_expert_block(
            E_idx,
            E_mask,
            M_in_idx,
            N_block,
            N_mask,
            X_ptr,
            stride_xm,
            stride_xk,
            W_ptr,
            stride_we,
            stride_wk,
            stride_wn,
            K,
            acc,
            no_k_mask,
            BLOCK_K,
        )
    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=M_boundary_mask[:, None] & N_mask[None, :])


@custom_op("transformers::scatter2scatter", mutates_args={"out"})
def scatter2scatter(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    out: torch.Tensor,
    FAN_OUT: int,
    x_grouped: bool = False,
    y_grouped: bool = False,
) -> None:
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * FAN_OUT
    assert out.size(0) == sorted_expert_idxs.size(0)
    assert out.size(1) == W.size(-1)

    def grid(meta):
        return (triton.cdiv(sorted_expert_idxs.size(0), meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)

    BLOCK_M = 128

    with torch.device(X.device):
        scatter2scatter_triton_kernel[grid](
            # X_ptr, stride_xm, stride_xk,
            X,
            X.stride(0),
            X.stride(1),
            # W_ptr, stride_we, stride_wk, stride_wn,
            W,
            W.stride(0),
            W.stride(1),
            W.stride(2),
            # Y_ptr, stride_ym, stride_yn,
            out,
            out.stride(0),
            out.stride(1),
            grouped_idx_ptr=sorted_scattered_idxs,
            expert_idxs_ptr=sorted_expert_idxs,
            FAN_OUT=FAN_OUT,
            M=X.size(0),
            K=X.size(1),
            N=out.size(1),
            E=W.size(0),
            BLOCK_M=BLOCK_M,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
        )
