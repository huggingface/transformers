# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op


@triton.autotune(configs=[triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=4)], key=["K"])
@triton.jit
def group_triton_kernel(
    src_ptr,
    stride_sn,
    stride_sk,
    has_coeff: tl.constexpr,
    coeff_ptr,
    FAN_OUT,
    tgt_ptr,
    stride_tn,
    stride_ti,
    grouped_idx_ptr,
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    N_block_id = tl.program_id(axis=0)

    N_blk = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_blk < N
    N_blk = tl.max_contiguous(tl.multiple_of(N_blk % N, BLOCK_N), BLOCK_N)
    N_idx = tl.load(grouped_idx_ptr + N_blk, mask=N_mask)

    K_blk = tl.arange(0, BLOCK_K)
    src_blk_ptrs = src_ptr + (N_idx // FAN_OUT)[:, None] * stride_sn + K_blk[None, :] * stride_sk
    tgt_blk_ptrs = tgt_ptr + N_blk[:, None] * stride_tn + K_blk[None, :] * stride_ti

    if has_coeff:
        c = tl.load(coeff_ptr + N_idx, mask=N_mask)[:, None]

    iters = tl.cdiv(K, BLOCK_K)
    no_k_mask = K % BLOCK_K == 0

    for i in range(iters):
        if no_k_mask or i < iters - 1:
            block = tl.load(src_blk_ptrs, mask=N_mask[:, None])

            if has_coeff:
                block *= c

            tl.store(tgt_blk_ptrs, block, mask=N_mask[:, None])
        else:
            K_mask = (i * BLOCK_K + K_blk) < K
            mask = N_mask[:, None] & K_mask[None, :]
            block = tl.load(src_blk_ptrs, mask=mask)

            if has_coeff:
                block *= c

            tl.store(tgt_blk_ptrs, block, mask=mask)

        src_blk_ptrs += BLOCK_K * stride_sk
        tgt_blk_ptrs += BLOCK_K * stride_ti


@custom_op("transformers::group", mutates_args={"out"})
def group(
    A: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    out: torch.Tensor,
    coeff: torch.Tensor | None = None,
    fan_out: int = 1,
) -> None:
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N

    def grid(meta):
        return (triton.cdiv(meta["N"], meta["BLOCK_N"]),)

    with torch.device(A.device):
        group_triton_kernel[grid](
            # A_ptr, stride_an, stride_ai,
            A,
            A.stride(0),
            A.stride(1),
            coeff is not None,
            coeff,
            fan_out,
            # Y_ptr, stride_yn, stride_yk,
            out,
            out.stride(0),
            out.stride(1),
            # grouped_idx_ptr,
            sorted_expert_idxs,
            # N: tl.constexpr, K: tl.constexpr,
            N,
            K,
        )
