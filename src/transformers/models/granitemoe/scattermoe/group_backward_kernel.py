# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 128}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def groupXtY_triton_kernel(
    DY_ptr,
    stride_dym,
    stride_dyk,
    X_ptr,
    stride_xm,
    stride_xn,
    DW_ptr,
    stride_dwe,
    stride_dwk,
    stride_dwn,
    expert_offsets_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, 128)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)

    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)

    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :] * stride_dyk

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

        iters = tl.cdiv(end_idx - start_idx, BLOCK_M)

        no_k_mask = K % BLOCK_K == 0
        no_n_mask = N % BLOCK_N == 0

        for i in range(iters):
            M_mask = (i * BLOCK_M + M_block) < end_idx

            if no_k_mask:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])

            if no_n_mask:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])

            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym
            acc = tl.dot(xt, dy, acc, allow_tf32=True)

        DW_blk_ptrs = DW_ptr + E_idx * stride_dwe + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn
        acc = acc.to(DW_blk_ptrs.dtype.element_ty)
        tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])


@custom_op("transformers::group_bwd_W", mutates_args={"DW"})
def group_bwd_W(DY: torch.Tensor, X: torch.Tensor, expert_offsets: torch.Tensor, DW: torch.Tensor, E: int) -> None:
    def grid(meta):
        return (E * triton.cdiv(meta["K"], meta["BLOCK_K"]), triton.cdiv(meta["N"], meta["BLOCK_N"]))

    with torch.device(X.device):
        groupXtY_triton_kernel[grid](
            # DY_ptr, stride_dym, stride_dyk,
            DY,
            DY.stride(0),
            DY.stride(1),
            # X_ptr, stride_xm, stride_xn,
            X,
            X.stride(0),
            X.stride(1),
            # DW_ptr, stride_dwe, stride_dwk, stride_dwn,
            DW,
            DW.stride(0),
            DW.stride(1),
            DW.stride(2),
            # expert_offsets_ptr,
            expert_offsets,
            # K: tl.constexpr, N: tl.constexpr,
            N=DY.size(-1),
            K=X.size(-1),
        )
