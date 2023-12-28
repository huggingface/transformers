from typing import Union, Optional
import torch
from dataclasses import dataclass
import triton
import triton.language as tl

# Based on https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py


@dataclass
class CVMMSel:
    raw_sel: torch.Tensor
    sel: torch.Tensor
    sel_index: torch.Tensor
    out_index: Optional[torch.Tensor] = None
    reduction_weight: Optional[torch.Tensor] = None

    def clone(self) -> 'CVMMSel':
        return CVMMSel(self.raw_sel, self.sel, self.sel_index, self.out_index, self.reduction_weight)


def cvmm_prepare_sel(sel: torch.Tensor, n_experts: int) -> CVMMSel:
    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()
    return CVMMSel(sel, ssel.view_as(sel), sel_index, None)



# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'float32', 'allow_tf32']
)
@triton.jit
def cvmm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bo, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_index, stride_sel, stride_out_index,
    float32: tl.constexpr, allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m = first_pid_m + (pid % group_size_m)

    sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
    sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel)
    sel_all = tl.load(sel_ptr + stride_sel * ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M))

    for matrix_id in range(sel_first, sel_last + 1):
        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetics` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        remap_offs_am = tl.load(index_ptr + stride_index * offs_am)

        # Create offset pointers
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (remap_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + matrix_id * stride_bo + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.

            if not float32:
                a = a.to(tl.float16)
                b = b.to(tl.float16)

            accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk


        if not float32:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        if out_index_ptr is not None:
            remap_offs_cm = tl.load(out_index_ptr + stride_out_index * offs_am)
        else:
            remap_offs_cm = remap_offs_am

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = ((offs_cm[:, None] < M) & (sel_all[:, None] == matrix_id)) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)







@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 4}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'float32_out', 'allow_tf32', 'op_float16'], reset_to_zero = ['c_ptr']
)
@triton.jit
def cvmm_backward_kernel3(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_co, stride_cm, stride_cn,
    stride_index, stride_sel, stride_out_index,
    float32_out: tl.constexpr, allow_tf32: tl.constexpr, op_float16: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, K_BLOCKS: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    k_block_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.

    a_ptrs_this = a_ptr + offs_am[:, None] * stride_am
    b_ptrs_this = b_ptr + offs_bn[None, :] * stride_bn

    # Kactual = end_i - start_i
    # Nblocks = (Kactual + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # WORK_PER_WORKER = (Nblocks + K_BLOCKS - 1) // K_BLOCKS
    # WORK_PER_WORKER = WORK_PER_WORKER if WORK_PER_WORKER > MIN_WORK_SIZE else MIN_WORK_SIZE


    # # Kloop_start = (Kactual + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # first_block_k = k_block_id * WORK_PER_WORKER
    # last_block_k = min((k_block_id+1) * WORK_PER_WORKER, Nblocks)

    block_start_index = k_block_id * BLOCK_SIZE_K * K_BLOCKS
    block_end_index = min(block_start_index + BLOCK_SIZE_K * K_BLOCKS, K) - 1

    first_mat = tl.load(sel_ptr + stride_sel * block_start_index)
    last_mat = tl.load(sel_ptr + stride_sel * block_end_index)


    for matrix_index in range(first_mat, last_mat + 1):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        start_i = block_start_index
        end_i = block_end_index + 1
        while start_i < end_i:
            middle = (start_i + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix < matrix_index:
                start_i = middle + 1
            else:
                end_i = middle


        # # Continue binary search: find the first matrix that is > matrix_index
        start_i2 = start_i
        end_i = block_end_index + 1
        while start_i2 < end_i:
            middle = (start_i2 + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix <= matrix_index:
                start_i2 = middle + 1
            else:
                end_i = middle

        end_i = start_i2

        count = end_i - start_i

        block_mem_indices_f_base = start_i  + tl.arange(0, BLOCK_SIZE_K)

        if count > 0:
            for k in range((count + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
                # block_mem_indices = (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
                block_mem_indices_f = block_mem_indices_f_base + k * BLOCK_SIZE_K
                block_mem_indices = block_mem_indices_f % K
                a_index = tl.load(index_ptr + stride_index * block_mem_indices)
                if out_index_ptr is not None:
                    b_index = tl.load(out_index_ptr + stride_out_index * block_mem_indices)
                else:
                    b_index = a_index
                sel_ok = block_mem_indices_f < end_i

                a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk

                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs, mask=sel_ok[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=sel_ok[:, None], other=0.0)

                if op_float16:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)

                # We accumulate along the K dimension.
                accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            if float32_out:
                c = accumulator
            else:
                c = accumulator.to(tl.float16)

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_co * matrix_index + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            # tl.store(c_ptrs, c, mask=c_mask)
            tl.atomic_add(c_ptrs, c, mask=c_mask)


torch.library.define("mylib::cvmm_triton", "(Tensor x, Tensor sel_idx, Tensor sel, Tensor keys, ScalarType out_dtype, Tensor? out_index) -> Tensor")

@torch.library.impl("mylib::cvmm_triton", "default")
def cvmm_triton(x: torch.Tensor, sel_index: torch.Tensor, sel: torch.Tensor, keys: torch.Tensor, out_dtype: torch.dtype, out_index: Optional[torch.Tensor] = None):
    xorig = x
    x = x.flatten(end_dim=-2)
    assert x.shape[-1] == keys.shape[1]

    sel_shape = sel.shape
    sel = sel.flatten()

    M = sel.shape[0]
    O, K, N = keys.shape
    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    # out = torch.zeros((M, N), device=x.device, dtype=out_dtype)
    # 1D launch kernel where each block gets its own program.

    # expected_m_per_matrix = int(math.ceil(M / O * 1.5))
    # expected_m_per_matrix = M

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    cvmm_kernel[grid](
        x, keys, out, sel_index, sel, out_index,
        M, N, K,
        x.stride(0), x.stride(1),
        keys.stride(0), keys.stride(1), keys.stride(2),
        out.stride(0), out.stride(1),
        sel_index.stride(0), sel.stride(0), out_index.stride(0) if out_index is not None else 0,
        float32=out.dtype==torch.float32, allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
    )

    return out.view(*sel_shape, N)


@torch.library.impl_abstract("mylib::cvmm_triton", cvmm_triton)
def cvmm_triton_abstract(x, sel_idx, sel, keys, out_dtype, out_index):
    sel_shape = sel.shape
    sel = sel.flatten()
    M = sel.shape[0]
    O, K, N = keys.shape
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    sel_shape = sel.shape
    return out.view(*sel_shape, N)


def cvmm_triton_backward(x: torch.Tensor, sel_index: torch.Tensor, sel: torch.Tensor, grads: torch.Tensor, n_experts: int, key_dtype: torch.dtype,
                         op_float16: bool, out_index: Optional[torch.Tensor] = None):
    x = x.flatten(end_dim=-2)
    x = x.transpose(0, 1)

    grads = grads.flatten(end_dim=-2)
    sel = sel.flatten()

    M, _ = x.shape
    K, N = grads.shape

    out = torch.zeros((n_experts, M, N), device=x.device, dtype=key_dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K'] * META['K_BLOCKS'])
    )

    cvmm_backward_kernel3[grid](
        x, grads, out, sel_index, sel, out_index,
        M, N, K,
        x.stride(0), x.stride(1),
        grads.stride(0), grads.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        sel_index.stride(0), sel.stride(0), out_index.stride(0) if out_index is not None else 0,
        float32_out=out.dtype == torch.float32,
        op_float16=op_float16,
        allow_tf32=False #torch.backends.cuda.matmul.allow_tf32
    )

    return out



class CVMM(torch.autograd.Function):
    warned = False

    @staticmethod
    def forward(ctx, x: torch.Tensor, sel_index: torch.Tensor, sel: torch.Tensor, keys: torch.Tensor, out_index: Optional[torch.Tensor] = None, reduction_weight: Optional[torch.Tensor] = None):
        ctx.save_for_backward(x, keys, sel, sel_index, out_index, reduction_weight)

        out_type = torch.float16 if torch.is_autocast_enabled() else x.dtype
        # if torch.is_autocast_enabled():
        #     x = x.half()
        #     keys = keys.half()
        res = torch.ops.mylib.cvmm_triton(x, sel_index, sel, keys, out_type, out_index)
        # res = cvmm_triton(x, sel_index, sel, keys, out_type, out_index)
        if reduction_weight is not None:
            res = res.view(*reduction_weight.shape, res.shape[-1])
            res = (reduction_weight.unsqueeze(-2).type_as(res) @ res).squeeze(-2)

        ctx.op_type = out_type
        ctx.keys_type = keys.dtype
        ctx.is_autocast = torch.is_autocast_enabled()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x, keys, sel, sel_index, out_index, reduction_weight = ctx.saved_tensors

        # if torch.is_autocast_enabled():
        #     x = x.half()
        #     keys = keys.half()
        #     grad_output = grad_output.half()

        # x = x.type(ctx.op_type)
        # keys_dt = keys.type_as(x)
        keys_dt = keys

        # Backward for weight
        if reduction_weight is not None:
            # Project back the grads with he reduction weight, so the grad for the weight matrix is ok
            grad_output_w = reduction_weight.unsqueeze(-1).type_as(grad_output) @ grad_output.unsqueeze(-2)
        else:
            grad_output_w  = grad_output

        grad_w = cvmm_triton_backward(x, sel_index, sel, grad_output_w, keys_dt.shape[0], ctx.keys_type, ctx.is_autocast, out_index=out_index)

        # Backward for input and reduction weight
        grad_w_off = None

        bw_index = sel_index if out_index is None else out_index
        bw_index_out = None
        if reduction_weight is not None:
            # Hack the output indices to emulate repeats
            bw_index_out = bw_index
            bw_index = bw_index // reduction_weight.shape[-1]

        grad_x_full = torch.ops.mylib.cvmm_triton(grad_output, bw_index, sel, keys_dt.transpose(1,2), ctx.op_type, bw_index_out)


        grad_x_full = grad_x_full.view(*x.shape[:-1], -1, x.shape[-1])
        if reduction_weight is not None:
            # grad_x_full is the unscaled grad. For the input, we have to scale it, for the reduction wegiht,
            # we have to compute dot products with the input.
            grad_x = (reduction_weight.view(*grad_x_full.shape[:-1]).unsqueeze(-2).type_as(grad_x_full) @ grad_x_full).squeeze(-2)
            grad_w_off = (grad_x_full.type_as(reduction_weight) @ x.unsqueeze(-1).type_as(reduction_weight)).squeeze(-1).view_as(reduction_weight)
        elif grad_x_full.shape[-2] != 1:
            grad_x = grad_x_full.sum(-2)
        else:
            grad_x = grad_x_full

        grad_x = grad_x.view_as(x)

        return grad_x, None, None, grad_w, None, grad_w_off


def cvmm(x: torch.Tensor, sel: Union[torch.Tensor, CVMMSel], keys: torch.Tensor):
    if not isinstance(sel, CVMMSel):
        sel = cvmm_prepare_sel(sel, keys.shape[0])

    return CVMM.apply(x, sel.sel_index, sel.sel, keys, sel.out_index, sel.reduction_weight)


def cvmm_prepare_sel2(sel: torch.Tensor, w: Optional[torch.Tensor] = None) -> CVMMSel:
    # Has multiple selections for each batch element
    n_per_batch = sel.shape[-1]

    # indices = torch.arange(sel.nelement() // n_per_batch, device=sel.device, dtype=torch.int32)
    # indices = indices.repeat_interleave(n_per_batch).flatten()

    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()

    # in_index = indices[sel_index]
    in_index = sel_index // n_per_batch

    return CVMMSel(sel, ssel.view_as(sel), in_index, sel_index, w)
