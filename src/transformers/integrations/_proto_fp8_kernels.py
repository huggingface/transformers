"""Prototype Triton kernels for the fp8 work package (candidates for `kernels-community/finegrained-fp8`).

- `act_quant_packed_ue8m0`: per-token, per-128-K-group activation quant to e4m3 with power-of-two (UE8M0) scales, four
  scales packed per int32 in the MN-major TMA-aligned layout Blackwell DeepGEMM consumes directly. One launch.
- `fp8_gemm_splitk`: block-scaled fp8 GEMM on pre-quantized activations, K split across programs (see REPORT, Q4).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _act_quant_packed_ue8m0_kernel(x_ptr, y_ptr, s_ptr, K, stride_xm, stride_ym, stride_sg):
    # one program = one row x four 128-wide K groups = one packed int32 of scales
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    cols = pid_g * 512 + tl.arange(0, 512)
    mask = cols < K
    x = tl.load(x_ptr + pid_m * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    xg = tl.reshape(x, (4, 128))
    # same recipe as deep_gemm.utils.per_token_cast_to_fp8(use_ue8m0=True): amax clamped at 1e-4, scale = amax / 448
    # rounded up to a power of two
    sf = tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-4) / 448.0
    bits = sf.to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    inv = 1.0 / (exp << 23).to(tl.float32, bitcast=True)
    y = tl.reshape(xg * inv[:, None], (512,)).to(tl.float8e4nv)
    tl.store(y_ptr + pid_m * stride_ym + cols, y, mask=mask)
    # groups past K carry the exponent of 1.0, like DeepGEMM's own padding
    group = pid_g * 4 + tl.arange(0, 4)
    exp = tl.where(group * 128 < K, exp, 127)
    packed = tl.sum(exp << (8 * tl.arange(0, 4)), axis=0)
    tl.store(s_ptr + pid_g * stride_sg + pid_m, packed)


def act_quant_packed_ue8m0(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] bf16/fp16/fp32 contiguous -> (e4m3 [M, K], int32 [M, ceil(K / 512)] with strides (1, align(M, 4)))."""
    assert x.dim() == 2 and x.is_contiguous()
    m, k = x.shape
    groups = (k + 511) // 512
    y = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty_strided((m, groups), (1, (m + 3) // 4 * 4), dtype=torch.int32, device=x.device)
    _act_quant_packed_ue8m0_kernel[(m, groups)](x, y, s, k, x.stride(0), y.stride(0), s.stride(1), num_warps=4)
    return y, s


@triton.jit
def _act_quant_fp32_kernel(x_ptr, y_ptr, s_ptr, K, stride_xm, stride_ym, stride_sm):
    # one program = one row x four 128-wide K groups; fp32 scales amax / 448 (the hub kernel's inline recipe)
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    cols = pid_g * 512 + tl.arange(0, 512)
    mask = cols < K
    x = tl.load(x_ptr + pid_m * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    xg = tl.reshape(x, (4, 128))
    sf = tl.max(tl.abs(xg), axis=1) / 448.0
    y = tl.reshape(xg / tl.maximum(sf[:, None], 1e-12), (512,)).to(tl.float8e4nv)
    tl.store(y_ptr + pid_m * stride_ym + cols, y, mask=mask)
    group = pid_g * 4 + tl.arange(0, 4)
    tl.store(s_ptr + pid_m * stride_sm + group, sf, mask=group * 128 < K)


def act_quant_fp32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] contiguous -> (e4m3 [M, K], fp32 [M, ceil(K / 128)]), one launch."""
    m, k = x.shape
    y = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, (k + 127) // 128), dtype=torch.float32, device=x.device)
    _act_quant_fp32_kernel[(m, (k + 511) // 512)](x, y, s, k, x.stride(0), y.stride(0), s.stride(0), num_warps=4)
    return y, s


@triton.jit
def _fp8_gemm_splitk_kernel(
    A, As, B, Bs, P, M, N, K,
    stride_am, stride_asm, stride_bn, stride_bsn, stride_ps, stride_pm,
    SPLIT_BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # one program = one (N tile, K split, M tile): a partial sum over SPLIT_BLOCKS 128-wide K blocks, stored in fp32
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(SPLIT_BLOCKS):
        kb = pid_s * SPLIT_BLOCKS + i
        k = kb * 128 + offs_k
        k_mask = k < K
        a = tl.load(A + offs_m[:, None] * stride_am + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(B + offs_n[None, :] * stride_bn + k[:, None], mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        a_s = tl.load(As + offs_m * stride_asm + kb, mask=m_mask & (kb * 128 < K), other=0.0)
        b_s = tl.load(Bs + (offs_n // 128) * stride_bsn + kb, mask=n_mask & (kb * 128 < K), other=0.0)
        acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
    tl.store(P + pid_s * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _sum_splits_kernel(P, C, MN, stride_ps, SPLITS: tl.constexpr):
    offs = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    mask = offs < MN
    acc = tl.zeros((2048,), dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        acc += tl.load(P + s * stride_ps + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=mask)


def splitk_plan(m: int, n: int, k: int, num_sms: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, K blocks per split, splits): enough programs to occupy the GPU, at most one wave."""
    block_m = min(max(triton.next_power_of_2(m), 16), 64)
    block_n = 128 if n >= 1024 else 64
    tiles = -(-m // block_m) * -(-n // block_n)
    k_blocks = -(-k // 128)
    splits = max(1, min(k_blocks, num_sms // tiles))
    split_blocks = -(-k_blocks // splits)
    return block_m, block_n, split_blocks, -(-k_blocks // split_blocks)


def fp8_gemm_splitk(xq, xs, w, ws, out_dtype=torch.bfloat16, num_sms=148, num_warps=4, num_stages=3):
    """C = dequant(xq, xs) @ dequant(w, ws).T for e4m3 xq [M, K] with fp32 scales [M, K/128] and e4m3 w [N, K] with
    fp32 block scales [N/128, K/128]. Three launches with the activation quant: quant, split-K GEMM, sum of splits."""
    m, k = xq.shape
    n = w.shape[0]
    block_m, block_n, split_blocks, splits = splitk_plan(m, n, k, num_sms)
    parts = torch.empty((splits, m, n), dtype=torch.float32, device=xq.device)
    grid = (-(-n // block_n), splits, -(-m // block_m))
    _fp8_gemm_splitk_kernel[grid](
        xq, xs, w, ws, parts, m, n, k,
        xq.stride(0), xs.stride(0), w.stride(0), ws.stride(0), parts.stride(0), parts.stride(1),
        SPLIT_BLOCKS=split_blocks, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages,
    )
    out = torch.empty((m, n), dtype=out_dtype, device=xq.device)
    _sum_splits_kernel[(-(-m * n // 2048),)](parts, out, m * n, parts.stride(0), SPLITS=splits, num_warps=4)
    return out


def triton_splitk_linear(x, w, ws, **kw):
    xq, xs = act_quant_fp32(x)
    return fp8_gemm_splitk(xq, xs, w, ws, out_dtype=x.dtype, **kw)
