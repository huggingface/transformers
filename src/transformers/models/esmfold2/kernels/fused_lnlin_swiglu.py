"""Fused LayerNorm + Linear(d, 2*d_inner) + SwiGLU(silu(x1) * x2) kernel.

Collapses the standard LayerNorm -> Linear -> chunk -> SiLU -> mul sequence
into a single Triton kernel:

    out[..., d_inner] = silu(x1) * x2     where (x1, x2) = chunk(LN(x) @ W12, 2)

Compared to the unfused PyTorch sequence this kernel:

  1. Eliminates the [M, 2*d_inner] HBM read between Linear and SwiGLU.
  2. Halves X reads in the matmul: each program produces both halves of the
     linear output (against W12_a and W12_b) in one pass over X.
  3. Fuses LayerNorm into the matmul k-loop.

The output ordering uses silu of the FIRST half of W12's output, so the
fused module matches the standard LN+SwiGLU MLP composition.

Backward: SwiGLU bwd is a small Triton kernel that mutates the saved [M, 2N]
linear-output buffer in place to produce dlin (no extra alloc). LN+Linear bwd
uses cuBLAS gemm + ATen's `native_layer_norm_backward` — fast on H100 and
avoids needing per-shape autotuned Triton bwd configs.

Tuned forward configs were autotuned on H100 80GB for the pair-transition
production shapes (d_pair=256, d_inner=1024, M=B*N*N at N=384 / 640).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ln_stats_kernel(
    X_ptr,
    X_row_stride,
    Mean_ptr,
    Mean_row_stride,
    Rstd_ptr,
    Rstd_row_stride,
    K,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-row LayerNorm reduction: writes mean and 1/sqrt(var+eps) for each row of X."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    x = tl.load(X_ptr + row * X_row_stride + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / K
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / K
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean_ptr + row * Mean_row_stride, mean)
    tl.store(Rstd_ptr + row * Rstd_row_stride, rstd)


@triton.jit
def _lnlin_swiglu_fwd_kernel(
    X_ptr,  # [M, K]
    W_ptr,  # [K, 2N]   — first half (cols [0:N]) feeds the silu input;
    #            second half (cols [N:2N]) is the gate output
    LN_W_ptr,  # [K]
    LN_B_ptr,  # [K]
    Lin_ptr,  # [M, 2N]   — full linear output (saved for backward)
    Out_ptr,  # [M, N]    — silu(x1) * x2  (input to the next Linear)
    Mean_ptr,
    Rstd_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_lin_m,
    stride_lin_n,
    stride_out_m,
    stride_out_n,
    HAS_LN_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """One program produces a [BLOCK_M, BLOCK_N] tile of `Out`. To avoid
    re-reading X for the two halves of W12, the program does TWO matmul
    accumulators (against W_a and W_b) in the same K-loop, producing both
    halves of the linear output for the same M tile. Then SwiGLU is applied
    in registers and both the [M, 2N] linear output and [M, N] swiglu output
    are written."""
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mean = tl.load(Mean_ptr + offs_m, mask=offs_m < M, other=0.0)
    rstd = tl.load(Rstd_ptr + offs_m, mask=offs_m < M, other=0.0)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    wa_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    wb_ptrs = W_ptr + (offs_k[:, None] * stride_wk + (N + offs_n[None, :]) * stride_wn)

    a_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    b_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kk = k * BLOCK_SIZE_K
        k_remaining = K - kk
        k_mask = offs_k < k_remaining

        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        ln_w = tl.load(LN_W_ptr + kk + offs_k, mask=k_mask, other=0.0)
        if HAS_LN_BIAS:
            ln_b = tl.load(LN_B_ptr + kk + offs_k, mask=k_mask, other=0.0)
            x_hat = ((x - mean[:, None]) * rstd[:, None]) * ln_w[None, :] + ln_b[
                None, :
            ]
        else:
            x_hat = ((x - mean[:, None]) * rstd[:, None]) * ln_w[None, :]

        wa = tl.load(
            wa_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        wb = tl.load(
            wb_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )

        a_acc = tl.dot(x_hat, wa, a_acc)
        b_acc = tl.dot(x_hat, wb, b_acc)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        wa_ptrs += BLOCK_SIZE_K * stride_wk
        wb_ptrs += BLOCK_SIZE_K * stride_wk

    a_bf = a_acc.to(Lin_ptr.type.element_ty)
    b_bf = b_acc.to(Lin_ptr.type.element_ty)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    lin_a_ptrs = (
        Lin_ptr + offs_m[:, None] * stride_lin_m + offs_n[None, :] * stride_lin_n
    )
    lin_b_ptrs = (
        Lin_ptr + offs_m[:, None] * stride_lin_m + (N + offs_n[None, :]) * stride_lin_n
    )
    tl.store(lin_a_ptrs, a_bf, mask=out_mask)
    tl.store(lin_b_ptrs, b_bf, mask=out_mask)

    # SwiGLU = silu(a) * b. SiLU computed in fp32 for numerical accuracy.
    sig = tl.sigmoid(a_acc)
    silu_a = a_acc * sig
    swiglu = silu_a * b_acc
    out_ptrs = Out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, swiglu.to(Out_ptr.type.element_ty), mask=out_mask)


# SwiGLU backward: in-place into the saved linear-output buffer (no alloc).
@triton.jit
def _swiglu_bwd_inplace_kernel(
    dout_ptr, lin_ptr, M, N, stride_d, stride_l, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    a_ptr = lin_ptr + row * stride_l + cols
    b_ptr = lin_ptr + row * stride_l + N + cols
    do_ptr = dout_ptr + row * stride_d + cols

    a = tl.load(a_ptr, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(a)
    silu = a * sig
    # d(silu)/da = sig + silu*(1 - sig) = sig*(1 + a*(1 - sig))
    silu_grad = sig * (1.0 + a * (1.0 - sig))

    da = do * b * silu_grad
    db = do * silu

    tl.store(a_ptr, da.to(lin_ptr.type.element_ty), mask=mask)
    tl.store(b_ptr, db.to(lin_ptr.type.element_ty), mask=mask)


_FWD_CONFIG_K128 = dict(
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=32,
    GROUP_SIZE_M=8,
    num_stages=4,
    num_warps=8,
)
_FWD_CONFIG_K256 = dict(
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=64,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=8,
    num_stages=3,
    num_warps=4,
)


def _pick_fwd_config(K: int) -> dict:
    if K == 128:
        return _FWD_CONFIG_K128
    if K == 256:
        return _FWD_CONFIG_K256
    # Reasonable default for other K. May be sub-optimal — autotune for new shapes.
    return _FWD_CONFIG_K256


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _ln_stats_settings(K: int) -> tuple[int, int]:
    BLOCK = _next_pow2(K)
    if BLOCK <= 256:
        num_warps = 4
    elif BLOCK <= 1024:
        num_warps = 8
    else:
        num_warps = 16
    return BLOCK, num_warps


def _lnlin_swiglu_fwd(
    x_2d: torch.Tensor, W12: torch.Tensor, LN_W: torch.Tensor, LN_B: torch.Tensor | None
):
    assert x_2d.is_contiguous(), "X must be contiguous"
    M, K = x_2d.shape
    K2, two_N = W12.shape
    assert K2 == K and two_N % 2 == 0, f"W12 shape mismatch: {W12.shape} vs K={K}"
    N = two_N // 2

    out = torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)
    lin = torch.empty((M, two_N), dtype=x_2d.dtype, device=x_2d.device)
    Mean = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)
    Rstd = torch.empty((M,), dtype=x_2d.dtype, device=x_2d.device)

    block, num_warps = _ln_stats_settings(K)
    _ln_stats_kernel[(M,)](
        x_2d,
        x_2d.stride(0),
        Mean,
        Mean.stride(0),
        Rstd,
        Rstd.stride(0),
        K,
        1e-5,
        BLOCK_SIZE=block,
        num_warps=num_warps,  # pyright: ignore[reportCallIssue]
    )

    cfg = _pick_fwd_config(K)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _lnlin_swiglu_fwd_kernel[grid](
        x_2d,
        W12,
        LN_W,
        LN_B if LN_B is not None else LN_W,  # ptr always non-null
        lin,
        out,
        Mean,
        Rstd,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        W12.stride(0),
        W12.stride(1),
        lin.stride(0),
        lin.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_LN_BIAS=(LN_B is not None),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        num_stages=cfg["num_stages"],  # pyright: ignore[reportCallIssue]
        num_warps=cfg["num_warps"],  # pyright: ignore[reportCallIssue]
    )
    return out, lin, Mean, Rstd


def _swiglu_bwd_inplace(dout: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
    """In-place SwiGLU backward: writes da, db into the [M, 2N] `lin` buffer.

    Returns the same tensor (now containing dlin values)."""
    M, N = dout.shape
    BLOCK = _next_pow2(N)
    _swiglu_bwd_inplace_kernel[(M,)](
        dout,
        lin,
        M,
        N,
        dout.stride(0),
        lin.stride(0),
        BLOCK=BLOCK,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return lin


class FusedLNLinearSwiGLUFunction(torch.autograd.Function):
    """
    Forward:  out = silu(x1) * x2 where (x1, x2) = chunk(LN(X) @ W12, 2)
    Backward: standard chain via cuBLAS gemms + ATen native_layer_norm_backward.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, X, W12, LN_W, LN_B):
        x_shape = X.shape
        x_2d = X.contiguous().view(-1, x_shape[-1])
        out, lin, mean, rstd = _lnlin_swiglu_fwd(x_2d, W12, LN_W, LN_B)
        ctx.save_for_backward(x_2d, W12, LN_W, LN_B, mean, rstd, lin)
        ctx.x_shape = x_shape
        ctx.has_ln_bias = LN_B is not None
        return out.view(*x_shape[:-1], out.shape[-1])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        x_2d, W12, LN_W, LN_B, mean, rstd, lin = ctx.saved_tensors
        dout_2d = dout.contiguous().view(-1, dout.shape[-1])
        K = x_2d.shape[1]

        # SwiGLU backward, in-place into the saved `lin` buffer (no new alloc).
        dlin = _swiglu_bwd_inplace(dout_2d, lin)

        # LN+Linear backward via cuBLAS + ATen LN bwd.
        # x_norm needed for dW12; recomputed via F.layer_norm (cuDNN, ~100 µs at d=256).
        x_norm = F.layer_norm(x_2d, (K,), LN_W, LN_B, eps=1e-5)
        dW12 = x_norm.transpose(0, 1) @ dlin  # [K, 2N]
        dx_norm = dlin @ W12.transpose(0, 1)  # [M, K]
        del x_norm

        # native_layer_norm_backward output_mask = [need_dX, need_dGamma, need_dBeta].
        # We always need dX and dGamma; dBeta only when LN bias was used.
        output_mask = [True, True, ctx.has_ln_bias]
        dX, dLN_W, dLN_B = torch.ops.aten.native_layer_norm_backward(
            dx_norm, x_2d, [K], mean.float(), rstd.float(), LN_W, LN_B, output_mask
        )
        # If LN_B was None (no bias), ATen returns dLN_B=None and we just pass it through.
        return dX.view(ctx.x_shape), dW12, dLN_W, dLN_B


class FusedLNLinearSwiGLU(nn.Module):
    """Fused module for `LayerNorm(d) -> Linear(d, 2*hidden) -> silu(x1)*x2`.

    Convention: silu of FIRST half. Output: [..., hidden]."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        has_ln_bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_inner = d_inner
        self.LN_W = nn.Parameter(torch.empty(d_model, **factory))
        self.LN_B = (
            nn.Parameter(torch.empty(d_model, **factory)) if has_ln_bias else None
        )
        self.W12 = nn.Parameter(torch.empty(d_model, 2 * d_inner, **factory))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.LN_W)
        if self.LN_B is not None:
            nn.init.zeros_(self.LN_B)
        bound = 1.0 / math.sqrt(self.d_model)
        nn.init.uniform_(self.W12, -bound, bound)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return FusedLNLinearSwiGLUFunction.apply(  # type: ignore[return-value]
            X, self.W12, self.LN_W, self.LN_B
        )
