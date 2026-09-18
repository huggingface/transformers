# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantized layers for Gemma: INT2/4/8 packed-weight Linear and Embedding,
plus SRQ (Static Range Quantization) activation rounding."""

import os
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import ACT2FN
from ..utils import is_triton_available


HAS_TRITON_GEMM = is_triton_available()

if HAS_TRITON_GEMM:
    import triton
    import triton.language as tl


try:
    from .gemma_quant_cpu import highway_table_gemm

    HAS_HIGHWAY_CPU = True
except ImportError:
    HAS_HIGHWAY_CPU = False
    highway_table_gemm = None


if HAS_TRITON_GEMM:

    @triton.jit
    def _triton_gemv_int4_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        N,
        K,
        K_PACKED,
        stride_wn,
        stride_wk,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_PACKED: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in range(0, K_PACKED, BLOCK_K_PACKED):
            offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
            mask_k = offs_kp < K_PACKED

            offs_k_low = offs_kp * 2
            offs_k_high = offs_kp * 2 + 1

            mask_low = offs_k_low < K
            mask_high = offs_k_high < K

            x_low = tl.load(x_ptr + offs_k_low, mask=mask_low, other=0.0).to(tl.float32)
            x_high = tl.load(x_ptr + offs_k_high, mask=mask_high, other=0.0).to(tl.float32)

            mask_b = mask_n[:, None] & mask_k[None, :]
            w_packed = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk, mask=mask_b, other=0
            )

            low = ((w_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
            high = (((w_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

            low = tl.where(mask_low[None, :], low, 0.0)
            high = tl.where(mask_high[None, :], high, 0.0)

            acc += tl.sum(low * x_low[None, :] + high * x_high[None, :], axis=1)

        scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        acc = acc * scale

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias

        tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=mask_n)

    @triton.jit
    def _triton_gemv_int2_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        N,
        K,
        K_PACKED,
        stride_wn,
        stride_wk,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_PACKED: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in range(0, K_PACKED, BLOCK_K_PACKED):
            offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
            mask_k = offs_kp < K_PACKED

            offs_k0 = offs_kp * 4
            offs_k1 = offs_kp * 4 + 1
            offs_k2 = offs_kp * 4 + 2
            offs_k3 = offs_kp * 4 + 3

            mask0 = offs_k0 < K
            mask1 = offs_k1 < K
            mask2 = offs_k2 < K
            mask3 = offs_k3 < K

            x0 = tl.load(x_ptr + offs_k0, mask=mask0, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + offs_k1, mask=mask1, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + offs_k2, mask=mask2, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + offs_k3, mask=mask3, other=0.0).to(tl.float32)

            mask_b = mask_n[:, None] & mask_k[None, :]
            w_packed = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk, mask=mask_b, other=0
            )

            v0 = ((w_packed & 0x03).to(tl.int8) - 2).to(tl.float32)
            v1 = (((w_packed >> 2) & 0x03).to(tl.int8) - 2).to(tl.float32)
            v2 = (((w_packed >> 4) & 0x03).to(tl.int8) - 2).to(tl.float32)
            v3 = (((w_packed >> 6) & 0x03).to(tl.int8) - 2).to(tl.float32)

            v0 = tl.where(mask0[None, :], v0, 0.0)
            v1 = tl.where(mask1[None, :], v1, 0.0)
            v2 = tl.where(mask2[None, :], v2, 0.0)
            v3 = tl.where(mask3[None, :], v3, 0.0)

            acc += tl.sum(v0 * x0[None, :] + v1 * x1[None, :] + v2 * x2[None, :] + v3 * x3[None, :], axis=1)

        scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        acc = acc * scale

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias

        tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=mask_n)

    @triton.jit
    def _triton_gemv_int8_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        N,
        K,
        stride_wn,
        stride_wk,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_val = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

            mask_b = mask_n[:, None] & mask_k[None, :]
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=mask_b, other=0).to(
                tl.float32
            )

            acc += tl.sum(w * x_val[None, :], axis=1)

        scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        acc = acc * scale

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias

        tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=mask_n)

    @triton.jit
    def _triton_gemm_int4_kernel(
        a_ptr,
        b_ptr,
        scale_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        K_PACKED,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_PACKED: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K_PACKED, BLOCK_K_PACKED):
            offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
            mask_k = offs_kp < K_PACKED

            offs_k_low = offs_kp * 2
            offs_k_high = offs_kp * 2 + 1

            mask_m = offs_m[:, None] < M
            mask_low = mask_m & (offs_k_low[None, :] < K)
            mask_high = mask_m & (offs_k_high[None, :] < K)

            a_low = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k_low[None, :] * stride_ak, mask=mask_low, other=0.0
            )
            a_high = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k_high[None, :] * stride_ak, mask=mask_high, other=0.0
            )

            mask_n = offs_n[:, None] < N
            mask_b = mask_n & mask_k[None, :]
            b_packed = tl.load(
                b_ptr + offs_n[:, None] * stride_bn + offs_kp[None, :] * stride_bk, mask=mask_b, other=0
            )

            low = ((b_packed & 0x0F).to(tl.int8) - 8).to(a_ptr.dtype.element_ty)
            high = (((b_packed >> 4) & 0x0F).to(tl.int8) - 8).to(a_ptr.dtype.element_ty)

            low = tl.where(mask_b, low, 0.0)
            high = tl.where(mask_b, high, 0.0)

            acc = tl.dot(a_low, tl.trans(low), acc)
            acc = tl.dot(a_high, tl.trans(high), acc)

        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
        acc = acc * scale[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(
            c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
            acc.to(c_ptr.dtype.element_ty),
            mask=mask_c,
        )

    @triton.jit
    def _triton_gemm_int2_kernel(
        a_ptr,
        b_ptr,
        scale_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        K_PACKED,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_PACKED: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K_PACKED, BLOCK_K_PACKED):
            offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
            mask_k = offs_kp < K_PACKED

            offs_k0 = offs_kp * 4
            offs_k1 = offs_kp * 4 + 1
            offs_k2 = offs_kp * 4 + 2
            offs_k3 = offs_kp * 4 + 3

            mask_m = offs_m[:, None] < M
            mask0 = mask_m & (offs_k0[None, :] < K)
            mask1 = mask_m & (offs_k1[None, :] < K)
            mask2 = mask_m & (offs_k2[None, :] < K)
            mask3 = mask_m & (offs_k3[None, :] < K)

            a0 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k0[None, :] * stride_ak, mask=mask0, other=0.0)
            a1 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k1[None, :] * stride_ak, mask=mask1, other=0.0)
            a2 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k2[None, :] * stride_ak, mask=mask2, other=0.0)
            a3 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k3[None, :] * stride_ak, mask=mask3, other=0.0)

            mask_n = offs_n[:, None] < N
            mask_b = mask_n & mask_k[None, :]
            b_packed = tl.load(
                b_ptr + offs_n[:, None] * stride_bn + offs_kp[None, :] * stride_bk, mask=mask_b, other=0
            )

            v0 = ((b_packed & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)
            v1 = (((b_packed >> 2) & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)
            v2 = (((b_packed >> 4) & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)
            v3 = (((b_packed >> 6) & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)

            v0 = tl.where(mask_b, v0, 0.0)
            v1 = tl.where(mask_b, v1, 0.0)
            v2 = tl.where(mask_b, v2, 0.0)
            v3 = tl.where(mask_b, v3, 0.0)

            acc = tl.dot(a0, tl.trans(v0), acc)
            acc = tl.dot(a1, tl.trans(v1), acc)
            acc = tl.dot(a2, tl.trans(v2), acc)
            acc = tl.dot(a3, tl.trans(v3), acc)

        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
        acc = acc * scale[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(
            c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
            acc.to(c_ptr.dtype.element_ty),
            mask=mask_c,
        )

    @triton.jit
    def _triton_gemm_int8_kernel(
        a_ptr,
        b_ptr,
        scale_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            mask_m = offs_m[:, None] < M
            mask_a = mask_m & mask_k[None, :]
            a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0)

            mask_n = offs_n[:, None] < N
            mask_b = mask_n & mask_k[None, :]
            b_tile = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk, mask=mask_b, other=0)

            b_cast = b_tile.to(a_ptr.dtype.element_ty)
            b_cast = tl.where(mask_b, b_cast, 0.0)

            acc = tl.dot(a_tile, tl.trans(b_cast), acc)

        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
        acc = acc * scale[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(
            c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
            acc.to(c_ptr.dtype.element_ty),
            mask=mask_c,
        )

    @triton.jit
    def _triton_grouped_gemm_int4_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        offsets_ptr,
        out_ptr,
        N,
        K,
        K_PACKED,
        stride_xm,
        stride_xk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_PACKED: tl.constexpr,
    ):
        expert_id = tl.program_id(0)
        pid_n = tl.program_id(1)

        start_idx = tl.load(offsets_ptr + expert_id)
        end_idx = tl.load(offsets_ptr + expert_id + 1)
        M_e = end_idx - start_idx
        if M_e <= 0:
            return

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        for m_start in range(0, M_e, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M_e

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k in range(0, K_PACKED, BLOCK_K_PACKED):
                offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
                mask_k = offs_kp < K_PACKED

                offs_k0 = offs_kp * 2
                offs_k1 = offs_kp * 2 + 1

                mask_a0 = mask_m[:, None] & (offs_k0[None, :] < K)
                mask_a1 = mask_m[:, None] & (offs_k1[None, :] < K)

                a0 = tl.load(
                    x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k0[None, :] * stride_xk,
                    mask=mask_a0,
                    other=0.0,
                )
                a1 = tl.load(
                    x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k1[None, :] * stride_xk,
                    mask=mask_a1,
                    other=0.0,
                )

                mask_b = mask_n[:, None] & mask_k[None, :]
                b_packed = tl.load(
                    w_ptr + expert_id * stride_we + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk,
                    mask=mask_b,
                    other=0,
                )

                low = ((b_packed & 0x0F).to(tl.int8) - 8).to(x_ptr.dtype.element_ty)
                high = (((b_packed >> 4) & 0x0F).to(tl.int8) - 8).to(x_ptr.dtype.element_ty)

                low = tl.where(mask_b, low, 0.0)
                high = tl.where(mask_b, high, 0.0)

                acc = tl.dot(a0, tl.trans(low), acc)
                acc = tl.dot(a1, tl.trans(high), acc)

            scale = tl.load(scale_ptr + expert_id * stride_se + offs_n * stride_sn, mask=mask_n, other=1.0).to(
                tl.float32
            )
            acc = acc * scale[None, :]

            tl.store(
                out_ptr + (start_idx + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on,
                acc.to(out_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )


def triton_grouped_gemm_int4(
    permuted_x: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Fused Grouped GEMM for low-bit MoE experts (int4). Dispatches all experts in a single kernel."""
    P = permuted_x.shape[0]
    num_experts = packed_w.shape[0]
    N = packed_w.shape[1]
    K = permuted_x.shape[1]
    K_PACKED = packed_w.shape[2]

    if scales.dim() == 3:
        scales = scales.squeeze(-1)

    out = torch.empty((P, N), dtype=permuted_x.dtype, device=permuted_x.device)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K_PACKED = 32

    grid = (num_experts, triton.cdiv(N, BLOCK_N))
    _triton_grouped_gemm_int4_kernel[grid](
        permuted_x,
        packed_w,
        scales,
        offsets,
        out,
        N,
        K,
        K_PACKED,
        permuted_x.stride(0),
        permuted_x.stride(1),
        packed_w.stride(0),
        packed_w.stride(1),
        packed_w.stride(2),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K_PACKED=BLOCK_K_PACKED,
    )
    return out


def triton_lowbit_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    num_bits: int = 4,
) -> torch.Tensor:
    """Fused low-bit GEMV/GEMM for quantized weights (int2, int4, int8).

    Streams packed integer weights directly from VRAM into registers, completely
    eliminating intermediate float weight allocations.
    """
    orig_shape = x.shape
    in_features = orig_shape[-1]
    out_features = weight_scale.shape[0]

    x_2d = x.contiguous().view(-1, in_features)
    M = x_2d.shape[0]
    N = out_features
    K = in_features

    weight_scale = weight_scale.squeeze(-1) if weight_scale.dim() > 1 else weight_scale
    scale_flat = weight_scale.contiguous()
    bias_flat = bias.contiguous() if bias is not None else None

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    if M == 1:
        # Fast path for single-token autoregressive decoding (GEMV)
        x_vec = x_2d.squeeze(0)
        out_vec = out.squeeze(0)
        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N),)
        if num_bits == 4:
            BLOCK_K_PACKED = 64
            K_PACKED = (K + 1) // 2
            _triton_gemv_int4_kernel[grid](
                x_vec,
                weight,
                scale_flat,
                bias_flat if bias_flat is not None else x_vec,
                out_vec,
                N,
                K,
                K_PACKED,
                weight.stride(0),
                weight.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_N=BLOCK_N,
                BLOCK_K_PACKED=BLOCK_K_PACKED,
            )
        elif num_bits == 2:
            BLOCK_K_PACKED = 64
            K_PACKED = (K + 3) // 4
            _triton_gemv_int2_kernel[grid](
                x_vec,
                weight,
                scale_flat,
                bias_flat if bias_flat is not None else x_vec,
                out_vec,
                N,
                K,
                K_PACKED,
                weight.stride(0),
                weight.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_N=BLOCK_N,
                BLOCK_K_PACKED=BLOCK_K_PACKED,
            )
        elif num_bits == 8:
            BLOCK_K = 64
            _triton_gemv_int8_kernel[grid](
                x_vec,
                weight,
                scale_flat,
                bias_flat if bias_flat is not None else x_vec,
                out_vec,
                N,
                K,
                weight.stride(0),
                weight.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}")
    else:
        # Tiled GEMM fast path using tl.dot (tensor cores) for prefill and batched decoding
        BLOCK_M = 16 if M <= 16 else 32
        BLOCK_N = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        if num_bits == 4:
            BLOCK_K_PACKED = 32
            K_PACKED = (K + 1) // 2
            _triton_gemm_int4_kernel[grid](
                x_2d,
                weight,
                scale_flat,
                bias_flat if bias_flat is not None else x_2d,
                out,
                M,
                N,
                K,
                K_PACKED,
                x_2d.stride(0),
                x_2d.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                out.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K_PACKED=BLOCK_K_PACKED,
            )
        elif num_bits == 2:
            BLOCK_K_PACKED = 32
            K_PACKED = (K + 3) // 4
            _triton_gemm_int2_kernel[grid](
                x_2d,
                weight,
                scale_flat,
                bias_flat if bias_flat is not None else x_2d,
                out,
                M,
                N,
                K,
                K_PACKED,
                x_2d.stride(0),
                x_2d.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                out.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K_PACKED=BLOCK_K_PACKED,
            )
        elif num_bits == 8:
            BLOCK_K = 32
            _triton_gemm_int8_kernel[grid](
                x_2d,
                weight,
                scale_flat,
                bias_flat if bias_flat is not None else x_2d,
                out,
                M,
                N,
                K,
                x_2d.stride(0),
                x_2d.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                out.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}")

    if len(orig_shape) > 2:
        return out.view(*orig_shape[:-1], out_features)
    return out


def apply_srq(x: torch.Tensor, scale: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Apply Static Range Quantization rounding and clipping (in x's dtype).

    A `scale` of 0 means the layer is uncalibrated, in which case this is a no-op. The guard uses
    `torch.where` rather than `scale.item()` so it stays on-device and `torch.compile`-friendly (an
    `.item()` would force a host-device sync and break `fullgraph=True`).
    """
    scale = scale.to(x.dtype)
    max_value = 2 ** (bits - 1) - 1
    min_value = -max_value - 1
    calibrated = scale != 0
    safe_scale = torch.where(calibrated, scale, torch.ones_like(scale))
    x_q = torch.clamp(torch.round(x / safe_scale), float(min_value), float(max_value)) * safe_scale
    return torch.where(calibrated, x_q, x)


def _unpack_int4(packed: torch.Tensor, original_width: int) -> torch.Tensor:
    """Unpack int4 values from uint8 storage. Two values per byte.

    Each byte: low nibble = first value, high nibble = second value.
    Values are stored unsigned in [0, 15] and shifted to signed [-8, 7].
    Cast to uint8 first so the right shift is logical, not arithmetic.
    """
    packed = packed.to(torch.uint8)
    low = (packed & 0x0F).to(torch.int8) - 8
    high = (packed >> 4).to(torch.int8) - 8
    interleaved = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return interleaved[..., :original_width]


def _unpack_int2(packed: torch.Tensor, original_width: int) -> torch.Tensor:
    """Unpack int2 values from uint8 storage. Four values per byte.

    Bits [1:0]/[3:2]/[5:4]/[7:6] hold values 0..3 each, shifted to signed [-2, 1].
    """
    packed = packed.to(torch.uint8)
    v0 = (packed & 0x03).to(torch.int8) - 2
    v1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
    v3 = (packed >> 6).to(torch.int8) - 2
    interleaved = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], -1)
    return interleaved[..., :original_width]


class QuantizedLinear(nn.Linear):
    """Linear layer with INT2/4/8 packed weights and SRQ activation rounding."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.num_bits = num_bits

        # int2/int4 packed in uint8 (4 / 2 values per byte); int8 stored directly.
        # Replace the inherited fp32 weight with packed-int storage.
        if num_bits == 2:
            packed_in = (in_features + 3) // 4
            weight_storage = torch.empty(out_features, packed_in, dtype=torch.uint8)
        elif num_bits == 4:
            packed_in = (in_features + 1) // 2
            weight_storage = torch.empty(out_features, packed_in, dtype=torch.uint8)
        else:
            weight_storage = torch.empty(out_features, in_features, dtype=torch.int8)
        self.weight = nn.Parameter(weight_storage, requires_grad=False)
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1, dtype=torch.float32))
        # SRQ activation scales — optional, loaded from checkpoint. 0 means uncalibrated, in which
        # case `apply_srq` is a no-op, so `forward` can apply it unconditionally.
        self.input_activation_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.output_activation_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _dequantize_weights(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize weights (handles int2/int4/int8 storage). If `dtype` is given,
        the math runs in that dtype; otherwise int×fp32 promotion gives fp32."""
        if self.num_bits == 2:
            int_weights = _unpack_int2(self.weight, self.in_features)
        elif self.num_bits == 4:
            int_weights = _unpack_int4(self.weight, self.in_features)
        else:
            int_weights = self.weight
        if dtype is None:
            return int_weights * self.weight_scale
        return int_weights.to(dtype) * self.weight_scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = apply_srq(x, self.input_activation_scale)

        # 1. Fast Path: Fused GPU Kernel (CUDA / Triton)
        if (
            x.is_cuda
            and self.weight.is_cuda
            and HAS_TRITON_GEMM
            and self.num_bits in (2, 4, 8)
            and os.environ.get("TRANSFORMERS_GEMMA_DISABLE_TRITON", "0") != "1"
        ):
            out = triton_lowbit_gemm(x, self.weight, self.weight_scale, self.bias, num_bits=self.num_bits)
        # 2. Fast Path: CPU SIMD Table Lookups (HL-GEMM via Highway/C++ extension)
        elif not x.is_cuda and HAS_HIGHWAY_CPU and highway_table_gemm is not None:
            out = highway_table_gemm(x, self.weight, self.weight_scale, self.bias, num_bits=self.num_bits)
        # 3. Eager PyTorch Fallback (Only if compiled kernels unavailable)
        else:
            out = F.linear(x, self._dequantize_weights(x.dtype), self.bias)

        return apply_srq(out, self.output_activation_scale)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, num_bits={self.num_bits}"
        )


class QuantizedEmbedding(nn.Module):
    """Embedding with INT2/4/8 packed table, per-row dequant scale, and architectural embed_scale.

    Does NOT subclass `nn.Embedding` because the packed-int storage isn't a usable
    embedding table on its own: indexing `.embedding_quantized[idx]` returns packed
    bytes, not a row of size `embedding_dim`. Callers expect `embed_tokens.weight[idx, :]`
    to return the *dequantized* row, so we expose `weight` as a property (below)
    that returns the dequantized table on demand.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        output_dtype: torch.dtype,
        embed_scale: float = 1.0,
        num_bits: int = 8,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scalar_embed_scale = embed_scale
        self.num_bits = num_bits
        self.output_dtype = output_dtype

        # int2/int4 packed in uint8 (4 / 2 values per byte); int8 stored directly.
        if num_bits == 2:
            packed_dim = (embedding_dim + 3) // 4
            embed_storage = torch.empty(num_embeddings, packed_dim, dtype=torch.uint8)
        elif num_bits == 4:
            packed_dim = (embedding_dim + 1) // 2
            embed_storage = torch.empty(num_embeddings, packed_dim, dtype=torch.uint8)
        else:
            embed_storage = torch.empty(num_embeddings, embedding_dim, dtype=torch.int8)
        self.embedding_quantized = nn.Parameter(embed_storage, requires_grad=False)
        self.embedding_scale = nn.Parameter(torch.ones(num_embeddings, 1, dtype=torch.float32))

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized embedding table (no architectural `embed_scale` applied).

        Mirrors `nn.Embedding.weight` so callers can do `weight[idx, :]` and get
        the same unscaled row they'd get from a non-quantized embedding.
        """
        return self._dequantize_weights(self.embedding_quantized, self.embedding_scale)

    def _dequantize_weights(self, quant_rows: torch.Tensor, scale_rows: torch.Tensor) -> torch.Tensor:
        """Unpack int2/int4/int8 + apply per-row block-wise dequantization scale."""
        if self.num_bits == 4:
            int_rows = _unpack_int4(quant_rows, self.embedding_dim)
        elif self.num_bits == 2:
            int_rows = _unpack_int2(quant_rows, self.embedding_dim)
        else:
            int_rows = quant_rows

        block_size = self.embedding_dim // scale_rows.shape[-1]
        scale = scale_rows.repeat_interleave(block_size, dim=-1)
        return int_rows.to(self.output_dtype) * scale.to(self.output_dtype)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        result = self._dequantize_weights(self.embedding_quantized[input_ids], self.embedding_scale[input_ids])
        return (result * self.scalar_embed_scale).to(self.output_dtype)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"num_bits={self.num_bits}, embed_scale={self.scalar_embed_scale}"
        )


class QuantizedGemma4TextExperts(nn.Module):
    """Quantized collection of expert weights stored as packed 3D integer tensors."""

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        hidden_activation: str = "gelu_pytorch_tanh",
        act_fn: Callable | None = None,
        num_bits: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_bits = num_bits
        self.act_fn = act_fn if act_fn is not None else ACT2FN[hidden_activation]

        if num_bits == 4:
            packed_hidden = (hidden_dim + 1) // 2
            packed_inter = (intermediate_dim + 1) // 2
            dtype = torch.uint8
        elif num_bits == 2:
            packed_hidden = (hidden_dim + 3) // 4
            packed_inter = (intermediate_dim + 3) // 4
            dtype = torch.uint8
        elif num_bits == 8:
            packed_hidden = hidden_dim
            packed_inter = intermediate_dim
            dtype = torch.int8
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}")

        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, packed_hidden, dtype=dtype),
            requires_grad=False,
        )
        self.gate_up_scale = nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, packed_inter, dtype=dtype),
            requires_grad=False,
        )
        self.down_scale = nn.Parameter(
            torch.ones(num_experts, hidden_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        use_triton = (
            HAS_TRITON_GEMM
            and hidden_states.is_cuda
            and os.environ.get("TRANSFORMERS_GEMMA_DISABLE_TRITON", "0") != "1"
        )

        if use_triton and self.num_bits == 4:
            orig_shape = hidden_states.shape
            hidden_flat = hidden_states.view(-1, self.hidden_dim)
            B = hidden_flat.shape[0]
            top_k = top_k_index.shape[-1]

            flat_topk = top_k_index.view(-1)
            flat_weights = top_k_weights.view(-1)
            flat_tok = torch.arange(B, device=hidden_states.device).repeat_interleave(top_k)

            sorted_expert, sort_perm = torch.sort(flat_topk)
            perm_tok = flat_tok[sort_perm]
            perm_weights = flat_weights[sort_perm]
            perm_x = hidden_flat[perm_tok]

            counts = torch.bincount(flat_topk, minlength=self.num_experts)
            offsets = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=hidden_states.device)
            offsets[1:] = torch.cumsum(counts, dim=0)

            # 1. Fused Grouped GEMM gate_up
            gu = triton_grouped_gemm_int4(perm_x, self.gate_up_proj, self.gate_up_scale, offsets)
            gate, up = gu.chunk(2, dim=-1)
            mid = self.act_fn(gate) * up

            # 2. Fused Grouped GEMM down
            down = triton_grouped_gemm_int4(mid, self.down_proj, self.down_scale, offsets)

            # 3. Scatter-add back into original token order
            final_flat = torch.zeros_like(hidden_flat)
            final_flat.index_add_(0, perm_tok, (down * perm_weights[:, None]).to(final_flat.dtype))
            return final_flat.view(orig_shape)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            if self.num_bits == 4:
                gu_int = _unpack_int4(self.gate_up_proj[expert_idx], self.hidden_dim)
                d_int = _unpack_int4(self.down_proj[expert_idx], self.intermediate_dim)
            elif self.num_bits == 2:
                gu_int = _unpack_int2(self.gate_up_proj[expert_idx], self.hidden_dim)
                d_int = _unpack_int2(self.down_proj[expert_idx], self.intermediate_dim)
            else:
                gu_int = self.gate_up_proj[expert_idx]
                d_int = self.down_proj[expert_idx]
            gu_w = gu_int.to(current_state.dtype) * self.gate_up_scale[expert_idx].to(current_state.dtype)
            d_w = d_int.to(current_state.dtype) * self.down_scale[expert_idx].to(current_state.dtype)
            gate, up = nn.functional.linear(current_state, gu_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, d_w)

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, "
            f"intermediate_dim={self.intermediate_dim}, num_bits={self.num_bits}"
        )


def replace_with_quant_layers(
    model: nn.Module,
    quantization_config=None,
    modules_to_not_convert: list[str] | None = None,
) -> None:
    """Replace `nn.Linear` / `nn.Embedding` / `Gemma4TextExperts` modules with quantized counterparts.

    Per-module bit widths come from `quantization_config.module_quant_configs`.
    `nn.Embedding` modules are only replaced when `quantize_embeddings` is True.
    `Gemma4TextExperts` modules are only replaced when `quantize_experts` is True.
    Modules whose name matches an entry in `modules_to_not_convert` are skipped.
    """
    import re

    from ..quantizers.quantizers_utils import should_convert_module

    quantize_embeddings = getattr(quantization_config, "quantize_embeddings", False)
    quantize_experts = getattr(quantization_config, "quantize_experts", False)
    num_bits = getattr(quantization_config, "num_bits", 4)
    module_quant_configs = getattr(quantization_config, "module_quant_configs", None) or {}

    # Join all the per-module patterns into one regex, compiled once, so each module name needs a
    # single search instead of a loop over patterns. Each pattern is a named group `g0`, `g1`, ...;
    # whichever group matches identifies its override.
    overrides_by_group = {f"g{i}": override for i, override in enumerate(module_quant_configs.values())}
    matcher = (
        re.compile("|".join(f"(?P<g{i}>{pattern})" for i, pattern in enumerate(module_quant_configs)))
        if module_quant_configs
        else None
    )

    for name, module in list(model.named_modules()):
        if not should_convert_module(name, modules_to_not_convert):
            continue
        opts = {"num_bits": num_bits}
        if matcher is not None and (match := matcher.search(name)) is not None:
            override = next(overrides_by_group[g] for g, v in match.groupdict().items() if v is not None)
            opts = {"num_bits": num_bits, **override}
        if isinstance(module, nn.Embedding):
            if not quantize_embeddings:
                continue
            new_module = QuantizedEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                embed_scale=getattr(module, "scalar_embed_scale", 1.0),
                output_dtype=module.weight.dtype,
                **opts,
            )
        elif isinstance(module, nn.Linear):
            new_module = QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                **opts,
            )
        elif type(module).__name__ == "Gemma4TextExperts":
            if not quantize_experts:
                continue
            new_module = QuantizedGemma4TextExperts(
                num_experts=module.num_experts,
                hidden_dim=module.hidden_dim,
                intermediate_dim=module.intermediate_dim,
                act_fn=getattr(module, "act_fn", None),
                **opts,
            )
        else:
            continue
        new_module.requires_grad_(False)
        model.set_submodule(name, new_module)
    return model
