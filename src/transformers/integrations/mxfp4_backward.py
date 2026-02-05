# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Backward kernels and autograd wrappers for MXFP4 quantized operations.

This module enables training/fine-tuning with MXFP4-stored weights by providing:
1. SwiGLU backward pass implementation
2. MatmulOGS autograd wrapper with gradient computation for activations (dX)

Note: Weight gradients (dW) are NOT currently supported. This enables LoRA/adapter
fine-tuning where base weights are frozen, but not full fine-tuning of quantized weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils import is_torch_available, is_triton_available, logging


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from torch import Tensor

if is_triton_available():
    import triton
    import triton.language as tl


# =============================================================================
# Constants
# =============================================================================

MXFP_BLOCK_SIZE = 32  # Number of elements per scale block in MXFP4

# FP4 dequantization lookup table (same values as in mxfp4.py)
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


# =============================================================================
# SwiGLU Backward Implementation
# =============================================================================


@dataclass(frozen=True)
class SwiGLUBackwardConfig:
    """Configuration for SwiGLU backward pass."""

    alpha: float = 1.702  # Swish scaling factor
    limit: float | None = 7.0  # Clamp limit for saturation


if is_triton_available():

    @triton.jit
    def _swiglu_backward_kernel(
        # Gradients
        DOut,  # [M, N/2] - gradient from upstream
        DA,  # [M, N] - gradient to downstream (output)
        # Forward inputs (for recomputation)
        A,  # [M, N] - original input
        # Strides
        stride_do_m,
        stride_do_n,
        stride_da_m,
        stride_da_n,
        stride_a_m,
        stride_a_n,
        # Dimensions
        M,
        N_half,  # N/2
        # Parameters
        alpha,
        limit,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_LIMIT: tl.constexpr,
    ):
        """
        Backward kernel for SwiGLU activation.

        Forward: out = swish(a_gelu) * (a_linear + 1)
        where:
            a_gelu = a[..., ::2].clamp(max=limit)
            a_linear = a[..., 1::2].clamp(min=-limit, max=limit)
            swish(x) = x * sigmoid(alpha * x)

        Backward:
            d_a_gelu = d_out * (a_linear + 1) * d_swish(a_gelu)
            d_a_linear = d_out * swish(a_gelu)

        where d_swish(x) = sigmoid(alpha*x) * (1 + alpha*x * (1 - sigmoid(alpha*x)))
                        = sigmoid(alpha*x) + alpha*x * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))

        For clamped values, gradients are zeroed.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Masks
        mask_m = offs_m < M
        mask_n = offs_n < N_half
        mask = mask_m[:, None] & mask_n[None, :]

        # Load upstream gradient
        dout_ptrs = DOut + offs_m[:, None] * stride_do_m + offs_n[None, :] * stride_do_n
        d_out = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Load original inputs (interleaved: gelu at even indices, linear at odd)
        # a_gelu = a[..., ::2], a_linear = a[..., 1::2]
        offs_n_gelu = offs_n * 2  # Even indices
        offs_n_linear = offs_n * 2 + 1  # Odd indices

        a_gelu_ptrs = A + offs_m[:, None] * stride_a_m + offs_n_gelu[None, :] * stride_a_n
        a_linear_ptrs = A + offs_m[:, None] * stride_a_m + offs_n_linear[None, :] * stride_a_n

        a_gelu_raw = tl.load(a_gelu_ptrs, mask=mask, other=0.0).to(tl.float32)
        a_linear_raw = tl.load(a_linear_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Apply clamping and compute masks for zero gradients
        if HAS_LIMIT:
            # a_gelu clamped at max=limit
            gelu_saturated = a_gelu_raw > limit
            a_gelu = tl.where(gelu_saturated, limit, a_gelu_raw)

            # a_linear clamped at min=-limit, max=limit
            linear_saturated_high = a_linear_raw > limit
            linear_saturated_low = a_linear_raw < -limit
            linear_saturated = linear_saturated_high | linear_saturated_low
            a_linear = tl.where(linear_saturated_high, limit, a_linear_raw)
            a_linear = tl.where(linear_saturated_low, -limit, a_linear)
        else:
            a_gelu = a_gelu_raw
            a_linear = a_linear_raw
            gelu_saturated = tl.zeros_like(a_gelu, dtype=tl.int1)
            linear_saturated = tl.zeros_like(a_linear, dtype=tl.int1)

        # Compute swish(a_gelu) = a_gelu * sigmoid(alpha * a_gelu)
        alpha_x = alpha * a_gelu
        sigmoid_alpha_x = tl.sigmoid(alpha_x)
        swish_gelu = a_gelu * sigmoid_alpha_x

        # Compute d_swish(a_gelu)
        # d_swish(x)/dx = sigmoid(alpha*x) + alpha*x * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))
        d_swish = sigmoid_alpha_x + alpha_x * sigmoid_alpha_x * (1.0 - sigmoid_alpha_x)

        # Compute gradients
        # d_a_linear = d_out * swish(a_gelu)
        d_a_linear = d_out * swish_gelu

        # d_a_gelu = d_out * (a_linear + 1) * d_swish(a_gelu)
        d_a_gelu = d_out * (a_linear + 1.0) * d_swish

        # Zero gradients where saturated
        if HAS_LIMIT:
            d_a_gelu = tl.where(gelu_saturated, 0.0, d_a_gelu)
            d_a_linear = tl.where(linear_saturated, 0.0, d_a_linear)

        # Store gradients back to interleaved positions
        da_gelu_ptrs = DA + offs_m[:, None] * stride_da_m + offs_n_gelu[None, :] * stride_da_n
        da_linear_ptrs = DA + offs_m[:, None] * stride_da_m + offs_n_linear[None, :] * stride_da_n

        tl.store(da_gelu_ptrs, d_a_gelu.to(DA.dtype.element_ty), mask=mask)
        tl.store(da_linear_ptrs, d_a_linear.to(DA.dtype.element_ty), mask=mask)


def swiglu_backward_triton(
    grad_output: Tensor,
    input_a: Tensor,
    alpha: float = 1.702,
    limit: float | None = 7.0,
) -> Tensor:
    """
    Compute backward pass for SwiGLU activation using Triton kernel.

    Args:
        grad_output: Gradient from upstream, shape [M, N/2]
        input_a: Original input to SwiGLU forward, shape [M, N]
        alpha: Swish scaling factor (default: 1.702)
        limit: Clamp limit for saturation (default: 7.0)

    Returns:
        Gradient w.r.t. input_a, shape [M, N]
    """
    M = input_a.shape[0] if input_a.ndim == 2 else input_a.numel() // input_a.shape[-1]
    N = input_a.shape[-1]
    N_half = N // 2

    # Reshape for kernel
    grad_output_2d = grad_output.view(M, N_half).contiguous()
    input_a_2d = input_a.view(M, N).contiguous()
    grad_input = torch.empty_like(input_a_2d)

    # Kernel configuration
    BLOCK_M = 32
    BLOCK_N = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_half, BLOCK_N))

    _swiglu_backward_kernel[grid](
        grad_output_2d,
        grad_input,
        input_a_2d,
        grad_output_2d.stride(0),
        grad_output_2d.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
        input_a_2d.stride(0),
        input_a_2d.stride(1),
        M,
        N_half,
        alpha,
        limit if limit is not None else 0.0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_LIMIT=limit is not None,
        num_warps=4,
    )

    return grad_input.view_as(input_a)


def swiglu_backward_torch(
    grad_output: Tensor,
    input_a: Tensor,
    alpha: float = 1.702,
    limit: float | None = 7.0,
) -> Tensor:
    """
    Compute backward pass for SwiGLU activation using PyTorch ops (reference implementation).

    This is used as fallback when Triton is not available or for testing.
    """
    a_gelu_raw = input_a[..., ::2]
    a_linear_raw = input_a[..., 1::2]

    # Apply clamping
    if limit is not None:
        gelu_saturated = a_gelu_raw > limit
        a_gelu = a_gelu_raw.clamp(max=limit)

        linear_saturated = (a_linear_raw > limit) | (a_linear_raw < -limit)
        a_linear = a_linear_raw.clamp(min=-limit, max=limit)
    else:
        a_gelu = a_gelu_raw
        a_linear = a_linear_raw
        gelu_saturated = torch.zeros_like(a_gelu, dtype=torch.bool)
        linear_saturated = torch.zeros_like(a_linear, dtype=torch.bool)

    # Compute swish and its derivative
    alpha_x = alpha * a_gelu
    sigmoid_alpha_x = torch.sigmoid(alpha_x)
    swish_gelu = a_gelu * sigmoid_alpha_x

    # d_swish(x)/dx = sigmoid(alpha*x) + alpha*x * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))
    d_swish = sigmoid_alpha_x + alpha_x * sigmoid_alpha_x * (1.0 - sigmoid_alpha_x)

    # Compute gradients
    d_a_linear = grad_output * swish_gelu
    d_a_gelu = grad_output * (a_linear + 1.0) * d_swish

    # Zero gradients where saturated
    if limit is not None:
        d_a_gelu = d_a_gelu.masked_fill(gelu_saturated, 0.0)
        d_a_linear = d_a_linear.masked_fill(linear_saturated, 0.0)

    # Interleave gradients back
    grad_input = torch.empty_like(input_a)
    grad_input[..., ::2] = d_a_gelu
    grad_input[..., 1::2] = d_a_linear

    return grad_input


class SwiGLUFunction(torch.autograd.Function):
    """
    Autograd function for SwiGLU with backward support.

    This wraps the triton_kernels SwiGLU forward and adds backward pass.
    """

    @staticmethod
    def forward(
        ctx,
        input_a: Tensor,
        alpha: float,
        limit: float | None,
        triton_kernels_hub,
        routing_data=None,
    ) -> Tensor:
        """
        Forward pass using triton_kernels SwiGLU.

        Args:
            input_a: Input tensor, shape [..., N] where N is even
            alpha: Swish scaling factor
            limit: Clamp limit
            triton_kernels_hub: The triton kernels module
            routing_data: Optional routing data for MoE

        Returns:
            Output tensor, shape [..., N/2]
        """
        # Import precision config from the hub
        PrecisionConfig = triton_kernels_hub.swiglu.PrecisionConfig

        # Create precision config
        precision_config = PrecisionConfig(limit=limit)

        # Call the original swiglu
        output = triton_kernels_hub.swiglu.swiglu(input_a, alpha, precision_config, routing_data)

        # Save for backward
        ctx.save_for_backward(input_a)
        ctx.alpha = alpha
        ctx.limit = limit

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        """
        Backward pass for SwiGLU.

        Returns gradients for: input_a, alpha, limit, triton_kernels_hub, routing_data
        Only input_a has a gradient.
        """
        (input_a,) = ctx.saved_tensors
        alpha = ctx.alpha
        limit = ctx.limit

        # Use Triton kernel if available, otherwise fallback to PyTorch
        if is_triton_available() and input_a.is_cuda:
            grad_input = swiglu_backward_triton(grad_output, input_a, alpha, limit)
        else:
            grad_input = swiglu_backward_torch(grad_output, input_a, alpha, limit)

        return grad_input, None, None, None, None


def swiglu_with_backward(
    input_a: Tensor,
    alpha: float,
    limit: float | None,
    triton_kernels_hub,
    routing_data=None,
) -> Tensor:
    """
    SwiGLU activation with backward support.

    This is a drop-in replacement for triton_kernels.swiglu.swiglu that supports
    gradient computation for training.
    """
    return SwiGLUFunction.apply(input_a, alpha, limit, triton_kernels_hub, routing_data)


# =============================================================================
# MXFP4 Dequantization Utilities
# =============================================================================


def dequantize_mxfp4_block(
    packed_weights: Tensor,
    scales: Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    Dequantize MXFP4 weights to higher precision for backward computation.

    This performs on-the-fly dequantization without permanently storing the
    full-precision weights.

    Args:
        packed_weights: Packed MXFP4 weights (uint8, 2 values per byte)
        scales: Per-block scales
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized weights in the specified dtype
    """
    device = packed_weights.device

    # Create lookup table
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=device)

    # Unpack: each byte contains 2 FP4 values (low 4 bits and high 4 bits)
    low_nibble = packed_weights & 0x0F
    high_nibble = packed_weights >> 4

    # Lookup values
    low_values = lut[low_nibble.long()]
    high_values = lut[high_nibble.long()]

    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([low_values, high_values], dim=-1).flatten(-2)

    # Apply scales (broadcast over the block)
    # scales shape: [..., num_blocks], unpacked shape: [..., num_blocks * MXFP_BLOCK_SIZE]
    scales_exp = (scales.to(torch.int32) - 127).to(dtype)
    scale_factors = torch.pow(2.0, scales_exp)

    # Reshape for broadcasting
    *prefix_shape, num_blocks = scales.shape
    unpacked = unpacked.view(*prefix_shape, num_blocks, MXFP_BLOCK_SIZE)
    dequantized = unpacked * scale_factors.unsqueeze(-1)
    dequantized = dequantized.view(*prefix_shape, num_blocks * MXFP_BLOCK_SIZE)

    return dequantized


# =============================================================================
# MatmulOGS Backward Implementation
# =============================================================================


class MatmulOGSFunction(torch.autograd.Function):
    """
    Autograd function for matmul_ogs with backward support.

    This enables gradient computation for the input activations (dX).
    Weight gradients (dW) are NOT supported - attempting to compute them
    will raise an error.

    The backward pass correctly handles:
    - Regular matmul (no routing)
    - MoE with gather/scatter indices (routing gradient inversion)
    - MXFP4 weights (on-the-fly dequantization in backward)
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        w,  # Can be Tensor or triton_kernels Tensor
        bias: Tensor | None,
        routing_data,
        gather_indx,
        scatter_indx,
        precision_config,
        gammas: Tensor | None,
        fused_activation,
        triton_kernels_hub,
        alpha: float = 1.702,
        limit: float | None = 7.0,
    ) -> Tensor:
        """
        Forward pass using matmul_ogs.

        Saves necessary tensors for backward computation.
        """
        matmul_ogs = triton_kernels_hub.matmul_ogs.matmul_ogs

        # Call original matmul_ogs
        output = matmul_ogs(
            x,
            w,
            bias,
            routing_data,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=precision_config,
            gammas=gammas,
            fused_activation=fused_activation,
        )

        # Determine if we need to save for backward
        needs_input_grad = x.requires_grad

        if needs_input_grad:
            # Save tensors needed for backward
            # We save the input x, weight info, and routing indices
            ctx.save_for_backward(x, bias, gammas)

            # Store non-tensor attributes
            ctx.w = w
            ctx.routing_data = routing_data
            ctx.gather_indx = gather_indx
            ctx.scatter_indx = scatter_indx
            ctx.precision_config = precision_config
            ctx.fused_activation = fused_activation
            ctx.triton_kernels_hub = triton_kernels_hub
            ctx.alpha = alpha
            ctx.limit = limit

            # If fused_activation was swiglu, we need to save pre-activation output
            # for the swiglu backward
            if fused_activation is not None and fused_activation.specs.name == "swiglu":
                # Re-run without activation to get pre-activation output
                # This is needed for swiglu backward
                output_no_act = matmul_ogs(
                    x,
                    w,
                    bias,
                    routing_data,
                    gather_indx=gather_indx,
                    precision_config=precision_config,
                    gammas=None,  # No scaling in this intermediate
                )
                ctx.pre_activation_output = output_no_act
            else:
                ctx.pre_activation_output = None

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass for matmul_ogs.

        Computes gradient w.r.t. input activations (dX).
        Does NOT compute gradient w.r.t. weights (dW).

        For MoE with routing:
        - Gradient of gather is scatter
        - Gradient of scatter is gather
        """
        x, bias, gammas = ctx.saved_tensors
        w = ctx.w
        routing_data = ctx.routing_data
        gather_indx = ctx.gather_indx
        scatter_indx = ctx.scatter_indx
        precision_config = ctx.precision_config
        fused_activation = ctx.fused_activation
        triton_kernels_hub = ctx.triton_kernels_hub
        alpha = ctx.alpha
        limit = ctx.limit

        grad_x = None

        if ctx.needs_input_grad[0]:  # Need gradient w.r.t. x
            # Handle fused swiglu activation backward first
            if fused_activation is not None and fused_activation.specs.name == "swiglu":
                pre_act = ctx.pre_activation_output
                # Backward through swiglu
                grad_after_act = swiglu_backward_torch(grad_output, pre_act, alpha, limit)
            else:
                grad_after_act = grad_output

            # Apply gamma scaling backward (if gammas were used)
            if gammas is not None:
                grad_after_gamma = grad_after_act * gammas.unsqueeze(-1)
            else:
                grad_after_gamma = grad_after_act

            # Backward through matmul: dX = dY @ W^T
            # For MoE, we need to handle gather/scatter inversion

            # Get weight tensor for backward
            # If MXFP4, we need to dequantize on-the-fly
            w_for_backward = _get_weight_for_backward(w, precision_config, triton_kernels_hub)

            if routing_data is None or routing_data.n_expts_act == 1:
                # Simple case: no MoE routing or single expert
                # dX = dY @ W^T
                if w_for_backward.ndim == 3:
                    # Batched matmul
                    grad_x = torch.bmm(grad_after_gamma.unsqueeze(0), w_for_backward.transpose(-1, -2)).squeeze(0)
                else:
                    grad_x = torch.mm(grad_after_gamma, w_for_backward.t())

            else:
                # MoE case: need to invert gather/scatter
                grad_x = _matmul_ogs_backward_moe(
                    grad_after_gamma,
                    w_for_backward,
                    x,
                    routing_data,
                    gather_indx,
                    scatter_indx,
                )

        # Return gradients for all inputs
        # (x, w, bias, routing_data, gather_indx, scatter_indx, precision_config, gammas, fused_activation, triton_kernels_hub, alpha, limit)
        return grad_x, None, None, None, None, None, None, None, None, None, None, None


def _get_weight_for_backward(w, precision_config, triton_kernels_hub) -> Tensor:
    """
    Get weight tensor in a format suitable for backward computation.

    For MXFP4 weights, this dequantizes on-the-fly to bfloat16.
    """
    # Check if this is a triton_kernels Tensor with MXFP4 data
    if hasattr(w, "storage") and hasattr(w.storage, "data"):
        # This is a triton_kernels Tensor
        w_data = w.storage.data

        # Check if we have MXFP4 scales
        if precision_config is not None and precision_config.weight_scale is not None:
            # MXFP4: need to dequantize
            scales = precision_config.weight_scale
            if hasattr(scales, "storage"):
                scales = scales.storage.data

            # Dequantize
            w_dequant = _dequantize_mxfp4_weight(w_data, scales)
            return w_dequant
        else:
            return w_data
    elif isinstance(w, Tensor):
        return w
    else:
        raise TypeError(f"Unsupported weight type: {type(w)}")


def _dequantize_mxfp4_weight(packed_weights: Tensor, scales: Tensor) -> Tensor:
    """
    Dequantize MXFP4 packed weights for backward computation.

    Args:
        packed_weights: Packed FP4 data, shape [n_experts, K, N/2] (uint8)
        scales: Per-block scales, shape [n_experts, K//32, N]

    Returns:
        Dequantized weights, shape [n_experts, K, N]
    """
    device = packed_weights.device
    dtype = torch.bfloat16

    # Create lookup table
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=device)

    # Get shapes
    n_experts = packed_weights.shape[0]
    K = packed_weights.shape[1]
    N_half = packed_weights.shape[2]
    N = N_half * 2

    # Unpack nibbles
    packed_flat = packed_weights.view(n_experts, K, N_half)
    low_nibble = (packed_flat & 0x0F).long()
    high_nibble = (packed_flat >> 4).long()

    # Lookup
    low_vals = lut[low_nibble]
    high_vals = lut[high_nibble]

    # Interleave to get [n_experts, K, N]
    unpacked = torch.empty(n_experts, K, N, dtype=dtype, device=device)
    unpacked[..., 0::2] = low_vals
    unpacked[..., 1::2] = high_vals

    # Apply scales
    # scales shape: [n_experts, K//32, N] -> need to broadcast over K dimension
    scales_exp = (scales.to(torch.int32).view(torch.uint8).to(torch.int32) - 127)
    scale_factors = torch.pow(2.0, scales_exp.to(dtype))

    # Broadcast scales over the block size (32 elements in K dimension)
    K_blocks = K // MXFP_BLOCK_SIZE
    scale_factors_expanded = scale_factors.unsqueeze(-2).expand(n_experts, K_blocks, MXFP_BLOCK_SIZE, N)
    scale_factors_expanded = scale_factors_expanded.reshape(n_experts, K, N)

    dequantized = unpacked * scale_factors_expanded

    return dequantized


def _matmul_ogs_backward_moe(
    grad_output: Tensor,
    w: Tensor,
    x: Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
) -> Tensor:
    """
    Backward pass for MoE matmul with gather/scatter routing.

    The forward pass is:
        gathered_x = x[gather_indx]
        y_experts = gathered_x @ W
        y = scatter(y_experts, scatter_indx)

    The backward pass is:
        grad_y_experts = gather(grad_y, scatter_indx^{-1})  # Inverse of scatter
        grad_gathered_x = grad_y_experts @ W^T
        grad_x = scatter(grad_gathered_x, gather_indx^{-1})  # Inverse of gather

    For our routing indices:
        - scatter_indx.src_indx: source indices for scatter (expert outputs -> final)
        - scatter_indx.dst_indx: destination indices
        - gather_indx.src_indx: source indices for gather (input -> expert inputs)
        - gather_indx.dst_indx: destination indices
    """
    n_expts_act = routing_data.n_expts_act
    n_expts_tot = routing_data.n_expts_tot
    expt_hist = routing_data.expt_hist

    # Get number of output rows (before scatter)
    if scatter_indx is not None:
        n_expert_outputs = scatter_indx.src_indx.shape[0]
    else:
        n_expert_outputs = grad_output.shape[0]

    # Step 1: Inverse scatter - gather gradients from scattered output positions
    # scatter_indx.dst_indx tells us where each expert output went
    # We need to gather from those positions
    if scatter_indx is not None:
        dst_idx = scatter_indx.dst_indx
        valid_mask = dst_idx != -1

        # Initialize gradient for expert outputs
        grad_y_experts = torch.zeros(
            n_expert_outputs, grad_output.shape[-1], dtype=grad_output.dtype, device=grad_output.device
        )

        # For valid indices, gather from grad_output
        valid_dst = dst_idx[valid_mask]
        valid_src_positions = torch.arange(n_expert_outputs, device=dst_idx.device)[valid_mask]

        # Each expert output position gets gradient from its scatter destination
        # Since scatter can accumulate, we need to be careful
        # dst_indx maps src -> dst, so we gather grad_output[dst] into expert positions
        grad_y_experts[valid_src_positions] = grad_output[valid_dst // n_expts_act]
    else:
        grad_y_experts = grad_output

    # Step 2: Backward through per-expert matmul
    # grad_gathered_x = grad_y_experts @ W^T
    # W shape: [n_experts, K, N] -> need [n_experts, N, K] for transpose

    # Compute expert offsets from histogram
    if expt_hist is not None:
        offs = torch.zeros(n_expts_tot + 1, dtype=torch.int32, device=expt_hist.device)
        offs[1:] = torch.cumsum(expt_hist, 0)

        grad_gathered_x = torch.zeros(
            n_expert_outputs, x.shape[-1], dtype=grad_output.dtype, device=grad_output.device
        )

        for expert_idx in range(n_expts_tot):
            lo, hi = offs[expert_idx].item(), offs[expert_idx + 1].item()
            if hi > lo:
                expert_grad_y = grad_y_experts[lo:hi]
                expert_w = w[expert_idx] if w.ndim == 3 else w
                grad_gathered_x[lo:hi] = torch.mm(expert_grad_y, expert_w.t())
    else:
        # Single batch processing
        if w.ndim == 3:
            grad_gathered_x = torch.bmm(grad_y_experts.unsqueeze(0), w.transpose(-1, -2)).squeeze(0)
        else:
            grad_gathered_x = torch.mm(grad_y_experts, w.t())

    # Step 3: Inverse gather - scatter gradients back to input positions
    # gather_indx.src_indx tells us where each gathered input came from
    # We need to scatter-add gradients back to those positions
    if gather_indx is not None:
        src_idx = gather_indx.src_indx
        valid_mask = src_idx != -1

        # Initialize gradient for input
        grad_x = torch.zeros(x.shape[0], x.shape[-1], dtype=grad_output.dtype, device=grad_output.device)

        # Scatter-add: for each gathered position, add gradient to original position
        valid_src = src_idx[valid_mask]
        valid_gathered_positions = torch.arange(n_expert_outputs, device=src_idx.device)[valid_mask]

        # Each input position may receive gradients from multiple experts
        grad_x.index_add_(0, valid_src // n_expts_act, grad_gathered_x[valid_gathered_positions])
    else:
        grad_x = grad_gathered_x

    return grad_x


def matmul_ogs_with_backward(
    x: Tensor,
    w,
    bias: Tensor | None,
    routing_data,
    gather_indx,
    scatter_indx,
    precision_config,
    gammas: Tensor | None,
    fused_activation,
    triton_kernels_hub,
    alpha: float = 1.702,
    limit: float | None = 7.0,
) -> Tensor:
    """
    matmul_ogs with backward support for training.

    This is a drop-in replacement for triton_kernels.matmul_ogs.matmul_ogs that
    supports gradient computation for the input activations.

    NOTE: Weight gradients (dW) are NOT supported. If w.requires_grad is True,
    this will raise an error.

    Args:
        x: Input activations
        w: Weight tensor (can be MXFP4 quantized)
        bias: Optional bias
        routing_data: MoE routing data
        gather_indx: Gather indices for MoE
        scatter_indx: Scatter indices for MoE
        precision_config: Precision configuration (contains MXFP4 scales)
        gammas: Optional scaling factors
        fused_activation: Fused activation configuration
        triton_kernels_hub: The triton kernels module
        alpha: Swish alpha for SwiGLU (default: 1.702)
        limit: Clamp limit for SwiGLU (default: 7.0)

    Returns:
        Output tensor

    Raises:
        NotImplementedError: If w.requires_grad is True
    """
    # Check if weight gradients are requested
    if hasattr(w, "requires_grad") and w.requires_grad:
        raise NotImplementedError(
            "Weight gradients (dW) are not supported for MXFP4 quantized weights. "
            "MXFP4 training currently only supports gradient computation for input activations (dX), "
            "which enables LoRA/adapter fine-tuning where base weights are frozen. "
            "To use MXFP4 for training, ensure your quantized weights have requires_grad=False."
        )

    return MatmulOGSFunction.apply(
        x,
        w,
        bias,
        routing_data,
        gather_indx,
        scatter_indx,
        precision_config,
        gammas,
        fused_activation,
        triton_kernels_hub,
        alpha,
        limit,
    )
