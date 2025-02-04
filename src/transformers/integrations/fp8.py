# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from ..utils import is_accelerate_available, logging

if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)

ACTIVATION_SCHEMES = ["static", "dynamic"]
quant_dtype = torch.float8_e4m3fn

def fp8_quantize(weight: torch.Tensor, scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to FP8."""
    if scale is None:
        # Calculate scale as max value divided by absmax
        scale = 448.0 / weight.abs().max().clamp(min=1e-12)
        # Scale and clamp tensor to FP8 range
        qweight = (weight * scale).clamp(min=-448.0, max=448.0)
        scale = scale.float().reciprocal()
    else:
        qweight = (weight * scale.reciprocal()).clamp(min=-448.0, max=448.0)

    qweight = qweight.to(quant_dtype)
    return qweight, scale


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-12,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs per-token-group quantization on input tensor, converting to FP8.
    
    Args:
        x (torch.Tensor): Input tensor to quantize (shape: [..., hidden_dim])
        group_size (int): Size of groups for quantization
        column_major_scales (bool): If True, returns scales in column-major format
        eps (float): Small value to avoid division by zero
        dtype (torch.dtype, optional): FP8 dtype to use. Defaults to platform-specific.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scaling factors
    """
    # Input validation
    assert x.ndim >= 2, "Input tensor must have at least 2 dimensions"
    assert x.shape[-1] % group_size == 0, f"Last dimension ({x.shape[-1]}) must be divisible by group_size ({group_size})"

    # Determine FP8 dtype and limits
    if dtype is None:
        dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
    finfo = torch.finfo(dtype)

    # Reshape input for group-wise operations
    orig_shape = x.shape
    num_groups = x.shape[-1] // group_size
    
    # Reshape to [*, num_groups, group_size]
    x_reshaped = x.view(-1, num_groups, group_size)
    
    # Calculate max absolute values per group
    max_abs = x_reshaped.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps)
    
    # Calculate scales as max_dtype / max_abs
    scales = finfo.max / max_abs
    
    # Quantize values
    x_scaled = (x_reshaped * scales)
    x_quant = x_scaled.clamp(min=finfo.min, max=finfo.max).to(dtype)
    
    # Reshape back to original shape
    x_quant = x_quant.view(orig_shape)
    
    # Process scales
    scales = scales.squeeze(-1)  # Remove the last singleton dimension

    scales = scales.view(-1, num_groups)
    
    # Return reciprocal of scales for compatibility with other operations
    return x_quant, scales.float().reciprocal()

def per_token_group_dequant_fp8(
    x: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantizes FP8 tensor back to floating point using group scales.
    
    Args:
        x (torch.Tensor): Quantized input tensor
        scales (torch.Tensor): Scale factors (reciprocal)
        group_size (int): Size of groups used in quantization
        output_dtype (torch.dtype): Output dtype
    
    Returns:
        torch.Tensor: Dequantized tensor
    """
    # Reshape input for group-wise operations
    orig_shape = x.shape
    num_groups = x.shape[-1] // group_size
    x_reshaped = x.view(-1, num_groups, group_size)
    
    # Ensure scales have correct shape
    if scales.ndim == 2:
        scales = scales.view(-1, num_groups, 1)
    else:
        scales = scales.view(*orig_shape[:-1], num_groups, 1)
    
    # Dequantize
    x_dequant = x_reshaped.to(torch.float32) * scales
    
    # Reshape back and convert to desired dtype
    return x_dequant.view(orig_shape).to(output_dtype)

@torch.compile
def w8a8_block_fp8_matmul(
    input_q: torch.Tensor,  # [batch, seq_len, hidden_dim]
    weight_q: torch.Tensor,  # [out_features, hidden_dim]
    input_scale: torch.Tensor,  # [batch * seq_len, num_input_groups]
    weight_scale: torch.Tensor,  # [num_weight_blocks_m, num_weight_blocks_n]
    block_size: Tuple[int, int],  # (M=128, N=128) for weights
    output_dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Performs blocked matrix multiplication with FP8 quantized matrices.
    
    Args:
        input_q: Quantized input tensor with 1x128 block quantization
        weight_q: Quantized weight tensor with 128x128 block quantization
        input_scale: Scaling factors for input blocks
        weight_scale: Scaling factors for weight blocks
        block_size: Tuple of (M, N) for weight block dimensions
        output_dtype: Desired output dtype
    """
    batch_size, seq_len, hidden_dim = input_q.shape
    out_features = weight_q.shape[0]
    
    # Reshape input for batched matmul
    input_reshaped = input_q.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
    
    # Calculate number of blocks
    num_weight_blocks_m = out_features // block_size[0]
    num_weight_blocks_n = hidden_dim // block_size[1]
    
    # Initialize output tensor
    output = torch.zeros((batch_size * seq_len, out_features), 
                        dtype=torch.float32, 
                        device=input_q.device)
    
    # Process each block
    for i in range(num_weight_blocks_m):
        m_start = i * block_size[0]
        m_end = m_start + block_size[0]
        
        for j in range(num_weight_blocks_n):
            n_start = j * block_size[1]
            n_end = n_start + block_size[1]
            
            # Extract current blocks
            input_block = input_reshaped[:, n_start:n_end]
            weight_block = weight_q[m_start:m_end, n_start:n_end]
            
            # Get corresponding scales
            curr_input_scale = input_scale[:, j:j+1]  # [batch*seq_len, 1]
            curr_weight_scale = weight_scale[i, j]  # scalar
            
            # Dequantize and multiply
            block_result = torch._scaled_mm(
                input_block,
                weight_block.t(),
                scale_a=curr_input_scale,
                scale_b=curr_weight_scale,
                out_dtype=x.dtype
            )
            # block_result = torch.matmul(
            #     input_block.to(torch.float32) * curr_input_scale,
            #     weight_block.to(torch.float32).t() * curr_weight_scale
            # )
            
            # Accumulate result
            output[:, m_start:m_end] += block_result
    
    # Reshape output back to original dimensions
    output = output.view(batch_size, seq_len, out_features)
    
    return output.to(output_dtype)


def fp8_quantize(weight, scale: Optional[torch.Tensor] = None, qdtype=torch.float8_e4m3fn):
    if scale is None:
        # weight, scale = quant_weights(weight, torch.int8, False)
        finfo = torch.finfo(qdtype)
        # Calculate the scale as dtype max divided by absmax
        scale = finfo.max / weight.abs().max().clamp(min=1e-12)
        # scale and clamp the tensor to bring it to
        # the representative range of float8 data type
        # (as default cast is unsaturated)
        qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
        scale = scale.float().reciprocal()
    else:
        qweight = (weight * scale.reciprocal()).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    return qweight, scale


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert e4m3fn weights and scales to e4m3fnuz format for ROCm compatibility."""
    if weight.dtype != torch.float8_e4m3fn:
        return weight, weight_scale, input_scale
        
    # Convert -128 (NaN in e4m3fnuz) to 0
    weight_as_int8 = weight.view(torch.int8)
    weight_as_int8[weight_as_int8 == -128] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # Double scales since e4m3fnuz values are half of e4m3fn
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale

class FP8Linear(nn.Module):
    """FP8 Linear layer implementation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device=None,
        dtype=None,
        activation_scheme="dynamic",
        weight_block_size=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=quant_dtype, device=device))
        self.weight_scale = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device))
        
        if activation_scheme == "static":
            self.input_scale = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device))
        else:
            self.register_parameter('input_scale', None)
            
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle ROCm compatibility
        # Standard FP8 matmul
        if self.activation_scheme == "dynamic":
            qinput, self.input_scale = per_token_group_quant_fp8(input, self.weight_block_size[1])

        weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
            self.weight, self.weight_scale, self.input_scale
        )
            
        output = w8a8_block_fp8_matmul(
                    qinput,
                    weight,
                    input_scale,
                    weight_scale,
                    self.weight_block_size,
                    output_dtype=input.dtype,
                )
        
        if self.bias is not None:
            output = output + self.bias
                
        return output

class FP8MoELinear(FP8Linear):
    """FP8 Linear layer for MoE implementation."""
    
    def __init__(
        self,
        n_experts: int,
        in_features: int,
        out_features: int,
        bias: bool,
        device=None,
        dtype=None,
        activation_scheme="dynamic",
        weight_block_size=None
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
            activation_scheme,
            weight_block_size
        )
        self.n_experts = n_experts
        
        # Reshape weight and scale for experts
        self.weight = nn.Parameter(
            torch.empty((n_experts, out_features, in_features), 
            dtype=quant_dtype, 
            device=device)
        )
        self.weight_scale = nn.Parameter(
            torch.empty((n_experts, 1), 
            dtype=torch.float32, 
            device=device)
        )

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        # Handle ROCm compatibility
        weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
            self.weight, self.weight_scale, self.input_scale
        )
        
        if self.activation_scheme == "dynamic":
            input_scale = x.abs().max() / torch.finfo(quant_dtype).max
        
        # Select expert weights and scales
        selected_weights = weight[expert_indices]
        selected_scales = weight_scale[expert_indices]
        
        # Perform FP8 matmul for each expert
        output = torch._scaled_mm(
            x,
            selected_weights.transpose(-1, -2),
            scale_a=input_scale,
            scale_b=selected_scales,
            out_dtype=x.dtype
        )
        
        if self.bias is not None:
            output = output + self.bias[expert_indices]
            
        return output
    
def _replace_with_fp8_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """Replace Linear layers with FP8Linear."""
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)
        
        if isinstance(module, nn.Linear) and name not in (modules_to_not_convert or []):
            current_key_name_str = ".".join(current_key_name)
            if not any(key in current_key_name_str for key in (modules_to_not_convert or [])):
                with init_empty_weights():
                    # Check if this is an MoE layer
                    is_moe = any(moe_key in current_key_name_str 
                               for moe_key in ["gate", "experts"])
                    is_moe = False
                    
                    if is_moe:
                        n_experts = getattr(model.config, "num_experts", 8)
                        model._modules[name] = FP8MoELinear(
                            n_experts=n_experts,
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias is not None,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            activation_scheme=quantization_config.activation_scheme,
                            weight_block_size=quantization_config.weight_block_size
                        )
                    else:
                        model._modules[name] = FP8Linear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias is not None,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            activation_scheme=quantization_config.activation_scheme,
                            weight_block_size=quantization_config.weight_block_size
                        )
                    has_been_replaced = True
                    
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_fp8_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
            
        current_key_name.pop(-1)
        
    return model, has_been_replaced

def replace_with_fp8_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
):
    """Helper function to replace model layers with FP8 versions."""
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    
    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    
    model, has_been_replaced = _replace_with_fp8_linear(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )
    
    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )
    
    return model