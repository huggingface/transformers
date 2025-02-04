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
from torch.nn import functional as F
import triton
import triton.language as tl
from triton import Config

if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)

ACTIVATION_SCHEMES = ["dynamic"]
quant_dtype = torch.float8_e4m3fn

# def fp8_quantize(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Quantize weights to FP8."""
   
#         # Calculate scale as max value divided by absmax
#     scale = 448.0 / weight.abs().max().clamp(min=1e-12)
#     # Scale and clamp tensor to FP8 range
#     qweight = (weight * scale).clamp(min=-448.0, max=448.0)
#     scale = scale.float().reciprocal()

#     qweight = qweight.to(quant_dtype)
#     return qweight, scale

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c

def linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: Optional[torch.Tensor] = None, block_size: Optional[Tuple[int, int]] = None, activation_scheme: str = "dynamic") -> torch.Tensor:
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    else:
        if block_size is None:
            block_size = 128
        else : 
            block_size = 128
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight_scale)
        if bias is not None:
            y += bias
        return y


class FP8Linear(nn.Module):
    dtype = torch.float8_e4m3fn

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, block_size: Optional[Tuple[int, int]] = None, device=None, activation_scheme="dynamic"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=FP8Linear.dtype, device=device))
        # if self.weight.element_size() == 1:
        if block_size is None:
            block_size = self.weight.shape
        scale_out_features = (out_features + block_size[0] - 1) // block_size[0]
        scale_in_features = (in_features + block_size[1] - 1) // block_size[1]
        self.weight_scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32, device=device))
        # else:
        #     self.register_parameter("weight_scale", None)

        self.block_size = block_size

        if activation_scheme == "dynamic":
            self.register_parameter("input_scale", None)
        else : 
            raise ValueError(f"Only dynamic activation scheme is supported for FP8Linear for now, you provided {activation_scheme}")
        self.activation_scheme = activation_scheme
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(self.weight_scale)
        return linear(x, self.weight, self.weight_scale, self.bias, self.block_size, self.activation_scheme)
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

        
        if self.activation_scheme == "dynamic":
            input_scale = x.abs().max() / torch.finfo(quant_dtype).max
        
        # Select expert weights and scales
        selected_weights = self.weight[expert_indices]
        selected_scales = self.weight_scale[expert_indices]
        
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
                    model._modules[name] = FP8Linear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        activation_scheme=quantization_config.activation_scheme,
                        block_size=quantization_config.weight_block_size
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