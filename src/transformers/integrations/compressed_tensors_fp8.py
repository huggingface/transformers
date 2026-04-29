# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Compressed-tensors FP8 integration for transformers.

Supports loading compressed-tensors FP8 checkpoints (per-channel and per-tensor)
via dequantization to BF16 followed by standard matmul. The primary benefit is
memory savings (FP8 weights use half the memory of BF16).

Supported models:
  - Per-channel dynamic: e.g. RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
  - Per-tensor static: e.g. RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..core_model_loading import ConversionOps, _IdentityOp
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_fbgemm_gpu_available, is_torch_xpu_available, logging


logger = logging.get_logger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max

_is_torch_xpu_available = is_torch_xpu_available()

if is_fbgemm_gpu_available() and not _is_torch_xpu_available:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

# Will be initialized lazily in replace_with_ct_fp8_linear
quantize_fp8_per_row = None


class CTFP8Linear(nn.Linear):
    """Linear layer for compressed-tensors FP8 models.

    Stores weights in FP8 format and uses row-wise FP8 matmul kernels for compute:
    - XPU: torch._scaled_mm
    - CUDA: fbgemm.f8f8bf16_rowwise

    Activation is dynamically quantized per-row via quantize_fp8_per_row.
    Weight scale (per-channel or per-tensor) is stored as weight_scale_inv.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_scheme: str = "dynamic",
        has_bias: bool = False,
        dtype=_FP8_DTYPE,
    ):
        super().__init__(in_features, out_features)

        self.has_bias = has_bias
        self.activation_scheme = activation_scheme
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

        # Weight scale: per-channel (out_features, 1) or per-tensor (scalar → expanded at load)
        self.weight_scale_inv = nn.Parameter(torch.zeros((out_features, 1), dtype=torch.float32))

        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If weights are not FP8 (e.g. already dequantized), just do normal linear
        if self.weight.element_size() > 1:
            return F.linear(input, self.weight, self.bias)

        # Save shape for restoring after squashing batch dims
        output_shape = (*input.shape[:-1], -1)

        # Dynamically quantize activation per-row to FP8
        x_quantized, x_scale = quantize_fp8_per_row(
            input.view(-1, input.shape[-1]).contiguous()
        )

        weight_scale_float32 = self.weight_scale_inv.to(torch.float32)

        # Ensure scale_b has shape (1, out_features) for row-wise _scaled_mm
        # Per-channel: (out_features, 1) → .t() → (1, out_features) ✓
        # Per-tensor: (1, 1) → need to expand to (1, out_features)
        scale_b = weight_scale_float32.t()
        if scale_b.shape[-1] == 1 and self.out_features > 1:
            scale_b = scale_b.expand(1, self.out_features).contiguous()

        if _is_torch_xpu_available:
            output = torch._scaled_mm(
                x_quantized,
                self.weight.t(),
                scale_a=x_scale.unsqueeze(-1),
                scale_b=scale_b,
                out_dtype=input.dtype,
                bias=self.bias,
            )
        else:
            output = torch.ops.fbgemm.f8f8bf16_rowwise(
                x_quantized, self.weight, x_scale, weight_scale_float32, use_fast_accum=True
            )
            output = output + self.bias if self.bias is not None else output

        output = output.to(input.device)
        output = output.reshape(output_shape)
        del x_quantized, x_scale
        return output


def replace_with_ct_fp8_linear(
    model, modules_to_not_convert=None, activation_scheme="dynamic", dequantize=False, pre_quantized=False
):
    """Replace all nn.Linear modules with CTFP8Linear for compressed-tensors FP8 loading."""
    from .fbgemm_fp8 import get_quantize_fp8_per_row

    global quantize_fp8_per_row
    quantize_fp8_per_row = get_quantize_fp8_per_row()

    if dequantize:
        return model

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        module_kwargs = {} if pre_quantized else {"dtype": None}
        if isinstance(module, nn.Linear):
            with torch.device("meta"):
                new_module = CTFP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    activation_scheme=activation_scheme,
                    has_bias=module.bias is not None,
                    **module_kwargs,
                )
            model.set_submodule(module_name, new_module)
            has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using compressed-tensors FP8 but no linear modules were found. "
            "Please double check your model architecture."
        )
    return model


# ─── Weight Converters ────────────────────────────────────────────────────────


class CompressedTensorsScaleConvert(ConversionOps):
    """Convert compressed-tensors `weight_scale` to `weight_scale_inv`.

    In compressed-tensors, `weight_scale` is the dequantization multiplier:
        bf16_weight = fp8_weight * weight_scale

    In our CTFP8Linear, `weight_scale_inv` has the same semantics (it's multiplied
    with the FP8 weight to get the dequantized value), so no inversion is needed.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, **kwargs):
        # The key in input_dict is the source_pattern string (e.g. "weight_scale$")
        scale_key = next(k for k in input_dict if "weight_scale" in k)
        scale = input_dict[scale_key][0]

        # Cast to float32, ensure (out_features, 1) shape for row-wise kernel
        dequant_scale = scale.to(torch.float32)
        if dequant_scale.dim() == 0:
            # Per-tensor scalar → expand to (out_features, 1)
            # out_features is inferred from the weight shape at runtime
            # For now store as (1, 1) and let _scaled_mm broadcast
            dequant_scale = dequant_scale.reshape(1, 1)
        elif dequant_scale.dim() == 1:
            # Per-channel (N,) → (N, 1)
            dequant_scale = dequant_scale.unsqueeze(-1)
        # else: already 2D (N, 1), keep as-is

        return {"weight_scale_inv": dequant_scale}

    @property
    def reverse_op(self):
        return _IdentityOp()


class CompressedTensorsActivationScaleConvert(ConversionOps):
    """Rename compressed-tensors `input_scale` to `activation_scale`."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, **kwargs):
        scale = input_dict["input_scale"][0]
        return {"activation_scale": scale.to(torch.float32)}

    @property
    def reverse_op(self):
        return _IdentityOp()


class CompressedTensorsFp8Dequantize(ConversionOps):
    """Dequantize compressed-tensors FP8 weights back to BF16.

    Used when `dequantize=True`: loads FP8 weights + scale, produces BF16 weights.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, full_layer_name=None, **kwargs):
        if len(input_dict) < 2:
            weight_key = next(k for k in input_dict if "weight" in k)
            return {full_layer_name: input_dict[weight_key]}

        weight_key = next(k for k in input_dict if k.endswith("weight") or k.endswith("weight$"))
        scale_key = next(k for k in input_dict if "weight_scale" in k and "inv" not in k)
        quantized = input_dict[weight_key][0]
        scale = input_dict[scale_key][0]

        quantized_float = quantized.to(torch.float32)
        if scale.dim() == 0:
            # Per-tensor: scalar scale
            dequantized = quantized_float * scale
        elif scale.dim() == 1:
            # Per-channel: (N,) scale, broadcast over K dimension
            dequantized = quantized_float * scale.unsqueeze(-1)
        else:
            dequantized = quantized_float * scale

        return {full_layer_name: dequantized.to(torch.bfloat16)}

    @property
    def reverse_op(self):
        return _IdentityOp()

