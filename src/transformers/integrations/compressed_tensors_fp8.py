# Copyright 2026 The HuggingFace Inc. team and Intel Corporation. All rights reserved.
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
and running inference with hardware-accelerated FP8 matmul kernels. Weights are
kept in FP8 format and activations are dynamically quantized per-row.

Supported models:
  - Per-channel dynamic: e.g. RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
  - Per-tensor static: e.g. RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8
"""

from functools import cache

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..core_model_loading import ConversionOps, _IdentityOp
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import logging


logger = logging.get_logger(__name__)

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


@cache
def _can_use_fp8_kernel():
    """Check if we can use FP8 matmul (XPU or CUDA SM89+)."""
    if torch.xpu.is_available():
        return True
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major > 8 or (major == 8 and minor >= 9):
            return True
    return False


def _quantize_fp8_per_row(x: torch.Tensor):
    """Quantize a 2D tensor to FP8 per-row using pure PyTorch (torch.compile compatible).

    Args:
        x: Input tensor of shape (num_rows, hidden_dim).

    Returns:
        Tuple of (x_fp8, scales) where x_fp8 is the quantized tensor and
        scales is a 1D tensor of shape (num_rows,) with per-row scales.
    """
    x_float = x.to(torch.float32)
    row_max = x_float.abs().amax(dim=-1)  # (num_rows,)
    # Avoid division by zero for all-zero rows
    safe_max = torch.where(row_max > 0, row_max, torch.ones_like(row_max))
    scales = safe_max / _FP8_MAX  # float32
    x_fp8 = torch.clamp(x_float / scales.unsqueeze(-1), min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
    return x_fp8, scales


class CompressedTensorsFP8Linear(nn.Linear):
    """Linear layer for compressed-tensors FP8 models.

    Stores weights in FP8 format and uses torch._scaled_mm for FP8 matmul.
    Activation is dynamically quantized per-row via quantize_fp8_per_row.
    Weight scale (per-channel or per-tensor) is stored as weight_scale.
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
        self.weight_scale = nn.Parameter(torch.zeros((out_features, 1), dtype=torch.float32))

        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If weights are not FP8 (e.g. already dequantized), just do normal linear
        if self.weight.element_size() > 1:
            return F.linear(input, self.weight, self.bias)

        # Save shape for restoring after squashing batch dims, then flatten once
        # (both the kernel and the fallback path operate on a 2D tensor).
        output_shape = (*input.shape[:-1], -1)
        x = input.reshape(-1, input.shape[-1])

        if _can_use_fp8_kernel():
            # XPU or CUDA SM89+: FP8 kernel path (quantize activation + scaled_mm)
            x_quantized, x_scale = _quantize_fp8_per_row(x)

            # Ensure scale_b has shape (1, out_features) for row-wise _scaled_mm.
            # weight_scale is already float32 (declared as such), so no cast needed.
            # Per-channel: (out_features, 1) → .t() → (1, out_features) ✓
            # Per-tensor:  (1, 1) → need to expand to (1, out_features)
            scale_b = self.weight_scale.t()
            is_per_tensor = scale_b.shape[-1] == 1 and self.out_features > 1
            if is_per_tensor:
                # expand() creates a stride-0 view; _scaled_mm requires contiguous scales.
                # The scale tensor is tiny so .contiguous() cost is negligible.
                scale_b = scale_b.expand(1, self.out_features).contiguous()

            output = torch._scaled_mm(
                x_quantized,
                self.weight.t(),
                scale_a=x_scale.unsqueeze(-1),
                scale_b=scale_b,
                out_dtype=input.dtype,
                bias=self.bias,
            )
            del x_quantized, x_scale
        else:
            # CUDA SM80 (A100): no FP8 hardware, dequantize weight to BF16 + normal matmul
            w = self.weight.to(input.dtype) * self.weight_scale.to(input.dtype)
            output = F.linear(x, w, self.bias)

        output = output.reshape(output_shape)
        return output


def replace_with_compressed_tensors_fp8_linear(
    model, modules_to_not_convert=None, activation_scheme="dynamic", pre_quantized=False
):
    """Replace all nn.Linear modules with CompressedTensorsFP8Linear for compressed-tensors FP8 loading."""
    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        module_kwargs = {} if pre_quantized else {"dtype": None}
        if isinstance(module, nn.Linear):
            # This method already runs under a `torch.device("meta")` context manager,
            # so the new module is created on meta automatically.
            new_module = CompressedTensorsFP8Linear(
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
    """Reshape the compressed-tensors `weight_scale` for the row-wise FP8 kernel.

    In compressed-tensors, `weight_scale` is the dequantization multiplier:
        bf16_weight = fp8_weight * weight_scale

    Our CompressedTensorsFP8Linear keeps the exact same `weight_scale` name and
    semantics, so no renaming/inversion is needed. We only reshape the scale so it
    matches the kernel layout: scalar → (1, 1), 1D (N,) → (N, 1).
    """

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

        return {"weight_scale": dequant_scale}

    @property
    def reverse_op(self):
        return _IdentityOp()


class CompressedTensorsFp8Dequantize(ConversionOps):
    """Dequantize compressed-tensors FP8 weights back to BF16.

    Folds the per-channel / per-tensor ``weight_scale`` into the FP8 weight,
    producing a BF16 tensor. Prepended to a converter chain for layers that
    cannot stay in FP8 (e.g. merged MoE experts, which are not ``nn.Linear``):
    it pairs each weight with its sibling scale *by index* and preserves the
    per-expert list structure so the downstream merge / concat ops still see
    one tensor per expert.
    """

    @staticmethod
    def _scale_pattern_for(weight_pattern: str) -> str:
        # Strip the optional ``$`` regex anchor so we can match the underlying name.
        anchored = weight_pattern.endswith("$")
        base = weight_pattern[:-1] if anchored else weight_pattern
        if base.endswith(".weight"):
            scale = base[: -len(".weight")] + ".weight_scale"
        elif base == "weight":
            scale = "weight_scale"
        else:
            scale = base + "_scale"
        return scale + "$" if anchored else scale

    @staticmethod
    def _dequantize_one(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        quantized_float = quantized.to(torch.float32)
        if scale.dim() == 1:
            # Per-channel: (N,) scale, broadcast over the K dimension.
            dequantized = quantized_float * scale.unsqueeze(-1)
        else:
            # Per-tensor scalar or already-broadcastable scale.
            dequantized = quantized_float * scale
        return dequantized.to(torch.bfloat16)

    def convert(self, input_dict, full_layer_name=None, **kwargs):
        weight_keys = [k for k in input_dict if "weight" in k and "weight_scale" not in k]
        scale_keys = [k for k in input_dict if "weight_scale" in k]

        # No scale alongside (e.g. RMSNorm weights that match the weight pattern but
        # ship no scale) — pass the weight through untouched.
        if not scale_keys:
            weight_key = weight_keys[0]
            return {full_layer_name: input_dict[weight_key]}

        # Dequantize each weight pattern that has a sibling scale, pairing per-expert
        # tensors by index and preserving the list structure for downstream merge ops.
        # Scale entries are dropped so only weights remain in the chain.
        result: dict = {}
        for key in weight_keys:
            scale_key = self._scale_pattern_for(key)
            if scale_key not in input_dict:
                result[key] = input_dict[key]
                continue
            weights = input_dict[key]
            scales = input_dict[scale_key]
            weights = weights if isinstance(weights, list) else [weights]
            scales = scales if isinstance(scales, list) else [scales]
            if len(weights) != len(scales):
                raise ValueError(
                    f"CompressedTensorsFp8Dequantize: weight/scale count mismatch for {key} "
                    f"({len(weights)} weights vs {len(scales)} scales)."
                )
            result[key] = [self._dequantize_one(w, s) for w, s in zip(weights, scales)]
        return result

    @property
    def reverse_op(self):
        # Dequantization is one-way here: we never re-quantize on save (online FP8
        # quantization is intentionally not supported — use finegrained-fp8 for that).
        return _IdentityOp()
