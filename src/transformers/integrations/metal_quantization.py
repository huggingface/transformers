# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
Metal affine quantization integration for transformers.

This module provides:
  - ``MetalLinear``: a drop-in replacement for ``nn.Linear`` that stores weights
    as affine-quantized uint32 packed tensors and uses the ``quantization-mlx``
    Metal kernels for the forward pass.
  - ``replace_with_metal_linear``: walks a model and swaps every eligible
    ``nn.Linear`` with ``MetalLinear``.
  - ``MetalQuantize`` / ``MetalDequantize``: weight conversion operations that
    participate in the new ``WeightConverter`` pipeline.

Weight layout (transposed, matching ``affine_qmm_t``):
  - ``weight``: ``[N, K_packed]`` (``uint32``) -- K is the packed dimension.
  - ``scales``:  ``[N, K // group_size]`` (``float16 / bfloat16``)
  - ``qbiases``: ``[N, K // group_size]`` (same dtype as scales)

The kernel call is ``affine_qmm_t(x, weight, scales, qbiases, group_size, bits)``
which computes ``y = x @ dequant(weight).T``, identical to ``nn.Linear``.
"""

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)

_metal_kernel = None


def _get_metal_kernel():
    """Lazily load the quantization-mlx kernel from Hugging Face Hub."""
    global _metal_kernel
    if _metal_kernel is None:
        try:
            from .hub_kernels import get_kernel

            _metal_kernel = get_kernel("kernels-community/mlx-quantization-metal-kernels")
        except Exception as e:
            raise ImportError(
                f"Failed to load the quantization-mlx kernel from the Hub: {e}. "
                "Make sure you have `kernels` installed (`pip install kernels`) "
                "and are running on an Apple Silicon machine."
            ) from e
    return _metal_kernel


# ---------------------------------------------------------------------------
# MetalLinear -- the quantized nn.Linear replacement
# ---------------------------------------------------------------------------


class MetalLinear(nn.Linear):
    """
    A quantized linear layer that stores weights in affine uint32 packed format
    and uses the ``quantization-mlx`` Metal kernels for the forward pass.

    Parameters match ``nn.Linear`` with additional quantization metadata.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=torch.uint32,
        bits: int = 4,
        group_size: int = 128,
    ):
        nn.Module.__init__(self)

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        elems_per_int = 32 // bits
        k_packed = in_features // elems_per_int
        n_groups = in_features // group_size

        if dtype == torch.uint32:
            self.weight = nn.Parameter(torch.zeros(out_features, k_packed, dtype=torch.uint32), requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=dtype), requires_grad=False)

        scales_dtype = torch.float32 if dtype == torch.uint32 else None
        self.scales = nn.Parameter(torch.zeros(out_features, n_groups, dtype=scales_dtype), requires_grad=False)
        self.qbiases = nn.Parameter(torch.zeros(out_features, n_groups, dtype=scales_dtype), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != torch.uint32:
            return nn.functional.linear(input, self.weight, self.bias)

        kernel = _get_metal_kernel()

        output = kernel.affine_qmm_t(
            input,
            self.weight,
            self.scales.to(input.dtype),
            self.qbiases.to(input.dtype),
            self.group_size,
            self.bits,
        )

        if self.bias is not None:
            output = output + self.bias
        return output


def replace_with_metal_linear(
    model,
    modules_to_not_convert: list[str] | None = None,
    quantization_config=None,
    pre_quantized: bool = False,
):
    """
    Replace every eligible ``nn.Linear`` with ``MetalLinear``.

    Args:
        model: the ``PreTrainedModel`` (on the meta device at this point).
        modules_to_not_convert: module names to leave untouched.
        quantization_config: the ``MetalConfig`` instance.
        pre_quantized: ``True`` when loading from a quantized checkpoint.
    """
    if quantization_config.dequantize:
        return model

    bits = quantization_config.bits
    group_size = quantization_config.group_size

    has_been_replaced = False

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        if isinstance(module, nn.Linear):
            module_kwargs = {} if pre_quantized else {"dtype": None}
            new_module = MetalLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                bits=bits,
                group_size=group_size,
                **module_kwargs,
            )

            model.set_submodule(module_name, new_module)
            has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading a model with Metal quantization but no nn.Linear modules were found. "
            "Please double check your model architecture."
        )

    return model


def _affine_quantize_tensor(weight: torch.Tensor, group_size: int, bits: int):
    """
    Quantize a 2-D float weight ``[N, K]`` into packed uint32 + scales + biases.

    Returns ``(w_packed, scales, biases)`` with:
      - ``w_packed``: ``[N, K // (32 // bits)]`` uint32
      - ``scales``:   ``[N, K // group_size]`` float32/float16/bfloat16
      - ``biases``:   ``[N, K // group_size]`` float32/float16/bfloat16
    """
    N, K = weight.shape
    elems_per_int = 32 // bits
    max_val = (1 << bits) - 1
    n_groups = K // group_size

    w_grouped = weight.float().reshape(N, n_groups, group_size)
    w_min = w_grouped.min(dim=-1).values  # [N, n_groups]
    w_max = w_grouped.max(dim=-1).values

    scales = ((w_max - w_min) / max_val).clamp(min=1e-8)
    biases = w_min

    w_int = (w_grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1)
    w_int = w_int.round().clamp(0, max_val).to(torch.int32).reshape(N, K)

    # Pack into uint32
    k_packed = K // elems_per_int
    w_packed = torch.zeros(N, k_packed, dtype=torch.int32, device=weight.device)
    for i in range(elems_per_int):
        w_packed |= w_int[:, i::elems_per_int] << (bits * i)

    return w_packed.to(torch.uint32), scales, biases


def _affine_dequantize_tensor(
    w_packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, group_size: int, bits: int
):
    """
    Dequantize a packed uint32 weight ``[N, K_packed]`` back to float.

    Returns a ``[N, K]`` float32 tensor.
    """
    N = w_packed.shape[0]
    elems_per_int = 32 // bits
    max_val = (1 << bits) - 1
    K = w_packed.shape[1] * elems_per_int

    w_packed_i = w_packed.to(torch.int32)
    w_flat = torch.zeros(N, K, dtype=torch.float32, device=w_packed.device)
    for i in range(elems_per_int):
        w_flat[:, i::elems_per_int] = ((w_packed_i >> (bits * i)) & max_val).float()

    w_grouped = w_flat.reshape(N, -1, group_size)
    w_deq = w_grouped * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    return w_deq.reshape(N, K)


class MetalQuantize(ConversionOps):
    """
    Quantize a full-precision weight tensor into (weight, scales, qbiases).

    Used during quantize-on-the-fly.  The float ``weight`` is replaced in-place
    by the packed uint32 tensor.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: dict, **kwargs) -> dict:
        target_key, value = next(iter(input_dict.items()))
        value = value[0] if isinstance(value, list) else value

        bits = self.hf_quantizer.quantization_config.bits
        group_size = self.hf_quantizer.quantization_config.group_size

        w_packed, scales, biases = _affine_quantize_tensor(value, group_size, bits)

        base = target_key.rsplit(".", 1)[0] if "." in target_key else ""
        scale_key = f"{base}.scales" if base else "scales"
        bias_key = f"{base}.qbiases" if base else "qbiases"

        orig_dtype = value.dtype
        return {
            target_key: w_packed,
            scale_key: scales.to(orig_dtype),
            bias_key: biases.to(orig_dtype),
        }


class MetalDequantize(ConversionOps):
    """
    Dequantize (weight, scales, qbiases) back to a full-precision tensor.

    Used when ``dequantize=True`` is set in the config to fall back to a normal
    ``nn.Linear`` on devices without MPS.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: dict, full_layer_name: str | None = None, **kwargs) -> dict:
        bits = self.hf_quantizer.quantization_config.bits
        group_size = self.hf_quantizer.quantization_config.group_size

        if len(input_dict) < 2:
            return {full_layer_name: input_dict["weight$"]}

        quantized = input_dict["weight$"][0]
        scales = input_dict["scales"][0]
        qbiases = input_dict["qbiases"][0]

        w_deq = _affine_dequantize_tensor(quantized, scales, qbiases, group_size, bits)
        return {full_layer_name: w_deq.to(scales.dtype)}
