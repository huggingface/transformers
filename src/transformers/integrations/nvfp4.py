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
from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

from ..core_model_loading import ConversionOps, _IdentityOp
from ..utils import is_torch_available
from ..utils.import_utils import (
    KERNELS_MAX_VERSION,
    KERNELS_MIN_VERSION,
    is_kernels_available,
)
from .hub_kernels import lazy_load_kernel
from .mxfp4 import FP4_VALUES


if is_torch_available():
    import torch
    import torch.nn as nn

    _FP8_DTYPE = torch.float8_e4m3fn  # the dtype of the NVFP4 block scales


# NVFP4 uses one FP8 scale factor for every 16 consecutive weight values.
SF_VEC_SIZE = 16


@dataclass(frozen=True)
class NVFP4Kernel:
    """Entry points exposed by the NVFP4 Hub kernel."""

    PackedWeight: type
    gemm: Callable
    pack: Callable
    # The raw ops, for the W4A4 GEMM with a static activation scale
    scaled_fp4_quant: Callable
    cutlass_scaled_fp4_mm: Callable


@functools.cache
def load_nvfp4_kernel() -> NVFP4Kernel:
    """Load and validate the NVFP4 Hub kernel once."""

    if not is_kernels_available():
        raise ImportError(
            "NVFP4 quantization requires the `kernels` package. "
            f"Install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
            f"for example with `pip install kernels=={KERNELS_MIN_VERSION}`."
        )

    kernel = lazy_load_kernel("nvfp4")
    if kernel is None:
        raise ImportError(
            "Failed to load the NVFP4 kernel. Check that the Hub kernel has a build matching the current "
            "PyTorch and CUDA versions."
        )

    ops = getattr(getattr(kernel, "_ops", None), "ops", None)
    entry_points = {
        "PackedWeight": getattr(kernel, "PackedWeight", None),
        "gemm": getattr(kernel, "gemm", None),
        "pack": getattr(kernel, "pack", None),
        "scaled_fp4_quant": getattr(ops, "scaled_fp4_quant", None),
        "cutlass_scaled_fp4_mm": getattr(ops, "cutlass_scaled_fp4_mm", None),
    }
    missing = [name for name, entry_point in entry_points.items() if entry_point is None]
    if missing:
        raise ImportError(f"The NVFP4 kernel is missing required entry points: {', '.join(missing)}.")

    return NVFP4Kernel(**entry_points)


def dequantize_nvfp4(packed: torch.Tensor, scales: torch.Tensor, scale_2) -> torch.Tensor:
    """
    Dense float32 `(out_features, in_features)`: `fp4 * scales * scale_2`, two e2m1 values per byte, low nibble
    first
    """
    values = torch.tensor(FP4_VALUES, device=packed.device)
    weight = torch.stack([values[(packed & 0xF).int()], values[(packed >> 4).int()]], dim=-1)
    weight = weight.view(packed.shape[0], -1, SF_VEC_SIZE)
    scales = scales.view(_FP8_DTYPE).float().unsqueeze(-1)
    return (weight * scales * scale_2).flatten(1)


def swizzle_sf(sf_rowmajor: torch.Tensor) -> torch.Tensor:
    """
    The block-scaled GEMM layout of row-major FP8 block scales `(..., rows, cols)`, flattened per leading index:
    padded to 128-row x 4-column tiles, each tile stored as 32x16 bytes (rows `r`, `r + 32`, `r + 64`, `r + 96`
    interleaved).
    """
    *batch, rows, cols = sf_rowmajor.shape
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    padded = sf_rowmajor.new_zeros(*batch, row_tiles * 128, col_tiles * 4)
    padded[..., :rows, :cols] = sf_rowmajor
    tiles = padded.view(*batch, row_tiles, 128, col_tiles, 4).transpose(-3, -2)
    return tiles.reshape(*batch, -1, 4, 32, 4).transpose(-3, -2).reshape(*batch, -1)


def nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_features: int,
    in_features: int,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply an NVFP4 linear operation using packed weights. Activations are quantized with the static `input_scale`
    (ModelOpt's `amax / (6 * 448)`) when given, else dynamically.
    """

    kernel = load_nvfp4_kernel()
    weight_global_scale = 1 / weight_scale_2
    packed_weight = kernel.PackedWeight(
        qweight=weight,
        sf=weight_sf,
        global_scale=weight_global_scale,
        n=out_features,
        k=in_features,
        sf_rowmajor=weight_scale,
    )
    if input_scale is None or input.numel() <= 2 * in_features:
        return kernel.gemm(packed_weight, input)  # dynamic activation scale, or the W4A16 decode GEMV

    # The kernel's W4A4 GEMM, with the static activation global scale instead of one from the input's amax
    qinput, input_sf = kernel.scaled_fp4_quant(
        input.reshape(-1, in_features).to(torch.bfloat16).contiguous(), 1 / input_scale, True
    )
    output = kernel.cutlass_scaled_fp4_mm(
        qinput, weight, input_sf.view(_FP8_DTYPE), weight_sf.view(_FP8_DTYPE), input_scale / weight_global_scale
    )
    return output.reshape(*input.shape[:-1], out_features)


class NVFP4Linear(nn.Module):
    """Bias-free linear layer backed by packed NVFP4 weights."""

    def __init__(
        self, in_features: int, out_features: int, weight_only: bool = False, static_input_scale: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight-only (W4A16): the weight is dequantized in each forward, the activations stay in full precision. This is used to keep accuracy!
        self.weight_only = weight_only

        # The BF16 placeholder matches the source checkpoint before the load-time conversion replaces it with
        # packed uint8 weights. Four FP8 scale bytes are packed into each int32 entry in the swizzled layout.
        self.weight = nn.Buffer(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        self.weight_sf = nn.Buffer(torch.empty(swizzled_sf_shape(out_features, in_features), dtype=torch.int32))
        # As ModelOpt saves them, `weight = fp4 * weight_scale * weight_scale_2`; the W4A16 GEMV reads row-major scales
        self.weight_scale = nn.Buffer(torch.empty(out_features, in_features // SF_VEC_SIZE, dtype=torch.uint8))
        self.weight_scale_2 = nn.Buffer(torch.empty(1, dtype=torch.float32))
        # ModelOpt checkpoints save a static activation scale (weight-only ones too, unused). Empty for dynamic
        # activation quantization, when quantizing a full-precision checkpoint
        self.input_scale = nn.Buffer(torch.empty(1 if static_input_scale else 0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_only:  # TODO: maybe we'll create a mapping W4A16 -> func?
            weight = dequantize_nvfp4(self.weight, self.weight_scale, self.weight_scale_2)
            return nn.functional.linear(input, weight.to(input.dtype))
        return nvfp4_linear(
            input,
            self.weight,
            self.weight_sf,
            self.weight_scale,
            self.weight_scale_2,
            self.out_features,
            self.in_features,
            self.input_scale if self.input_scale.numel() else None,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4"


class NVFP4Quantize(ConversionOps):
    """Pack a floating-point checkpoint weight into the buffers used by `NVFP4Linear`."""

    def __init__(self, device: torch.device):
        self.device = device

    def convert(self, input_dict, **kwargs):
        kernel = load_nvfp4_kernel()
        output = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            if not key.endswith("weight"):
                output[key] = tensor
                continue

            packed_weight = kernel.pack(tensor.to(torch.bfloat16), device=self.device)
            base = key.rsplit(".", 1)[0]
            output[key] = packed_weight.qweight
            output[f"{base}.weight_sf"] = packed_weight.sf
            output[f"{base}.weight_scale"] = packed_weight.sf_rowmajor
            output[f"{base}.weight_scale_2"] = 1 / packed_weight.global_scale
            output[f"{base}.input_scale"] = torch.empty(0, device=self.device)
        return output


def swizzled_sf_shape(out_features: int, in_features: int) -> tuple[int, int]:
    """
    The int32 shape of the swizzled block scales of an `(out_features, in_features)` weight: 128-row x 4-column tiles
    of fp8 scales, four scales per int32.
    """
    return -(-out_features // 128) * 128, -(-in_features // (SF_VEC_SIZE * 4))


class NVFP4Deserialize(ConversionOps):
    """
    ModelOpt's FP8 block scales `*_scale` as bytes, plus their swizzled `*_sf` for the kernel
    TODO: check why we need to rename weight_scale_2 to input scale?

    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, full_layer_name=None, **kwargs) -> dict[str, torch.Tensor]:
        if len(input_dict) == 1:  # a param of its own, named by its target
            input_dict = {full_layer_name: next(iter(input_dict.values()))}
        output = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            if key.endswith("_scale") and not key.endswith("input_scale"):  # the block scales
                output[key] = tensor = tensor.view(torch.uint8)
                sf_shape = swizzled_sf_shape(tensor.shape[-2], tensor.shape[-1] * SF_VEC_SIZE)
                swizzled = swizzle_sf(tensor).view(torch.int32).view(*tensor.shape[:-2], *sf_shape)
                output[key.removesuffix("scale") + "sf"] = swizzled
            elif key in ("weight_scale_2", "input_scale"):  # an `NVFP4Linear`'s scalars, saved 0-d
                output[key] = tensor.float().reshape(-1)
            else:
                output[key] = tensor
        if "weight_scale_2" in output:  # an `NVFP4Linear`
            output.setdefault("input_scale", torch.empty(0, device=output["weight"].device))
        return output


class NVFP4Dequantize(ConversionOps):
    """
    Dense weights, `fp4 * weight_scale * weight_scale_2` (two e2m1 values per byte, low nibble first)
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        output = {}
        # In the dtype of the dense param it loads into, a weight without scales too (e.g. a norm saved in float32)
        dtype = model.get_parameter_or_buffer(full_layer_name).dtype
        for key, weights in input_dict.items():
            if not key.endswith("weight"):
                continue  # the scales, read along with their weight
            weights = weights if isinstance(weights, list) else [weights]
            scales, scales_2 = input_dict.get(f"{key}_scale"), input_dict.get(f"{key}_scale_2")
            if scales is not None and scales_2 is not None:
                scales = scales if isinstance(scales, list) else [scales]
                scales_2 = scales_2 if isinstance(scales_2, list) else [scales_2]
                weights = [dequantize_nvfp4(*tensors) for tensors in zip(weights, scales, scales_2)]
            output[key] = [weight.to(dtype) for weight in weights]
        return output

    @property
    def reverse_op(self) -> ConversionOps:
        # TODO: A dequantized model has no quantization config anymore: it saves unquantized
        return _IdentityOp()


def is_nvfp4_compatible(module: nn.Module) -> bool:
    """A bias-free `nn.Linear` whose dims are 16-aligned."""
    if not isinstance(module, nn.Linear) or module.bias is not None:
        return False
    return module.weight.shape[-1] % SF_VEC_SIZE == 0 and module.weight.shape[-2] % 16 == 0
