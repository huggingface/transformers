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

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available
from ..utils.import_utils import (
    KERNELS_MAX_VERSION,
    KERNELS_MIN_VERSION,
    is_kernels_available,
)
from .hub_kernels import lazy_load_kernel


if is_torch_available():
    import torch
    import torch.nn as nn


# NVFP4 uses one FP8 scale factor for every 16 consecutive weight values.
SF_VEC_SIZE = 16
# The e2m1 values of the 16 FP4 codes: sign bit, 2 exponent bits, 1 mantissa bit.
FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@dataclass(frozen=True)
class NVFP4Kernel:
    """Entry points exposed by the NVFP4 Hub kernel."""

    PackedWeight: type
    gemm: Callable
    pack: Callable
    swizzled_sf_shape: Callable


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

    entry_points = {
        "PackedWeight": getattr(kernel, "PackedWeight", None),
        "gemm": getattr(kernel, "gemm", None),
        "pack": getattr(kernel, "pack", None),
        "swizzled_sf_shape": getattr(kernel, "swizzled_sf_shape", None),
    }
    missing = [name for name, entry_point in entry_points.items() if entry_point is None]
    if missing:
        raise ImportError(f"The NVFP4 kernel is missing required entry points: {', '.join(missing)}.")

    return NVFP4Kernel(**entry_points)


def nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    weight_sf_rowmajor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Apply an NVFP4 linear operation using packed weights and dynamic activation quantization."""

    kernel = load_nvfp4_kernel()
    packed_weight = kernel.PackedWeight(
        qweight=weight,
        sf=weight_sf,
        global_scale=weight_global_scale,
        n=out_features,
        k=in_features,
        sf_rowmajor=weight_sf_rowmajor,
    )
    return kernel.gemm(packed_weight, input)


class NVFP4Linear(nn.Module):
    """Bias-free linear layer backed by packed NVFP4 weights."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The BF16 placeholder matches the source checkpoint before the load-time conversion replaces it with
        # packed uint8 weights. Four FP8 scale bytes are packed into each int32 entry in the swizzled layout.
        self.weight = nn.Buffer(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        sf_m, sf_n = load_nvfp4_kernel().swizzled_sf_shape(out_features, in_features)
        self.weight_sf = nn.Buffer(torch.empty(sf_m, sf_n, dtype=torch.int32))
        # The W4A16 decode GEMV consumes row-major scales, while the W4A4 prefill GEMM consumes swizzled scales.
        self.weight_sf_rowmajor = nn.Buffer(torch.empty(out_features, in_features // SF_VEC_SIZE, dtype=torch.uint8))
        self.weight_global_scale = nn.Buffer(torch.empty(1, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nvfp4_linear(
            input,
            self.weight,
            self.weight_sf,
            self.weight_sf_rowmajor,
            self.weight_global_scale,
            self.out_features,
            self.in_features,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4"


class NVFP4Quantize(ConversionOps):
    """Pack a floating-point checkpoint weight into the buffers used by `NVFP4Linear`."""

    def __init__(self, hf_quantizer):
        self.device = hf_quantizer.quantization_device

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
            output[f"{base}.weight_sf_rowmajor"] = packed_weight.sf_rowmajor
            output[f"{base}.weight_global_scale"] = packed_weight.global_scale
        return output


class NVFP4Dequantize(ConversionOps):
    """
    Dense weights from the packed `weight` (two e2m1 values per byte, low nibble first), its row-major FP8 block
    scales `weight_sf_rowmajor` and `weight_global_scale`: `weight = fp4 * block_scale / global_scale`.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        output = {}
        for key, weights in input_dict.items():
            if not key.endswith("weight"):
                continue  # the scales, read along with their weight
            weights = weights if isinstance(weights, list) else [weights]
            scales = input_dict.get(f"{key}_sf_rowmajor")
            global_scales = input_dict.get(f"{key}_global_scale")
            if scales is not None and global_scales is not None:
                scales = scales if isinstance(scales, list) else [scales]
                global_scales = global_scales if isinstance(global_scales, list) else [global_scales]
                # In the dtype of the dense param it loads into
                dtype = model.get_parameter_or_buffer(full_layer_name).dtype
                weights = [self._dequantize(*tensors).to(dtype) for tensors in zip(weights, scales, global_scales)]
            output[key] = weights
        return output

    def _dequantize(self, packed, scales, global_scale):
        values = torch.tensor(FP4_E2M1_VALUES, device=packed.device)
        weight = torch.stack([values[(packed & 0xF).int()], values[(packed >> 4).int()]], dim=-1)
        weight = weight.view(packed.shape[0], -1, SF_VEC_SIZE)
        scales = scales.view(torch.float8_e4m3fn).float().unsqueeze(-1)
        return (weight * scales / global_scale).flatten(1)


def _is_nvfp4_compatible(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Linear)
        and module.bias is None
        and module.in_features % SF_VEC_SIZE == 0
        and module.out_features % 16 == 0
    )


def replace_with_nvfp4_linear(model: nn.Module, modules_to_not_convert: list[str] | None = None) -> nn.Module:
    """Replace eligible bias-free linear layers with `NVFP4Linear`."""

    for name, module in list(model.named_modules()):
        if not should_convert_module(name, modules_to_not_convert) or not _is_nvfp4_compatible(module):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, NVFP4Linear(module.in_features, module.out_features))
    return model
