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


# NVFP4 uses one FP8 scale factor for every 16 consecutive weight values.
SF_VEC_SIZE = 16


@dataclass(frozen=True)
class NVFP4Kernel:
    """Entry points exposed by the NVFP4 Hub kernel."""

    PackedWeight: type
    gemm: Callable
    pack: Callable
    swizzled_sf_shape: Callable
    ops: object  # the raw ops, for the W4A4 GEMM with a static activation scale


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
        "ops": getattr(getattr(kernel, "_ops", None), "ops", None),
    }
    missing = [name for name, entry_point in entry_points.items() if entry_point is None]
    if missing:
        raise ImportError(f"The NVFP4 kernel is missing required entry points: {', '.join(missing)}.")

    return NVFP4Kernel(**entry_points)


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
    qinput, input_sf = kernel.ops.scaled_fp4_quant(
        input.reshape(-1, in_features).to(torch.bfloat16).contiguous(), 1 / input_scale, True
    )
    fp8 = torch.float8_e4m3fn
    output = kernel.ops.cutlass_scaled_fp4_mm(
        qinput, weight, input_sf.view(fp8), weight_sf.view(fp8), input_scale / weight_global_scale
    )
    return output.reshape(*input.shape[:-1], out_features)


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
        # As ModelOpt saves them, `weight = fp4 * weight_scale * weight_scale_2`; the W4A16 GEMV reads row-major scales
        self.weight_scale = nn.Buffer(torch.empty(out_features, in_features // SF_VEC_SIZE, dtype=torch.uint8))
        self.weight_scale_2 = nn.Buffer(torch.empty(1, dtype=torch.float32))
        # Empty for dynamic activation quantization
        self.input_scale = nn.Buffer(torch.empty(0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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


def _scale_halves(output: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    "Scale each half of `output` (e.g. gate, up) by its own column of `scales` `(rows, num_halves)`"
    return (output.unflatten(-1, (scales.shape[-1], -1)) * scales.unsqueeze(-1)).flatten(-2)


class NVFP4Experts(nn.Module):
    """
    Fused experts from per-expert ModelOpt NVFP4 checkpoints, merged by the model's converters. Per projection of
    `(num_experts, out_features, in_features)`: e2m1 `{proj}` (two per byte), FP8 block scales `{proj}_scale` and
    their swizzled `{proj}_sf`, per-tensor `{proj}_scale_2` and `{proj}_input_scale` (`(num_experts, 2)` for gate, up).
    """

    def __init__(self, experts: nn.Module):
        super().__init__()
        self.config = experts.config
        self.has_gate = getattr(experts, "has_gate", True)
        self.act_fn, self.limit = experts.act_fn, getattr(experts, "limit", None)
        self.in_proj = "gate_up_proj" if self.has_gate else "up_proj"
        self.projections = {name: getattr(experts, name).shape for name in (self.in_proj, "down_proj")}
        self.num_experts = experts.down_proj.shape[0]
        for name, (num_experts, out_features, in_features) in self.projections.items():
            sf_shape = load_nvfp4_kernel().swizzled_sf_shape(out_features, in_features)
            self.register_buffer(name, torch.empty(num_experts, out_features, in_features // 2, dtype=torch.uint8))
            self.register_buffer(f"{name}_sf", torch.empty(num_experts, *sf_shape, dtype=torch.int32))
            block_scales = torch.empty(num_experts, out_features, in_features // SF_VEC_SIZE, dtype=torch.uint8)
            self.register_buffer(f"{name}_scale", block_scales)
            scales_shape = (num_experts, 2) if name == "gate_up_proj" else (num_experts,)
            self.register_buffer(f"{name}_scale_2", torch.empty(scales_shape, dtype=torch.float32))
            self.register_buffer(f"{name}_input_scale", torch.empty(scales_shape, dtype=torch.float32))

    def _activate(self, up: torch.Tensor) -> torch.Tensor:
        if not self.has_gate:
            return self.act_fn(up)
        gate, up = up.chunk(2, dim=-1)
        if self.limit is not None:
            gate, up = gate.clamp(max=self.limit), up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def scales(self, name: str, expert_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        "The activation scale `(rows,)` and the weight scales `(rows, num_halves)` of the experts `expert_ids`"
        # gate and up read the same activations: one scale, the largest (they match in ModelOpt)
        input_scale = getattr(self, f"{name}_input_scale").view(self.num_experts, -1).amax(-1)
        return input_scale[expert_ids], getattr(self, f"{name}_scale_2").view(self.num_experts, -1)[expert_ids]

    def linear(self, name: str, expert_idx: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        _, out_features, in_features = self.projections[name]
        input_scale, scale_2 = self.scales(name, expert_idx.reshape(1))
        output = nvfp4_linear(
            input,
            getattr(self, name)[expert_idx],
            getattr(self, f"{name}_sf")[expert_idx],
            getattr(self, f"{name}_scale")[expert_idx],
            torch.ones(1, device=input.device),  # the global scales are applied to the output, per half
            out_features,
            in_features,
            input_scale,
        )
        return _scale_halves(output, scale_2).to(input.dtype)

    def grouped_linear(self, name, input, expert_ids, tokens_per_expert, offsets) -> torch.Tensor:
        "W4A4 `scaled_grouped_mm` over rows sorted by expert, each divided by its expert's static activation scale"
        from torch.nn.functional import ScalingType, SwizzleType, scaled_grouped_mm

        input_scale, scale_2 = self.scales(name, expert_ids)
        qinput, input_sf = load_nvfp4_kernel().ops.scaled_fp4_quant(
            (input.float() / input_scale[:, None]).to(torch.bfloat16).contiguous(),
            torch.ones(1, device=input.device),
            False,
        )
        # The grouped GEMM wants each expert's block scales on a new 128-row tile; static size, so no host sync
        input_sf = input_sf.view(torch.uint8)
        padding = -tokens_per_expert % 128
        rows = torch.arange(input_sf.shape[0], device=input.device) + (torch.cumsum(padding, 0) - padding)[expert_ids]
        grouped_sf = input_sf.new_zeros(input_sf.shape[0] + 127 * self.num_experts, input_sf.shape[1])
        grouped_sf[rows] = input_sf
        fp4, fp8 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
        output = scaled_grouped_mm(
            qinput.view(fp4),
            getattr(self, name).view(fp4).transpose(-2, -1),
            swizzle_sf(grouped_sf).view(-1, -(-input_sf.shape[1] // 4) * 4).view(fp8),
            ScalingType.BlockWise1x16,
            getattr(self, f"{name}_sf").view(self.num_experts, -1).view(fp8),
            ScalingType.BlockWise1x16,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            offs=offsets,
            output_dtype=torch.bfloat16,
        )
        return _scale_halves(output, input_scale[:, None] * scale_2).to(input.dtype)

    def eager_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        "One NVFP4 GEMM per hit expert"
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx in top_k_index.unique():
            token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
            current = self._activate(self.linear(self.in_proj, expert_idx, hidden_states[token_idx]))
            current = self.linear("down_proj", expert_idx, current) * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def grouped_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        "Tokens sorted by expert, one grouped NVFP4 GEMM per projection"
        expert_ids, perm = torch.sort(top_k_index.reshape(-1))
        # histc instead of bincount avoids cuda graph issues; CPU and MPS only support float input
        histc_input = expert_ids.float() if hidden_states.device.type in ("cpu", "mps") else expert_ids.int()
        tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1).long()
        grouped = (expert_ids, tokens_per_expert, torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32))
        hidden = hidden_states[perm // top_k_index.shape[-1]]
        hidden = self._activate(self.grouped_linear(self.in_proj, hidden, *grouped))
        hidden = self.grouped_linear("down_proj", hidden, *grouped) * top_k_weights.reshape(-1)[perm, None]
        output = torch.empty_like(hidden, dtype=torch.float32)
        output[perm] = hidden  # back to the token order
        return output.view(*top_k_index.shape, -1).sum(dim=1).to(hidden_states.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        grouped = self.config._experts_implementation == "grouped_mm"
        # torch's grouped NVFP4 GEMM only runs on sm90 / sm100 (e.g. not on sm120): one GEMM per expert elsewhere
        if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device) not in ((9, 0), (10, 0)):
            grouped = False
        return (self.grouped_forward if grouped else self.eager_forward)(hidden_states, top_k_index, top_k_weights)


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


class NVFP4Deserialize(ConversionOps):
    "ModelOpt's FP8 block scales `*_scale` as bytes, plus their swizzled `*_sf` for the kernel"

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, **kwargs) -> dict[str, torch.Tensor]:
        output = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            if key.endswith("_scale") and not key.endswith("input_scale"):  # the block scales
                output[key] = tensor = tensor.view(torch.uint8)
                rows, columns = tensor.shape[-2], tensor.shape[-1] * SF_VEC_SIZE
                sf_shape = load_nvfp4_kernel().swizzled_sf_shape(rows, columns)
                swizzled = swizzle_sf(tensor).view(torch.int32).view(*tensor.shape[:-2], *sf_shape)
                output[key.removesuffix("scale") + "sf"] = swizzled
            elif key in ("weight_scale_2", "input_scale"):  # an `NVFP4Linear`'s scalars, saved 0-d
                output[key] = tensor.float().reshape(-1)
            else:
                output[key] = tensor
        if "weight_sf" in output:  # an `NVFP4Linear`
            output.setdefault("input_scale", torch.empty(0, device=output["weight"].device))
        return output


class NVFP4Dequantize(ConversionOps):
    "Dense weights, `fp4 * weight_scale * weight_scale_2` (two e2m1 values per byte, low nibble first)"

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        output = {}
        for key, weights in input_dict.items():
            if not key.endswith("weight"):
                continue  # the scales, read along with their weight
            weights = weights if isinstance(weights, list) else [weights]
            scales, scales_2 = input_dict.get(f"{key}_scale"), input_dict.get(f"{key}_scale_2")
            if scales is not None and scales_2 is not None:
                scales = scales if isinstance(scales, list) else [scales]
                scales_2 = scales_2 if isinstance(scales_2, list) else [scales_2]
                # In the dtype of the dense param it loads into
                dtype = model.get_parameter_or_buffer(full_layer_name).dtype
                weights = [self._dequantize(*tensors).to(dtype) for tensors in zip(weights, scales, scales_2)]
            output[key] = weights
        return output

    @property
    def reverse_op(self) -> ConversionOps:
        # A dequantized model has no quantization config anymore: it saves unquantized
        return _IdentityOp()

    def _dequantize(self, packed, scales, scale_2):
        values = torch.tensor(FP4_VALUES, device=packed.device)
        weight = torch.stack([values[(packed & 0xF).int()], values[(packed >> 4).int()]], dim=-1)
        weight = weight.view(packed.shape[0], -1, SF_VEC_SIZE)
        scales = scales.view(torch.float8_e4m3fn).float().unsqueeze(-1)
        return (weight * scales * scale_2).flatten(1)


def is_nvfp4_compatible(module: nn.Module) -> bool:
    "A bias-free linear, or fused experts with `(num_experts, out_features, in_features)` weights, with 16-aligned dims"
    if isinstance(module, nn.Linear):
        weights = [module.weight] if module.bias is None else []
    elif getattr(module, "has_bias", False) or getattr(module, "is_transposed", False):
        weights = []
    else:
        weights = [module._parameters.get(name) for name in ("gate_up_proj", "up_proj", "down_proj")]
        weights = [weight for weight in weights if weight is not None]
    return bool(weights) and all(w.shape[-1] % SF_VEC_SIZE == 0 and w.shape[-2] % 16 == 0 for w in weights)
