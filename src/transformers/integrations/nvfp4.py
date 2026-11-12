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

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps, _IdentityOp
from ..utils import is_torch_available
from ..utils.import_utils import (
    KERNELS_MAX_VERSION,
    KERNELS_MIN_VERSION,
    is_kernels_available,
)
from .hub_kernels import lazy_load_kernel
from .moe import ExpertsInterface
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
    swizzled_sf_shape: Callable
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
        "swizzled_sf_shape": getattr(kernel, "swizzled_sf_shape", None),
        "scaled_fp4_quant": getattr(ops, "scaled_fp4_quant", None),
        "cutlass_scaled_fp4_mm": getattr(ops, "cutlass_scaled_fp4_mm", None),
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
        # Weight-only (W4A16): the weight is dequantized in each forward, no kernel nor activation scale
        self.weight_only = weight_only
        # The BF16 placeholder matches the source checkpoint before the load-time conversion replaces it with
        # packed uint8 weights. Four FP8 scale bytes are packed into each int32 entry in the swizzled layout.
        self.weight = nn.Buffer(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        if not weight_only:
            sf_m, sf_n = load_nvfp4_kernel().swizzled_sf_shape(out_features, in_features)
            self.weight_sf = nn.Buffer(torch.empty(sf_m, sf_n, dtype=torch.int32))
        # As ModelOpt saves them, `weight = fp4 * weight_scale * weight_scale_2`; the W4A16 GEMV reads row-major scales
        self.weight_scale = nn.Buffer(torch.empty(out_features, in_features // SF_VEC_SIZE, dtype=torch.uint8))
        self.weight_scale_2 = nn.Buffer(torch.empty(1, dtype=torch.float32))
        # ModelOpt checkpoints save a static activation scale (weight-only ones too, unused). Empty for dynamic
        # activation quantization, when quantizing a full-precision checkpoint
        self.input_scale = nn.Buffer(torch.empty(1 if static_input_scale else 0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_only:
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


def _frozen(*shape, dtype: torch.dtype) -> nn.Parameter:
    "An uninitialized parameter loading quantized data, so without gradients"
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


def _scale_halves(output: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    "Scale each half of `output` (e.g. gate, up) by its own column of `scales` `(rows, num_halves)`"
    return (output.unflatten(-1, (scales.shape[-1], -1)) * scales.unsqueeze(-1)).flatten(-2)


class NVFP4Experts(nn.Module):
    """
    Fused experts from per-expert ModelOpt NVFP4 checkpoints, merged by the model's converters. Per projection of
    `(num_experts, out_features, in_features)`: e2m1 `{proj}` (two per byte), FP8 block scales `{proj}_scale` and
    their swizzled `{proj}_sf`, per-tensor `{proj}_scale_2` and `{proj}_input_scale` (`(num_experts, 2)` for gate, up).
    """

    def __init__(self, config, has_gate: bool = True, has_bias: bool = False, weight_only: bool = False):
        super().__init__()
        if has_bias:
            raise ValueError("NVFP4Experts does not support bias")
        self.config = config
        self.has_gate = has_gate
        # Weight-only (W4A16): each hit expert's weights are dequantized in its forward, no kernel nor activation scale
        self.weight_only = weight_only
        self.act_fn = ACT2FN[getattr(config, "hidden_activation", None) or config.hidden_act]
        self.limit = getattr(config, "swiglu_limit", None)
        self.num_experts = getattr(config, "num_local_experts", None) or config.num_experts
        hidden = config.hidden_size
        intermediate = getattr(config, "moe_intermediate_size", None) or config.intermediate_size
        self.in_proj = "gate_up_proj" if has_gate else "up_proj"
        in_proj_shape = (self.num_experts, 2 * intermediate if has_gate else intermediate, hidden)
        self.projections = {self.in_proj: in_proj_shape, "down_proj": (self.num_experts, hidden, intermediate)}
        # Parameters, not buffers: expert parallelism shards them along their first axis, the experts'
        for name, (num_experts, out_features, in_features) in self.projections.items():
            self.register_parameter(name, _frozen(num_experts, out_features, in_features // 2, dtype=torch.uint8))
            if not weight_only:
                sf_shape = load_nvfp4_kernel().swizzled_sf_shape(out_features, in_features)
                self.register_parameter(f"{name}_sf", _frozen(num_experts, *sf_shape, dtype=torch.int32))
            block_scales = _frozen(num_experts, out_features, in_features // SF_VEC_SIZE, dtype=torch.uint8)
            self.register_parameter(f"{name}_scale", block_scales)
            scales_shape = (num_experts, 2) if name == "gate_up_proj" else (num_experts,)
            self.register_parameter(f"{name}_scale_2", _frozen(*scales_shape, dtype=torch.float32))
            # Weight-only checkpoints save it too, unused
            self.register_parameter(f"{name}_input_scale", _frozen(*scales_shape, dtype=torch.float32))

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
        if self.weight_only:
            weight = dequantize_nvfp4(getattr(self, name)[expert_idx], getattr(self, f"{name}_scale")[expert_idx], 1.0)
            scale_2 = getattr(self, f"{name}_scale_2").view(self.num_experts, -1)[expert_idx.reshape(1)]
            return _scale_halves(nn.functional.linear(input, weight.to(input.dtype)), scale_2).to(input.dtype)
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
        qinput, input_sf = load_nvfp4_kernel().scaled_fp4_quant(
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
        fp4 = torch.float4_e2m1fn_x2
        output = scaled_grouped_mm(
            qinput.view(fp4),
            getattr(self, name).view(fp4).transpose(-2, -1),
            swizzle_sf(grouped_sf).view(-1, -(-input_sf.shape[1] // 4) * 4).view(_FP8_DTYPE),
            ScalingType.BlockWise1x16,
            getattr(self, f"{name}_sf").view(self.num_experts, -1).view(_FP8_DTYPE),
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
            if expert_idx >= self.num_experts:  # another rank's, under expert parallelism
                continue
            token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
            current = self._activate(self.linear(self.in_proj, expert_idx, hidden_states[token_idx]))
            current = self.linear("down_proj", expert_idx, current) * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        # The eager implementation: `quantized_module` wraps the class to dispatch to `ALL_NVFP4_EXPERTS_FUNCTIONS`
        return self.eager_forward(hidden_states, top_k_index, top_k_weights)


def nvfp4_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    "Tokens sorted by expert, one grouped NVFP4 GEMM per projection"
    if self.weight_only:  # the grouped GEMM quantizes activations
        return self.eager_forward(hidden_states, top_k_index, top_k_weights)
    # torch's grouped NVFP4 GEMM only runs on sm90 / sm100 (e.g. not on sm120): one GEMM per expert elsewhere
    if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device) not in ((9, 0), (10, 0)):
        return self.eager_forward(hidden_states, top_k_index, top_k_weights)
    expert_ids, perm = torch.sort(top_k_index.reshape(-1))
    # histc instead of bincount avoids cuda graph issues; CPU and MPS only support float input
    histc_input = expert_ids.float() if hidden_states.device.type in ("cpu", "mps") else expert_ids.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1).long()
    # Expert parallelism marks other ranks' routes with ids past `num_experts`: sorted last, past every group, so
    # the GEMM leaves their rows unwritten. Their ids are clamped for the scale lookups and their rows zeroed.
    on_rank = expert_ids < self.num_experts
    expert_ids = expert_ids.clamp(max=self.num_experts - 1)
    grouped = (expert_ids, tokens_per_expert, torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32))
    hidden = hidden_states[perm // top_k_index.shape[-1]]
    hidden = self._activate(self.grouped_linear(self.in_proj, hidden, *grouped))
    hidden = self.grouped_linear("down_proj", hidden, *grouped) * top_k_weights.reshape(-1)[perm, None]
    hidden = hidden.masked_fill(~on_rank[:, None], 0.0)
    output = torch.empty_like(hidden, dtype=torch.float32)
    output[perm] = hidden  # back to the token order
    return output.view(*top_k_index.shape, -1).sum(dim=1).to(hidden_states.dtype)


class NVFP4ExpertsInterface(ExpertsInterface):
    """Interface for registering custom NVFP4 experts forward functions; `eager` is `NVFP4Experts.eager_forward`."""

    # No batched NVFP4 GEMM: `generate`'s decode-time switch to `batched_mm` keeps the grouped one
    _global_mapping = {"grouped_mm": nvfp4_grouped_mm_experts_forward, "batched_mm": nvfp4_grouped_mm_experts_forward}


ALL_NVFP4_EXPERTS_FUNCTIONS = NVFP4ExpertsInterface()


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
        # Weight-only layers run without the kernels, which are the only readers of the swizzled scales
        self.weight_only = hf_quantizer is not None and hf_quantizer.quantization_config.weight_only

    def convert(self, input_dict, full_layer_name=None, **kwargs) -> dict[str, torch.Tensor]:
        if len(input_dict) == 1:  # a param of its own, e.g. merged experts' `gate_up_proj_scale`: named by its target
            input_dict = {full_layer_name: next(iter(input_dict.values()))}
        output = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            if key.endswith("_scale") and not key.endswith("input_scale"):  # the block scales
                output[key] = tensor = tensor.view(torch.uint8)
                if self.weight_only:
                    continue
                rows, columns = tensor.shape[-2], tensor.shape[-1] * SF_VEC_SIZE
                sf_shape = load_nvfp4_kernel().swizzled_sf_shape(rows, columns)
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
    "Dense weights, `fp4 * weight_scale * weight_scale_2` (two e2m1 values per byte, low nibble first)"

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
                weights = [self._dequantize(*tensors) for tensors in zip(weights, scales, scales_2)]
            output[key] = [weight.to(dtype) for weight in weights]
        return output

    @property
    def reverse_op(self) -> ConversionOps:
        # A dequantized model has no quantization config anymore: it saves unquantized
        return _IdentityOp()

    def _dequantize(self, packed, scales, scale_2):
        return dequantize_nvfp4(packed, scales, scale_2)


def dequantize_nvfp4(packed: torch.Tensor, scales: torch.Tensor, scale_2) -> torch.Tensor:
    "Dense float32 `(out_features, in_features)`: `fp4 * scales * scale_2`, two e2m1 values per byte, low nibble first"
    values = torch.tensor(FP4_VALUES, device=packed.device)
    weight = torch.stack([values[(packed & 0xF).int()], values[(packed >> 4).int()]], dim=-1)
    weight = weight.view(packed.shape[0], -1, SF_VEC_SIZE)
    scales = scales.view(_FP8_DTYPE).float().unsqueeze(-1)
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
