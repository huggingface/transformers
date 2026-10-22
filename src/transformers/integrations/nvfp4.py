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
"""NVFP4 linears and MoE experts on the `kernels-community/finegrained-kernels` Hub kernel.

NVFP4 weights are E2M1 values packed two per byte, with an E4M3 scale per 16 values and an fp32 global per tensor.
They load pre-quantized from a ModelOpt export, or quantize while a full-precision checkpoint loads.
"""

from __future__ import annotations

import functools
import re

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps, _IdentityOp
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging
from .hub_kernels import _MISSING_KERNELS_MESSAGE, lazy_load_kernel
from .moe import ExpertsInterface, use_experts_implementation


if is_torch_available():
    import torch
    import torch.nn as nn

    from .finegrained_fp8 import Fp8Dequantize


logger = logging.get_logger(__name__)

# one E4M3 scale per 16 values
GROUP_SIZE = 16
# the projections of a fused experts module
EXPERT_PROJECTIONS = ("gate_up_proj", "up_proj", "down_proj")
# the ModelOpt formats this integration runs. `NVFP4` quantizes the activations, `W4A16_NVFP4` does not.
NVFP4_FORMATS = ("NVFP4", "W4A16_NVFP4")


@functools.cache
def load_finegrained_kernels():
    """Load the `finegrained-kernels` Hub kernel once."""
    kernel = lazy_load_kernel("finegrained-kernels")
    if kernel is None:
        raise ImportError(
            f"NVFP4 needs the `kernels-community/finegrained-kernels` kernel. {_MISSING_KERNELS_MESSAGE}"
        )
    return kernel


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _first_attr(config, *names):
    return next(getattr(config, name) for name in names if getattr(config, name, None) is not None)


def _parameter(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)


def _tiles_swizzled(rows: int, scale_cols: int) -> bool:
    """Check whether the kernels read a `(rows, scale_cols)` scale grid swizzled: whole 128-row blocks, two bands."""
    return _is_sm100() and rows % 128 == 0 and scale_cols % 4 == 0 and scale_cols >= 8


@functools.cache
def supported_fused_norms() -> dict[type, str]:
    """Map each post-expert norm class the kernels fuse to its kernel name.

    Match the exact class, since a subclass may compute something else. Build the map on first use, so importing this
    module does not import a model.
    """
    from ..models.muse_spark.modeling_muse_spark import MuseSparkTextInputScaledRMSNorm

    return {nn.RMSNorm: "rms_norm", MuseSparkTextInputScaledRMSNorm: "input_scaled_rms_norm"}


def quantize_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a `(N, K)` weight to `(packed int8, E4M3 scales per 16, fp32 global)`.

    The global is `amax / (6 * 448)`, the smallest that keeps every E4M3 scale in range.
    """
    global_scale = (weight.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float()
    packed, scales = load_finegrained_kernels().nvfp4_act_quant((weight.float() / global_scale).contiguous())
    return packed.view(torch.int8), scales, global_scale


class NVFP4Linear(nn.Module):
    """Linear layer with NVFP4 weights.

    Activations quantize to NVFP4 against `input_global_scale`, the checkpoint's calibrated `input_scale`. A layer
    quantized on the fly has no calibration, so its activations quantize with a global of 1. `weight_only=True`
    keeps the activations in the model dtype.

    Quantized on the fly, the weight starts as a full-precision placeholder, since the loader casts a checkpoint
    tensor to its slot's dtype before quantizing it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pre_quantized: bool = False,
        weight_only: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_format = "bf16" if weight_only else None
        weight = torch.empty(out_features, in_features // 2, dtype=torch.int8)
        self.weight = _parameter(weight if pre_quantized else torch.empty(out_features, in_features))
        self.weight_scale_inv = _parameter(
            torch.empty(out_features, in_features // GROUP_SIZE, dtype=torch.float8_e4m3fn)
        )
        self.weight_global_scale = _parameter(torch.ones((), dtype=torch.float32))
        calibrated = pre_quantized and not weight_only
        self.input_global_scale = _parameter(torch.ones((), dtype=torch.float32)) if calibrated else None
        self.bias = _parameter(torch.empty(out_features)) if bias else None

    def swizzle_scales(self):
        """Hold the scale grid swizzled on SM100, the layout its scaled MMA reads. It runs once the weights load."""
        rows, scale_cols = self.weight_scale_inv.shape[-2:]
        # the weight-only arm reads row-major scales
        if self.activation_format != "bf16" and self.weight_scale_inv.ndim == 2 and _tiles_swizzled(rows, scale_cols):
            swizzled = load_finegrained_kernels().swizzle_mx_scales(self.weight_scale_inv.data.unsqueeze(0))
            self.weight_scale_inv = _parameter(swizzled)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # under tensor parallelism the module holds a shard, so the shapes come from the tensors
        output = load_finegrained_kernels().matmul_2d(
            input.reshape(-1, input.shape[-1]),
            self.weight,
            None,
            self.weight_scale_inv,
            bias=self.bias,
            activation_format=self.activation_format,
            output_dtype=input.dtype,
            a_global_scale=self.input_global_scale,
            b_global_scale=self.weight_global_scale,
        )
        return output.reshape(*input.shape[:-1], output.shape[-1])

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4"


class NVFP4Experts(nn.Module):
    """MoE experts with NVFP4 weights, run on the kernels' fused MoE chain.

    Each projection holds packed E2M1 weights, E4M3 scales and one weight global per expert. Activations quantize
    against one calibrated input global per expert, with a global of 1 on the fly, or not at all with
    `weight_only=True`. gate|up rows are interleaved, so the kernels fuse the gated activation. On SM100 the scales
    are swizzled, the layout the scaled MMA reads. Quantized on the fly, the weights start as full-precision
    placeholders, like `NVFP4Linear`'s.
    """

    def __init__(
        self,
        config,
        has_gate: bool = True,
        has_bias: bool = False,
        pre_quantized: bool = False,
        weight_only: bool = False,
        post_expert_norm: nn.Module | None = None,
    ):
        super().__init__()
        if has_bias:
            raise ValueError("NVFP4 experts do not support biases.")
        self.config = config
        self.has_gate = has_gate
        self.num_experts = _first_attr(config, "num_local_experts", "num_experts")
        self.hidden_dim = _first_attr(config, "moe_hidden_size", "hidden_size")
        self.intermediate_dim = _first_attr(config, "moe_intermediate_size", "intermediate_size")
        self.act_fn_name = _first_attr(config, "hidden_activation", "hidden_act")
        self.act_fn = ACT2FN[self.act_fn_name]
        self.activation_format = "bf16" if weight_only else None
        # the replaced experts' per-expert output norm, applied to each expert's output before the weighted sum
        self.post_expert_norm = post_expert_norm
        self.gate_up_name = "gate_up_proj" if has_gate else "up_proj"
        up_rows = (2 if has_gate else 1) * self.intermediate_dim
        self._register_projection(self.gate_up_name, up_rows, self.hidden_dim, pre_quantized, weight_only)
        self._register_projection("down_proj", self.hidden_dim, self.intermediate_dim, pre_quantized, weight_only)

    def _register_projection(self, proj: str, rows: int, cols: int, pre_quantized: bool, weight_only: bool):
        scale_cols = cols // GROUP_SIZE
        scale_shape = (self.num_experts, rows, scale_cols)
        # the weight-only arm reads row-major scales
        if not weight_only and _tiles_swizzled(rows, scale_cols):
            # `SWIZZLE_32_4_4` with the expert axis first, so expert parallelism shards it on dim 0
            scale_shape = (self.num_experts, rows // 128, scale_cols // 4, 2, 256)
        packed = torch.empty(self.num_experts, rows, cols // 2, dtype=torch.int8)
        setattr(self, proj, _parameter(packed if pre_quantized else torch.empty(self.num_experts, rows, cols)))
        setattr(self, f"{proj}_scale_inv", _parameter(torch.empty(scale_shape, dtype=torch.float8_e4m3fn)))
        setattr(self, f"{proj}_weight_global_scale", _parameter(torch.ones(self.num_experts, dtype=torch.float32)))
        calibrated = pre_quantized and not weight_only
        input_global = _parameter(torch.ones(self.num_experts, dtype=torch.float32)) if calibrated else None
        setattr(self, f"{proj}_input_global_scale", input_global)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        return self.act_fn(gate_up[..., 0::2]) * gate_up[..., 1::2]

    def _moe_operands(self, kernel) -> dict:
        """Return the arguments of the kernels' fused MoE forwards."""
        if self.act_fn_name in kernel.get_supported_act_fns():
            act_fn = self.act_fn_name
        else:
            act_fn = self._apply_gate if self.has_gate else self.act_fn
        operands = {
            "act_fn": act_fn,
            "gate": self.has_gate,
            "swiglu_alpha": None,
            "swiglu_limit": None,
            "activation_format": self.activation_format,
            "post_expert_norm": self.post_expert_norm,
            "post_expert_norm_weight": None,
            "post_expert_norm_eps": None,
        }
        # the kernels fuse a norm they know into the reduce, by name. Any other runs as the module's own callable.
        norm = self.post_expert_norm
        if norm is not None and (kind := supported_fused_norms().get(type(norm))) in kernel.get_supported_norms():
            eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
            operands["post_expert_norm"] = kind
            operands["post_expert_norm_weight"] = norm.weight
            # `nn.RMSNorm` leaves `eps` as `None` and uses the dtype's epsilon
            operands["post_expert_norm_eps"] = float(eps) if eps is not None else torch.finfo(norm.weight.dtype).eps
        # The kernels name the up projection `gate_up_proj` whether or not it is gated
        for name, proj in (("gate_up_proj", self.gate_up_name), ("down_proj", "down_proj")):
            for slot in ("", "_scale_inv", "_weight_global_scale", "_input_global_scale"):
                operands[f"{name}{slot}"] = getattr(self, f"{proj}{slot}")
            operands[f"{name}_activation_scale"] = None
            operands[f"{name}_bias"] = None
        return operands

    def expert_linear(self, input: torch.Tensor, proj: str, expert_idx: int) -> torch.Tensor:
        """Run one expert's `proj` as a dense NVFP4 linear."""
        scale = getattr(self, f"{proj}_scale_inv")
        input_global = getattr(self, f"{proj}_input_global_scale")
        return load_finegrained_kernels().matmul_2d(
            input,
            getattr(self, proj)[expert_idx],
            None,
            # `matmul_2d` reads a swizzled expert slice with a leading dim of 1
            scale[expert_idx : expert_idx + 1] if scale.ndim == 5 else scale[expert_idx],
            activation_format=self.activation_format,
            output_dtype=input.dtype,
            a_global_scale=None if input_global is None else input_global[expert_idx],
            b_global_scale=getattr(self, f"{proj}_weight_global_scale")[expert_idx],
        )

    def forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            # expert parallelism routes the other ranks' experts to the extra index `num_experts`
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            hidden = self.expert_linear(hidden_states[token_idx], self.gate_up_name, expert_idx)
            hidden = self._apply_gate(hidden) if self.has_gate else self.act_fn(hidden)
            hidden = self.expert_linear(hidden, "down_proj", expert_idx)
            if self.post_expert_norm is not None:
                hidden = self.post_expert_norm(hidden)
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, (hidden * routing_weights).float())
        return final_hidden_states.to(hidden_states.dtype)


def nvfp4_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    kernel = load_finegrained_kernels()
    return kernel.moe_fused_grouped(hidden_states, top_k_index, top_k_weights, **self._moe_operands(kernel))


def nvfp4_batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    kernel = load_finegrained_kernels()
    return kernel.moe_fused_batched(hidden_states, top_k_index, top_k_weights, **self._moe_operands(kernel))


class NVFP4ExpertsInterface(ExpertsInterface):
    """Interface for registering custom NVFP4 experts forward functions."""

    _global_mapping = {
        "grouped_mm": nvfp4_grouped_mm_experts_forward,
        "batched_mm": nvfp4_batched_mm_experts_forward,
    }


ALL_NVFP4_EXPERTS_FUNCTIONS = NVFP4ExpertsInterface()


def replace_with_nvfp4(
    model: nn.Module,
    modules_to_not_convert: list[str] | None = None,
    formats: dict[str, str] | None = None,
    pre_quantized: bool = False,
    weight_only: bool = False,
) -> bool:
    """Replace the linears and fused experts that NVFP4 covers. Return whether any module was replaced.

    `formats` maps model module names to ModelOpt formats. Without it, every eligible module is NVFP4. Only modules in
    an `NVFP4_FORMATS` format are replaced. The other quantized ones load dequantized. `weight_only=True` keeps every
    module's activations in the model dtype.
    """
    # MoE routers stay in full precision, unless the checkpoint quantizes them
    parents = [name.removesuffix(".experts") for name, _ in model.named_modules() if name.endswith(".experts")]
    routers = {f"{parent}.{key}" for parent in parents for key in ("gate", "router")}
    replaced = False
    for name, module in list(model.named_modules()):
        if not should_convert_module(name, modules_to_not_convert):
            continue
        is_experts = name.endswith(".experts")
        if not is_experts and (
            type(module) is not nn.Linear or module.in_features % GROUP_SIZE or module.out_features % GROUP_SIZE
        ):
            continue
        if formats is None and name in routers:
            continue
        quant_format = "NVFP4" if formats is None else formats.get(name)
        if quant_format not in NVFP4_FORMATS:
            continue
        options = {"pre_quantized": pre_quantized, "weight_only": weight_only or quant_format == "W4A16_NVFP4"}
        with torch.device("meta"):
            if is_experts:
                has_gate = getattr(module, "has_gate", True)
                has_bias = getattr(module, "has_bias", False)
                experts_class = use_experts_implementation(
                    experts_class=NVFP4Experts,
                    experts_interface=ALL_NVFP4_EXPERTS_FUNCTIONS,
                    has_gate=has_gate,
                    has_bias=has_bias,
                )
                norm = module.post_expert_norm if getattr(module, "has_post_expert_norm", False) else None
                config = getattr(module, "config", model.config.get_text_config())
                new_module = experts_class(config, has_gate, has_bias, post_expert_norm=norm, **options)
            else:
                new_module = NVFP4Linear(module.in_features, module.out_features, module.bias is not None, **options)
        # the kernels take raw pointers, so tensor parallelism passes them local shards and not DTensors
        new_module._hf_quantized_needs_local_tp = True
        model.set_submodule(name, new_module)
        replaced = True
    return replaced


def update_nvfp4_parallel_plan(model: nn.Module, plan: dict[str, str]):
    """Shard the NVFP4 experts by expert, with masked expert parallelism.

    Each rank runs its own experts on every token. The router masks the other ranks' experts, and an all-reduce sums
    the outputs, so there is no all-to-all dispatch. The per-expert scales and globals shard with their weights.
    Tensor parallelism replicates the experts. Dense NVFP4 linears keep their styles, and their scalar globals are
    replicated.
    """
    from ..distributed.tensor_parallel import replace_layer_number_by_wildcard

    # the plans are built for every load, and only a distributed one applies them
    if getattr(model.config, "distributed_config", None) is None:
        return
    modules = {replace_layer_number_by_wildcard(name): module for name, module in model.named_modules()}
    for glob, module in modules.items():
        if isinstance(module, NVFP4Linear) and plan.get(glob) is not None:
            for slot in ("weight_global_scale", "input_global_scale"):
                plan[f"{glob}.{slot}"] = "replicate"
        if not isinstance(module, NVFP4Experts):
            continue
        if plan.get(glob) == "ep_dispatch_experts":
            parent = glob.rpartition(".")[0]
            router = next((f"{parent}.{key}" for key in ("gate", "router") if f"{parent}.{key}" in modules), None)
            if router is None:
                raise ValueError(f"Cannot find a gate or router for masked expert parallelism at `{glob}`.")
            plan[glob] = "moe_tp_experts"
            plan[router] = "ep_router"
        styles = {plan.get(f"{glob}.{proj}") for proj in (module.gate_up_name, "down_proj")} - {None}
        if styles == {"grouped_gemm"}:
            for proj in (module.gate_up_name, "down_proj"):
                for slot in ("_scale_inv", "_weight_global_scale", "_input_global_scale"):
                    plan[f"{glob}.{proj}{slot}"] = "grouped_gemm"
        elif styles:
            # Splitting within an expert would cut the scale groups and the interleaved gate|up rows. So the experts
            # are replicated, and only expert parallelism shards them.
            for key in [key for key in plan if key == glob or key.startswith(f"{glob}.")]:
                plan.pop(key)


def _layout_expert_tensor(experts: NVFP4Experts, slot: str, value: torch.Tensor) -> torch.Tensor:
    """Return an `(E, rows, ...)` expert tensor in the layout `experts` holds for `slot`.

    gate|up rows go from stacked to interleaved. A scale grid is swizzled when the module holds it swizzled.
    """
    if experts.has_gate and slot.startswith("gate_up_proj") and value.ndim == 3:
        gate, up = value.chunk(2, dim=1)
        value = torch.stack([gate, up], dim=2).reshape(value.shape)
    if getattr(experts, slot).ndim == 5 and value.ndim == 3:
        value = load_finegrained_kernels().swizzle_mx_scales(value.contiguous())
    return value


class NVFP4Quantize(ConversionOps):
    """Quantize a full-precision linear or experts weight into the NVFP4 tensors its module holds."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
        ((key, value),) = input_dict.items()
        value = value[0] if isinstance(value, list) else value
        module_name, _, slot = full_layer_name.rpartition(".")
        module = model.get_submodule(module_name)
        if not value.is_cuda:
            value = value.to(torch.device("cuda", torch.cuda.current_device()))
        if isinstance(module, NVFP4Linear):
            packed, scales, global_scale = quantize_nvfp4(value)
            return {
                key: packed,
                f"{module_name}.weight_scale_inv": scales,
                f"{module_name}.weight_global_scale": global_scale,
            }
        # a checkpoint may ship the stack transposed, as `(E, in, out)`, where the placeholder is `(E, out, in)`
        if value.shape != getattr(module, slot).shape:
            value = value.transpose(-1, -2)
        packed, scales, global_scales = (torch.stack(tensors) for tensors in zip(*map(quantize_nvfp4, value)))
        return {
            key: _layout_expert_tensor(module, slot, packed),
            f"{full_layer_name}_scale_inv": _layout_expert_tensor(module, f"{slot}_scale_inv", scales),
            f"{full_layer_name}_weight_global_scale": global_scales.reshape(-1),
        }


class NVFP4ExpertLayout(ConversionOps):
    """Lay out a pre-quantized NVFP4 expert tensor the way `NVFP4Experts` holds it.

    Packed E2M1 bytes ship as uint8 and are viewed as int8, since a copy would convert the values. Experts that stay
    in full precision pass through.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
        experts_name, _, slot = full_layer_name.rpartition(".")
        experts = model.get_submodule(experts_name)
        out = {}
        for key, value in input_dict.items():
            value = value[0] if isinstance(value, list) else value
            if isinstance(experts, NVFP4Experts):
                value = _layout_expert_tensor(
                    experts, slot, value.view(torch.int8) if value.dtype == torch.uint8 else value
                )
            out[key] = value
        return out

    @property
    def reverse_op(self) -> ConversionOps:
        # a save gets the experts' tensors in the checkpoint layout, from `nvfp4_checkpoint_state_dict`
        return _IdentityOp()


class NVFP4ExpertGlobals(ConversionOps):
    """Reduce the per-expert globals of the checkpoint's projections to the one per expert the experts hold.

    gate and up read the same activations, so their input globals reduce to the max. A stacked gate|up keeps one
    weight global: the gate half's, as vLLM does. It warns when the up half's differs.
    """

    def __init__(self, hf_quantizer, reduce: str):
        self.hf_quantizer = hf_quantizer
        self.reduce = reduce

    def convert(self, input_dict, full_layer_name=None, **kwargs):
        values = [
            (value[0] if isinstance(value, list) else value).float().reshape(-1) for value in input_dict.values()
        ]
        if self.reduce == "max":
            return {full_layer_name: torch.stack(values).amax(dim=0)}
        if any(not torch.equal(values[0], value) for value in values[1:]):
            logger.warning_once(
                "The gate and up projections of these NVFP4 experts have different weight globals. Both load with "
                "the gate's, as vLLM does, which may cost some accuracy."
            )
        return {full_layer_name: values[0].contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        return NVFP4CopyGlobals()


class NVFP4CopyGlobals(ConversionOps):
    """Give each of the checkpoint's projections the experts' global, the reverse of `NVFP4ExpertGlobals`."""

    def convert(self, input_dict, target_patterns=None, **kwargs):
        ((_, value),) = input_dict.items()
        value = value[0] if isinstance(value, list) else value
        return {target: value.clone() for target in target_patterns}


class NVFP4Deserialize(Fp8Dequantize):
    """Load a module's checkpoint tensors.

    An NVFP4 module takes them as they are. Any other quantized weight is dequantized to the model dtype, such as FP8
    attention, or NVFP4 itself under `dequantize=True`. Block FP8 (also as a `[N / 128, 1, K / 128, 1]` grid),
    per-tensor FP8, MXFP8 and NVFP4 with its global are read.
    """

    def convert(self, input_dict, full_layer_name=None, model=None, **kwargs):
        module_name = full_layer_name.rpartition(".")[0]
        if isinstance(model.get_submodule(module_name), (NVFP4Linear, NVFP4Experts)):
            out = {}
            for slot, value in input_dict.items():
                value = value[0] if isinstance(value, list) else value
                out[f"{module_name}.{slot}"] = value.view(torch.int8) if value.dtype == torch.uint8 else value
            return out
        output_dtype = self._get_target_dtype(model, full_layer_name)
        out = {}
        for key, value in input_dict.items():
            if key.endswith(("_scale_inv", "_global_scale")):
                continue
            scales = input_dict.get(f"{key}_scale_inv")
            if scales is None:
                out[key] = value
                continue
            # a module's global is `weight_global_scale`, a fused projection's `{proj}_weight_global_scale`
            globals_ = input_dict.get(f"{key}_global_scale", input_dict.get(f"{key}_weight_global_scale"))
            weights = value if isinstance(value, list) else [value]
            scales = scales if isinstance(scales, list) else [scales]
            globals_ = globals_ if isinstance(globals_, list) else [globals_] * len(weights)
            dequantized = [self._dequantize(*args, output_dtype) for args in zip(weights, scales, globals_)]
            out[key] = dequantized if isinstance(value, list) else dequantized[0]
        if list(out) == ["weight"]:
            weight = out["weight"]
            return {full_layer_name: weight[0] if isinstance(weight, list) else weight}
        return out

    def _dequantize(self, weight, scales, global_scale, output_dtype):
        if weight.dtype == torch.uint8:
            # packed E2M1 ships as uint8, which the FP4 unpack reads as int8
            weight = weight.view(torch.int8)
        if scales.ndim == 4:
            scales = scales.reshape(scales.shape[0], scales.shape[2])
        dequantized = self._dequantize_one(weight, scales, output_dtype=torch.float32)
        if global_scale is not None:
            # one per tensor, or one per expert of a stack
            dequantized = dequantized * global_scale.float().reshape(-1, *([1] * (dequantized.ndim - 1)))
        return dequantized.to(output_dtype or torch.bfloat16)

    @property
    def reverse_op(self) -> ConversionOps:
        # NVFP4 tensors save as they are held, and the layers loaded dequantized save in the model dtype
        return _IdentityOp()


def nvfp4_checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the state dict of `model` with its NVFP4 tensors in the checkpoint layout.

    Scales are unswizzled, gate|up rows go back to stacked, and packed E2M1 bytes are uint8. A layer quantized on the
    fly runs its activations with a global of 1, which it saves as its input global.
    """
    state_dict = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, NVFP4Linear):
            slots = [("weight", "weight_scale_inv", "weight_global_scale", "input_global_scale")]
        elif isinstance(module, NVFP4Experts):
            slots = [
                (proj, f"{proj}_scale_inv", f"{proj}_weight_global_scale", f"{proj}_input_global_scale")
                for proj in (module.gate_up_name, "down_proj")
            ]
        else:
            continue
        for weight_slot, scale_slot, global_slot, input_slot in slots:
            weight, scale = getattr(module, weight_slot), getattr(module, scale_slot)
            if scale.ndim == 5:
                experts = scale.shape[0] if isinstance(module, NVFP4Experts) else None
                rows, cols = weight.shape[-2], weight.shape[-1] * 2 // GROUP_SIZE
                plain = load_finegrained_kernels().unswizzle_mx_scales(
                    scale.view(torch.uint8), rows, cols, num_experts=experts
                )
                scale = plain.view(scale.dtype)
            if weight_slot == "gate_up_proj" and module.has_gate:
                # interleaved rows `g0, u0, g1, u1, ...` back to `gate | up`
                weight, scale = (
                    value.unflatten(1, (-1, 2)).transpose(1, 2).flatten(1, 2) for value in (weight, scale)
                )
            state_dict[f"{name}.{weight_slot}"] = weight.view(torch.uint8).contiguous()
            state_dict[f"{name}.{scale_slot}"] = scale.contiguous()
            if module.activation_format is None and getattr(module, input_slot) is None:
                state_dict[f"{name}.{input_slot}"] = torch.ones_like(getattr(module, global_slot))
    return state_dict


def nvfp4_checkpoint_conversions(
    hf_quantizer, weight_conversions: list, dequantize: bool = False, for_save: bool = False
) -> list:
    """Return the conversions that load a ModelOpt NVFP4 checkpoint.

    ModelOpt names a module's tensors `weight_scale`, `weight_scale_2` and `input_scale`, and those of a fused experts
    stack `{proj}_weight_scale` and so on. They take the module slots' names. Per-expert keys merge into the fused
    experts through the model's own converters, which also merge their globals here. With `dequantize=True` every
    converter dequantizes, globals included.
    """
    from ..core_model_loading import MergeModulelist, WeightConverter, WeightRenaming

    projections = "|".join(EXPERT_PROJECTIONS)
    renames = []
    for shipped, slot in (
        ("weight_scale", "scale_inv"),
        ("weight_scale_2", "weight_global_scale"),
        ("input_scale", "input_global_scale"),
    ):
        module_slot = "weight_scale_inv" if slot == "scale_inv" else slot
        renames.append(WeightRenaming(source_patterns=rf"^(.+)\.{shipped}$", target_patterns=rf"\1.{module_slot}"))
        renames.append(
            WeightRenaming(
                source_patterns=rf"^(.+experts\.(?:{projections}))_{shipped}$", target_patterns=rf"\1_{slot}"
            )
        )
    if dequantize:
        return renames + _dequantize_conversions(hf_quantizer, weight_conversions)
    conversions = []
    for converter in weight_conversions:
        conversions.append(converter)
        if not isinstance(converter, WeightConverter):
            continue
        target = converter._original_target_patterns[0]
        if not re.search(rf"experts\.({projections})(_scale_inv)?$", target):
            continue
        converter.operations = [*converter.operations, NVFP4ExpertLayout(hf_quantizer)]
        if target.endswith("_scale_inv"):
            continue
        for slot, reduce in (("weight_global_scale", "first"), ("input_global_scale", "max")):
            conversions.append(
                converter.copy(
                    [re.sub(r"weight$", slot, source) for source in converter._original_source_patterns],
                    [f"{target}_{slot}"],
                    operations=[MergeModulelist(dim=0), NVFP4ExpertGlobals(hf_quantizer, reduce)],
                )
            )
    if for_save:
        # a save writes the modules' own names. The fused stacks' converters would match first and keep the weights
        # fused, where the model's converters split them.
        return conversions
    # a fused experts stack ships in the model's layout, and only needs the experts' own
    conversions += [
        WeightConverter(f"experts.{proj}{slot}", f"experts.{proj}{slot}", [NVFP4ExpertLayout(hf_quantizer)])
        for proj in EXPERT_PROJECTIONS
        for slot in ("", "_scale_inv")
    ]
    return renames + conversions


def _dequantize_conversions(hf_quantizer, weight_conversions: list) -> list:
    """Return `weight_conversions` with the globals the dequantization needs, and the fused experts' converters."""
    from ..core_model_loading import WeightConverter

    # a fused experts stack dequantizes with its scale and global. It comes first, as the first matching converter wins
    conversions = [
        WeightConverter(
            [f"experts.{proj}", f"experts.{proj}_scale_inv", f"experts.{proj}_weight_global_scale"],
            f"experts.{proj}",
            [NVFP4Deserialize(hf_quantizer)],
        )
        for proj in EXPERT_PROJECTIONS
    ]
    for converter in weight_conversions:
        if isinstance(converter, WeightConverter) and isinstance(converter.operations[0], NVFP4Deserialize):
            sources = converter._original_source_patterns
            globals_ = [
                re.sub(r"weight$", "weight_global_scale", source) for source in sources if source.endswith("weight")
            ]
            converter = converter.copy(
                [*sources, *globals_], converter._original_target_patterns, operations=converter.operations
            )
        conversions.append(converter)
    return conversions
