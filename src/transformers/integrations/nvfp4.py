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
from ..core_model_loading import ConversionOps, Interleave, _IdentityOp
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging
from .hub_kernels import _MISSING_KERNELS_MESSAGE, lazy_load_kernel
from .moe import ExpertsInterface, use_experts_implementation


if is_torch_available():
    import torch
    import torch.nn as nn
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    from .finegrained_fp8 import Fp8Dequantize


logger = logging.get_logger(__name__)

GROUP_SIZE = 16
# the projections of a fused experts module
EXPERT_PROJECTIONS = ("gate_up_proj", "up_proj", "down_proj")


@functools.cache
def load_finegrained_kernels():
    """Load the `finegrained-kernels` Hub kernel once."""
    kernel = lazy_load_kernel("finegrained-kernels")
    if kernel is None:
        raise ImportError(
            f"NVFP4 needs the `kernels-community/finegrained-kernels` kernel. {_MISSING_KERNELS_MESSAGE}"
        )
    return kernel


def _first_attr(config, *names):
    return next(getattr(config, name) for name in names if getattr(config, name, None) is not None)


def _parameter(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _fits_swizzled_layout(rows: int, scale_cols: int) -> bool:
    """Check whether `(rows, scale_cols)` scales can be swizzled. Only Blackwell GPUs read that layout.

    A weight has one scale per 16 values, so its scales form a `(rows, cols / 16)` table. The layout cuts that table
    into blocks of 128 rows by 4 columns. So the rows must be a multiple of 128 and the columns a multiple of 4. The
    kernels also read at least two blocks across, so the scales need 8 columns or more.

    The scales of a whole weight usually fit. Tensor parallelism gives each rank a smaller piece, which may not fit.
    So the check runs on each rank's piece.
    """
    return _is_blackwell() and rows % 128 == 0 and scale_cols % 4 == 0 and scale_cols >= 8


@functools.cache
def supported_fused_norms() -> dict[type, str]:
    """Map each post-expert norm class the kernels fuse to its kernel name. Built lazily to avoid importing a model."""
    from ..models.muse_spark.modeling_muse_spark import MuseSparkTextInputScaledRMSNorm

    return {nn.RMSNorm: "rms_norm", MuseSparkTextInputScaledRMSNorm: "input_scaled_rms_norm"}


def quantize_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a `(N, K)` weight to `(packed int8, E4M3 scales, fp32 global)`, with a global of `amax / (6 * 448)`."""
    global_scale = (weight.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float()
    packed, scales = load_finegrained_kernels().nvfp4_act_quant((weight.float() / global_scale).contiguous())
    return packed.view(torch.int8), scales, global_scale


class NVFP4Linear(nn.Module):
    """Linear layer with NVFP4 weights.

    Activations quantize against the calibrated `input_global_scale`, or a global of 1 when quantized on the fly.
    `weight_only=True` keeps them in the model dtype. On the fly, the weight starts as a full-precision placeholder.
    On SM100 the scales may load swizzled, as a 5-D grid, see `NVFP4SwizzleScales`.
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
        scale_shape = (out_features, in_features // GROUP_SIZE)
        self.weight_scale_inv = _parameter(torch.empty(scale_shape, dtype=torch.float8_e4m3fn))
        self.weight_global_scale = _parameter(torch.ones((), dtype=torch.float32))
        self.input_global_scale = None if weight_only else _parameter(torch.ones((), dtype=torch.float32))
        self.bias = _parameter(torch.empty(out_features)) if bias else None

    def swizzles_scales(self, slot: str, rows: int, scale_cols: int) -> bool:
        """Return whether scales of this shape should be swizzled. A linear only needs its own scales to fit.

        Under tensor parallelism, the shape is this rank's piece of the scales.
        """
        return self.activation_format != "bf16" and _fits_swizzled_layout(rows, scale_cols)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # under tensor parallelism the shapes come from the shards
        output = load_finegrained_kernels().matmul_2d(
            A=input.reshape(-1, input.shape[-1]).contiguous(),
            B=self.weight,
            Bs=self.weight_scale_inv,
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
    """MoE experts with NVFP4 weights and one weight and input global per expert.

    gate|up rows are interleaved, so the kernels fuse the gated activation. On Blackwell GPUs the scales may be
    swizzled, see `swizzles_scales`.
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
        # an optional clamped SwiGLU, see `_apply_gate`
        self.swiglu_alpha = getattr(config, "swiglu_alpha", None)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.activation_format = "bf16" if weight_only else None
        # normalizes each expert's output before the weighted sum
        self.post_expert_norm = post_expert_norm
        self.gate_up_name = "gate_up_proj" if has_gate else "up_proj"
        up_rows = (2 if has_gate else 1) * self.intermediate_dim
        self._register_projection(self.gate_up_name, up_rows, self.hidden_dim, pre_quantized, weight_only)
        self._register_projection("down_proj", self.hidden_dim, self.intermediate_dim, pre_quantized, weight_only)

    def swizzles_scales(self, slot: str, rows: int, scale_cols: int) -> bool:
        """Return whether scales of this shape should be swizzled.

        Down only needs its own scales to fit. A swizzled gate_up writes the activations for down in the swizzled
        layout, which only a swizzled down reads. So gate_up also needs the down scales to fit, and its rows give the
        size down has on this rank. Under tensor parallelism, the shape is this rank's piece of the scales.
        """
        if self.activation_format == "bf16" or not _fits_swizzled_layout(rows, scale_cols):
            return False
        if slot == "down_proj_scale_inv":
            return True
        intermediate = rows // (2 if self.has_gate else 1)
        return _fits_swizzled_layout(self.hidden_dim, intermediate // GROUP_SIZE)

    def _register_projection(self, proj: str, rows: int, cols: int, pre_quantized: bool, weight_only: bool):
        scale_shape = (self.num_experts, rows, cols // GROUP_SIZE)
        packed = torch.empty(self.num_experts, rows, cols // 2, dtype=torch.int8)
        setattr(self, proj, _parameter(packed if pre_quantized else torch.empty(self.num_experts, rows, cols)))
        setattr(self, f"{proj}_scale_inv", _parameter(torch.empty(scale_shape, dtype=torch.float8_e4m3fn)))
        setattr(self, f"{proj}_weight_global_scale", _parameter(torch.ones(self.num_experts, dtype=torch.float32)))
        input_global = None if weight_only else _parameter(torch.ones(self.num_experts, dtype=torch.float32))
        setattr(self, f"{proj}_input_global_scale", input_global)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        if self.swiglu_alpha is not None:
            return (up + 1.0) * gate * torch.sigmoid(gate * self.swiglu_alpha)
        return self.act_fn(gate) * up

    def _moe_operands(self, kernel) -> dict:
        """Return the arguments of the kernels' fused MoE forwards."""
        if self.act_fn_name in kernel.get_supported_act_fns():
            act_fn = self.act_fn_name
        else:
            act_fn = self._apply_gate if self.has_gate else self.act_fn
        operands = {
            "act_fn": act_fn,
            "gate": self.has_gate,
            "swiglu_alpha": self.swiglu_alpha,
            "swiglu_limit": self.swiglu_limit,
            "activation_format": self.activation_format,
            "post_expert_norm": self.post_expert_norm,
            "post_expert_norm_weight": None,
            "post_expert_norm_eps": None,
        }
        # the kernels fuse the norms they know, by name
        norm = self.post_expert_norm
        if norm is not None and (kind := supported_fused_norms().get(type(norm))) in kernel.get_supported_norms():
            eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
            operands["post_expert_norm"] = kind
            operands["post_expert_norm_weight"] = norm.weight
            # `nn.RMSNorm` leaves `eps` as `None` and uses the dtype's epsilon
            operands["post_expert_norm_eps"] = float(eps) if eps is not None else torch.finfo(norm.weight.dtype).eps
        # the kernels call the up projection `gate_up_proj`, gated or not
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
            A=input,
            B=getattr(self, proj)[expert_idx],
            # `matmul_2d` reads a swizzled expert slice with a leading dim of 1
            Bs=scale[expert_idx : expert_idx + 1] if scale.ndim == 5 else scale[expert_idx],
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
    """Replace the linears and fused experts in an NVFP4 format. Return whether any module was replaced.

    `formats` maps module names to how they load: `"nvfp4"`, `"nvfp4_weight_only"` or `"dequantize"`. Without it, every
    eligible module is NVFP4. A pre-quantized linear that dequantizes gets a slot for its scale, see
    `dequantize_fp8_linears`.
    """
    # routers stay in full precision unless the checkpoint quantizes them
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
        loads = "nvfp4" if formats is None else formats.get(name)
        if loads == "dequantize" and pre_quantized:
            # an FP8 layer loads its scale into a slot of its own, see `dequantize_fp8_linears`
            module.register_parameter("weight_scale_inv", _parameter(torch.empty((), device="meta")))
            continue
        if loads not in ("nvfp4", "nvfp4_weight_only"):
            continue
        options = {"pre_quantized": pre_quantized, "weight_only": weight_only or loads == "nvfp4_weight_only"}
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
    """Shard NVFP4 experts and replicate the dense globals.

    Expert parallelism is masked: each rank runs its own experts on every token, and an all-reduce sums the outputs.
    Tensor parallelism alone splits inside each expert.
    """
    from ..distributed.tensor_parallel import replace_layer_number_by_wildcard

    # only a distributed load applies the plan
    if getattr(model.config, "distributed_config", None) is None:
        return
    modules = {replace_layer_number_by_wildcard(name): module for name, module in model.named_modules()}
    for glob, module in modules.items():
        if isinstance(module, NVFP4Linear) and plan.get(glob) is not None:
            for slot in ("weight_global_scale", "input_global_scale"):
                plan[f"{glob}.{slot}"] = "replicate"
        # an FP8 layer's scale grid is small, so each rank holds it whole and slices its shard when it dequantizes
        if type(module) is nn.Linear and "weight_scale_inv" in module._parameters and plan.get(glob) is not None:
            plan[f"{glob}.weight_scale_inv"] = "replicate"
        if not isinstance(module, NVFP4Experts):
            continue
        if plan.get(glob) == "ep_dispatch_experts":
            parent = glob.rpartition(".")[0]
            router = next((f"{parent}.{key}" for key in ("gate", "router") if f"{parent}.{key}" in modules), None)
            if router is None:
                raise ValueError(f"Cannot find a gate or router for masked expert parallelism at `{glob}`.")
            plan[glob] = "moe_tp_experts"
            plan[router] = "ep_router"
        # The base plan gives the scales their weight's style. Expert parallelism splits the per-expert globals too,
        # while tensor parallelism gives each rank a slice of every expert, so every global.
        for proj in (module.gate_up_name, "down_proj"):
            if plan.get(f"{glob}.{proj}") == "grouped_gemm":
                for slot in ("_weight_global_scale", "_input_global_scale"):
                    plan[f"{glob}.{proj}{slot}"] = "grouped_gemm"


class _NVFP4Op(ConversionOps):
    """Base op that carries the quantizer. Its reverse is the same op with `inverse` flipped."""

    def __init__(self, hf_quantizer=None, inverse: bool = False):
        self.hf_quantizer = hf_quantizer
        self.inverse = inverse

    @property
    def reverse_op(self) -> ConversionOps:
        return type(self)(self.hf_quantizer, inverse=not self.inverse)


def _single(value):
    return value[0] if isinstance(value, list) else value


def _nvfp4_module(model, name: str, renames: list | None = None):
    """Return the name and the NVFP4 module that own the tensor `name`, or `(None, None)`.

    On save `name` is a checkpoint key, e.g. `block_sparse_moe.experts.0.w1.weight`. The model's `renames` give its
    model name, and its module is the closest parent.
    """
    for rule in renames or ():
        name, _ = rule.rename_source_key(name)
    while name:
        name = name.rpartition(".")[0]
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        return (name, module) if isinstance(module, (NVFP4Linear, NVFP4Experts)) else (None, None)
    return None, None


def _apply_param_ops(op, model, full_name: str, value):
    """Run `value` through the layout ops of `full_name`. On save, run them in reverse."""
    ops = op.hf_quantizer.get_param_ops(model, full_name)
    if op.inverse:
        ops = [param_op.reverse_op for param_op in ops[::-1]]
    tensors = {full_name: value}
    for param_op in ops:
        tensors = param_op.convert(tensors, model=model, full_layer_name=full_name)
    return tensors[full_name]


def _reciprocal_globals(hf_quantizer) -> bool:
    return hf_quantizer is not None and hf_quantizer.quantization_config.checkpoint_format == "nvfp4-pack-quantized"


class NVFP4Reciprocal(_NVFP4Op):
    """Invert the globals of an `nvfp4-pack-quantized` checkpoint. It ships `448 * 6 / amax`, and the kernels take the
    reciprocal. The op is its own inverse, so a save writes them back as they ship.
    """

    def convert(self, input_dict, **kwargs):
        return {key: 1 / _single(value) for key, value in input_dict.items()}


class NVFP4ViewPackedInt8(_NVFP4Op):
    """View packed E2M1 uint8 bytes as the int8 the modules hold. A copy would convert the values instead."""

    def convert(self, input_dict, **kwargs):
        held, want = (torch.int8, torch.uint8) if self.inverse else (torch.uint8, torch.int8)
        out = {}
        for key, value in input_dict.items():
            value = _single(value)
            out[key] = value.view(want) if value.dtype == held else value
        return out


class NVFP4InterleaveGateUp(_NVFP4Op):
    """Reorder expert gate|up rows between the checkpoint's stacked order and the interleaved order the experts hold."""

    def convert(self, input_dict, **kwargs):
        # `Interleave(inverse=True)` goes from stacked to interleaved
        interleave = Interleave(dim=1, inverse=not self.inverse)
        return {key: interleave.convert({key: _single(value)}, [key], [key])[key] for key, value in input_dict.items()}


class NVFP4SwizzleScales(_NVFP4Op):
    """Swizzle scales into the 5-D layout Blackwell GPUs read, when their module says so.

    Under tensor or expert parallelism, the op works on this rank's piece. The kernels read the layout from the shape.
    The reverse turns any 5-D scales back into plain ones.
    """

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
        module_name, _, slot = full_layer_name.rpartition(".")
        module = model.get_submodule(module_name)
        dense = isinstance(module, NVFP4Linear)
        kernel = load_finegrained_kernels()
        out = {}
        for key, value in input_dict.items():
            value = _single(value).contiguous()
            if self.inverse and value.ndim == 5:
                matrices, row_blocks, column_groups = value.shape[:3]
                plain = kernel.unswizzle_mx_scales(value, row_blocks * 128, column_groups * 4, num_experts=matrices)
                value = plain[0] if dense else plain
            elif not self.inverse and module.swizzles_scales(slot, *value.shape[-2:]):
                # the kernels swizzle a stack of scales, so a linear's scales are a stack of one
                value = kernel.swizzle_mx_scales(value.unsqueeze(0) if dense else value)
            elif not self.inverse and module.activation_format != "bf16" and _is_blackwell():
                logger.warning_once(
                    "Some NVFP4 scales don't fit the swizzled layout Blackwell GPUs read fastest, so those layers run "
                    "slower. Tensor parallelism makes this more likely, since each GPU holds a smaller piece of each "
                    "weight's scales. A smaller `tp_size`, or expert parallelism for MoE experts, can avoid it."
                )
            out[key] = value
        return out


class NVFP4Quantize(_NVFP4Op):
    """Quantize a full-precision linear or experts weight into the NVFP4 tensors its module holds."""

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
        value = _single(next(iter(input_dict.values())))
        module_name, _, slot = full_layer_name.rpartition(".")
        module = model.get_submodule(module_name)
        # the NVFP4 quantize kernel runs on the accelerator
        if value.device.type == "cpu":
            value = value.to(torch.accelerator.current_accelerator())
        if isinstance(module, NVFP4Linear):
            packed, scales, global_scale = quantize_nvfp4(value)
            names = ("weight", "weight_scale_inv", "weight_global_scale", "input_global_scale")
        else:
            # a checkpoint may ship the stack transposed, as `(E, in, out)`, where the placeholder is `(E, out, in)`
            if value.shape != getattr(module, slot).shape:
                value = value.transpose(-1, -2)
            packed, scales, global_scale = (torch.stack(tensors) for tensors in zip(*map(quantize_nvfp4, value)))
            global_scale = global_scale.reshape(-1)
            names = (slot, f"{slot}_scale_inv", f"{slot}_weight_global_scale", f"{slot}_input_global_scale")
        out = {}
        for name, tensor in zip(names, (packed, scales, global_scale)):
            out[f"{module_name}.{name}"] = _apply_param_ops(self, model, f"{module_name}.{name}", tensor)
        # no calibration, so the activations quantize with a global of 1
        if getattr(module, names[3]) is not None:
            out[f"{module_name}.{names[3]}"] = torch.ones_like(global_scale)
        return out


class NVFP4ExpertGlobals(ConversionOps):
    """Reduce the gate and up globals to one per expert: the max for input globals, the gate's for weight globals."""

    def __init__(self, hf_quantizer, reduce: str):
        self.hf_quantizer = hf_quantizer
        self.reduce = reduce

    def convert(self, input_dict, full_layer_name=None, **kwargs):
        values = [_single(value).float().reshape(-1) for value in input_dict.values()]
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
        value = _single(next(iter(input_dict.values())))
        return {target: value.clone() for target in target_patterns}


class NVFP4Dequantize(Fp8Dequantize):
    """Dequantize a weight to the model dtype: FP8, block FP8, MXFP8, or NVFP4 with its global.

    Dequantized layers save in the model dtype.
    """

    def convert(self, input_dict, full_layer_name=None, model=None, **kwargs):
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
            return {full_layer_name: _single(out["weight"])}
        return out

    def _dequantize(self, weight, scales, global_scale, output_dtype):
        if weight.dtype == torch.uint8:
            if global_scale is None:
                raise ValueError("An NVFP4 layer that loads dequantized needs its global; pass `dequantize=True`.")
            # packed E2M1 ships as uint8, which the FP4 unpack reads as int8
            weight = weight.view(torch.int8)
        if scales.ndim == 4:
            scales = scales.reshape(scales.shape[0], scales.shape[2])
        dequantized = self._dequantize_one(weight, scales, output_dtype=torch.float32)
        if global_scale is not None:
            global_scale = global_scale.float()
            if _reciprocal_globals(self.hf_quantizer):
                global_scale = 1 / global_scale
            dequantized = dequantized * global_scale.reshape(-1, *([1] * (dequantized.ndim - 1)))
        return dequantized.to(output_dtype or torch.bfloat16)

    @property
    def reverse_op(self) -> ConversionOps:
        return _IdentityOp()


def dequantize_fp8_linears(model: nn.Module):
    """Dequantize the FP8 layers that loaded with their scale, and drop the scale.

    The scale is per tensor or a block grid, whose block size follows from its shape. Under tensor parallelism each rank
    holds the whole grid and dequantizes its shard of the weight.
    """
    for module in model.modules():
        scale = module._parameters.get("weight_scale_inv") if type(module) is nn.Linear else None
        if scale is None:
            continue
        # an unquantized layer that matched a catch-all pattern loads no scale
        if scale.is_meta:
            del module.weight_scale_inv
            continue
        weight, grid = module.weight.data, scale.data
        offsets = [0, 0]
        if isinstance(weight, DTensor):
            _, offsets = compute_local_shape_and_global_offset(weight.shape, weight.device_mesh, weight.placements)
            grid = grid.to_local() if isinstance(grid, DTensor) else grid
        local = weight.to_local() if isinstance(weight, DTensor) else weight
        if grid.numel() > 1:
            # ModelOpt ships a block grid as `[N / 128, 1, K / 128, 1]`
            grid = grid.reshape(grid.shape[0], grid.shape[2]) if grid.ndim == 4 else grid
            if weight.shape[0] % grid.shape[0] or weight.shape[1] % grid.shape[1]:
                raise ValueError(
                    f"A {tuple(weight.shape)} weight does not split into a {tuple(grid.shape)} scale grid."
                )
            block = (weight.shape[0] // grid.shape[0], weight.shape[1] // grid.shape[1])
            rows, cols = (
                slice(start // size, -(-(start + length) // size))
                for start, length, size in zip(offsets, local.shape, block)
            )
            grid = grid[rows, cols]
        local.copy_(NVFP4Dequantize(None)._dequantize_one(local, grid, output_dtype=local.dtype))
        del module.weight_scale_inv


class NVFP4Deserialize(_NVFP4Op):
    """Load a module's checkpoint tensors after a converter merges them, and write them back on save.

    An NVFP4 module's tensors run its layout ops, in reverse on save. Other quantized layers, such as FP8 attention,
    load dequantized.
    """

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
        renames = self.hf_quantizer.model_renames if self.inverse else None
        module_name, module = _nvfp4_module(model, full_layer_name, renames)
        if module is None:
            if self.inverse:
                return input_dict
            holder_name, _, slot = full_layer_name.rpartition(".")
            holder = model.get_submodule(holder_name)
            if "weight_scale_inv" in holder._parameters:
                # an FP8 layer loads as it ships, its weight in the model dtype, see `dequantize_fp8_linears`
                value = _single(next(iter(input_dict.values())))
                return {full_layer_name: value if slot == "weight_scale_inv" else value.to(holder.weight.dtype)}
            return NVFP4Dequantize(self.hf_quantizer).convert(
                input_dict, full_layer_name=full_layer_name, model=model, **kwargs
            )
        out = {}
        for key, value in input_dict.items():
            full_name = f"{module_name}.{key.rpartition('.')[2]}"
            # on save the converter splits the tensors again, so they keep their keys
            out[key if self.inverse else full_name] = _apply_param_ops(self, model, full_name, _single(value))
        return out


def checkpoint_renames() -> list:
    """Return the renames that give a checkpoint's NVFP4 tensors the slots' names.

    ModelOpt names them `weight_scale`, `weight_scale_2` and `input_scale`. `nvfp4-pack-quantized` names the packed
    weight `weight_packed`.
    """
    from ..core_model_loading import WeightRenaming

    projections = "|".join(EXPERT_PROJECTIONS)
    renames = [WeightRenaming(source_patterns=r"^(.+)\.weight_packed$", target_patterns=r"\1.weight")]
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
    return renames


def nvfp4_checkpoint_conversions(hf_quantizer, weight_conversions: list, dequantize: bool = False) -> list:
    """Return the conversions that load an NVFP4 checkpoint. Per-expert globals merge through copies of the model's
    own expert converters.
    """
    from ..core_model_loading import MergeModulelist, WeightConverter

    projections = "|".join(EXPERT_PROJECTIONS)
    # a dequantized load needs nothing more, as `serialized_names` lists the globals it folds in
    if dequantize:
        return weight_conversions
    conversions = []
    for converter in weight_conversions:
        conversions.append(converter)
        if not isinstance(converter, WeightConverter):
            continue
        # every converter loads its tensors into their module's layout, e.g. a fused gate|up linear's scales
        if not any(isinstance(op, NVFP4Deserialize) for op in converter.operations):
            converter.operations = [*converter.operations, NVFP4Deserialize(hf_quantizer)]
        target = converter._original_target_patterns[0]
        if not re.search(rf"experts\.({projections})$", target):
            continue
        for slot, reduce in (("weight_global_scale", "first"), ("input_global_scale", "max")):
            conversions.append(
                converter.copy(
                    [re.sub(r"weight$", slot, source) for source in converter._original_source_patterns],
                    [f"{target}_{slot}"],
                    # the reduce takes the multipliers, so a reciprocal global inverts before it
                    operations=[
                        MergeModulelist(dim=0),
                        *([NVFP4Reciprocal(hf_quantizer)] if _reciprocal_globals(hf_quantizer) else []),
                        NVFP4ExpertGlobals(hf_quantizer, reduce),
                    ],
                )
            )
    return conversions
