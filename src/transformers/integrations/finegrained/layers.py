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
"""Quantized modules and forwards for the block-scaled formats (block-FP8, MXFP8, MXFP4, NVFP4), served by
`kernels-community/finegrained-kernels`. The ops in `conversions` move a checkpoint's layout onto them."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...quantizers.quantizers_utils import FUSED_STACKS
from ...utils import logging
from ...utils.import_utils import is_kernels_available
from ...utils.quantization_config import BlockScaledConfig, Fp8Config
from ..deepgemm import (
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
    deepgemm_fp8_fp4_megamoe_experts_forward,
    is_sm100,
    prefers_deepgemm_linear,
)
from ..hub_kernels import _MISSING_KERNELS_MESSAGE, lazy_load_kernel
from ..moe import ExpertsInterface, use_experts_implementation
from .formats import FP8_DTYPE, WeightFormat, cdiv, resolve_weight_format


logger = logging.get_logger(__name__)


def _first_attr(obj, *names, raise_error: bool = True):
    """The first of `names` the object defines; `None` when it defines none and the caller passes
    `raise_error=False` because it has its own fallback."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if raise_error:
        raise AttributeError(f"{type(obj).__name__} has none of: {names}")
    return None


@dataclass(frozen=True)
class FineGrained:
    """Entry points of the `kernels-community/finegrained-kernels` Triton kernel, all required.

    The kernels resolve the weight format from the weight/scale dtypes and take the module's `activation_format` as
    is (`None` = the weights' format, `"bf16"` = weight-only). The MoE forwards run the whole chain (scheduling, fused
    GLU and requant, biases, the routing-weighted reduce); an activation or a post-expert norm they do not fuse is
    passed as a callable.
    """

    matmul_2d: Callable
    matmul_grouped: Callable
    moe_fused_batched: Callable
    moe_fused_grouped: Callable
    swizzle_mx_scales: Callable
    unswizzle_mx_scales: Callable
    mxfp8_act_quant: Callable
    mxfp4_act_quant: Callable
    nvfp4_act_quant: Callable
    get_supported_act_fns: Callable
    get_supported_norms: Callable


# Successes are cached, failures are not: the env can change between attempts. A module global rather than
# `@functools.cache`, which Dynamo warns about tracing on every compile.
_FINEGRAINED: FineGrained | None = None


@torch._dynamo.allow_in_graph
def _load_finegrained_kernel() -> None:
    """Load the kernel once into `_FINEGRAINED`. `@allow_in_graph` keeps the untraceable hub download out of
    `torch.compile`'s trace; the node's output must be proxyable, so this returns `None`, never the bundle."""
    global _FINEGRAINED
    if _FINEGRAINED is not None:
        return

    if not is_kernels_available():
        raise ImportError(f"finegrained-kernels kernel unavailable: {_MISSING_KERNELS_MESSAGE}")

    kernel = lazy_load_kernel("finegrained-kernels")
    if kernel is None:
        raise ImportError(
            "Failed to load the finegrained-kernels kernel — check that `kernels-community/finegrained-kernels` "
            "has a build matching the current torch/CUDA."
        )

    symbols = {name: getattr(kernel, name, None) for name in FineGrained.__dataclass_fields__}
    missing = [name for name, attr in symbols.items() if attr is None]
    if missing:
        raise ImportError(
            f"finegrained-kernels build is missing required symbols: {', '.join(missing)}. {_MISSING_KERNELS_MESSAGE}"
        )
    _FINEGRAINED = FineGrained(**symbols)


def load_finegrained_kernel() -> FineGrained:
    _load_finegrained_kernel()
    return _FINEGRAINED


def _set_optional_parameter(module: nn.Module, name: str, tensor: torch.Tensor | None) -> None:
    """Register `name` as a Parameter over `tensor`, or as an absent one when `tensor` is None —
    a slot a checkpoint may or may not fill (a bias, an NVFP4 global, a static activation scale)."""
    module.register_parameter(name, None if tensor is None else nn.Parameter(tensor))


def _alloc_expert_proj(
    num_experts: int,
    proj_out: int,
    proj_in: int,
    format_spec: WeightFormat,
    scale_dtype: torch.dtype,
    scale_group: tuple[int, int] | None,
    min_scale_out: int = 1,
    swizzled: bool = False,
) -> tuple[nn.Parameter, nn.Parameter]:
    """`(weight, weight_scale_inv)` for one expert projection, the grid at the format's granularity (one scale per
    expert without one). `min_scale_out` floors the grid's rows so a gate|up stack keeps room for both halves.
    `swizzled` lays the grid out `SWIZZLE_32_4_4`, expert axis leading so EP shards it on dim 0; a projection that
    does not tile whole blocks stays row-major."""
    weight = torch.empty(num_experts, proj_out, proj_in // format_spec.values_per_byte, dtype=format_spec.weight_dtype)
    if scale_group is None:
        scale_shape = (num_experts, max(1, min_scale_out), 1)
    else:
        rows, cols = max(cdiv(proj_out, scale_group[0]), min_scale_out), cdiv(proj_in, scale_group[1])
        scale_shape = (num_experts, rows, cols)
        if swizzled and rows % 128 == 0 and cols % 4 == 0:
            scale_shape = (num_experts, rows // 128, cols // 4, 2, 256)
    scale = torch.empty(scale_shape, dtype=scale_dtype)
    return (
        nn.Parameter(weight, requires_grad=weight.is_floating_point()),
        nn.Parameter(scale, requires_grad=scale.is_floating_point() and not swizzled),
    )


def _holds_swizzled_scales(format_spec: WeightFormat, activation_format: str | None) -> bool:
    """Whether triton-layout experts hold their group scales `SWIZZLE_32_4_4`, the layout SM100's scaled MMA reads
    directly (row-major forces a per-tile gather): a group-scaled format whose activations are quantized."""
    return format_spec.scale_group is not None and activation_format != "bf16" and is_sm100()


def finegrained_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    allow_deepgemm: bool = True,
    weight_global_scale: torch.Tensor | None = None,
    input_global_scale: torch.Tensor | None = None,
    activation_format: str | None = None,
) -> torch.Tensor:
    """FP8/FP4 linear of `FineGrainedLinear` and the eager `FineGrainedExperts` loop: DeepGEMM where
    `prefers_deepgemm_linear` admits the call (pre-SM100 128x128 block-FP8, dynamic, inference), Triton otherwise.

    Args:
        input: `(..., K)` bf16/fp16 activations.
        weight: `(N, K)` `float8_e4m3fn`, or `(N, K // 2)` `int8` packing two FP4 values per byte.
        weight_scale_inv: the weight's block scales, `float32` or UE8M0.
        block_size: `[block_n, block_k]` of block-FP8, `None` for a per-tensor scale; FP4 formats ignore it.
        bias: added to the output.
        activation_scale: a calibrated per-tensor activation scale, `None` to quantize per token.
        allow_deepgemm: `False` forces Triton (see `disable_deepgemm_on_multi_device`).
        weight_global_scale: NVFP4's per-matrix weight global.
        input_global_scale: NVFP4's calibrated activation global, the checkpoint's `input_scale`.
        activation_format: the activation format where the weights leave it open (`"bf16"` = weight-only).
    """
    if prefers_deepgemm_linear(
        input,
        weight,
        weight_scale_inv,
        block_size=block_size,
        activation_scale=activation_scale,
        weight_global_scale=weight_global_scale,
        input_global_scale=input_global_scale,
        activation_format=activation_format,
        allow_deepgemm=allow_deepgemm,
    ):
        return deepgemm_fp8_fp4_linear(input, weight, weight_scale_inv, block_size=block_size, bias=bias)

    kernel = load_finegrained_kernel()
    original_shape = input.shape
    output = kernel.matmul_2d(
        input.reshape(-1, original_shape[-1]),
        weight,
        activation_scale,
        weight_scale_inv,
        activation_format=activation_format,
        output_dtype=input.dtype,
        a_global_scale=input_global_scale,
        b_global_scale=weight_global_scale,
    )
    output = output.reshape(*original_shape[:-1], output.shape[-1])
    if bias is not None:
        output.add_(bias)
    return output


@functools.cache
def supported_fused_norms() -> dict[type[nn.Module], str]:
    """The kernels' fused form of a post-expert norm, by the norm's exact class: a subclass may compute something
    else. Built on first use, so that importing this module does not import a model."""
    from ...models.muse_spark.modeling_muse_spark import MuseSparkTextInputScaledRMSNorm

    return {nn.RMSNorm: "rms_norm", MuseSparkTextInputScaledRMSNorm: "input_scaled_rms_norm"}


class _FineGrainedModule:
    """What every finegrained module (dense linear or experts) carries besides its parameters."""

    # the kernels take raw pointers, so the TP styles hand them local shards rather than DTensors
    tp_uses_local_tensors = True
    # set by `disable_deepgemm_on_multi_device`
    _deepgemm_disabled = False

    quantization_config: BlockScaledConfig
    # the dtype NVFP4's per-matrix weight global is held in, `None` for a single-level format
    global_scale_dtype: torch.dtype | None = None

    @property
    def two_level(self) -> bool:
        """NVFP4's two levels: a per-matrix weight global, and the calibrated activation scale read as the
        activation global rather than as a static scale."""
        return self.global_scale_dtype is not None

    @property
    def holds_activation_scale(self) -> bool:
        """Whether the module holds the checkpoint's calibrated `input_scale`: a static FP8 scale, or NVFP4's
        activation global unless the activations stay unquantized."""
        return self.activation_scheme == "static" or (self.two_level and self.activation_format != "bf16")

    def _activation_operands(self, activation: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``(activation_scale, input_global_scale)``: the calibrated activation scale in the kernels' slot."""
        return (None, activation) if self.two_level else (activation, None)

    @property
    def weight_format(self) -> str:
        return self.quantization_config.method.value

    @property
    def activation_format(self) -> str | None:
        return self.quantization_config.activation_format

    @property
    def activation_scheme(self) -> str:
        return self.quantization_config.activation_scheme

    @property
    def block_size(self) -> tuple[int, int] | None:
        return self.quantization_config.weight_block_size


class FineGrainedLinear(_FineGrainedModule, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        quantization_config: BlockScaledConfig | None = None,
        has_bias: bool = False,
    ):
        super().__init__(in_features, out_features)

        self.quantization_config = quantization_config or Fp8Config(weight_block_size=None)
        self.has_bias = has_bias

        format_spec, sf_dtype, scale_group = resolve_weight_format(
            self.weight_format, self.quantization_config.scale_fmt, self.block_size
        )
        self.global_scale_dtype = format_spec.global_scale_dtype
        in_storage = in_features // format_spec.values_per_byte
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_storage, dtype=format_spec.weight_dtype),
            requires_grad=format_spec.weight_dtype.is_floating_point,
        )
        _set_optional_parameter(
            self,
            "weight_global_scale",
            torch.tensor(1.0, dtype=self.global_scale_dtype) if self.two_level else None,
        )

        # no group and no block: one per-tensor scale
        scale = (
            torch.tensor(1.0, dtype=torch.float32)
            if scale_group is None
            else torch.empty(cdiv(out_features, scale_group[0]), cdiv(in_features, scale_group[1]), dtype=sf_dtype)
        )
        self.weight_scale_inv = nn.Parameter(scale, requires_grad=scale.dtype.is_floating_point)

        _set_optional_parameter(
            self, "activation_scale", torch.tensor(1.0, dtype=torch.float32) if self.holds_activation_scale else None
        )
        _set_optional_parameter(self, "bias", torch.empty(self.out_features) if self.has_bias else None)

    @classmethod
    def from_module(
        cls, module: nn.Linear, quantization_config: BlockScaledConfig, model: nn.Module
    ) -> FineGrainedLinear:
        return cls(
            module.in_features,
            module.out_features,
            quantization_config=quantization_config,
            has_bias=module.bias is not None,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        activation_scale, input_global_scale = self._activation_operands(self.activation_scale)
        return finegrained_linear(
            input,
            self.weight,
            self.weight_scale_inv,
            block_size=self.block_size,
            activation_scale=activation_scale,
            bias=self.bias,
            allow_deepgemm=not self._deepgemm_disabled,
            weight_global_scale=self.weight_global_scale,
            input_global_scale=input_global_scale,
            activation_format=self.activation_format,
        )


class FineGrainedEmbedding(_FineGrainedModule, nn.Embedding):
    """An embedding table stored in FP8 with one per-tensor `weight_scale`, rescaled on the rows a lookup gathers, for
    tables too big to dequantize."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        quantization_config: BlockScaledConfig | None = None,
    ):
        nn.Module.__init__(self)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if not -num_embeddings <= padding_idx < num_embeddings:
                raise ValueError("padding_idx must be within num_embeddings.")
            padding_idx %= num_embeddings
        self.padding_idx = padding_idx
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self.quantization_config = quantization_config or Fp8Config(weight_block_size=None)
        # a block-FP8 config names its tables in `modules_to_convert`, which still take one scale each
        if self.weight_format != "fp8" or (
            self.quantization_config.weight_block_size is not None and not self.quantization_config.modules_to_convert
        ):
            raise ValueError("Fine-grained embeddings hold per-tensor FP8 tables.")
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=FP8_DTYPE), requires_grad=False)
        # `(1,)` rather than a scalar: the shape checkpoints serialize per-tensor scales with
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)

    @classmethod
    def from_module(
        cls, module: nn.Embedding, quantization_config: BlockScaledConfig, model: nn.Module
    ) -> FineGrainedEmbedding:
        return cls(
            module.num_embeddings,
            module.embedding_dim,
            module.padding_idx,
            quantization_config=quantization_config,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        rows = super().forward(input)
        return rows.to(self.weight_scale.dtype) * self.weight_scale


class FineGrainedGroupedLinear(FineGrainedLinear):
    """Quantized block-diagonal grouped linear: `n_groups` sub-matrices stacked in one `(n_groups * out_per_group,
    in_per_group)` weight, each run on every token's own group. Takes `(..., n_groups, in_per_group)` and returns
    `(..., n_groups, out_per_group)`, as the grouped linear it replaces."""

    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        *,
        quantization_config: BlockScaledConfig | None = None,
        has_bias: bool = False,
    ):
        super().__init__(
            in_features_per_group, out_features, quantization_config=quantization_config, has_bias=has_bias
        )
        self.n_groups = n_groups

    @classmethod
    def from_module(
        cls, module: nn.Linear, quantization_config: BlockScaledConfig, model: nn.Module
    ) -> FineGrainedGroupedLinear:
        return cls(
            in_features_per_group=module.in_features,
            out_features=module.out_features,
            n_groups=module.n_groups,
            quantization_config=quantization_config,
            has_bias=module.bias is not None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]

        # the stored K, which packed FP4 halves
        w = self.weight.view(self.n_groups, -1, self.weight.shape[-1])
        scale_inv = self.weight_scale_inv
        x = x.movedim(-2, 0).reshape(-1, hidden_dim)
        if scale_inv.numel() == 1:
            # one scale over every group, which the grouped kernel takes per group
            scale_inv = scale_inv.reshape(1, 1, 1).expand(self.n_groups, 1, 1).contiguous()
        else:
            scale_inv = scale_inv.view(self.n_groups, scale_inv.size(0) // self.n_groups, scale_inv.size(1))

        tokens_per_group = x.size(0) // self.n_groups
        # the kernels pad the schedule to a power of two of groups, the empty ones at the final row
        padded_groups = 1 << (self.n_groups - 1).bit_length()
        expert_start = (
            torch.arange(padded_groups + 1, device=x.device, dtype=torch.int32).clamp_max(self.n_groups)
            * tokens_per_group
        )

        activation_scale, input_global_scale = self._activation_operands(self.activation_scale)
        kernel = load_finegrained_kernel()
        y = kernel.matmul_grouped(
            x,
            w,
            activation_scale,
            scale_inv,
            expert_start=expert_start,
            activation_format=self.activation_format,
            a_global_scale=input_global_scale,
            b_global_scale=self.weight_global_scale,
        )
        y = y.reshape(self.n_groups, *input_shape, -1).movedim(0, -2)
        if self.has_bias:
            y.add_(self.bias.view(self.n_groups, -1))
        return y


def finegrained_batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """Batched (decode) experts forward: one program per routed row."""
    kernel = load_finegrained_kernel()
    return kernel.moe_fused_batched(hidden_states, top_k_index, top_k_weights, **self._moe_operands(kernel))


def finegrained_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """Grouped (prefill) experts forward: the chain over the expert-sorted schedule."""
    kernel = load_finegrained_kernel()
    return kernel.moe_fused_grouped(hidden_states, top_k_index, top_k_weights, **self._moe_operands(kernel))


class ProjectionOperands(NamedTuple):
    """One expert projection's tensors as the kernels take them, `None` where the checkpoint fills no such slot."""

    weight: torch.Tensor
    scale_inv: torch.Tensor
    weight_global_scale: torch.Tensor | None
    input_global_scale: torch.Tensor | None
    activation_scale: torch.Tensor | None
    bias: torch.Tensor | None


# the names the module and the kernels give each projection's tensors, spelled once rather than on every forward
_PROJECTION_SLOTS = {
    proj: tuple(
        f"{proj}{suffix}" for suffix in ("", "_scale_inv", "_weight_global_scale", "_activation_scale", "_bias")
    )
    for proj in FUSED_STACKS
}
_KERNEL_OPERAND_KEYS = {
    proj: tuple(proj if field == "weight" else f"{proj}_{field}" for field in ProjectionOperands._fields)
    for proj in ("gate_up_proj", "down_proj")
}


class FineGrainedExpertsInterface(ExpertsInterface):
    """Interface for registering custom finegrained experts forward functions."""

    _global_mapping = {
        "deepgemm": deepgemm_fp8_fp4_experts_forward,
        "batched_mm": finegrained_batched_mm_experts_forward,
        "grouped_mm": finegrained_grouped_mm_experts_forward,
        "deepgemm_megamoe": deepgemm_fp8_fp4_megamoe_experts_forward,
    }
    # the triton chains read gate|up interleaved (group scales possibly swizzled), the DeepGEMM arms stacked
    weight_layouts = {
        "batched_mm": "interleaved",
        "grouped_mm": "interleaved",
        "deepgemm": "stacked",
        "deepgemm_megamoe": "stacked",
    }


ALL_FINEGRAINED_EXPERTS_FUNCTIONS = FineGrainedExpertsInterface()


@use_experts_implementation(experts_interface=ALL_FINEGRAINED_EXPERTS_FUNCTIONS)
class FineGrainedExperts(_FineGrainedModule, nn.Module):
    def __init__(
        self,
        config,
        *,
        quantization_config: BlockScaledConfig | None = None,
        has_gate: bool = True,
        has_bias: bool = False,
        is_concatenated: bool = True,
        post_expert_norm: nn.Module | None = None,
        has_post_expert_norm: bool = False,
        ships_transposed: bool = False,
    ):
        super().__init__()

        self.config = config
        self.swiglu_alpha = getattr(config, "swiglu_alpha", None)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.act_fn_name = _first_attr(config, "hidden_activation", "hidden_act")
        self.hidden_dim = _first_attr(config, "moe_hidden_size", "hidden_size")
        self.num_experts = _first_attr(config, "num_local_experts", "num_experts")
        self.intermediate_dim = _first_attr(config, "moe_intermediate_size", "intermediate_size")
        self.act_fn = ACT2FN[self.act_fn_name]

        self.quantization_config = quantization_config or Fp8Config(weight_block_size=None)
        activation_format = self.activation_format
        format_spec, scale_dtype, scale_group = resolve_weight_format(
            self.weight_format, self.quantization_config.scale_fmt, self.block_size
        )
        self.global_scale_dtype = format_spec.global_scale_dtype

        # The conversion ops lay gate|up rows and scales out at load in the layout the experts implementation reads
        # (`weight_layouts`); the eager loop reads either.
        self.has_gate = has_gate
        self.has_bias = has_bias
        self.is_concatenated = is_concatenated
        implementation = getattr(config, "_experts_implementation", None)
        triton_layout = ALL_FINEGRAINED_EXPERTS_FUNCTIONS.weight_layouts.get(implementation) != "stacked"
        self.holds_interleaved_gate_up = self.has_gate and triton_layout
        swizzled = triton_layout and _holds_swizzled_scales(format_spec, activation_format)
        self.gate_up_name = "gate_up_proj" if self.has_gate else "up_proj"
        up_rows = (2 if self.has_gate else 1) * self.intermediate_dim
        storage = (format_spec, scale_dtype, scale_group, swizzled)
        self._register_projection(self.gate_up_name, up_rows, self.hidden_dim, 2 if self.has_gate else 1, storage)
        self._register_projection("down_proj", self.hidden_dim, self.intermediate_dim, 1, storage)

        # the model's own per-expert output norm, which `from_module` takes from the experts this replaces
        self.post_expert_norm = post_expert_norm
        self.has_post_expert_norm = has_post_expert_norm
        # a full-precision stack quantized on load arrives in the layout of the module this replaces
        self.ships_transposed = ships_transposed

        # set by `register_shared_gate_up_input_globals`
        self.register_buffer("shared_gate_up_input_global", None, persistent=False)
        # what `_moe_operands` hands the kernels besides tensors, resolved on the first MoE forward
        self._static_moe_operands = None

    @classmethod
    def from_module(
        cls, module: nn.Module, quantization_config: BlockScaledConfig, model: nn.Module
    ) -> FineGrainedExperts:
        """A `FineGrainedExperts` holding `module`'s weights, in the model's own gate|up convention and with its
        per-expert output norm."""
        config = getattr(module, "config", None) or model.config.get_text_config()
        return cls(
            config=config,
            quantization_config=quantization_config,
            has_gate=getattr(module, "has_gate", True),
            has_bias=getattr(module, "has_bias", False),
            is_concatenated=getattr(module, "is_concatenated", True),
            post_expert_norm=getattr(module, "post_expert_norm", None),
            has_post_expert_norm=getattr(module, "has_post_expert_norm", False),
            ships_transposed=getattr(module, "is_transposed", False),
        )

    @property
    def experts_weight_layout(self) -> str | None:
        """The layout this module's weights were laid out in when it loaded (see `FineGrainedExpertsInterface`), `None`
        for a non-gated stack with row-major scales, which any implementation reads."""
        swizzled = any(getattr(self, f"{proj}_scale_inv").ndim == 5 for proj in (self.gate_up_name, "down_proj"))
        if self.holds_interleaved_gate_up or swizzled:
            return "interleaved"
        return "stacked" if self.has_gate else None

    def _register_projection(self, proj, rows, in_dim, min_scale_out, storage):
        """One expert projection: its weight and scale grid, then the slots a checkpoint may fill (a bias, NVFP4's
        weight global, a calibrated activation scale)."""
        format_spec, scale_dtype, scale_group, swizzled = storage
        weight, scale = _alloc_expert_proj(
            self.num_experts, rows, in_dim, format_spec, scale_dtype, scale_group, min_scale_out, swizzled
        )
        self.register_parameter(proj, weight)
        self.register_parameter(f"{proj}_scale_inv", scale)

        # in the model dtype, as the experts it replaces hold it
        bias = torch.empty(self.num_experts, rows) if self.has_bias else None
        # one per expert: `FineGrainedWeightGlobals` merges a gate|up stack's separately calibrated halves
        weight_global = torch.ones(self.num_experts, dtype=self.global_scale_dtype) if self.two_level else None
        activation = torch.ones(self.num_experts, dtype=torch.float32) if self.holds_activation_scale else None
        _set_optional_parameter(self, f"{proj}_bias", bias)
        _set_optional_parameter(self, f"{proj}_weight_global_scale", weight_global)
        _set_optional_parameter(self, f"{proj}_activation_scale", activation)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # interleaved rows give alternating output columns, the split the fused epilogue does; stacked rows two halves
        if self.holds_interleaved_gate_up:
            gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
        else:
            gate, up = gate_up.chunk(2, dim=-1)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        if self.swiglu_alpha is not None:
            # SwiGLU-OAI gate (same math as the model's non-quantized experts)
            glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
            return (up + 1.0) * glu
        return self.act_fn(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # index_add_ will accumulate using the dtype of the tensor we write into
        # so we use float32 for the accumulation to avoid numerical issues in bf16/fp16
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        # one sync here instead of one per iteration: indexing with a 0-dim cuda tensor syncs
        for expert_idx in expert_hit.tolist():
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            proj_out = self.expert_linear(hidden_states[token_idx], self.gate_up_name, expert_idx)
            proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
            proj_out = self.expert_linear(proj_out, "down_proj", expert_idx)
            if self.has_post_expert_norm:
                proj_out = self.post_expert_norm(proj_out)
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            weighted_out = proj_out * routing_weights.to(proj_out.dtype)
            final_hidden_states.index_add_(0, token_idx, weighted_out.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def _projection_operands(self, proj: str) -> ProjectionOperands:
        """``proj``'s tensors as the kernels take them: the calibrated activation scale goes in as NVFP4's
        activation global (the gate_up's shared one where it is set) or as a static scale."""
        weight, scale_inv, weight_global, activation, bias = (
            self._parameters[name] for name in _PROJECTION_SLOTS[proj]
        )
        shared = self._buffers["shared_gate_up_input_global"]
        if shared is not None and proj == self.gate_up_name:
            activation = shared
        activation_scale, input_global_scale = self._activation_operands(activation)
        return ProjectionOperands(weight, scale_inv, weight_global, input_global_scale, activation_scale, bias)

    def expert_linear(self, input: torch.Tensor, proj: str, expert_idx: int) -> torch.Tensor:
        """One expert's ``proj`` as a dense linear, over that expert's slice of the projection's operands."""
        weight, scale, weight_global, input_global, activation_scale, bias = self._projection_operands(proj)
        weight = weight[expert_idx]
        # one expert's slice of a swizzled stack is the `(1, ...)` artifact `matmul_2d` reads
        scale = scale[expert_idx : expert_idx + 1] if scale.ndim == 5 else scale[expert_idx]
        # None where the checkpoint fills no such slot
        weight_global, activation_scale, bias = (
            operand[expert_idx] if operand is not None else None for operand in (weight_global, activation_scale, bias)
        )
        # one per expert, or one shared by all
        if input_global is not None and input_global.numel() > 1:
            input_global = input_global[expert_idx : expert_idx + 1]

        linear = functools.partial(
            finegrained_linear,
            block_size=self.block_size,
            activation_scale=activation_scale,
            allow_deepgemm=not self._deepgemm_disabled,
            weight_global_scale=weight_global,
            input_global_scale=input_global,
            activation_format=self.activation_format,
        )
        # Separately calibrated gate/up grids interleave per block, rather than per weight row:
        # run each half with its own blocks, then interleave the output columns. A single scale row
        # is shared by both halves and can be read directly.
        if (
            proj == self.gate_up_name
            and self.holds_interleaved_gate_up
            and scale.ndim == 2
            and 1 < scale.shape[0] < weight.shape[0]
        ):
            halves = [
                linear(
                    input,
                    weight[half::2].contiguous(),
                    scale[half::2].contiguous(),
                    bias=bias[half::2] if bias is not None else None,
                )
                for half in (0, 1)
            ]
            return torch.stack(halves, dim=-1).flatten(-2)
        return linear(input, weight, scale, bias=bias)

    def _moe_operands(self, kernel) -> dict:
        """Everything the kernels' MoE forwards take from the module; only the tensors are gathered per call. A
        method so an adapter can extend it: PEFT adds its LoRA factors, which the kernels apply between the GEMMs."""
        static = self._static_moe_operands
        if static is None:
            static = self._static_moe_operands = self._resolve_static_moe_operands(kernel)
        operands = dict(static)
        operands.update(zip(_KERNEL_OPERAND_KEYS["gate_up_proj"], self._projection_operands(self.gate_up_name)))
        operands.update(zip(_KERNEL_OPERAND_KEYS["down_proj"], self._projection_operands("down_proj")))
        if isinstance(static["post_expert_norm"], str):
            weight = operands["post_expert_norm_weight"] = self.post_expert_norm.weight
            if static["post_expert_norm_eps"] is None:
                # `nn.RMSNorm` leaves epsilon None and resolves it to the dtype's own
                operands["post_expert_norm_eps"] = torch.finfo(weight.dtype).eps
        return operands

    def _resolve_static_moe_operands(self, kernel) -> dict:
        """The non-tensor half of `_moe_operands`. An activation or post-expert norm the kernels support is passed by
        name and fused; any other as the module's own callable, run between the kernels."""
        act_name = self.act_fn_name
        if act_name in kernel.get_supported_act_fns() and (self.swiglu_alpha is None or act_name == "silu"):
            act_fn = act_name
        else:
            act_fn = self._apply_gate if self.has_gate else self.act_fn

        post_expert_norm, norm_eps = None, 1e-6
        if self.has_post_expert_norm:
            norm_name = supported_fused_norms().get(type(self.post_expert_norm))
            if norm_name in kernel.get_supported_norms():
                # the model owns this module, so take epsilon under either name
                post_expert_norm = norm_name
                eps = _first_attr(self.post_expert_norm, "eps", "variance_epsilon", raise_error=False)
                norm_eps = float(eps) if eps is not None else None
            else:
                post_expert_norm = self.post_expert_norm
        return {
            "act_fn": act_fn,
            "gate": self.has_gate,
            "swiglu_alpha": self.swiglu_alpha,
            "swiglu_limit": self.swiglu_limit,
            "activation_format": self.activation_format,
            "post_expert_norm": post_expert_norm,
            "post_expert_norm_weight": None,
            "post_expert_norm_eps": norm_eps,
        }


def assert_modules_are_quantized(model: nn.Module) -> None:
    """Refuse a full-precision weight in a finegrained module once the checkpoint is in: the checkpoint does not
    quantize that module, and serving it would hand back a model that is not quantized."""
    for name, module in model.named_modules():
        if not isinstance(module, _FineGrainedModule):
            continue
        for attr in ("weight", *FUSED_STACKS):
            weight = getattr(module, attr, None)
            if weight is not None and weight.element_size() > 1:
                raise ValueError(
                    f"{name}.{attr} was converted for quantized compute but holds a {weight.dtype} "
                    "weight — this checkpoint does not quantize it. Name the module in the "
                    "quantization config's `modules_to_not_convert` to leave it as it is."
                )


def register_shared_gate_up_input_globals(model: nn.Module) -> None:
    """Share the experts' gate_up NVFP4 activation global, as the max of their calibrated ones, where the config's
    `shared_gate_up_activation_global` asks for it (see `NVFP4Config`)."""
    for module in model.modules():
        if not isinstance(module, FineGrainedExperts):
            continue
        scope = getattr(module.quantization_config, "shared_gate_up_activation_global", None)
        if scope is None:
            continue
        input_global = getattr(module, f"{module.gate_up_name}_activation_scale")
        if input_global is not None:
            if hasattr(input_global, "full_tensor"):  # sharded over experts
                input_global = input_global.full_tensor() if scope == "global" else input_global.to_local()
            module.shared_gate_up_input_global = input_global.detach().amax().reshape(1)


def disable_deepgemm_on_multi_device(model: nn.Module) -> None:
    """Route the quantized modules through Triton when the model spans several CUDA devices in one process: DeepGEMM
    binds each cached kernel to the CUDA context live when it loaded, and corrupts on another device."""
    quantized_modules = [m for m in model.modules() if isinstance(m, _FineGrainedModule)]
    cuda_devices = set()
    for m in quantized_modules:
        param = next(m.parameters(), None)
        if param is not None and param.device.type == "cuda":
            cuda_devices.add(param.device.index)
    if len(cuda_devices) <= 1:
        return
    for m in quantized_modules:
        m._deepgemm_disabled = True
    logger.warning_once(
        "This finegrained quantized model spans multiple CUDA devices in one process; routing its linear and experts "
        "layers through our triton kernels instead of DeepGEMM (DeepGEMM's cached kernels are bound to a "
        "single CUDA context and corrupt across devices). Run tensor/expert parallel (one device per "
        "process) to use the faster DeepGEMM path."
    )
