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
"""Quantized modules and forwards for the block-scaled formats.

The formats are block-FP8, MXFP8, MXFP4 and NVFP4, served by `kernels-community/finegrained-kernels`.
The ops in `conversions` load a checkpoint's layout into these modules.
"""

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
from ...utils.quantization_config import BlockScaledConfig
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


def _first_attr(obj, *names):
    """Return the first attribute of `names` that `obj` defines."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of: {names}")


@dataclass(frozen=True)
class FineGrained:
    """Entry points of the `kernels-community/finegrained-kernels` Triton kernel. All of them are required.

    The kernels infer the weight format from the weight and scale dtypes.
    They take the module's `activation_format` as is. `None` means the weights' format and `"bf16"` means weight-only.
    The MoE forwards run the whole expert chain, from scheduling to the routing-weighted reduce.
    An activation or post-expert norm they cannot fuse is passed as a callable.
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


# Cache successes but not failures, since the environment can change between attempts.
# A module global, because Dynamo warns about tracing `functools.cache` on every compile.
_FINEGRAINED: FineGrained | None = None


@torch._dynamo.allow_in_graph
def _load_finegrained_kernel() -> None:
    """Load the kernel once into `_FINEGRAINED`.

    `@allow_in_graph` keeps the hub download out of the `torch.compile` trace.
    The graph node's output must be proxyable, so this returns `None` and not the kernel.
    """
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
    """Register `name` as a parameter over `tensor`, or as `None` when `tensor` is `None`.

    Use it for slots a checkpoint may or may not fill, such as a bias or an NVFP4 global.
    """
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
    """Return `(weight, weight_scale_inv)` for one expert projection.

    The scale grid follows the format's granularity. A format without one gets one scale per expert.
    `min_scale_out` sets a floor on the grid rows, so a gate|up stack keeps room for both halves.
    `swizzled` lays the grid out as `SWIZZLE_32_4_4` with the expert axis first, so EP shards it on dim 0.
    A grid that `_tiles_swizzled` rejects stays row-major.
    """
    weight = torch.empty(num_experts, proj_out, proj_in // format_spec.values_per_byte, dtype=format_spec.weight_dtype)
    if scale_group is None:
        scale_shape = (num_experts, max(1, min_scale_out), 1)
    else:
        rows, cols = max(cdiv(proj_out, scale_group[0]), min_scale_out), cdiv(proj_in, scale_group[1])
        scale_shape = (num_experts, rows, cols)
        if swizzled and _tiles_swizzled(rows, cols):
            scale_shape = (num_experts, rows // 128, cols // 4, 2, 256)
    scale = torch.empty(scale_shape, dtype=scale_dtype)
    return (
        nn.Parameter(weight, requires_grad=weight.is_floating_point()),
        nn.Parameter(scale, requires_grad=scale.is_floating_point() and not swizzled),
    )


def _tiles_swizzled(rows: int, cols: int) -> bool:
    """Check whether the kernels can read a `(rows, cols)` scale grid as `SWIZZLE_32_4_4`.

    They read whole 128-row blocks and whole 4-column bands. They never run a single K step, so a grid needs two bands.
    """
    return rows % 128 == 0 and cols % 4 == 0 and cols >= 8


def _holds_swizzled_scales(format_spec: WeightFormat, activation_format: str | None) -> bool:
    """Check whether triton-layout experts hold their group scales as `SWIZZLE_32_4_4`.

    SM100's scaled MMA reads that layout directly. Row-major scales need a gather per tile.
    Only group-scaled formats with quantized activations use it.
    """
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
    """Run the FP8/FP4 linear of `FineGrainedLinear` and of the eager `FineGrainedExperts` loop.

    DeepGEMM runs the call when `prefers_deepgemm_linear` accepts it. Triton runs it otherwise.

    Args:
        input (`torch.Tensor`):
            The `(..., K)` bf16 or fp16 activations.
        weight (`torch.Tensor`):
            A `(N, K)` `float8_e4m3fn` weight, or a `(N, K // 2)` `int8` weight with two FP4 values per byte.
        weight_scale_inv (`torch.Tensor`):
            The weight's block scales, in `float32` or UE8M0.
        block_size (`list[int]`, *optional*):
            The `[block_n, block_k]` of block-FP8. `None` means a per-tensor scale. FP4 formats ignore it.
        bias (`torch.Tensor`, *optional*):
            Added to the output.
        activation_scale (`torch.Tensor`, *optional*):
            A calibrated per-tensor activation scale. `None` quantizes per token.
        allow_deepgemm (`bool`, *optional*, defaults to `True`):
            `False` forces Triton. See `disable_deepgemm_on_multi_device`.
        weight_global_scale (`torch.Tensor`, *optional*):
            NVFP4's per-matrix weight global.
        input_global_scale (`torch.Tensor`, *optional*):
            NVFP4's calibrated activation global. It is the checkpoint's `input_scale`.
        activation_format (`str`, *optional*):
            The activation format when the weights leave it open. `"bf16"` means weight-only.
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
    """Map each post-expert norm class to the name of its fused kernel form.

    Match on the exact class, since a subclass may compute something else.
    Build the map on first use, so importing this module does not import a model.
    """
    from ...models.muse_spark.modeling_muse_spark import MuseSparkTextInputScaledRMSNorm

    return {
        nn.RMSNorm: "rms_norm",
        # Scales by `(1 + weight)` before the RMS norm and rounds to bf16 between the two steps
        MuseSparkTextInputScaledRMSNorm: "input_scaled_rms_norm",
    }


class _FineGrainedModule:
    """State shared by every finegrained module, dense linear or experts."""

    # the kernels take raw pointers, so the TP styles pass them local shards and not DTensors
    tp_uses_local_tensors = True
    # set by `disable_deepgemm_on_multi_device`
    _deepgemm_disabled = False

    quantization_config: BlockScaledConfig
    # dtype of NVFP4's per-matrix weight global, `None` for single-level formats
    global_scale_dtype: torch.dtype | None = None

    @property
    def two_level(self) -> bool:
        """Whether the format has NVFP4's two scale levels.

        The weight has a per-matrix global.
        The calibrated activation scale is read as the activation global, not as a static scale.
        """
        return self.global_scale_dtype is not None

    @property
    def holds_activation_scale(self) -> bool:
        """Whether the module holds the checkpoint's calibrated `input_scale`.

        It does for a static FP8 scale. It also does for NVFP4's activation global, unless activations stay in bf16.
        """
        return self.activation_scheme == "static" or (self.two_level and self.activation_format != "bf16")

    def _activation_operands(self, activation: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return `(activation_scale, input_global_scale)` with the calibrated scale in the kernels' slot."""
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
        quantization_config: BlockScaledConfig,
        has_bias: bool = False,
    ):
        super().__init__(in_features, out_features)

        self.quantization_config = quantization_config
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
    """FP8 embedding table with one per-tensor `weight_scale`.

    A lookup rescales only the rows it gathers. Use it for tables too big to dequantize, such as the ~95 GiB
    `ple.ple_embedding.ngram_embedding` of Qwen4Exp.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        quantization_config: BlockScaledConfig,
    ):
        if quantization_config.method.value != "fp8" or quantization_config.weight_block_size is not None:
            raise ValueError("Fine-grained embeddings hold per-tensor FP8 tables.")
        weight = torch.empty(num_embeddings, embedding_dim, dtype=FP8_DTYPE)
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx, _weight=weight, _freeze=True)
        self.quantization_config = quantization_config
        # `(1,)` and not a scalar, since checkpoints serialize per-tensor scales with that shape
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
    """Quantized block-diagonal grouped linear.

    It stacks `n_groups` sub-matrices in one `(n_groups * out_per_group, in_per_group)` weight.
    Each group of a token's input runs through its own sub-matrix.
    It maps `(..., n_groups, in_per_group)` to `(..., n_groups, out_per_group)`, like the grouped linear it replaces.
    """

    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        *,
        quantization_config: BlockScaledConfig,
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

        # the stored K, halved for packed FP4
        w = self.weight.view(self.n_groups, -1, self.weight.shape[-1])
        scale_inv = self.weight_scale_inv
        x = x.movedim(-2, 0).reshape(-1, hidden_dim)
        if scale_inv.numel() == 1:
            # the grouped kernel takes one scale per group
            scale_inv = scale_inv.reshape(1, 1, 1).expand(self.n_groups, 1, 1).contiguous()
        else:
            scale_inv = scale_inv.view(self.n_groups, scale_inv.size(0) // self.n_groups, scale_inv.size(1))

        tokens_per_group = x.size(0) // self.n_groups
        # the kernels pad the group count to a power of two, with the empty groups last
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
    """Run the batched experts forward used for decode. One program handles each routed row."""
    kernel = load_finegrained_kernel()
    return kernel.moe_fused_batched(hidden_states, top_k_index, top_k_weights, **self._moe_operands(kernel))


def finegrained_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """Run the grouped experts forward used for prefill. It runs over the expert-sorted schedule."""
    kernel = load_finegrained_kernel()
    return kernel.moe_fused_grouped(hidden_states, top_k_index, top_k_weights, **self._moe_operands(kernel))


class ProjectionOperands(NamedTuple):
    """One expert projection's tensors as the kernels take them. A slot the checkpoint does not fill is `None`."""

    weight: torch.Tensor
    scale_inv: torch.Tensor
    weight_global_scale: torch.Tensor | None
    input_global_scale: torch.Tensor | None
    activation_scale: torch.Tensor | None
    bias: torch.Tensor | None


class FineGrainedExpertsInterface(ExpertsInterface):
    """Interface for registering custom finegrained experts forward functions."""

    _global_mapping = {
        "deepgemm": deepgemm_fp8_fp4_experts_forward,
        "batched_mm": finegrained_batched_mm_experts_forward,
        "grouped_mm": finegrained_grouped_mm_experts_forward,
        "deepgemm_megamoe": deepgemm_fp8_fp4_megamoe_experts_forward,
    }


ALL_FINEGRAINED_EXPERTS_FUNCTIONS = FineGrainedExpertsInterface()


@use_experts_implementation(experts_interface=ALL_FINEGRAINED_EXPERTS_FUNCTIONS)
class FineGrainedExperts(_FineGrainedModule, nn.Module):
    # `_experts_implementation` -> {default TP/EP plan style: impl-specific style}, applied by the quantizer.
    _impl_tp_layer_overrides: dict[str, dict[str, str]] = {
        "deepgemm_megamoe": {
            "ep_dispatch_experts": "megamoe_experts",
            "moe_tp_experts": "megamoe_experts",
            "ep_router": "megamoe_router",
        },
    }

    def __init__(
        self,
        config,
        *,
        quantization_config: BlockScaledConfig,
        has_gate: bool = True,
        has_bias: bool = False,
        is_concatenated: bool = True,
        post_expert_norm: nn.Module | None = None,
        has_post_expert_norm: bool = False,
        source_is_transposed: bool = False,
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

        self.quantization_config = quantization_config
        activation_format = self.activation_format
        format_spec, scale_dtype, scale_group = resolve_weight_format(
            self.weight_format, self.quantization_config.scale_fmt, self.block_size
        )
        self.global_scale_dtype = format_spec.global_scale_dtype

        self.has_gate = has_gate
        self.has_bias = has_bias
        self.is_concatenated = is_concatenated
        # The Triton kernels' fused GLU reads gate|up interleaved row by row.
        # A scale spanning several rows would then mix gate and up, which DeepGEMM cannot read.
        # So off SM100, where DeepGEMM serves block FP8, block-scaled gate|up stay stacked.
        scale_spans_rows = self.block_size is not None and self.block_size[0] > 1
        self.holds_interleaved_gate_up = self.has_gate and (is_sm100() or not scale_spans_rows)
        swizzled = _holds_swizzled_scales(format_spec, activation_format)
        self.gate_up_name = "gate_up_proj" if self.has_gate else "up_proj"
        up_rows = (2 if self.has_gate else 1) * self.intermediate_dim
        storage = (format_spec, scale_dtype, scale_group, swizzled)
        self._register_projection(self.gate_up_name, up_rows, self.hidden_dim, 2 if self.has_gate else 1, storage)
        self._register_projection("down_proj", self.hidden_dim, self.intermediate_dim, 1, storage)
        if getattr(self, f"{self.gate_up_name}_scale_inv").ndim != self.down_proj_scale_inv.ndim:
            self.hold_row_major_scales()

        # the per-expert output norm of the replaced experts
        # a model may hold a norm it never applies, whose weights the checkpoint still ships
        self.post_expert_norm = post_expert_norm
        self.has_post_expert_norm = has_post_expert_norm
        # `self.is_transposed` stays False because this module always holds `(E, rows, K)`
        # a full-precision stack quantized on load is transposed first if the replaced module held it transposed
        self.source_is_transposed = source_is_transposed

        # the non-tensor kernel arguments, resolved on the first MoE forward
        self._static_moe_operands = None

    @classmethod
    def from_module(
        cls, module: nn.Module, quantization_config: BlockScaledConfig, model: nn.Module
    ) -> FineGrainedExperts:
        """Build a `FineGrainedExperts` for `module`. Keep the model's gate|up convention and its output norm."""
        config = getattr(module, "config", None) or model.config.get_text_config()
        return cls(
            config=config,
            quantization_config=quantization_config,
            has_gate=getattr(module, "has_gate", True),
            has_bias=getattr(module, "has_bias", False),
            is_concatenated=getattr(module, "is_concatenated", True),
            post_expert_norm=getattr(module, "post_expert_norm", None),
            has_post_expert_norm=getattr(module, "has_post_expert_norm", False),
            source_is_transposed=getattr(module, "is_transposed", False),
        )

    def hold_row_major_scales(self) -> None:
        """Hold both scale grids row-major.

        The fused kernels read both grids in one layout. So when one grid cannot be swizzled, neither is.
        """
        for proj in (self.gate_up_name, "down_proj"):
            scale = getattr(self, f"{proj}_scale_inv")
            if scale.ndim == 5:
                experts, row_blocks, col_bands = scale.shape[:3]
                grid = scale.new_empty(experts, row_blocks * 128, col_bands * 4)
                self.register_parameter(
                    f"{proj}_scale_inv", nn.Parameter(grid, requires_grad=grid.is_floating_point())
                )

    def _register_projection(self, proj, rows, in_dim, min_scale_out, storage):
        """Register one expert projection's weight and scale grid.

        Also register the optional slots a checkpoint may fill.
        Those are a bias, NVFP4's weight global and a calibrated activation scale.
        """
        format_spec, scale_dtype, scale_group, swizzled = storage
        weight, scale = _alloc_expert_proj(
            self.num_experts, rows, in_dim, format_spec, scale_dtype, scale_group, min_scale_out, swizzled
        )
        self.register_parameter(proj, weight)
        self.register_parameter(f"{proj}_scale_inv", scale)

        # in the model dtype, like the replaced experts
        bias = torch.empty(self.num_experts, rows) if self.has_bias else None
        # one per expert, since `FineGrainedWeightGlobals` merges the separately calibrated gate and up halves
        weight_global = torch.ones(self.num_experts, dtype=self.global_scale_dtype) if self.two_level else None
        activation = torch.ones(self.num_experts, dtype=torch.float32) if self.holds_activation_scale else None
        _set_optional_parameter(self, f"{proj}_bias", bias)
        _set_optional_parameter(self, f"{proj}_weight_global_scale", weight_global)
        _set_optional_parameter(self, f"{proj}_activation_scale", activation)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # interleaved rows give alternating output columns, like the fused epilogue, and stacked rows give two halves
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
        """Return `proj`'s tensors as the kernels take them.

        The calibrated activation scale goes in as NVFP4's activation global, or else as a static scale.
        """
        activation = self._parameters[f"{proj}_activation_scale"]
        activation_scale, input_global_scale = self._activation_operands(activation)
        return ProjectionOperands(
            weight=self._parameters[proj],
            scale_inv=self._parameters[f"{proj}_scale_inv"],
            weight_global_scale=self._parameters[f"{proj}_weight_global_scale"],
            input_global_scale=input_global_scale,
            activation_scale=activation_scale,
            bias=self._parameters[f"{proj}_bias"],
        )

    def expert_linear(self, input: torch.Tensor, proj: str, expert_idx: int) -> torch.Tensor:
        """Run one expert's `proj` as a dense linear over that expert's slice of the operands."""
        weight, scale, weight_global, input_global, activation_scale, bias = self._projection_operands(proj)
        weight = weight[expert_idx]
        # `matmul_2d` reads a swizzled expert slice with a leading dim of 1
        scale = scale[expert_idx : expert_idx + 1] if scale.ndim == 5 else scale[expert_idx]
        weight_global, activation_scale, bias = (
            operand[expert_idx] if operand is not None else None for operand in (weight_global, activation_scale, bias)
        )
        # either one per expert or one shared by all experts
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
        # Separately calibrated gate and up grids interleave per block, not per weight row.
        # Run each half with its own blocks, then interleave the output columns.
        # A single scale row is shared by both halves and can be read directly.
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
        """Return the arguments of the kernels' MoE forwards. Only the tensors are gathered on each call."""
        static = self._static_moe_operands
        if static is None:
            static = self._static_moe_operands = self._resolve_static_moe_operands(kernel)
        operands = dict(static)
        # The kernels name the up projection `gate_up_proj` whether or not it is gated
        for name, projection in (
            ("gate_up_proj", self._projection_operands(self.gate_up_name)),
            ("down_proj", self._projection_operands("down_proj")),
        ):
            operands[name] = projection.weight
            operands[f"{name}_scale_inv"] = projection.scale_inv
            operands[f"{name}_weight_global_scale"] = projection.weight_global_scale
            operands[f"{name}_input_global_scale"] = projection.input_global_scale
            operands[f"{name}_activation_scale"] = projection.activation_scale
            operands[f"{name}_bias"] = projection.bias
        if isinstance(static["post_expert_norm"], str):
            weight = operands["post_expert_norm_weight"] = self.post_expert_norm.weight
            if static["post_expert_norm_eps"] is None:
                # `nn.RMSNorm` leaves `eps` as `None` and uses the dtype's epsilon
                operands["post_expert_norm_eps"] = torch.finfo(weight.dtype).eps
        return operands

    def _resolve_static_moe_operands(self, kernel) -> dict:
        """Return the non-tensor half of `_moe_operands`.

        The kernels fuse a supported activation or post-expert norm, which is passed by name.
        Any other is passed as the module's own callable and runs between the kernels.
        """
        act_name = self.act_fn_name
        # the kernels gate over interleaved gate|up only; stacked gate|up run ungated and the module gates
        gate = self.holds_interleaved_gate_up
        fusable = act_name in kernel.get_supported_act_fns() and (self.swiglu_alpha is None or act_name == "silu")
        if fusable and (gate or not self.has_gate):
            act_fn = act_name
        else:
            act_fn = self._apply_gate if self.has_gate else self.act_fn

        post_expert_norm, norm_eps = None, 1e-6
        if self.has_post_expert_norm:
            norm_name = supported_fused_norms().get(type(self.post_expert_norm))
            if norm_name in kernel.get_supported_norms():
                post_expert_norm = norm_name
                eps = getattr(self.post_expert_norm, "eps", getattr(self.post_expert_norm, "variance_epsilon", None))
                norm_eps = float(eps) if eps is not None else None
            else:
                post_expert_norm = self.post_expert_norm
        return {
            "act_fn": act_fn,
            "gate": gate,
            "swiglu_alpha": self.swiglu_alpha,
            "swiglu_limit": self.swiglu_limit,
            "activation_format": self.activation_format,
            "post_expert_norm": post_expert_norm,
            "post_expert_norm_weight": None,
            "post_expert_norm_eps": norm_eps,
        }


def assert_modules_are_quantized(model: nn.Module) -> None:
    """Raise if a finegrained module still holds a full-precision weight after loading.

    It means the checkpoint does not quantize that module. Serving it would return a model that is not quantized.
    """
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


def disable_deepgemm_on_multi_device(model: nn.Module) -> None:
    """Route the quantized modules through Triton when the model spans several CUDA devices in one process.

    DeepGEMM binds each cached kernel to the CUDA context active at load time. It corrupts results on another device.
    """
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
