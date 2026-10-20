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
"""Quantized modules and forwards for the block-scaled formats — block-FP8, MXFP8, MXFP4, NVFP4
and weight-only — served by `kernels-community/finegrained-kernels`. The checkpoint's layout is
moved onto these modules by the ops in `conversions`.
"""

from __future__ import annotations

import functools
import os
import sys
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


logger = logging.get_logger(__name__)


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


@functools.cache
def _get_ue8m0_dtype() -> torch.dtype:
    """Use native UE8M0 when available, otherwise store its biased exponent bytes."""
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


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
    """Entry points exposed by the `kernels-community/finegrained-kernels` Triton kernel.

    Every format (block-FP8, MXFP8, MXFP4, NVFP4, weight-only) flows through the matmuls and
    the two MoE forwards, with the weight format resolved off the weight/scale dtypes inside the
    kernel and the module's ``activation_format`` passed through unchanged — the kernels speak the
    same vocabulary (``None`` = the weights' format, ``"bf16"`` = weight-only). The experts
    forwards hand the kernels' MoE forwards the module's tensors and let them run the whole chain:
    scheduling, fused GLU + intermediate requant, biases, the routing-weighted reduce. What they
    cannot fuse runs in between — an activation outside ``get_supported_act_fns()`` as the module's
    own callable, a per-expert output norm by name where ``get_supported_norms()`` has it. The
    quantizers serve on-the-fly quantization into the group formats. All symbols are required; a
    build missing any raises at load with the full list.
    """

    matmul_2d: Callable
    matmul_batched: Callable
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
    rms_norm_rows: Callable
    embedding_lookup: Callable


# Cache the loaded kernel but not failures: re-checking each call is cheap and intended, since the env
# can change between attempts. A module global (not `@functools.cache`) avoids Dynamo warning about
# tracing a cache-wrapped function on every compile.
_FINEGRAINED: FineGrained | None = None


@torch._dynamo.allow_in_graph
def _load_finegrained_kernel() -> None:
    """
    Load the finegrained-kernels Triton kernel once into the `_FINEGRAINED` module global.

    `@allow_in_graph` makes `torch.compile` treat the untraceable hub download + dynamic import as a
    single opaque node instead of tracing into it; it returns `None` (proxyable) and populates the
    global, which `load_finegrained_kernel` then returns.

    Under NO circumstances may this function return a value: an `@allow_in_graph` fx node's
    return must be proxyable, and returning the bundle (e.g. from the warm-cache
    short-circuit) breaks torch.compile with `Unsupported: torch.* op returned non-Tensor`.

    Raises `ImportError` if the `kernels` package is missing, or the kernel or required
    symbols cannot be found.
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

    # the bundle's fields ARE the required symbols, under the kernel's own names
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


def _cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


@dataclass(frozen=True)
class WeightFormat:
    """Storage layout of one quantized weight format — the single source both module classes
    derive their parameter shapes from. ``scale_dtype`` ``None`` defers to the config's
    ``scale_fmt`` (block-FP8 ships fp32 or UE8M0 containers; the group formats pin theirs).
    ``scale_group`` ``None`` means the block comes from the quant config's ``weight_block_size``.
    ``global_scale_dtype`` ``None`` means the format has no second level; NVFP4 names the dtype its
    per-matrix global is held in."""

    weight_dtype: torch.dtype
    values_per_byte: int = 1
    scale_dtype: torch.dtype | None = None
    scale_group: tuple[int, int] | None = None
    global_scale_dtype: torch.dtype | None = None


def get_weight_format(weight_format: str) -> WeightFormat:
    """Resolve only the requested format, so FP8 and NVFP4 do not require UE8M0 support."""
    if weight_format == "fp8":
        return WeightFormat(weight_dtype=_FP8_DTYPE)
    if weight_format in ("mxfp8", "mxfp4"):
        packed = weight_format == "mxfp4"
        return WeightFormat(
            weight_dtype=torch.int8 if packed else _FP8_DTYPE,
            values_per_byte=2 if packed else 1,
            scale_dtype=_get_ue8m0_dtype(),
            scale_group=(1, 32),
        )
    if weight_format == "nvfp4":
        return WeightFormat(
            weight_dtype=torch.int8,
            values_per_byte=2,
            scale_dtype=_FP8_DTYPE,
            scale_group=(1, 16),
            global_scale_dtype=torch.float32,
        )
    raise ValueError(f"unknown weight_format {weight_format!r}; expected fp8, mxfp8, mxfp4 or nvfp4")


def resolve_weight_format(
    weight_format: str,
    scale_fmt: str = "float",
    block_size: tuple[int, int] | None = None,
) -> tuple[WeightFormat, torch.dtype, tuple[int, int] | None]:
    """``(format, scale_dtype, (sf_gran_n, sf_gran_k))`` for one format name, with the config's
    ``scale_fmt``/``weight_block_size`` filling the slots the format leaves open."""
    format_spec = get_weight_format(weight_format)
    scale_dtype = format_spec.scale_dtype
    if scale_dtype is None:
        scale_dtype = _get_ue8m0_dtype() if scale_fmt == "ue8m0" else torch.float32
    return format_spec, scale_dtype, format_spec.scale_group if format_spec.scale_group is not None else block_size


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
    """``(weight, weight_scale_inv)`` Parameters for one expert projection: the weight in the
    format's storage (FP4 packs two values per byte along K), the scale grid at the format's
    granularity (``None`` = one scale per expert). ``min_scale_out`` floors the grid's output dim
    so a fused gate_up keeps room for both halves. ``swizzled`` allocates the grid in the
    ``SWIZZLE_32_4_4`` layout instead — expert axis leading, so EP shards it on dim 0 — which needs
    whole blocks; a projection that does not tile stays affine."""
    weight = torch.empty(num_experts, proj_out, proj_in // format_spec.values_per_byte, dtype=format_spec.weight_dtype)
    if scale_group is None:
        scale_shape = (num_experts, max(1, min_scale_out), 1)
    else:
        rows, cols = max(_cdiv(proj_out, scale_group[0]), min_scale_out), _cdiv(proj_in, scale_group[1])
        scale_shape = (num_experts, rows, cols)
        if swizzled and rows % 128 == 0 and cols % 4 == 0:
            scale_shape = (num_experts, rows // 128, cols // 4, 2, 256)
    scale = torch.empty(scale_shape, dtype=scale_dtype)
    return (
        nn.Parameter(weight, requires_grad=weight.is_floating_point()),
        nn.Parameter(scale, requires_grad=scale.is_floating_point() and not swizzled),
    )


# The layout each experts backend reads, which a module's storage is fixed to when it loads. `True`: the triton
# chains, gate|up interleaved (the pairing their fused GLU epilogue reads) and group scales swizzled where
# `_holds_swizzled_scales` allows. `False`: the DeepGEMM arms, gate|up stacked and block scales affine (Mega MoE
# does its own interleave, and the grouped GEMM reads a scale grid row-major, whose blocks an interleaved stack
# splits between the halves). The eager loop, and a backend not listed, reads either.
_EXPERTS_TRITON_LAYOUT = {"batched_mm": True, "grouped_mm": True, "deepgemm": False, "deepgemm_megamoe": False}


def _holds_swizzled_scales(format_spec: WeightFormat, activation_format: str | None) -> bool:
    """Whether a triton-layout experts module holds its block scales in the ``SWIZZLE_32_4_4`` layout the
    Blackwell tcgen05 scaled-MMA reads directly (row-major forces a per-tile gather). Needs SM100, a
    group-scaled format and a chain that quantizes activations. Block-FP8 reaches the scaled-MMA under
    UE8M0 by broadcasting one 128-block scalar in-register, so it has no per-group grid to lay out."""
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
    """End-to-end FP8/FP4 linear used by `FineGrainedLinear` and the eager `FineGrainedExperts` loop.

    Dispatch order:
      1. DeepGEMM (`deepgemm_fp8_fp4_linear`) where `prefers_deepgemm_linear` admits the call: the
         pre-SM100 128x128 block-FP8 shape it is faster on, dynamic and inference-only. Never on SM100,
         where Triton is tuned per shape and is the only path reading pre-swizzled scales.
      2. Triton finegrained everywhere else: SM100, FP4 weights, an ``activation_scale`` (DeepGEMM is
         dynamic-only), a call that needs a gradient (DeepGEMM has no backward).

    Args:
        input: (..., K) bf16/fp16 activations.
        weight: (N, K) `float8_e4m3fn` or (N, K // 2) `int8` (FP4-packed).
        weight_scale_inv: per-block weight scales — `float32` (as DeepSeek-V3 ships them) or
            `float8_e8m0fnu` (DeepSeek-V4; reinterpreted as int32 at the DeepGEMM kernel boundary).
        block_size: [block_n, block_k] for FP8 block-wise quant, or None/[N, K] for per-tensor.
            Ignored for FP4 weights (the kernel infers SF granularity from the dtype).
        bias: optional bias added to the matmul output.
        activation_scale: pass a per-tensor scalar to use static activation quant; leave `None`
            for dynamic (per-token) quant.
        allow_deepgemm: set ``False`` to force the Triton fallback for this call. Used when the
            model spans multiple CUDA devices in one process — DeepGEMM's cached kernels are bound
            to a single CUDA context and produce garbage across devices (see
            ``disable_deepgemm_on_multi_device``).
        weight_global_scale: the NVFP4 two-level weight global, one value for this matrix
            (None for other formats).
        input_global_scale: the NVFP4 calibrated activation global, the checkpoint's
            ``input_scale`` (None = quantize against the block scales alone).
        activation_format: the activation format where the weights leave it open
            (``"bf16"`` = weight-only).
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
    else. Built on first use, as the models import `modeling_utils`, which imports this module."""
    from ...models.muse_spark.modeling_muse_spark import MuseSparkTextInputScaledRMSNorm

    return {nn.RMSNorm: "rms_norm", MuseSparkTextInputScaledRMSNorm: "input_scaled_rms_norm"}


class _FineGrainedModule:
    """What every finegrained module (dense linear or experts) carries besides its parameters."""

    # the kernels take raw pointers, so the TP layer must hand local shards and not DTensors, whatever
    # `should_use_local_tensors` would decide from grad mode alone
    _hf_quantized_needs_local_tp = True

    # Internal, temporary flag — not public API, don't set it directly. `disable_deepgemm_on_multi_device`
    # flips it True at load when the model spans >1 CUDA device in one process (DeepGEMM's context-bound
    # kernels corrupt across devices); removable once the kernel ships a context-free loader.
    _deepgemm_disabled = False

    quantization_config: BlockScaledConfig
    # the dtype NVFP4's per-matrix weight global is held in, `None` for a single-level format
    global_scale_dtype: torch.dtype | None = None

    @property
    def two_level(self) -> bool:
        """NVFP4's two levels: a per-matrix weight global over the block scales, and the calibrated
        activation scale read as the activation global rather than as a static scale."""
        return self.global_scale_dtype is not None

    @property
    def holds_activation_scale(self) -> bool:
        """Whether the module holds the checkpoint's calibrated activation scale (its ``input_scale``): a
        static FP8 scale, or NVFP4's activation global unless the activations stay unquantized."""
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

        # the format table decides storage: value dtype (packed E2M1 = 2 values per int8 byte
        # along K), scale dtype/granularity, and whether a two-level global exists
        format_spec, sf_dtype, scale_group = resolve_weight_format(
            self.weight_format, self.quantization_config.scale_fmt, self.block_size
        )
        self.global_scale_dtype = format_spec.global_scale_dtype
        in_storage = in_features // format_spec.values_per_byte
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_storage, dtype=format_spec.weight_dtype),
            requires_grad=format_spec.weight_dtype.is_floating_point,
        )
        # NVFP4 two-level: the per-tensor fp32 global the kernel recovers on the accumulator
        _set_optional_parameter(
            self,
            "weight_global_scale",
            torch.tensor(1.0, dtype=self.global_scale_dtype) if self.two_level else None,
        )

        # no group and no block: one per-tensor scale
        scale = (
            torch.tensor(1.0, dtype=torch.float32)
            if scale_group is None
            else torch.empty(_cdiv(out_features, scale_group[0]), _cdiv(in_features, scale_group[1]), dtype=sf_dtype)
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
    """FP8 or NVFP4 table decoded only for the rows selected by a lookup."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        quantization_config: BlockScaledConfig | None = None,
        dtype: torch.dtype | None = None,
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
        if self.weight_format not in ("fp8", "nvfp4") or (
            self.weight_format == "fp8"
            and self.quantization_config.weight_block_size is not None
            and not self.quantization_config.modules_to_convert
        ):
            raise ValueError("Fine-grained embeddings support per-tensor FP8 or NVFP4 weights.")
        self.compute_dtype = dtype or torch.get_default_dtype()
        if self.weight_format == "nvfp4":
            self._hf_quantized_embedding_group_size = 16
            if embedding_dim % 16:
                raise ValueError("NVFP4 embedding width must be divisible by its scale group size (16).")
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim // 2, dtype=torch.int8), requires_grad=False
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim // 16, dtype=_FP8_DTYPE), requires_grad=False
            )
            self.weight_global_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim, dtype=_FP8_DTYPE), requires_grad=False
            )
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
            dtype=module.weight.dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.device.type in ("cuda", "xpu") and not (
            self.weight_format == "fp8" and torch.compiler.is_compiling()
        ):
            scale = self.weight_scale if self.weight_format == "fp8" else self.weight_scale_inv
            global_scale = None if self.weight_format == "fp8" else self.weight_global_scale
            if input.device == self.weight.device == scale.device and (
                global_scale is None or global_scale.device == self.weight.device
            ):
                dtype = scale.dtype if global_scale is None else self.compute_dtype
                return load_finegrained_kernel().embedding_lookup(self.weight, input, scale, global_scale, dtype)
        if self.weight_format == "fp8":
            rows = super().forward(input)
            return rows.to(self.weight_scale.dtype) * self.weight_scale
        packed = nn.functional.embedding(input, self.weight).view(torch.uint8)
        scales = nn.functional.embedding(input.to(self.weight_scale_inv.device), self.weight_scale_inv).to(
            device=packed.device, dtype=torch.float32
        )
        low, high = packed & 15, packed >> 4
        codes = torch.stack((low, high), dim=-1).flatten(-2).long()
        values = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            device=packed.device,
        )
        rows = values[codes] * scales.repeat_interleave(16, dim=-1) * self.weight_global_scale.to(packed.device)
        return rows.to(self.compute_dtype)


class FineGrainedGroupedLinear(FineGrainedLinear):
    """Quantized drop-in for block-diagonal grouped linears. NOT the MoE experts path
    (`FineGrainedExperts`): every group runs on every token, with no routing.

    The underlying nn.Linear stores a single `(n_groups * out_per_group, in_per_group)` weight;
    logically that is `n_groups` independent `(out_per_group, in_per_group)` sub-matrices, each
    consuming a disjoint slice of the input's last-but-one dim. Forward takes `(..., n_groups,
    in_per_group)` and returns `(..., n_groups, out_per_group)` — the same contract as the bf16
    grouped linear it replaces."""

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
            # A per-tensor checkpoint calibrates the whole stored matrix, so every
            # group uses the same scale. The grouped kernel expects an expert axis.
            scale_inv = scale_inv.reshape(1, 1, 1).expand(self.n_groups, 1, 1).contiguous()
        else:
            scale_inv = scale_inv.view(self.n_groups, scale_inv.size(0) // self.n_groups, scale_inv.size(1))

        tokens_per_group = x.size(0) // self.n_groups
        # Kernels pad the expert schedule to a power of two. Empty trailing groups
        # point at the final row, including when n_groups itself is not a power of two.
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
    """Grouped (prefill) experts forward: one on-device routing pass, then the chain over the
    expert-sorted schedule, scattering back to routed rows (EP-sentinel rows skipped)."""
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


class FineGrainedExperts(_FineGrainedModule, nn.Module):
    # the plan kinds an `_experts_implementation` needs in place of the impl-agnostic default
    # (megamoe wants no gradient-sync hooks and an EP `process_group`); extend when adding an impl
    _impl_tp_layer_overrides: dict[str, dict[str, str]] = {
        "deepgemm_megamoe": {
            "moe_tp_experts": "megamoe_experts",
            "ep_dispatch_experts": "megamoe_experts",
            "ep_router": "megamoe_router",
        },
    }

    def __init__(
        self,
        config,
        *,
        quantization_config: BlockScaledConfig | None = None,
        has_gate: bool = True,
        has_bias: bool = False,
        is_concatenated: bool = True,
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

        # Checkpoint and module layouts may differ: conversion ops normalize stacked or interleaved
        # gate/up rows and swizzle scales at load. The triton and eager paths use interleaved rows;
        # DeepGEMM paths use stacked rows.
        self.has_gate = has_gate
        self.has_bias = has_bias
        self.is_concatenated = is_concatenated
        triton_layout = _EXPERTS_TRITON_LAYOUT.get(getattr(config, "_experts_implementation", None), True)
        self.holds_interleaved_gate_up = self.has_gate and triton_layout
        swizzled = triton_layout and _holds_swizzled_scales(format_spec, activation_format)
        self.gate_up_name = "gate_up_proj" if self.has_gate else "up_proj"
        up_rows = (2 if self.has_gate else 1) * self.intermediate_dim
        storage = (format_spec, scale_dtype, scale_group, swizzled)
        self._register_projection(self.gate_up_name, up_rows, self.hidden_dim, 2 if self.has_gate else 1, storage)
        self._register_projection("down_proj", self.hidden_dim, self.intermediate_dim, 1, storage)

        self.post_expert_norm = None
        self.has_post_expert_norm = False
        self.ships_transposed = False

        # optimization knob, set by `register_shared_gate_up_input_globals`: one gate_up activation global
        self.register_buffer("shared_gate_up_input_global", None, persistent=False)
        # what `_moe_operands` hands the kernels besides tensors, resolved on the first MoE forward
        self._static_moe_operands = None

    @classmethod
    def from_module(
        cls, module: nn.Module, quantization_config: BlockScaledConfig, model: nn.Module
    ) -> FineGrainedExperts:
        """A `FineGrainedExperts` holding `module`'s weights, in the model's own gate|up convention.

        The flags travel twice over: `use_experts_implementation` stamps them on the instance after
        `__init__`, so the class it builds needs them as well as the constructor. The model's per-expert
        output norm is the model's own module, so it is taken from the experts this replaces.
        """
        flags = {
            "has_gate": getattr(module, "has_gate", True),
            "has_bias": getattr(module, "has_bias", False),
            "is_concatenated": getattr(module, "is_concatenated", True),
        }
        experts_class = _experts_class_with(cls, **flags)
        config = getattr(module, "config", None) or model.config.get_text_config()
        experts = experts_class(config=config, quantization_config=quantization_config, **flags)
        experts.post_expert_norm = getattr(module, "post_expert_norm", None)
        experts.has_post_expert_norm = getattr(module, "has_post_expert_norm", False)
        # a full-precision stack quantized on load arrives in the layout of the module this replaces
        experts.ships_transposed = getattr(module, "is_transposed", False)
        return experts

    def check_experts_implementation(self, implementation: str | None) -> None:
        """Refuse a backend that reads another layout than the one this module was loaded in, which
        `set_experts_implementation` asks before switching: the GLU would pair the wrong rows, or swizzled
        scales would be read as affine."""
        triton_layout = _EXPERTS_TRITON_LAYOUT.get(implementation)
        if triton_layout is None:
            return
        swizzled = any(getattr(self, f"{proj}_scale_inv").ndim == 5 for proj in (self.gate_up_name, "down_proj"))
        if (self.has_gate and self.holds_interleaved_gate_up != triton_layout) or (swizzled and not triton_layout):
            raise NotImplementedError(
                f"`experts_implementation={implementation!r}` reads "
                f"{'gate|up interleaved' if triton_layout else 'gate|up stacked and block scales row-major'}, "
                "but this module was loaded in another layout. Switching between the triton and DeepGEMM experts "
                "backends after load is not supported — pass `experts_implementation=...` to `from_pretrained` so "
                "the model loads for it."
            )

    def _register_projection(self, proj, rows, in_dim, min_scale_out, storage):
        """One expert projection: the packed weight and its scale grid, then the slots a
        checkpoint may or may not fill — a bias, the NVFP4 second-level globals, a calibrated
        activation scale. `storage` is the layout `resolve_weight_format` settled on."""
        format_spec, scale_dtype, scale_group, swizzled = storage
        weight, scale = _alloc_expert_proj(
            self.num_experts, rows, in_dim, format_spec, scale_dtype, scale_group, min_scale_out, swizzled
        )
        self.register_parameter(proj, weight)
        self.register_parameter(f"{proj}_scale_inv", scale)

        # the model dtype (the default dtype under `from_pretrained`), like the bf16 experts it
        # replaces: the kernels add it on the fp32 accumulator, and a save keeps the checkpoint dtype
        bias = torch.empty(self.num_experts, rows) if self.has_bias else None
        # NVFP4 two-level: the fp32 globals the kernels recover on the accumulator, one per expert
        # for the weight (`FineGrainedWeightGlobals` merges a separately calibrated gate|up stack
        # down to that). The calibrated activation scale is the checkpoint's `input_scale`: a static FP8 scale,
        # or NVFP4's activation global.
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
        # straight off the Parameter table, which every slot is registered in (empty ones as `None`)
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
        """Everything the kernels' MoE forwards take from the module: the two projections with their
        scales (held swizzled for dot_scaled chains), the per-expert globals and biases, the activation
        and the post-expert norm. Only the tensors are gathered per call; the rest is fixed once the
        module holds its norm, so it is resolved on the first call.

        A method rather than a function so an adapter library can extend it per module: PEFT overrides
        it to add its LoRA factors, which the kernels then apply on the routed rows between the GEMMs
        (`expert_linear` is the eager loop's counterpart)."""
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
        """The non-tensor half of `_moe_operands`. A supported activation NAME is fused into the gate_up
        epilogue; any other runs as the module's own GLU, a callable the kernels call on the host between
        the GEMMs, so any activation works without a kernel change. A supported norm NAME is folded into
        the routing-weighted reduce, one pass ahead of it computing the rows' rsqrt; any other callable is
        applied to the routed rows before it. Both read whole expert rows: the TP styles never split an
        expert that has a post-expert norm."""
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


@functools.cache
def _experts_class_with(cls, *, has_gate: bool, has_bias: bool, is_concatenated: bool) -> type:
    """`cls` with the experts dispatch and these flags: `use_experts_implementation` wraps the class it is handed in
    place, so each flag set wraps a subclass of its own, once, and `cls` itself stays as defined. The subclass is
    registered on `cls`'s module under its own name, so a pickled model finds it."""
    qualname = f"{cls.__qualname__}_gate{int(has_gate)}_bias{int(has_bias)}_concatenated{int(is_concatenated)}"
    subclass = type(cls.__name__, (cls,), {"__module__": cls.__module__, "__qualname__": qualname})
    setattr(sys.modules[cls.__module__], qualname, subclass)
    return use_experts_implementation(
        experts_class=subclass,
        experts_interface=ALL_FINEGRAINED_EXPERTS_FUNCTIONS,
        has_gate=has_gate,
        has_bias=has_bias,
        is_concatenated=is_concatenated,
    )


class FineGrainedExpertsInterface(ExpertsInterface):
    """Interface for registering custom finegrained experts forward functions."""

    _global_mapping = {
        "deepgemm": deepgemm_fp8_fp4_experts_forward,
        "batched_mm": finegrained_batched_mm_experts_forward,
        "grouped_mm": finegrained_grouped_mm_experts_forward,
        "deepgemm_megamoe": deepgemm_fp8_fp4_megamoe_experts_forward,
    }


ALL_FINEGRAINED_EXPERTS_FUNCTIONS = FineGrainedExpertsInterface()


def assert_modules_are_quantized(model: nn.Module) -> None:
    """Every finegrained module must hold a quantized weight once the checkpoint is in.

    A full-precision weight in a quantized slot means the swap and the checkpoint disagree: the
    module was allocated quantized storage and the loader replaced it with what the checkpoint
    actually ships. Serving that in full precision would hand back a model the caller believes is
    quantized and is not, so it is a load failure. Checked once here rather than per forward —
    the dtype cannot change between them.
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


def register_shared_gate_up_input_globals(model: nn.Module) -> None:
    """Share the gate_up's NVFP4 activation global across experts, as the max of their calibrated ones, so
    its rows quantize once per token before routing and the GEMM gathers them. Per-expert globals, the
    calibration exactly, quantize per routed row (top_k times the rows), which the GEMM then reads
    materialized: 3-11% slower on a GLM NVFP4 MoE layer at prefill, the most at small batches. Decode is
    unaffected. A max never clips, but experts below it lose block-scale precision.

    ``TRANSFORMERS_FINEGRAINED_SHARED_GATE_UP_ACTIVATION_GLOBAL`` turns it on and picks the experts the max
    runs over under expert parallelism: ``global`` all of them, so the numerics do not depend on the EP
    size, or ``rank`` this rank's, a tighter global. Unset keeps every expert's own."""
    scope = os.environ.get("TRANSFORMERS_FINEGRAINED_SHARED_GATE_UP_ACTIVATION_GLOBAL")
    if scope is None:
        return
    if scope not in ("global", "rank"):
        raise ValueError(
            f"TRANSFORMERS_FINEGRAINED_SHARED_GATE_UP_ACTIVATION_GLOBAL takes `global` or `rank`, got {scope!r}"
        )
    for module in model.modules():
        if not isinstance(module, FineGrainedExperts):
            continue
        input_global = getattr(module, f"{module.gate_up_name}_activation_scale")
        if input_global is not None:
            if hasattr(input_global, "full_tensor"):  # sharded over experts
                input_global = input_global.full_tensor() if scope == "global" else input_global.to_local()
            module.shared_gate_up_input_global = input_global.detach().amax().reshape(1)


def disable_deepgemm_on_multi_device(model: nn.Module) -> None:
    """Flag every quantized module to skip DeepGEMM when the model spans >1 CUDA device in one
    process. DeepGEMM loads each kernel via `cuKernelGetFunction`, which binds the `CUfunction`
    handle to the CUDA context live at load time; driving that cached handle from another device
    launches it against the wrong context and produces garbage. (Build-time fix: compile DeepGEMM
    with `DG_JIT_USE_RUNTIME_API=1` for a context-free `cudaKernel_t` loader; until our wheel picks
    that up we avoid single-process multi-device.) Setting `_deepgemm_disabled` routes both the
    linear and experts paths through our triton kernels. A model that fits on one device keeps
    DeepGEMM even with other GPUs visible; TP/EP put one device per process, so this is a no-op
    there."""
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
