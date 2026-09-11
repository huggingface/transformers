# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps, Interleave
from ..quantizers.quantizers_utils import get_module_from_name, should_convert_module
from ..utils import logging
from ..utils.import_utils import is_kernels_available
from .deepgemm import (
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
    deepgemm_fp8_fp4_megamoe_experts_forward,
    is_deepgemm_loadable,
    is_sm100,
)
from .hub_kernels import _MISSING_KERNELS_MESSAGE, lazy_load_kernel
from .moe import ExpertsInterface, use_experts_implementation
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


@functools.cache
def _get_ue8m0_dtype() -> torch.dtype:
    """Return ``torch.float8_e8m0fnu`` or raise a clear error on torch without FP8 support.

    UE8M0 scales are always stored/consumed as this single dtype — the kernels (Triton
    finegrained + DeepGEMM) read it natively, and supporting the same scales in mixed
    container dtypes would be a mess — so fail loudly rather than fall back."""
    if not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError(
            "scale_fmt='ue8m0' requires torch.float8_e8m0fnu, which is only available in "
            f"PyTorch >= 2.7 (found {torch.__version__}). Upgrade torch to use UE8M0 FP8 checkpoints."
        )
    return torch.float8_e8m0fnu


def _first_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of: {names}")


@dataclass(frozen=True)
class FineGrained:
    """Entry points exposed by the `kernels-community/finegrained-kernels` Triton kernel.

    Every recipe (block-FP8, MXFP8, MXFP4, NVFP4, weight-only) flows through the matmuls and
    the two MoE forwards, with the recipe resolved off the weight/scale dtypes inside the
    kernel; the per-recipe quant helpers and the ``Epilogue``/``Quantization`` op-boundary
    classes ship in the same build. The experts forwards hand the kernels' MoE forwards the
    module's tensors and let them run the chain (scheduling, fused GLU + intermediate requant,
    biases, the routing-weighted reduce); an activation outside ``get_supported_act_fns()`` is passed as
    the module's own callable and runs on the host between the two GEMMs. All symbols are
    required — a build missing any raises at load with the full list.
    """

    matmul_2d: Callable
    matmul_batched: Callable
    matmul_grouped: Callable
    moe_fused_batched: Callable
    moe_fused_grouped: Callable
    swizzle_mx_scales: Callable
    unswizzle_mx_scales: Callable
    get_supported_act_fns: Callable
    Quantization: Callable
    Epilogue: Callable


# Cache the loaded kernel but not failures: re-checking each call is cheap and intended, since the env
# can change between attempts. A module global (not `@functools.cache`) avoids Dynamo warning about
# tracing a cache-wrapped function on every compile.
_FINEGRAINED: FineGrained | None = None


def _import_local_finegrained():
    """A locally importable `finegrained_kernels` package takes precedence over the hub build:
    `FINEGRAINED_KERNELS_PATH` (a checkout's `torch-ext` directory, or any directory containing
    the package) is prepended to `sys.path`, then a plain import is attempted either way — an
    installed / already-on-path package also wins. Returns the module, or `None` to fall back
    to the `kernels` hub load."""
    path = os.environ.get("FINEGRAINED_KERNELS_PATH")
    if path:
        if not os.path.isdir(path):
            raise ImportError(f"FINEGRAINED_KERNELS_PATH does not exist: {path}")
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        return importlib.import_module("finegrained_kernels")
    except ImportError:
        if path:
            raise  # an explicit local path that fails to import is a setup error, not a fallback
        return None


@torch._dynamo.allow_in_graph
def _load_finegrained_kernel() -> None:
    """
    Load the finegrained-fp8 Triton kernel once into the `_FINEGRAINED` module global.

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

    kernel = _import_local_finegrained()
    if kernel is None:
        if not is_kernels_available():
            raise ImportError(f"finegrained-fp8 kernel unavailable: {_MISSING_KERNELS_MESSAGE}")
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
class _WeightFormat:
    """Storage layout of one quantized weight format — the single source both module classes
    derive their parameter shapes from. ``scale_dtype`` ``None`` defers to the config's
    ``scale_fmt`` (block-FP8 ships fp32 or UE8M0 containers; the group formats pin theirs).
    ``scale_group`` ``None`` means the block comes from the quant config's ``weight_block_size``."""

    weight_dtype: torch.dtype
    values_per_byte: int = 1
    scale_dtype: torch.dtype | None = None
    scale_group: tuple[int, int] | None = None
    has_global_scale: bool = False


def _weight_formats() -> dict[str, _WeightFormat]:
    return {
        # block-scaled E4M3, block from the quant config, fp32/UE8M0 scale container
        "fp8": _WeightFormat(weight_dtype=_FP8_DTYPE),
        # E4M3 values, UE8M0 group-32 scales
        "mxfp8": _WeightFormat(weight_dtype=_FP8_DTYPE, scale_dtype=_get_ue8m0_dtype(), scale_group=(1, 32)),
        # packed E2M1 values (2/byte), UE8M0 group-32 scales
        "mxfp4": _WeightFormat(
            weight_dtype=torch.int8, values_per_byte=2, scale_dtype=_get_ue8m0_dtype(), scale_group=(1, 32)
        ),
        # packed E2M1 values, E4M3 group-16 block scales, per-tensor/per-expert fp32 global
        "nvfp4": _WeightFormat(
            weight_dtype=torch.int8,
            values_per_byte=2,
            scale_dtype=_FP8_DTYPE,
            scale_group=(1, 16),
            has_global_scale=True,
        ),
    }


def resolve_weight_format(
    weight_format: str,
    scale_fmt: str = "float",
    block_size: tuple[int, int] | None = None,
) -> tuple[_WeightFormat, torch.dtype, tuple[int, int] | None]:
    """``(format, scale_dtype, (sf_gran_n, sf_gran_k))`` for one format name, with the config's
    ``scale_fmt``/``weight_block_size`` filling the slots the format leaves open."""
    formats = _weight_formats()
    if weight_format not in formats:
        raise ValueError(f"unknown weight_format {weight_format!r}; expected one of {sorted(formats)}")
    fmt = formats[weight_format]
    scale_dtype = fmt.scale_dtype
    if scale_dtype is None:
        scale_dtype = _get_ue8m0_dtype() if scale_fmt == "ue8m0" else torch.float32
    return fmt, scale_dtype, fmt.scale_group if fmt.scale_group is not None else block_size


def _alloc_expert_proj(
    num_experts: int,
    proj_out: int,
    proj_in: int,
    fmt: _WeightFormat,
    scale_dtype: torch.dtype,
    scale_group: tuple[int, int] | None,
    min_scale_out: int = 1,
    swizzled: bool = False,
) -> tuple[nn.Parameter, nn.Parameter]:
    """``(weight, weight_scale_inv)`` Parameters for one expert projection: the weight in the
    format's storage (FP4 packs two values per byte along K), the scale grid at the format's
    granularity (``None`` = one scale per expert). ``min_scale_out`` floors the grid's output dim —
    the fused gate_up projection keeps room for both halves (``2``) even when ``proj_out`` is
    smaller than the block. ``swizzled`` allocates the grid in the ``SWIZZLE_32_4_4`` artifact
    layout instead — 128-row blocks x 4-column groups of ``(2, 256)`` byte tiles, the expert axis
    leading so EP shards and gathers it on dim 0 — which needs whole blocks; a projection that
    does not tile stays affine, which the kernels read directly."""
    weight = torch.empty(num_experts, proj_out, proj_in // fmt.values_per_byte, dtype=fmt.weight_dtype)
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


def _swizzles_scales(config, fmt: _WeightFormat, activation_format: str | None) -> bool:
    """Whether an experts module holds its block scales in the ``SWIZZLE_32_4_4`` layout the Blackwell
    tcgen05 scaled-MMA reads directly (plain row-major forces a per-tile gather that caps the scaled
    dot below the fp8/fp4 peak): SM100, a triton dispatch (the fused forwards and the eager loop both
    read it; the DeepGEMM backends read affine scales), a group-scaled format (MX group-32, NVFP4
    group-16 — block-FP8's grid never reaches a scaled-MMA, even with UE8M0 scales) and a chain that
    quantizes activations (weight-only reads scales per group affinely).
    ``TRANSFORMERS_FINEGRAINED_NO_SWIZZLE=1`` keeps every module affine (debug / A/B)."""
    if os.environ.get("TRANSFORMERS_FINEGRAINED_NO_SWIZZLE", "0") == "1":
        return False
    return (
        getattr(config, "_experts_implementation", None) not in ("deepgemm", "deepgemm_megamoe")
        and fmt.scale_group is not None
        and _block_recipe(activation_format) is not None
        and torch.cuda.is_available()
        and is_sm100()
    )


def finegrained_triton_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    activation_format: str | None = None,
) -> torch.Tensor:
    """Triton fine-grained linear: fused act-quant + matmul, then optional bias add.

    Serves every weight recipe the kernel resolves off the tensors themselves — block-FP8
    (fp32 or UE8M0 scales), MXFP8, MXFP4 and NVFP4 (``int8``-packed values; the two-level
    per-tensor ``weight_global_scale`` recovers on the accumulator). ``activation_scale=None`` →
    dynamic activation quant (inline); a per-tensor scalar → static quant against it.
    ``activation_format`` picks the activation recipe where the weights leave it open (``None`` =
    the weight-native choice; ``"bf16"`` = weight-only, no activation quant — W4A16).
    """
    kernel = load_finegrained_kernel()
    quantization = _kernel_quantization(kernel, activation_format)
    original_shape = input.shape
    output = kernel.matmul_2d(
        input.reshape(-1, original_shape[-1]),
        weight,
        activation_scale,
        weight_scale_inv,
        quantization=quantization,
        output_dtype=input.dtype,
        b_global_scale=weight_global_scale,
    )
    output = output.reshape(*original_shape[:-1], output.shape[-1])
    if bias is not None:
        output.add_(bias)
    return output


def finegrained_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    allow_deepgemm: bool = True,
    weight_global_scale: torch.Tensor | None = None,
    activation_format: str | None = None,
) -> torch.Tensor:
    """End-to-end FP8/FP4 linear used by `FineGrainedLinear` and the eager `FineGrainedExperts` loop.

    Dispatch order — both backends handle FP8 and FP4 weights with fp32 or UE8M0 scales:
      1. DeepGEMM (`deepgemm_fp8_fp4_linear`) — on the pre-SM100 shapes it supports (FP4,
         UE8M0 SFs, 128×128 block FP8). Never preferred on SM100: the Triton path is tuned
         per shape there and is the only one that reads the pre-swizzled SWIZZLE_32_4_4
         scales, and DeepGEMM's context-bound kernels are the multi-device hazard below.
      2. Triton finegrained fallback — everywhere else: SM100, an ``activation_scale``
         (DeepGEMM is dynamic-only), or any shape DeepGEMM declined.

    Args:
        input: (..., K) bf16/fp16 activations.
        weight: (N, K) `float8_e4m3fn` or (N, K // 2) `int8` (FP4-packed).
        weight_scale_inv: per-block weight scales — `float32` (V3-style) or `float8_e8m0fnu`
            (V4-style; reinterpreted as int32 at the DeepGEMM kernel boundary).
        block_size: [block_n, block_k] for FP8 block-wise quant, or None/[N, K] for per-tensor.
            Ignored for FP4 weights (the kernel infers SF granularity from the dtype).
        bias: optional bias added to the matmul output.
        activation_scale: pass a per-tensor scalar to use static activation quant; leave `None`
            for dynamic (per-token) quant.
        allow_deepgemm: set ``False`` to force the Triton fallback for this call. Used when the
            model spans multiple CUDA devices in one process — DeepGEMM's cached kernels are bound
            to a single CUDA context and produce garbage across devices (see
            ``disable_deepgemm_on_multi_device``).
        weight_global_scale: the NVFP4 two-level per-tensor global (None for other recipes).
        activation_format: the activation recipe where the weights leave it open (see
            ``finegrained_triton_linear``).
    """
    # DeepGEMM is CUDA-only, dynamic-only, SM90+, FP4 or 128x128-block FP8; the combos it would
    # silently corrupt (float32 scales on SM100, #47030) are skipped up front rather than attempted.
    # ``TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`` forces Triton for this dispatcher only.
    deepgemm_preferred = (
        # SM100 perf: Triton measures 2.9x (decode) / 1.45x (prefill) over DeepGEMM on the DSV4
        # qkv linear — the 128x128 block-FP8 shape DeepGEMM is otherwise preferred for, which the
        # gate above never catches (block-FP8's (N/128, K/128) grid has no swizzled layout).
        not is_sm100()
        # If the model is on multiple devices, DeepGEMM's context-bound kernels corrupt across devices. The
        # multi-device guard `disable_deepgemm_on_multi_device` flips `_deepgemm_disabled` True at load, which
        # this dispatcher sees and respects. The Triton fallback is context-free and safe.
        and allow_deepgemm
        # A pre-swizzled (SWIZZLE_32_4_4) scale is not readable as row-major: DeepGEMM would
        # consume the permuted buffer as affine and silently return garbage. Correctness gate,
        # true on any arch, for whatever the loader's `FineGrainedSwizzleScales` swizzled.
        and weight_scale_inv.ndim <= 2
        and is_deepgemm_loadable()
        and activation_scale is None
        # DeepGEMM serves neither the NVFP4 two-level global nor an explicit activation format
        and weight_global_scale is None
        and activation_format is None
        and (weight.dtype == torch.int8 or (block_size is not None and block_size[0] == block_size[1] == 128))
        and os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "0") != "1"
    )

    if deepgemm_preferred:
        try:
            return deepgemm_fp8_fp4_linear(
                input,
                weight,
                weight_scale_inv,
                block_size=block_size,
                activation_scale=activation_scale,
                bias=bias,
            )
        except (ImportError, NotImplementedError, ValueError) as e:
            # DeepGEMM is loadable but declined this specific input; fall back to Triton, which is more
            # permissive (handles FP8/FP4 with float32 or UE8M0 scales on any arch, plus input dtypes
            # DeepGEMM rejects). If Triton can't serve it either, it raises its own error.
            #   - NotImplementedError: an arch/input combo DeepGEMM has no kernel for (FP4 on Hopper —
            #     `is_deepgemm_loadable` is dtype-agnostic and passes there — or float32 scales on Blackwell);
            #   - ValueError: an input DeepGEMM rejects but Triton supports (e.g. activations it won't quantize);
            #   - ImportError: a symbol/build gap.
            logger.warning_once(
                f"DeepGEMM declined this call, falling back to Triton. Reason: {e} "
                "Set `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` to skip DeepGEMM for FP8 linear entirely."
            )

    return finegrained_triton_linear(
        input,
        weight,
        weight_scale_inv,
        bias,
        activation_scale,
        weight_global_scale=weight_global_scale,
        activation_format=activation_format,
    )


class _FineGrainedModule:
    """What every finegrained module (dense linear or experts) carries besides its parameters."""

    # the dtype the checkpoint shipped this module's block scales in, when it differs from the held
    # one (uint8 exponent bytes, float32 values) — set by the loader's `FineGrainedScaleContainer`,
    # read back by its reverse so a save restores the checkpoint's container
    scale_container_dtype: torch.dtype | None = None

    # Internal, temporary flag — not public API, don't set it directly. `disable_deepgemm_on_multi_device`
    # flips it True at load when the model spans >1 CUDA device in one process (DeepGEMM's context-bound
    # kernels corrupt across devices); removable once the kernel ships a context-free loader.
    _deepgemm_disabled = False


class FineGrainedLinear(_FineGrainedModule, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        scale_fmt: str = "float",
        has_bias: bool = False,
        weight_format: str = "fp8",
        activation_format: str | None = None,
        has_global_scale: bool | None = None,
    ):
        super().__init__(in_features, out_features)

        self.has_bias = has_bias
        self.weight_format = weight_format
        self.block_size = block_size
        self.activation_scheme = activation_scheme
        self.activation_format = activation_format
        # the format table decides storage: value dtype (packed E2M1 = 2 values per int8 byte
        # along K), scale dtype/granularity, and whether a two-level global exists
        fmt, sf_dtype, scale_group = resolve_weight_format(weight_format, scale_fmt, block_size)
        if has_global_scale is None:
            has_global_scale = fmt.has_global_scale
        in_storage = in_features // fmt.values_per_byte
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_storage, dtype=fmt.weight_dtype),
            requires_grad=fmt.weight_dtype.is_floating_point,
        )
        if has_global_scale:
            # NVFP4 two-level: the per-tensor fp32 global the kernel recovers on the accumulator
            self.weight_global_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_parameter("weight_global_scale", None)

        if scale_group is None:
            # no group and no block: one per-tensor scale
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = _cdiv(out_features, scale_group[0])
            scale_in_features = _cdiv(in_features, scale_group[1])
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=sf_dtype),
                requires_grad=sf_dtype.is_floating_point,
            )

        if self.activation_scheme == "static":
            self.activation_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_parameter("activation_scale", None)

        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.element_size() > 1:
            return F.linear(input, self.weight, self.bias)

        weight = to_local(self.weight)
        scale_inv = to_local(self.weight_scale_inv)

        return finegrained_linear(
            input,
            weight,
            scale_inv,
            block_size=self.block_size,
            activation_scale=self.activation_scale,
            bias=self.bias,
            allow_deepgemm=not self._deepgemm_disabled,
            weight_global_scale=to_local(self.weight_global_scale) if self.weight_global_scale is not None else None,
            activation_format=self.activation_format,
        )


class FineGrainedEmbedding(nn.Embedding):
    """``nn.Embedding`` whose table is stored in FP8 with one per-tensor ``weight_scale``. A table
    worth quantizing is one too big to dequantize at load (Qwen4-Exp's hashed n-gram table is
    ~48 GiB in FP8), so the rescale runs on the rows a lookup gathers. Built on the ``meta``
    device: ``nn.Embedding`` allocates a full-precision table first."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=_FP8_DTYPE), requires_grad=False)
        # `(1,)` rather than a scalar: the shape checkpoints serialize per-tensor scales with
        self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        rows = super().forward(input)
        return rows.to(self.weight_scale.dtype) * self.weight_scale


class FineGrainedGroupedLinear(FineGrainedLinear):
    """FP8 drop-in for block-diagonal grouped linears.

    The underlying nn.Linear stores a single `(n_groups * out_per_group, in_per_group)`
    weight; logically that's `n_groups` independent `(out_per_group, in_per_group)`
    sub-matrices, each consuming a disjoint slice of the input's last-but-one dim.
    Forward expects input of shape `(..., n_groups, in_per_group)` and returns
    `(..., n_groups, out_per_group)` — same contract as the vanilla bf16 grouped
    linear it replaces.

    """

    def __init__(
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        scale_fmt: str = "float",
        has_bias: bool = False,
    ):
        super().__init__(
            in_features=in_features_per_group,
            out_features=out_features,
            block_size=block_size,
            activation_scheme=activation_scheme,
            scale_fmt=scale_fmt,
            has_bias=has_bias,
        )
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]

        if self.weight.element_size() > 1:
            w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
            x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
            y = torch.bmm(x, w).transpose(0, 1)
            y = y.reshape(*input_shape, self.n_groups, -1)
            if self.has_bias:
                y.add_(self.bias.view(self.n_groups, -1))
            return y

        w = to_local(self.weight)
        scale_inv = to_local(self.weight_scale_inv)

        w = w.view(self.n_groups, -1, hidden_dim)
        x = x.movedim(-2, 0).reshape(-1, hidden_dim)
        scale_inv = scale_inv.view(self.n_groups, scale_inv.size(0) // self.n_groups, scale_inv.size(1))

        tokens_per_group = x.size(0) // self.n_groups
        # (E+1,) row boundaries — the kernels' expert_start schedule
        expert_start = torch.arange(0, self.n_groups + 1, device=x.device, dtype=torch.int32) * tokens_per_group

        kernel = load_finegrained_kernel()
        y = kernel.matmul_grouped(
            x,
            w,
            None,
            scale_inv,
            expert_start=expert_start,
            quantization=_kernel_quantization(kernel, self.activation_format),
            b_global_scale=to_local(self.weight_global_scale) if self.weight_global_scale is not None else None,
        )
        y = y.reshape(self.n_groups, *input_shape, -1).movedim(0, -2)
        if self.has_bias:
            y.add_(self.bias.view(self.n_groups, -1))
        return y


def _kernel_quantization(kernel, activation_format: str | None):
    """Map the module-level ``activation_format`` onto the kernel's ``Quantization``. ``None``
    keeps the kernel's weight-native default; ``"bf16"`` = weight-only (no activation quant)."""
    if activation_format is None:
        return None
    recipe = None if activation_format == "bf16" else activation_format
    return kernel.Quantization(input_recipe=recipe)


def _block_recipe(activation_format: str | None) -> str | None:
    """The module-level ``activation_format`` as the kernels' block ``recipe``: ``None`` keeps
    the weight family's own format, ``"bf16"`` is weight-only (no activation quant), an explicit
    name is itself."""
    if activation_format is None:
        return "weights"
    return None if activation_format == "bf16" else activation_format


def _moe_operands(kernel, module) -> dict:
    """Everything the kernels' MoE forwards take from the module: the two projections with their
    scales (held swizzled for dot_scaled chains), the per-expert globals and biases,
    and the activation — a ``get_supported_act_fns()`` name the gate_up epilogue fuses, else the module's
    own GLU as a callable the kernels run on the host, so any activation works without a kernel
    change."""
    up = "gate_up_proj" if module.has_gate else "up_proj"

    def local(name):
        t = getattr(module, name, None)
        return to_local(t) if t is not None else None

    act_name = module.act_fn_name
    fused = act_name in kernel.get_supported_act_fns() and (module.swiglu_alpha is None or act_name == "silu")
    if fused:
        act_fn = act_name
    else:
        act_fn = module._apply_gate if module.has_gate else module.act_fn
    return {
        "gate_up_proj": local(up),
        "down_proj": local("down_proj"),
        "gate_up_proj_scale_inv": local(f"{up}_scale_inv"),
        "down_proj_scale_inv": local("down_proj_scale_inv"),
        "gate_up_proj_global_scale": local(f"{up}_global_scale"),
        "down_proj_global_scale": local("down_proj_global_scale"),
        "gate_up_proj_bias": local(f"{up}_bias"),
        "down_proj_bias": local("down_proj_bias"),
        "act_fn": act_fn,
        "swiglu_alpha": module.swiglu_alpha,
        "swiglu_limit": module.swiglu_limit,
        "recipe": _block_recipe(module.activation_format),
        "gate": module.has_gate,
    }


def _fused_experts_forward(module, kernel_forward: str, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """The kernels' fused MoE chain (gate_up with the fused GLU epilogue and intermediate requant
    where supported -> down -> the routing-weighted top-k reduce) over the module's tensors."""
    if module.activation_scheme == "static":
        raise NotImplementedError(
            f"the {kernel_forward} experts dispatch does not support activation_scheme='static'; use "
            "experts_implementation='eager' or activation_scheme='dynamic'."
        )
    kernel = load_finegrained_kernel()
    return getattr(kernel, kernel_forward)(hidden_states, top_k_index, top_k_weights, **_moe_operands(kernel, module))


def finegrained_batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """Batched (decode) experts forward: one program per routed row."""
    return _fused_experts_forward(self, "moe_fused_batched", hidden_states, top_k_index, top_k_weights)


def finegrained_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """Grouped (prefill) experts forward: one on-device routing pass, then the chain over the
    expert-sorted schedule, scattering back to routed rows (EP-sentinel rows skipped)."""
    return _fused_experts_forward(self, "moe_fused_grouped", hidden_states, top_k_index, top_k_weights)


class FineGrainedExperts(_FineGrainedModule, nn.Module):
    # Per-`_experts_implementation` rewrite of parallel-layer kinds in the TP/EP plan.
    # The plan dicts store `{module-path-pattern: parallel-layer-kind}`; this maps an
    # old kind to a new kind, and the quantizer rewrites every plan VALUE that matches.
    # The default `MoeTensorParalellExperts` kind is impl-agnostic; some impls need a
    # distinct TP layer (e.g. megamoe needs no gradient-sync hooks and an EP
    # `process_group` injection). Declared here so the quantizer doesn't have to know
    # about impl-specific TP needs — extend this dict when adding new impls.
    _impl_tp_layer_overrides: dict[str, dict[str, str]] = {
        "deepgemm_megamoe": {
            "moe_tp_experts": "megamoe_experts",
            "ep_router": "megamoe_router",
        },
    }

    def __init__(
        self,
        config,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        scale_fmt: str = "float",
        has_bias: bool = False,
        has_gate: bool = True,
        weight_format: str | None = None,
        activation_format: str | None = None,
        has_global_scale: bool | None = None,
        is_concatenated: bool = True,
    ):
        super().__init__()

        self.config = config
        self.has_bias = has_bias
        self.has_gate = has_gate
        # the model's own gate|up row order (transformers' experts flag): stacked ``[gate; up]`` or,
        # like GPT-OSS, already the kernels' interleaved ``[g0, u0, ...]`` — decides whether the
        # loader interleaves
        self.is_concatenated = is_concatenated
        self.block_size = block_size
        self.hidden_dim = config.hidden_size
        self.activation_format = activation_format
        self.activation_scheme = activation_scheme
        self.swiglu_alpha = getattr(config, "swiglu_alpha", None)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.num_experts = _first_attr(config, "num_local_experts", "num_experts")
        self.intermediate_dim = _first_attr(config, "moe_intermediate_size", "intermediate_size")
        self.act_fn_name = _first_attr(config, "hidden_activation", "hidden_act")
        self.act_fn = ACT2FN[self.act_fn_name]

        # Expert weight storage is declared by the QUANT config's format key, not the model
        # config: `weight_format` arrives from `replace_with_finegrained_layer` as the
        # checkpoint's quant_method ("fp8", "mxfp8", "mxfp4", "nvfp4"). The DeepSeek V4-style
        # `config.expert_dtype = "fp4"` model-config side-channel predates it and is kept as
        # the legacy fallback when no format is passed (dsv4 fp4 experts under the "fp8" key).
        if weight_format is None:
            weight_format = "mxfp4" if getattr(config, "expert_dtype", "fp8") == "fp4" else "fp8"
        self.weight_format = weight_format
        fmt, scale_dtype, scale_group = resolve_weight_format(weight_format, scale_fmt, block_size)
        # the forwards gate on this attribute, so it carries the format-resolved value (a None ctor
        # arg would silently drop the nvfp4 global at every forward)
        self.has_global_scale = fmt.has_global_scale if has_global_scale is None else has_global_scale

        # what the module HOLDS, as the loader's conversion ops deliver it (and their reverses
        # restore for the checkpoint): gate|up rows in the kernels' interleaved order unless the
        # backend packs gate|up itself (DeepGEMM Mega MoE), block scales swizzled where the
        # scaled-MMA reads them
        impl = getattr(config, "_experts_implementation", None)
        self._gate_up_interleaved = self.has_gate and impl != "deepgemm_megamoe"
        swizzled = _swizzles_scales(config, fmt, activation_format)

        up_name = "gate_up_proj" if self.has_gate else "up_proj"
        up_rows = (2 if self.has_gate else 1) * self.intermediate_dim
        for proj, rows, in_dim, min_scale_out in (
            (up_name, up_rows, self.hidden_dim, 2 if self.has_gate else 1),
            ("down_proj", self.hidden_dim, self.intermediate_dim, 1),
        ):
            weight, scale = _alloc_expert_proj(
                self.num_experts, rows, in_dim, fmt, scale_dtype, scale_group, min_scale_out, swizzled
            )
            setattr(self, proj, weight)
            setattr(self, f"{proj}_scale_inv", scale)
            if self.has_bias:
                # the model dtype (the default dtype under `from_pretrained`), like the bf16 experts it
                # replaces: the kernels add it on the fp32 accumulator, and a save keeps the checkpoint dtype
                setattr(self, f"{proj}_bias", nn.Parameter(torch.empty(self.num_experts, rows)))
            else:
                self.register_parameter(f"{proj}_bias", None)
            # NVFP4 two-level: per-expert fp32 globals the kernels recover on the accumulator
            if self.has_global_scale:
                setattr(self, f"{proj}_global_scale", nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32)))
            else:
                self.register_parameter(f"{proj}_global_scale", None)

        if self.activation_scheme == "static":
            self.gate_up_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))
            self.down_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # the GEMM's output columns alternate gate/up; mirrors the fused epilogue's
        # `split_gate_up`, which the fused/unfused parity tests compare against
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
        if self.swiglu_alpha is not None:
            # Clamped SwiGLU-OAI gate (same math as the model's non-quantized experts).
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
            return (up + 1.0) * glu
        elif self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
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

        for expert_idx in expert_hit:
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up_act_scale = (
                self.gate_up_proj_activation_scale[expert_idx] if self.activation_scheme == "static" else None
            )
            proj_out = self.linear(
                current_state,
                "gate_up_proj" if self.has_gate else "up_proj",
                expert_idx,
                activation_scale=gate_up_act_scale,
            )
            proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
            down_act_scale = (
                self.down_proj_activation_scale[expert_idx] if self.activation_scheme == "static" else None
            )
            proj_out = self.linear(proj_out, "down_proj", expert_idx, activation_scale=down_act_scale)
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            weighted_out = proj_out * routing_weights.to(proj_out.dtype)
            final_hidden_states.index_add_(0, token_idx, weighted_out.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def linear(
        self, input: torch.Tensor, proj: str, expert_idx, activation_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One expert's ``proj`` as a dense linear: the same weight, scale (one expert's slice of a
        swizzled stack is the ``(1, ...)`` artifact ``matmul_2d`` reads), bias, NVFP4 global and
        activation format the fused forwards hand the kernels."""
        weight = getattr(self, proj)[expert_idx]
        bias = getattr(self, f"{proj}_bias")
        bias = bias[expert_idx] if bias is not None else None
        if weight.element_size() > 1:
            return F.linear(input, weight, bias)
        scale = getattr(self, f"{proj}_scale_inv")
        global_scale = getattr(self, f"{proj}_global_scale")
        return finegrained_linear(
            input,
            weight,
            scale[expert_idx : expert_idx + 1] if scale.ndim == 5 else scale[expert_idx],
            self.block_size,
            bias=bias,
            activation_scale=activation_scale,
            allow_deepgemm=not self._deepgemm_disabled,
            weight_global_scale=global_scale[expert_idx] if global_scale is not None else None,
            activation_format=self.activation_format,
        )


class FineGrainedExpertsInterface(ExpertsInterface):
    """Interface for registering custom FP8 experts forward functions."""

    _global_mapping = {
        "deepgemm": deepgemm_fp8_fp4_experts_forward,
        "batched_mm": finegrained_batched_mm_experts_forward,
        "grouped_mm": finegrained_grouped_mm_experts_forward,
        "deepgemm_megamoe": deepgemm_fp8_fp4_megamoe_experts_forward,
    }


ALL_FINEGRAINED_EXPERTS_FUNCTIONS = FineGrainedExpertsInterface()


def disable_deepgemm_on_multi_device(model: nn.Module) -> None:
    """Flag every quantized module to skip DeepGEMM when the model spans >1 CUDA device in one
    process. DeepGEMM loads each kernel via `cuKernelGetFunction`, which binds the `CUfunction`
    handle to the CUDA context live at load time; driving that cached handle from another device
    launches it against the wrong context and produces garbage. (Build-time fix: compile DeepGEMM
    with `DG_JIT_USE_RUNTIME_API=1` for a context-free `cudaKernel_t` loader; until our wheel picks
    that up we avoid single-process multi-device.) Setting `_deepgemm_disabled` routes both the
    linear and experts paths through Triton/grouped_mm. A model that fits on one device keeps
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
        "This FP8 model spans multiple CUDA devices in one process; routing its FP8 linear and experts "
        "layers through Triton/grouped_mm instead of DeepGEMM (DeepGEMM's cached kernels are bound to a "
        "single CUDA context and corrupt across devices). Run tensor/expert parallel (one device per "
        "process) to use the faster DeepGEMM path."
    )


def replace_with_finegrained_embedding(model, patterns: list[str], modules_to_not_convert: list[str] | None = None):
    """Swap every ``nn.Embedding`` whose name ends with one of ``patterns`` (the quant config's
    ``modules_to_convert``) for ``FineGrainedEmbedding``. Runs under ``dequantize=True`` too: a table
    worth quantizing is one we cannot afford to expand."""
    for name, module in list(model.named_modules()):
        if type(module) is nn.Embedding and any(name.endswith(p) for p in patterns):
            if not should_convert_module(name, modules_to_not_convert):
                continue
            with torch.device("meta"):
                new_module = FineGrainedEmbedding(module.num_embeddings, module.embedding_dim, module.padding_idx)
            new_module._hf_quantized_needs_local_tp = True
            model.set_submodule(name, new_module)
    return model


def replace_with_finegrained_layer(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    """Swap the model's ``nn.Linear`` (and grouped-linear) modules for ``FineGrainedLinear`` and its
    ``.experts`` modules for ``FineGrainedExperts``, both allocated in the quantized storage the
    checkpoint's format declares. ``modules_to_not_convert`` names the modules kept as they are
    (typically ``lm_head``); nothing is replaced under ``dequantize=True``."""

    if quantization_config.dequantize:
        return model

    # The checkpoint's quant_method IS the weight format ("fp8", "mxfp8", "mxfp4", "nvfp4").
    # Under the "fp8" key pass None: dsv4-style checkpoints declare fp4 EXPERTS through the
    # legacy `config.expert_dtype` model-config side-channel, which the ctor falls back to.
    quant_method = getattr(quantization_config, "quant_method", None)
    quant_method = getattr(quant_method, "value", quant_method)
    # modelopt exports are NVFP4-only (validated by the config translation)
    if quant_method == "modelopt":
        quant_method = "nvfp4"
    weight_format = quant_method if quant_method in ("mxfp8", "mxfp4", "nvfp4") else None

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        new_module = None
        with torch.device("meta"):
            if module_name.endswith(".experts"):
                has_gate = getattr(module, "has_gate", True)
                has_bias = getattr(module, "has_bias", False)
                is_concatenated = getattr(module, "is_concatenated", True)
                config = getattr(module, "config", model.config.get_text_config())
                # the decorator stamps these flags on the instance after __init__, so the model's
                # own gate|up convention has to travel through it too
                new_class = use_experts_implementation(
                    experts_class=FineGrainedExperts,
                    experts_interface=ALL_FINEGRAINED_EXPERTS_FUNCTIONS,
                    has_bias=has_bias,
                    has_gate=has_gate,
                    is_concatenated=is_concatenated,
                )
                new_module = new_class(
                    config=config,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    scale_fmt=quantization_config.scale_fmt,
                    has_bias=has_bias,
                    has_gate=has_gate,
                    weight_format=weight_format,
                    activation_format=getattr(quantization_config, "activation_format", None),
                    is_concatenated=is_concatenated,
                )
                # GPT-OSS hardcodes its clamped-SwiGLU parameters as MODULE attributes
                # (alpha=1.702, limit=7.0) rather than config fields — carry them over so
                # `_apply_gate` (and the fused epilogue) keep the reference numerics.
                if getattr(module, "alpha", None) is not None and new_module.swiglu_alpha is None:
                    new_module.swiglu_alpha = module.alpha
                    new_module.swiglu_limit = getattr(module, "limit", None)
            elif type(module) is nn.Linear:
                # Vanilla `nn.Linear` → standard FineGrainedLinear swap.
                new_module = FineGrainedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    scale_fmt=quantization_config.scale_fmt,
                    has_bias=module.bias is not None,
                    weight_format=weight_format or "fp8",
                    activation_format=getattr(quantization_config, "activation_format", None),
                )
            elif isinstance(module, nn.Linear) and "GroupedLinear" in type(module).__name__:
                # Block-diagonal grouped linear (e.g. DSv4's `DeepseekV4GroupedLinear`):
                # one underlying weight conceptually split into `n_groups` independent
                # sub-matmuls fed by disjoint input slices. Vanilla `FineGrainedLinear` would
                # collapse those groups into one giant linear and yield the wrong
                # output dim, so swap to `FineGrainedGroupedLinear` which keeps the per-group
                # bmm contract and runs each block as its own FP8 matmul.
                new_module = FineGrainedGroupedLinear(
                    in_features_per_group=module.in_features,
                    out_features=module.out_features,
                    n_groups=module.n_groups,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    scale_fmt=quantization_config.scale_fmt,
                    has_bias=module.bias is not None,
                )
            if new_module is not None:
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )
    return model


# ---------------------------------------------------------------------------------------------------
# Conversion ops. The loader runs a converter's ops on the rank-local tensors it collected for one
# target, and `save_pretrained` runs their reverses on the gathered tensors, so everything below
# maps a checkpoint layout onto the layout the modules above hold — and back.
# ---------------------------------------------------------------------------------------------------


def _keyed_by_target(input_dict: dict, target_patterns) -> dict:
    """A converter's ops see their single tensor under the SOURCE pattern until an op re-keys it to the
    target (the core ops do; the loader then expands the target into the full name) — so a
    layout op that is first in its converter re-keys the same way. Multi-tensor dicts (a
    deserializer's fully named outputs) pass through."""
    if target_patterns and len(input_dict) == 1:
        value = next(iter(input_dict.values()))
        return {target_patterns[0]: value[0] if isinstance(value, list) else value}
    return input_dict


def _held_scale(model, full_layer_name, key):
    """The finegrained module and its ``*_scale_inv`` Parameter a converter output fills, else
    ``(None, None)``. A converter emitting several tensors names them fully (key); a single target's
    name arrives as the pattern, its full name (with the ``_scale_inv`` suffix) in ``full_layer_name``."""
    param_name = (key if key.endswith("_scale_inv") else full_layer_name).rpartition(".")[2]
    if not param_name.endswith("_scale_inv") or model is None:
        return None, None
    module = model.get_submodule(full_layer_name.rpartition(".")[0])
    if not isinstance(module, _FineGrainedModule):
        return None, None
    return module, getattr(module, param_name, None)


def _recontain(scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The same bytes under a same-width dtype (uint8 <-> e8m0), an exact numeric cast otherwise."""
    return scale.view(dtype) if scale.element_size() == dtype.itemsize else scale.to(dtype)


def _checkpoint_scale_container(model) -> torch.dtype | None:
    """The scale container the checkpoint shipped, as recorded on its finegrained modules at load
    (one checkpoint, one container); None when it was already the held dtype."""
    if model is None:
        return None
    return next((m.scale_container_dtype for m in model.modules() if getattr(m, "scale_container_dtype", None)), None)


class FineGrainedInterleaveGateUp(ConversionOps):
    """Stacked ``[gate; up]`` expert rows into the kernels' ``[g0, u0, g1, u1, ...]`` order (weight,
    scale grid and bias share the row axis) — the core ``Interleave`` along dim 1. Decided off the
    model's experts modules, on load and on save alike: skipped when the model's own rows already
    come interleaved (``is_concatenated=False``, GPT-OSS) or the backend holds them stacked
    (``_gate_up_interleaved`` False, DeepGEMM Mega MoE)."""

    def __init__(self, hf_quantizer=None, inverse: bool = False):
        self.hf_quantizer = hf_quantizer
        self.inverse = inverse

    def convert(self, input_dict, model=None, target_patterns=None, **kwargs):
        input_dict = _keyed_by_target(input_dict, target_patterns)
        experts = next((m for m in model.modules() if isinstance(m, FineGrainedExperts)), None) if model else None
        if experts is None or not experts.is_concatenated or not experts._gate_up_interleaved:
            return input_dict
        interleave = Interleave(dim=1, inverse=not self.inverse)  # inverse=True is stacked -> interleaved
        return {key: interleave.convert({key: value}, None, [key])[key] for key, value in input_dict.items()}

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedInterleaveGateUp(self.hf_quantizer, inverse=not self.inverse)


class FineGrainedScaleContainer(ConversionOps):
    """A block scale into the dtype its module holds. Two containers exist for UE8M0: the same
    exponent byte under ``uint8`` (MiniMax-style mxfp8 checkpoints) — a bit reinterpretation into
    ``float8_e8m0fnu`` — and the power-of-two value in a float32 tensor (dsv4-flash-base) — an exact
    numeric cast. Other tensors, and scales already in the held dtype, pass through. The module
    records the checkpoint's container (``scale_container_dtype``), so the reverse restores it
    exactly on save."""

    def __init__(self, hf_quantizer=None, inverse: bool = False):
        self.hf_quantizer = hf_quantizer
        self.inverse = inverse

    def convert(self, input_dict, model=None, full_layer_name=None, target_patterns=None, **kwargs):
        out = {}
        for key, value in _keyed_by_target(input_dict, target_patterns).items():
            value = value[0] if isinstance(value, list) else value
            if self.inverse:
                container = _checkpoint_scale_container(model)
                # e8m0 is only ever a scale, so the dtype alone picks the tensors to restore
                if container is not None and value.dtype == _get_ue8m0_dtype():
                    value = _recontain(value, container)
            else:
                module, held = _held_scale(model, full_layer_name, key)
                if held is not None and value.dtype != held.dtype:
                    module.scale_container_dtype = value.dtype
                    value = _recontain(value, held.dtype)
            out[key] = value
        return out

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedScaleContainer(self.hf_quantizer, inverse=not self.inverse)


class FineGrainedSwizzleScales(ConversionOps):
    """Expert block scales into the ``SWIZZLE_32_4_4`` layout their module holds (a 5-D scale
    Parameter, see ``_swizzles_scales``): one triton launch per rank-local shard, values unchanged.
    Every other tensor the converter carries (weights, biases, the scales of a module kept affine)
    passes through. The reverse restores the checkpoint's affine grid from a 5-D tensor's own shape
    (whole 128-row blocks and 4-column groups, so no padding to trim), so saving is exact."""

    def __init__(self, hf_quantizer=None, inverse: bool = False):
        self.hf_quantizer = hf_quantizer
        self.inverse = inverse

    def convert(self, input_dict, model=None, full_layer_name=None, target_patterns=None, **kwargs):
        out = {}
        for key, value in _keyed_by_target(input_dict, target_patterns).items():
            value = value[0] if isinstance(value, list) else value
            out[key] = self._unswizzle(value) if self.inverse else self._swizzle(key, value, model, full_layer_name)
        return out

    @staticmethod
    def _unswizzle(value):
        if value.ndim != 5:
            return value
        experts, row_blocks, col_groups = value.shape[:3]
        kernel = load_finegrained_kernel()
        return kernel.unswizzle_mx_scales(value, row_blocks * 128, col_groups * 4, num_experts=experts)

    @staticmethod
    def _swizzle(key, value, model, full_layer_name):
        _, held = _held_scale(model, full_layer_name, key)
        if held is None or held.ndim != 5 or value.ndim == 5:
            return value
        return load_finegrained_kernel().swizzle_mx_scales(value)

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedSwizzleScales(self.hf_quantizer, inverse=not self.inverse)


class FineGrainedPackedBlocks(ConversionOps):
    """GPT-OSS's ``{proj}_blocks`` ``(E, N, K/32, 16)`` uint8 — two low-nibble-first E2M1 values per
    byte, 16 bytes per group of 32 — as the packed ``(E, N, K/2)`` int8 the kernels read: the same
    bytes, grouped per row. The reverse folds the rows back into 16-byte groups under uint8. The
    ``{proj}_scales`` companion goes through ``FineGrainedScaleContainer`` (exponent bytes ->
    ``float8_e8m0fnu``) in its own converter."""

    def __init__(self, hf_quantizer=None, inverse: bool = False):
        self.hf_quantizer = hf_quantizer
        self.inverse = inverse

    def convert(self, input_dict, target_patterns=None, **kwargs):
        out = {}
        for key, value in _keyed_by_target(input_dict, target_patterns).items():
            value = value[0] if isinstance(value, list) else value
            if self.inverse:
                out[key] = value.view(torch.uint8).reshape(*value.shape[:-1], value.shape[-1] // 16, 16)
            else:
                out[key] = value.reshape(*value.shape[:-2], -1).view(torch.int8)
        return out

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedPackedBlocks(self.hf_quantizer, inverse=not self.inverse)


class FineGrainedViewPackedInt8(ConversionOps):
    """Bitcast packed-FP4 uint8 checkpoint bytes to the int8 view the finegrained modules
    store (same bytes — ``copy_`` into an int8 param would numerically CONVERT and corrupt
    values >= 128). Passes non-uint8 tensors through untouched, so it can ride converters
    whose pattern also matches unquantized modules."""

    def __init__(self, hf_quantizer=None):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, target_patterns=None, **kwargs):
        out = {}
        for key, value in _keyed_by_target(input_dict, target_patterns).items():
            value = value[0] if isinstance(value, list) else value
            out[key] = value.view(torch.int8) if torch.is_tensor(value) and value.dtype == torch.uint8 else value
        return out


class FineGrainedFuseEqualGlobals(ConversionOps):
    """Reduce modelopt's per-half second-level globals to the single per-expert global the
    stacked gate_up GEMM applies: the halves are calibrated together and ship bit-identical
    (verified across the GLM-5.2-NVFP4 checkpoint), so this asserts equality and keeps one.
    Also handles the single-source (down_proj) case, where it just flattens to ``(E,)``."""

    def __init__(self, hf_quantizer=None):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, missing_keys=None, **kwargs):
        stacks = []
        for v in input_dict.values():
            v = torch.stack(v, dim=0) if isinstance(v, list) else v
            stacks.append(v.reshape(-1).float())
        first = stacks[0]
        for other in stacks[1:]:
            if not torch.equal(first, other):
                raise ValueError(
                    f"modelopt per-half globals differ for {full_layer_name} — the stacked "
                    "gate_up GEMM applies one global per expert; this checkpoint needs a "
                    "block-scale refold, which is not implemented."
                )
        return {full_layer_name: first}


class FineGrainedQuantize(ConversionOps):
    """
    A quantization operation that creates two tensors, weight and scale out of a weight.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def _resolve_block_size(self, value: torch.Tensor) -> tuple[int, int]:
        block_size = None
        if self.hf_quantizer.quantization_config is not None:
            if isinstance(self.hf_quantizer.quantization_config, dict):
                block_size = self.hf_quantizer.quantization_config.get("weight_block_size")
            else:
                block_size = getattr(self.hf_quantizer.quantization_config, "weight_block_size", None)
        if block_size is None:
            block_size = (value.shape[-2], value.shape[-1])
        return tuple(block_size)

    def _quantize_one(self, key: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
        # Pass through tensors that aren't tileable (1D norms / biases, or shapes
        # that don't divide cleanly by the configured block) — they were never
        # FP8-quantized on the load side, so the reverse op shouldn't touch them.
        if value.ndim < 2:
            return {key: value}
        block_m, block_n = self._resolve_block_size(value)
        rows, cols = value.shape[-2], value.shape[-1]
        if rows % block_m != 0 or cols % block_n != 0:
            return {key: value}

        # Leading dims can be empty (2D) or include num_experts/... (3D+)
        leading_shape = value.shape[:-2]
        rows_tiles = rows // block_m
        cols_tiles = cols // block_n
        original_shape = value.shape
        value_fp32 = value.to(torch.float32)
        # Reshape to (..., rows_tiles, block_m, cols_tiles, block_n)
        reshaped = value_fp32.reshape(*leading_shape, rows_tiles, block_m, cols_tiles, block_n)
        # Per-tile max-abs over the block dims (block_m at -3, block_n at -1)
        max_abs = reshaped.abs().amax(dim=(-3, -1))
        safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        # We store inverse scale to match the upstream ``weight_scale_inv`` convention
        scales = _FP8_MAX / safe_max_abs
        scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))  # keep zeros stable
        inv_scales = (1.0 / scales).to(torch.float32)
        # ue8m0 stores weight_scale_inv as a power of two. Round it before quantizing and derive the
        # forward scale from it, so dequant multiplies by the exact scale the weight was divided by.
        if self.hf_quantizer.quantization_config.scale_fmt == "ue8m0":
            inv_scales = torch.pow(2.0, torch.ceil(torch.log2(inv_scales.clamp(min=torch.finfo(torch.float32).tiny))))
            inv_scales = inv_scales.to(_get_ue8m0_dtype())
            scales = 1.0 / inv_scales.to(torch.float32)  # forward scale = exact reciprocal of the stored inverse
        # Broadcast scales over the block dims and quantize
        scales_broadcast = scales.unsqueeze(-1).unsqueeze(-3)  # (..., rows_tiles, 1, cols_tiles, 1)
        scaled = reshaped * scales_broadcast
        quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        quantized = quantized.reshape(original_shape)
        scale_key = key.rsplit(".", 1)[0] + ".weight_scale_inv" if key.endswith(".weight") else key + "_scale_inv"
        return {key: quantized, scale_key: inv_scales}

    def convert(self, input_dict: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # every (key, tensor) entry — one key, or one per expert from the reverse of a
        # ``MergeModulelist`` — quantizes the same way
        result: dict[str, torch.Tensor] = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            result.update(self._quantize_one(key, tensor))
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedDequantize(self.hf_quantizer)


class FineGrainedDequantize(ConversionOps):
    """Dequantize FP8 weights using their per-block ``weight_scale_inv``.

    Designed to run as the *first* op in any :class:`WeightConverter` chain when
    loading with ``dequantize=True`` — :meth:`update_weight_conversions` on the
    FP8 quantizer attaches it to each existing model-specific converter so that
    per-expert (weight, scale) pairs are folded into full-precision tensors before
    the chain's merge / concat ops collapse the per-expert structure.

    Pattern semantics
        Input ``input_dict`` carries one entry per source pattern; each value is a
        list of tensors (one per ``*`` match). For every weight pattern that has a
        sibling ``*.weight_scale_inv`` pattern in the dict, this op pairs them up by
        index, dequantizes per-pair, and emits the dequantized list under the
        original *weight* key. Scale entries are dropped from the output so the
        remaining ops only see weights.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def _scale_pattern_for(self, weight_pattern: str) -> str:
        # Strip the optional ``$`` regex anchor so we can match the underlying name.
        anchored = weight_pattern.endswith("$")
        base = weight_pattern[:-1] if anchored else weight_pattern
        if base.endswith(".weight"):
            scale = base[: -len(".weight")] + ".weight_scale_inv"
        elif base == "weight":
            scale = "weight_scale_inv"
        else:
            scale = base + "_scale_inv"
        return scale + "$" if anchored else scale

    # E2M1 (FP4) value table — checkpoints sometimes ship MoE experts as packed FP4
    # (two e2m1 nibbles per int8 byte), so the "weight" dtype lands as ``int8`` /
    # ``float4_e2m1fn_x2`` and we have to unpack before applying the scale grid.
    _FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

    def _unpack_fp4(self, packed: torch.Tensor) -> torch.Tensor:
        """Two ``e2m1`` FP4 values per byte → float32 tensor twice as wide on the last dim."""
        lut = torch.tensor(self._FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
        u8 = packed.contiguous().view(torch.uint8)
        low = (u8 & 0xF).long()
        high = ((u8 >> 4) & 0xF).long()
        unpacked = torch.stack([lut[low], lut[high]], dim=-1)
        return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])

    def _dequantize_one(
        self, quantized: torch.Tensor, scales: torch.Tensor, output_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # FP4 path: int8 / float4_e2m1fn_x2 stores two nibbles per byte. Unpack to fp32
        # first so the rest of the routine sees a normal (rows, cols) float matrix.
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        rows, cols = quantized_fp32.shape[-2:]
        # Derive block size from the scale grid rather than the global config: MoE experts
        # ship MXFP4 with a ``[1, 32]`` block, dense linears ship FP8 with ``[128, 128]``,
        # and the same dequant has to handle both within one checkpoint.
        try:
            scale_rows, scale_cols = scales.shape[-2:]
        except Exception:
            # scale can be a single tensor in extreme cases where it was not wrapped properly but is [1,0].
            scale_rows, scale_cols = 1, 1
        if rows % scale_rows or cols % scale_cols:
            raise ValueError(
                f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols})."
            )
        block_m = rows // scale_rows
        block_n = cols // scale_cols
        # ``ue8m0`` (``float8_e8m0fnu``) scales have no CUDA ``mul`` kernel, and casting
        # the FP8 weight to that dtype loses precision. Promote both sides to fp32 for
        # the math; prefer the destination parameter's dtype when known so eager modules
        # (e.g. plain ``nn.Linear``) keep the model's compute dtype after load.
        if output_dtype is None:
            output_dtype = (
                scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
            )
        # MXFP8 checkpoints ship E8M0 exponents stored as ``torch.uint8`` (one byte per
        # block) — the actual scale is `2 ** (byte - 127)`. Interpreting the raw bytes
        # as scalar multipliers would be silently wrong, so unpack to fp32 here.
        if scales.dtype == torch.uint8:
            s_fp32 = (scales.to(torch.float32) - 127.0).exp2()
        else:
            s_fp32 = scales.to(torch.float32)
        original_shape = quantized_fp32.shape
        q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
        s = s_fp32.reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
        return (q * s).to(output_dtype).reshape(original_shape)

    def _get_target_dtype(self, model: torch.nn.Module | None, full_layer_name: str | None) -> torch.dtype | None:
        if model is None or full_layer_name is None:
            return None
        module, tensor_name = get_module_from_name(model, full_layer_name)
        param = getattr(module, tensor_name, None)
        return getattr(param, "dtype", None)

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor] | torch.Tensor],
        full_layer_name: str | None = None,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        output_dtype = self._get_target_dtype(model, full_layer_name)
        # The dense converter (``["weight$", "weight_scale_inv", "activation_scale"] -> weight``)
        # hands one weight, with or without a scale (RMSNorm weights match ``weight$`` too).
        if "weight$" in input_dict:
            # the loader derives prefix/suffix from the output KEY; without a `full_layer_name`
            # (direct invocation) key it by the converter's target
            target_key = full_layer_name if full_layer_name is not None else "weight"
            quantized = input_dict["weight$"]
            quantized = quantized[0] if isinstance(quantized, list) else quantized
            if "weight_scale_inv" in input_dict:
                scales = input_dict["weight_scale_inv"]
                scales = scales[0] if isinstance(scales, list) else scales
                return {target_key: self._dequantize_one(quantized, scales, output_dtype=output_dtype)}
            return {target_key: quantized}

        # Generic chain path: dequantize every weight pattern that has a sibling scale.
        result: dict[str, list[torch.Tensor] | torch.Tensor] = {}
        for key, value in input_dict.items():
            if "activation_scale" in key or "weight_scale_inv" in key:
                continue  # consumed by the dequant; drop from the chain
            scale_key = self._scale_pattern_for(key)
            if scale_key not in input_dict:
                # No scale to apply (e.g. unrelated entry) — pass through untouched.
                result[key] = value
                continue
            weights = value if isinstance(value, list) else [value]
            scales = input_dict[scale_key]
            scales = scales if isinstance(scales, list) else [scales]
            if len(weights) != len(scales):
                raise ValueError(
                    f"FineGrainedDequantize: weight/scale count mismatch for {key} "
                    f"({len(weights)} weights vs {len(scales)} scales)."
                )
            result[key] = [self._dequantize_one(w, s, output_dtype=output_dtype) for w, s in zip(weights, scales)]
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        # Round-trip: dequantize on load -> re-quantize on save, so the saved
        # checkpoint preserves the FP8 format (weight + per-block ``weight_scale_inv``)
        # whether the in-memory state stayed quantized or was dequantized for compute.
        return FineGrainedQuantize(self.hf_quantizer)
