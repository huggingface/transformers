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
"""Quantized modules and forwards for the fine-grained family — block-FP8, MXFP8, MXFP4, NVFP4
and weight-only — served by `kernels-community/finegrained-kernels`. The checkpoint's layout is
moved onto these modules by `finegrained_conversions`.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..activations import ACT2FN
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import logging
from ..utils.import_utils import is_kernels_available
from .deepgemm import (
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
    deepgemm_fp8_fp4_megamoe_experts_forward,
    is_sm100,
    prefers_deepgemm_linear,
)
from .hub_kernels import _MISSING_KERNELS_MESSAGE, lazy_load_kernel
from .moe import ExpertsInterface, use_experts_implementation


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


def weight_formats() -> dict[str, WeightFormat]:
    return {
        # block-scaled E4M3, block from the quant config, fp32/UE8M0 scale container
        "fp8": WeightFormat(weight_dtype=_FP8_DTYPE),
        # E4M3 values, UE8M0 group-32 scales
        "mxfp8": WeightFormat(weight_dtype=_FP8_DTYPE, scale_dtype=_get_ue8m0_dtype(), scale_group=(1, 32)),
        # packed E2M1 values (2/byte), UE8M0 group-32 scales
        "mxfp4": WeightFormat(
            weight_dtype=torch.int8, values_per_byte=2, scale_dtype=_get_ue8m0_dtype(), scale_group=(1, 32)
        ),
        # packed E2M1 values, E4M3 group-16 block scales, per-tensor/per-expert fp32 global
        "nvfp4": WeightFormat(
            weight_dtype=torch.int8,
            values_per_byte=2,
            scale_dtype=_FP8_DTYPE,
            scale_group=(1, 16),
            global_scale_dtype=torch.float32,
        ),
    }


def resolve_weight_format(
    weight_format: str,
    scale_fmt: str = "float",
    block_size: tuple[int, int] | None = None,
) -> tuple[WeightFormat, torch.dtype, tuple[int, int] | None]:
    """``(format, scale_dtype, (sf_gran_n, sf_gran_k))`` for one format name, with the config's
    ``scale_fmt``/``weight_block_size`` filling the slots the format leaves open."""
    formats = weight_formats()
    if weight_format not in formats:
        raise ValueError(f"unknown weight_format {weight_format!r}; expected one of {sorted(formats)}")
    format_spec = formats[weight_format]
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


def _holds_swizzled_scales(config, format_spec: WeightFormat, activation_format: str | None) -> bool:
    """Whether an experts module holds its block scales in the ``SWIZZLE_32_4_4`` layout the
    Blackwell tcgen05 scaled-MMA reads directly (row-major forces a per-tile gather). Needs SM100,
    a triton dispatch (the DeepGEMM backends read affine scales), a group-scaled format and a chain
    that quantizes activations. Block-FP8 reaches the scaled-MMA under UE8M0 by broadcasting one
    128-block scalar in-register, so it has no per-group grid to lay out."""
    return (
        getattr(config, "_experts_implementation", None) not in ("deepgemm", "deepgemm_megamoe")
        and format_spec.scale_group is not None
        and activation_format != "bf16"
        and is_sm100()
    )


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

    Dispatch order — both backends handle FP8 and FP4 weights with fp32 or UE8M0 scales:
      1. DeepGEMM (`deepgemm_fp8_fp4_linear`) where it can actually run and is faster — the
         pre-SM100 shapes it supports (FP4, UE8M0 SFs, 128×128 block FP8). Never on SM100,
         where Triton is tuned per shape and is the only path reading pre-swizzled scales.
      2. Triton finegrained fallback everywhere else: SM100, an ``activation_scale``
         (DeepGEMM is dynamic-only), or any shape DeepGEMM declined.

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
        weight,
        weight_scale_inv,
        block_size=block_size,
        activation_scale=activation_scale,
        weight_global_scale=weight_global_scale,
        input_global_scale=input_global_scale,
        activation_format=activation_format,
        allow_deepgemm=allow_deepgemm,
    ):
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

    # Triton arm: fused act-quant + matmul, then optional bias add. `activation_scale` None
    # quantizes inline per token, a scalar quantizes statically against it; the two NVFP4 globals
    # both come back on the accumulator.
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
    ):
        super().__init__(in_features, out_features)

        self.has_bias = has_bias
        self.weight_format = weight_format
        self.block_size = block_size
        self.activation_scheme = activation_scheme
        self.activation_format = activation_format
        # the format table decides storage: value dtype (packed E2M1 = 2 values per int8 byte
        # along K), scale dtype/granularity, and whether a two-level global exists
        format_spec, sf_dtype, scale_group = resolve_weight_format(weight_format, scale_fmt, block_size)
        # the format's own dtype both decides whether there is a second level and says what it
        # is held in
        global_scale_dtype = format_spec.global_scale_dtype
        in_storage = in_features // format_spec.values_per_byte
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_storage, dtype=format_spec.weight_dtype),
            requires_grad=format_spec.weight_dtype.is_floating_point,
        )
        # NVFP4 two-level: the per-tensor fp32 global the kernel recovers on the accumulator
        _set_optional_parameter(
            self,
            "weight_global_scale",
            torch.tensor(1.0, dtype=global_scale_dtype) if global_scale_dtype is not None else None,
        )

        # no group and no block: one per-tensor scale
        scale = (
            torch.tensor(1.0, dtype=torch.float32)
            if scale_group is None
            else torch.empty(_cdiv(out_features, scale_group[0]), _cdiv(in_features, scale_group[1]), dtype=sf_dtype)
        )
        self.weight_scale_inv = nn.Parameter(scale, requires_grad=scale.dtype.is_floating_point)

        static = self.activation_scheme == "static"
        _set_optional_parameter(self, "activation_scale", torch.tensor(1.0, dtype=torch.float32) if static else None)
        _set_optional_parameter(self, "bias", torch.empty(self.out_features) if self.has_bias else None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return finegrained_linear(
            input,
            self.weight,
            self.weight_scale_inv,
            block_size=self.block_size,
            activation_scale=self.activation_scale,
            bias=self.bias,
            allow_deepgemm=not self._deepgemm_disabled,
            weight_global_scale=self.weight_global_scale,
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

        w = self.weight.view(self.n_groups, -1, hidden_dim)
        scale_inv = self.weight_scale_inv
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
            activation_format=self.activation_format,
            b_global_scale=self.weight_global_scale,
        )
        y = y.reshape(self.n_groups, *input_shape, -1).movedim(0, -2)
        if self.has_bias:
            y.add_(self.bias.view(self.n_groups, -1))
        return y


def _moe_operands(kernel, module) -> dict:
    """Everything the kernels' MoE forwards take from the module: the two projections with their
    scales (held swizzled for dot_scaled chains), the per-expert globals and biases,
    and the activation — a ``get_supported_act_fns()`` name the gate_up epilogue fuses, else the module's
    own GLU as a callable the kernels run on the host, so any activation works without a kernel
    change."""
    up = "gate_up_proj" if module.has_gate else "up_proj"

    act_name = module.act_fn_name
    fused = act_name in kernel.get_supported_act_fns() and (module.swiglu_alpha is None or act_name == "silu")
    if fused:
        act_fn = act_name
    else:
        act_fn = module._apply_gate if module.has_gate else module.act_fn

    # a `get_supported_norms()` NAME is folded into the routing-weighted reduce; any other
    # callable runs on the routed rows before it. The name is only usable where nothing has to
    # happen in between: fusing puts the norm INSIDE the launch that produced the rows, so under
    # intra-expert TP — where they are a partial sum over the sharded intermediate — the
    # all-reduce would have nowhere to land, and the module's wrapped forward must do it first.
    post_expert_norm, norm_weight, norm_eps = None, None, 1e-6
    if module.has_post_expert_norm:
        fusable = module.post_expert_norm_name in kernel.get_supported_norms()
        if not fusable or getattr(module.post_expert_norm, "_hf_tp_input_reduce", False):
            post_expert_norm = module._apply_post_norm
        else:
            norm = module.post_expert_norm
            post_expert_norm, norm_weight = module.post_expert_norm_name, norm.weight
            # the model owns this module, so take epsilon under either name; `nn.RMSNorm` leaves
            # it None and resolves to the dtype's own
            eps = _first_attr(norm, "eps", "variance_epsilon", raise_error=False)
            norm_eps = float(eps) if eps is not None else torch.finfo(norm.weight.dtype).eps

    # The kernels always say `gate_up_proj`; the module names that weight `up_proj` when the model
    # has no gate, which is the only reason these are not plain attribute reads. A slot the module
    # does not hold reads back as None — the argument the kernels take for an absent one.
    return {
        **{
            f"{kernel_proj}{slot}": getattr(module, f"{held_proj}{slot}")
            for kernel_proj, held_proj in (("gate_up_proj", up), ("down_proj", "down_proj"))
            for slot in ("", "_scale_inv", "_weight_global_scale", "_input_global_scale", "_activation_scale", "_bias")
        },
        # a supported activation NAME is fused into the gate_up epilogue; any other callable
        # leaves that GEMM plain and runs on the host between the two
        "act_fn": act_fn,
        "gate": module.has_gate,
        "swiglu_alpha": module.swiglu_alpha,
        "swiglu_limit": module.swiglu_limit,
        "activation_format": module.activation_format,
        # a supported norm NAME is folded into the routing-weighted reduce, one pass ahead of it
        # computing the rows' rsqrt; any other callable is applied to the routed rows before it
        "post_expert_norm": post_expert_norm,
        "post_expert_norm_weight": norm_weight,
        "post_expert_norm_eps": norm_eps,
    }


def _fused_experts_forward(module, kernel_forward: str, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """The kernels' fused MoE chain (gate_up with the fused GLU epilogue and intermediate requant
    where supported -> down -> the routing-weighted top-k reduce) over the module's tensors."""
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
    # Per-`_experts_implementation` rewrite of parallel-layer kinds in the TP/EP plan. The default
    # kind is impl-agnostic; some impls need a distinct one (megamoe needs no gradient-sync hooks
    # and an EP `process_group` injection). Declared here so the quantizer does not have to know
    # about impl-specific TP needs — extend when adding an impl.
    _impl_tp_layer_overrides: dict[str, dict[str, str]] = {
        "deepgemm_megamoe": {"moe_tp_experts": "megamoe_experts", "ep_router": "megamoe_router"},
    }

    def __init__(
        self,
        config,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        scale_fmt: str = "float",
        has_bias: bool = False,
        has_gate: bool = True,
        weight_format: str = "fp8",
        activation_format: str | None = None,
        is_concatenated: bool = True,
    ):
        super().__init__()

        self.config = config
        self.has_bias = has_bias
        self.has_gate = has_gate
        self.block_size = block_size
        # The model's own gate|up row order (transformers' experts flag): stacked ``[gate; up]``
        # or, like GPT-OSS, already the kernels' interleaved ``[g0, u0, ...]``. It describes the
        # CHECKPOINT only — what this module holds is `holds_interleaved_gate_up` below.
        self.is_concatenated = is_concatenated
        self.activation_format = activation_format
        self.activation_scheme = activation_scheme
        self.hidden_dim = _first_attr(config, "moe_hidden_size", "hidden_size")
        self.num_experts = _first_attr(config, "num_local_experts", "num_experts")
        self.intermediate_dim = _first_attr(config, "moe_intermediate_size", "intermediate_size")
        self.act_fn_name = _first_attr(config, "hidden_activation", "hidden_act")
        self.swiglu_alpha = getattr(config, "swiglu_alpha", None)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.act_fn = ACT2FN[self.act_fn_name]

        # This module's format, as its group settled it. Kept because `FineGrainedQuantize` reads
        # it back: on-the-fly quantization asks the MODULE what to quantize to, never the config,
        # which is what lets a mixed checkpoint (DeepSeek-V4: mxfp4 experts over block-FP8
        # linears) quantize each group to its own format from one `from_pretrained`.
        self.weight_format = weight_format
        format_spec, scale_dtype, scale_group = resolve_weight_format(weight_format, scale_fmt, block_size)
        # the format decides the second level: None means there is none, else the dtype it is in
        self.global_scale_dtype = format_spec.global_scale_dtype

        # What the module HOLDS, as the loader's conversion ops deliver it (and their reverses
        # restore for the checkpoint): gate|up rows in the kernels' interleaved order unless the
        # backend packs gate|up itself (DeepGEMM Mega MoE), block scales swizzled where the
        # scaled-MMA reads them. `is_concatenated` above is the checkpoint's own order: the
        # loader interleaves where the two disagree, and everything at runtime reads this one.
        # FROZEN at load, not a property over `_experts_implementation`: it describes the bytes
        # the tensors now hold, so re-deriving it would claim a layout they do not have.
        impl = getattr(config, "_experts_implementation", None)
        self.holds_interleaved_gate_up = self.has_gate and impl != "deepgemm_megamoe"
        swizzled = _holds_swizzled_scales(config, format_spec, activation_format)

        up_name = "gate_up_proj" if self.has_gate else "up_proj"
        up_rows = (2 if self.has_gate else 1) * self.intermediate_dim
        # gate_up's activation global is ONE value — its rows are the pre-routing hidden states,
        # quantized once before routing; down's rows belong to an expert each.
        for proj, rows, in_dim, min_scale_out, input_globals in (
            (up_name, up_rows, self.hidden_dim, 2 if self.has_gate else 1, 1),
            ("down_proj", self.hidden_dim, self.intermediate_dim, 1, self.num_experts),
        ):
            self._register_projection(
                proj,
                rows,
                in_dim,
                min_scale_out,
                input_globals,
                (format_spec, scale_dtype, scale_group, swizzled),
            )

        # The model's per-expert output norm, carried over by the swap. A submodule slot, not a
        # class attribute: a class attribute of the same name shadows what
        # `nn.Module.__setattr__` registers, so the norm would never be read back.
        self.post_expert_norm = None
        self.has_post_expert_norm = False
        # The form it implements, when the model names one it wants a backend to FUSE
        # (`get_supported_norms`); unnamed norms still run through `_apply_post_norm`.
        self.post_expert_norm_name = None

    def _register_projection(self, proj, rows, in_dim, min_scale_out, input_globals, storage):
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
        # down to that). The activation's is the checkpoint's `input_scale`.
        two_level = self.global_scale_dtype is not None
        calibrated = two_level and self.activation_format != "bf16"
        for name, tensor in (
            ("bias", bias),
            (
                "weight_global_scale",
                torch.ones(self.num_experts, dtype=self.global_scale_dtype) if two_level else None,
            ),
            ("input_global_scale", torch.ones(input_globals, dtype=torch.float32) if calibrated else None),
            (
                "activation_scale",
                torch.ones(self.num_experts, dtype=torch.float32) if self.activation_scheme == "static" else None,
            ),
        ):
            _set_optional_parameter(self, f"{proj}_{name}", tensor)

    def _apply_post_norm(self, expert_out: torch.Tensor) -> torch.Tensor:
        """The standard hook every backend applies: the model's norm on one expert application's
        rows, before the routing weights."""
        return self.post_expert_norm(expert_out)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # interleaved rows -> alternating output columns, the same split the fused epilogue does
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
            if self.has_post_expert_norm:
                proj_out = self._apply_post_norm(proj_out)
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
        scale = getattr(self, f"{proj}_scale_inv")
        weight_globals = getattr(self, f"{proj}_weight_global_scale")
        input_globals = getattr(self, f"{proj}_input_global_scale")
        weight_global = None if weight_globals is None else weight_globals[expert_idx]
        return finegrained_linear(
            input,
            weight,
            scale[expert_idx : expert_idx + 1] if scale.ndim == 5 else scale[expert_idx],
            self.block_size,
            bias=bias,
            activation_scale=activation_scale,
            allow_deepgemm=not self._deepgemm_disabled,
            weight_global_scale=weight_global,
            input_global_scale=(
                None
                if input_globals is None or self.activation_format == "bf16"
                else input_globals.reshape(-1)[expert_idx if input_globals.numel() > 1 else 0].reshape(1)
            ),
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


def raise_if_unquantized(model: nn.Module) -> None:
    """Fail the load if a finegrained module came out of it holding a full-precision weight.

    The only way to get one is a checkpoint that targets a module and then ships no scales for
    it, which the swap cannot see in advance. There is no forward for that module — the kernels
    take a byte-wide weight — so the run is already over; raising here names the module instead
    of leaving an unreadable error to come out of a kernel launch later.
    """
    unquantized = [
        f"{name}.{attr}"
        for name, module in model.named_modules()
        if isinstance(module, _FineGrainedModule)
        for attr in ("weight", "gate_up_proj", "up_proj", "down_proj")
        if getattr(module, attr, None) is not None and getattr(module, attr).element_size() > 1
    ]
    if unquantized:
        shown = ", ".join(unquantized[:3]) + (f" (+{len(unquantized) - 3} more)" if len(unquantized) > 3 else "")
        raise ValueError(
            f"{len(unquantized)} module(s) were converted for quantized compute but hold "
            f"full-precision weights: {shown}. This checkpoint targets them without shipping "
            "their scales; add them to the quantization config's `ignore` list to keep them "
            "in full precision."
        )


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

    def storage_for(module_name: str) -> dict:
        """The format THIS module is quantized in. A single-format checkpoint has one group
        covering everything; DeepSeek-V4 is mxfp4 experts over block-FP8 linears, so its experts
        and its attention projections resolve differently."""
        group = quantization_config.group_for(module_name)
        return {
            "weight_format": group.quant_method,
            "activation_format": group.activation_format,
            "block_size": group.weight_block_size,
            "activation_scheme": group.activation_scheme,
            "scale_fmt": group.scale_fmt,
        }

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        new_module = None
        with torch.device("meta"):
            if module_name.endswith(".experts"):
                # the decorator stamps these on the instance after __init__, so the model's own
                # gate|up convention has to travel through it as well as into the ctor
                flags = {
                    "has_gate": getattr(module, "has_gate", True),
                    "has_bias": getattr(module, "has_bias", False),
                    "is_concatenated": getattr(module, "is_concatenated", True),
                }
                new_class = use_experts_implementation(
                    experts_class=FineGrainedExperts,
                    experts_interface=ALL_FINEGRAINED_EXPERTS_FUNCTIONS,
                    **flags,
                )
                new_module = new_class(
                    config=getattr(module, "config", model.config.get_text_config()),
                    **storage_for(module_name),
                    **flags,
                )
                # A per-expert output norm belongs to the model, not the quantization, so the
                # swap carries the submodule over; `_apply_post_norm` is the standard hook and
                # needs no rebinding. A named form is what a backend can fuse.
                if getattr(module, "post_expert_norm", None) is not None:
                    new_module.post_expert_norm = module.post_expert_norm
                    # `use_experts_implementation` owns this flag on every experts class and sets
                    # it from the class declaration, so the swap updates it rather than deriving it
                    new_module.has_post_expert_norm = True
                    new_module.post_expert_norm_name = getattr(module, "post_expert_norm_name", None)
            elif type(module) is nn.Linear:
                new_module = FineGrainedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    has_bias=module.bias is not None,
                    **storage_for(module_name),
                )
            elif isinstance(module, nn.Linear) and hasattr(module, "n_groups"):
                # Block-diagonal grouped linear (DSv4's `DeepseekV4GroupedLinear`), recognised by
                # the attribute the swap needs rather than by its class NAME: a plain
                # `FineGrainedLinear` would collapse the groups into one giant linear and yield
                # the wrong output dim.
                storage = storage_for(module_name)
                new_module = FineGrainedGroupedLinear(
                    in_features_per_group=module.in_features,
                    out_features=module.out_features,
                    n_groups=module.n_groups,
                    has_bias=module.bias is not None,
                    **{k: storage[k] for k in ("block_size", "activation_scheme", "scale_fmt")},
                )
            if new_module is not None:
                # The kernels take raw pointers, so every operand must be a plain tensor. This is
                # what tells the TP layer to hand local shards rather than DTensors
                # (`should_use_local_tensors`); without it a dense linear sees DTensors whenever
                # grad is enabled, since its inference fast path only covers `not is_grad_enabled`.
                new_module._hf_quantized_needs_local_tp = True
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading a fine-grained quantized model but no linear or experts modules were "
            "found to convert. Please double check your model architecture."
        )
    return model
