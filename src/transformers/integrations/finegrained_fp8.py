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
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps, _IdentityOp
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
from .moe import ExpertsInterface, _grouped_mm, use_experts_implementation


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
class FineGrainedFP8:
    """Entry points exposed by the `kernels-community/finegrained-fp8` Triton kernel."""

    matmul: Callable
    batched_matmul: Callable
    grouped_matmul: Callable


# Cache the loaded kernel but not failures: re-checking each call is cheap and intended, since the env
# can change between attempts. A module global (not `@functools.cache`) avoids Dynamo warning about
# tracing a cache-wrapped function on every compile.
_FINEGRAINED_FP8: FineGrainedFP8 | None = None


@torch._dynamo.allow_in_graph
def _load_finegrained_fp8_kernel() -> None:
    """
    Load the finegrained-fp8 Triton kernel once into the `_FINEGRAINED_FP8` module global.

    `@allow_in_graph` makes `torch.compile` treat the untraceable hub download + dynamic import as a
    single opaque node instead of tracing into it; it returns `None` (proxyable) and populates the
    global, which `load_finegrained_fp8_kernel` then returns.

    Under NO circumstances may this function return a value: an `@allow_in_graph` fx node's
    return must be proxyable, and returning the bundle (e.g. from the warm-cache
    short-circuit) breaks torch.compile with `Unsupported: torch.* op returned non-Tensor`.

    Raises `ImportError` if the `kernels` package is missing, or the kernel or required
    symbols cannot be found.
    """
    global _FINEGRAINED_FP8
    if _FINEGRAINED_FP8 is not None:
        return

    if not is_kernels_available():
        raise ImportError(f"finegrained-fp8 kernel unavailable: {_MISSING_KERNELS_MESSAGE}")

    kernel = lazy_load_kernel("finegrained-fp8")
    if kernel is None:
        raise ImportError(
            "Failed to load the finegrained-fp8 kernel — check that `kernels-community/finegrained-fp8` "
            "has a build matching the current torch/CUDA."
        )

    matmul = getattr(kernel, "matmul_2d", None)
    batched_matmul = getattr(kernel, "matmul_batched", None)
    grouped_matmul = getattr(kernel, "matmul_grouped", None)

    missing = [
        name
        for name, attr in [
            ("matmul_2d", matmul),
            ("matmul_batched", batched_matmul),
            ("matmul_grouped", grouped_matmul),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"finegrained-fp8 kernel is missing required symbols: {', '.join(missing)}. {_MISSING_KERNELS_MESSAGE}"
        )

    _FINEGRAINED_FP8 = FineGrainedFP8(
        matmul=matmul,
        batched_matmul=batched_matmul,
        grouped_matmul=grouped_matmul,
    )


def load_finegrained_fp8_kernel() -> FineGrainedFP8:
    _load_finegrained_fp8_kernel()
    return _FINEGRAINED_FP8


def _cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def _alloc_expert_proj(
    num_experts: int,
    proj_out: int,
    proj_in: int,
    weight_dtype: torch.dtype,
    sf_dtype: torch.dtype,
    weight_k_div: int = 1,
    sf_gran_n: int | None = None,
    sf_gran_k: int | None = None,
    min_sf_out: int = 1,
) -> tuple[nn.Parameter, nn.Parameter]:
    """Allocate `(weight, weight_scale_inv)` parameters for one expert projection.

    `weight_k_div` halves the K dim for FP4-packed storage (2 e2m1 values per byte).
    `sf_gran_n` / `sf_gran_k` set per-block (None → per-row/per-tensor) SF granularity.
    `min_sf_out` floors the SF tensor's output dim — used by the fused gate_up
    projection to keep room for both halves (pass `2`) even when `proj_out < sf_gran_n`
    would otherwise collapse the SF dim to 1.
    """
    weight_t = torch.empty(num_experts, proj_out, proj_in // weight_k_div, dtype=weight_dtype)
    weight = nn.Parameter(weight_t, requires_grad=weight_t.is_floating_point())
    sf_out = max(_cdiv(proj_out, sf_gran_n) if sf_gran_n is not None else 1, min_sf_out)
    sf_in = _cdiv(proj_in, sf_gran_k) if sf_gran_k is not None else 1
    sf_t = torch.empty(num_experts, sf_out, sf_in, dtype=sf_dtype)
    sf = nn.Parameter(sf_t, requires_grad=sf_t.is_floating_point())
    return weight, sf


def finegrained_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton FP8/FP4 linear: fused act-quant + matmul, then optional bias add.

    ``activation_scale=None`` → dynamic per-K-block scales (inline); set it for
    static per-tensor quant. ``weight_scale_inv`` accepts fp32 or UE8M0; the
    dispatcher routes FP4 (``int8``-packed) weights automatically. ``output_dtype``
    defaults to ``input.dtype`` when left ``None``.
    """
    finegrained_fp8 = load_finegrained_fp8_kernel()
    output = finegrained_fp8.matmul(
        input,
        weight,
        weight_scale_inv,
        block_size,
        input.dtype,
        activation_scale=activation_scale,
    )
    if bias is not None:
        output.add_(bias)
    return output


def fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    allow_deepgemm: bool = True,
) -> torch.Tensor:
    """End-to-end FP8/FP4 linear used by `FP8Linear` and the eager `FP8Experts` loop.

    Dispatch order — both backends handle FP8 and FP4 weights with fp32 or UE8M0 scales:
      1. DeepGEMM (`deepgemm_fp8_fp4_linear`) — 3-6× faster on the shapes it supports.
         Preferred for FP4, UE8M0 SFs, and 128×128 block FP8.
      2. Triton finegrained-fp8 fallback — used when DeepGEMM is unavailable, when the
         caller passes ``activation_scale`` (DeepGEMM is dynamic-only), or for any
         shape DeepGEMM declined.

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
            to a single CUDA context and produce garbage across devices (see the multi-device guard
            in ``quantizer_finegrained_fp8.py``).
    """
    # DeepGEMM is CUDA-only, dynamic-only, SM90+ only, FP4/FP8-block-128-only. On SM100 its FP8 GEMM only
    # consumes UE8M0 scales; float32 scales would be ceil-rounded to UE8M0 without requantizing and
    # silently corrupt the output (#47030), so we skip DeepGEMM up front for that combo rather than
    # attempt-then-fall-back every call — `_assert_sm100_requirements` still guards it as a backstop.
    # ``TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`` forces the Triton fallback for this single dispatcher (the
    # experts ``"deepgemm"`` impl is unaffected — use ``set_experts_implementation`` for that). Used by the
    # FP8 MoE batched_mm / grouped_mm paths to avoid a still-unexplained DeepGEMM-vs-Triton interaction that
    # degrades end-to-end generation on B200 (per-row kernel outputs still measure bit-perfect, but final
    # tokens drift; not reproducible with the DeepGEMM linear off).
    deepgemm_preferred = (
        allow_deepgemm
        and is_deepgemm_loadable()
        and activation_scale is None
        and not (is_sm100() and weight_scale_inv.dtype == torch.float32)
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

    return finegrained_fp8_linear(input, weight, weight_scale_inv, block_size, bias, activation_scale)


def _dequantize_fp8_weight(
    weight: torch.Tensor, weight_scale_inv: torch.Tensor, block_size: list[int] | None, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize `(..., N, K)` FP8 weights with their per-block (or per-tensor) scales, for the backward pass."""
    if weight.dtype == torch.int8:
        raise NotImplementedError("Backward through FP4 weights is not supported.")
    scale = weight_scale_inv.to(dtype)
    weight = weight.to(dtype)
    if block_size is None:
        return weight.mul_(scale)
    rows, cols = weight.shape[-2:]
    full, rem = divmod(cols, block_size[1])
    # Only expand the scales over rows, and broadcast them over columns through a view of the weight: expanding them
    # over both would materialize a second weight-sized tensor
    scale = scale.repeat_interleave(block_size[0], dim=-2)[..., :rows, :]
    weight[..., : full * block_size[1]].unflatten(-1, (full, block_size[1])).mul_(scale[..., :full, None])
    if rem:
        weight[..., full * block_size[1] :].mul_(scale[..., full:])
    return weight


class FP8LinearFunction(torch.autograd.Function):
    """
    `fp8_linear` with a backward for the input, so adapters (e.g. LoRA) can train on top of FP8 layers. The FP8 weights
    are frozen: the input gradient goes through the dequantized weight, and the weight, scales and bias get none.
    """

    @staticmethod
    def forward(ctx, input, weight, weight_scale_inv, block_size, bias, activation_scale, allow_deepgemm):
        ctx.save_for_backward(weight, weight_scale_inv)
        ctx.block_size = block_size
        return fp8_linear(input, weight, weight_scale_inv, block_size, bias, activation_scale, allow_deepgemm)

    @staticmethod
    def backward(ctx, grad_output):
        weight, weight_scale_inv = ctx.saved_tensors
        weight = _dequantize_fp8_weight(weight, weight_scale_inv, ctx.block_size, grad_output.dtype)
        return grad_output @ weight, None, None, None, None, None, None


# WP glue (bench only), off by default: the grouped kernel's tile layout (7 tiny kernels) is computed once per
# `tokens_per_expert` tensor instead of once per call (the up and down projections of a layer share it), and the
# expert outputs are scattered back to token order instead of building the inverse permutation and gathering
_GLUE_EXPERTS = os.environ.get("GLUE_EXPERTS", "0") == "1"
_GLUE_TILE_LAYOUT = {}


def _glue_memoize_tile_layout(grouped_matmul):
    import sys

    kernel_module = sys.modules[grouped_matmul.__module__]
    original = kernel_module._grouped_tile_layout
    if getattr(original, "_glue_memoized", False):
        return

    def memoized(tokens_per_expert, block_size_m, S, num_experts):
        if not _GLUE_EXPERTS:
            return original(tokens_per_expert, block_size_m, S, num_experts)
        key = (id(tokens_per_expert), block_size_m, S, num_experts)
        if _GLUE_TILE_LAYOUT.get("key") != key:
            _GLUE_TILE_LAYOUT["key"] = key
            _GLUE_TILE_LAYOUT["ref"] = tokens_per_expert  # keeps the id from being reused
            _GLUE_TILE_LAYOUT["value"] = original(tokens_per_expert, block_size_m, S, num_experts)
        return _GLUE_TILE_LAYOUT["value"]

    memoized._glue_memoized = True
    kernel_module._grouped_tile_layout = memoized


# [scratch, bench only] FUSE_COMPILE_OPAQUE_EXPERTS=1: under torch.compile the hub's grouped kernel stays one opaque op
# with a fake implementation. As a `triton_op` it is inlined into every compiled artifact, where its autotuner
# benchmarks its configs again (about 80 s per artifact at GLM-5.3's expert shapes)
_FUSE_OPAQUE_GROUPED = os.environ.get("FUSE_COMPILE_OPAQUE_EXPERTS", "0") == "1"


def _fuse_grouped_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    return load_finegrained_fp8_kernel().grouped_matmul(
        input, weight, weight_scale_inv, offsets=offsets, tokens_per_expert=tokens_per_expert, block_size=block_size
    )


def _fuse_grouped_matmul_fake(input, weight, weight_scale_inv, offsets, tokens_per_expert, block_size):
    return input.new_empty((input.shape[0], weight.shape[1]))


torch.library.custom_op("transformers::fuse_fp8_grouped_matmul", _fuse_grouped_matmul, mutates_args=())
torch.library.register_fake("transformers::fuse_fp8_grouped_matmul", _fuse_grouped_matmul_fake)


def fp8_grouped_matmul(input, weight, weight_scale_inv, offsets, tokens_per_expert, block_size):
    if _FUSE_OPAQUE_GROUPED and block_size is not None:
        return torch.ops.transformers.fuse_fp8_grouped_matmul(
            input, weight, weight_scale_inv, offsets, tokens_per_expert, list(block_size)
        )
    if _GLUE_EXPERTS:
        _glue_memoize_tile_layout(load_finegrained_fp8_kernel().grouped_matmul)
    return load_finegrained_fp8_kernel().grouped_matmul(
        input,
        weight,
        weight_scale_inv,
        offsets=offsets,
        tokens_per_expert=tokens_per_expert,
        block_size=block_size,
    )


class FP8GroupedMatmulFunction(torch.autograd.Function):
    """The finegrained-fp8 grouped matmul with a backward for the input, like `FP8LinearFunction`."""

    # Max bytes of experts dequantized at once in backward. Stacks under it (e.g. Qwen3-30B-A3B: 0.8 GB) go in one shot
    # with no host sync; bigger ones are chunked (DeepSeek-V3 up proj: 28.9 GiB peak -> 2.8 GiB, same time)
    dequant_budget_bytes = 2**30

    @staticmethod
    def forward(ctx, input, weight, weight_scale_inv, offsets, tokens_per_expert, block_size):
        ctx.save_for_backward(weight, weight_scale_inv, offsets)
        ctx.block_size = block_size
        return fp8_grouped_matmul(input, weight, weight_scale_inv, offsets, tokens_per_expert, block_size)

    @staticmethod
    def backward(ctx, grad_output):
        weight, weight_scale_inv, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        num_experts = weight.size(0)
        expert_bytes = weight[0].numel() * grad_output.element_size()
        chunk = max(1, FP8GroupedMatmulFunction.dequant_budget_bytes // expert_bytes)
        if chunk >= num_experts:
            w = _dequantize_fp8_weight(weight, weight_scale_inv, ctx.block_size, grad_output.dtype)
            # (S, output_dim) @ grouped (num_experts, output_dim, input_dim) -> (S, input_dim)
            return _grouped_mm(grad_output, w, offs=offsets), None, None, None, None, None

        ends = offsets.tolist()
        # zeros also cover the EP sentinel rows past `ends[-1]`
        grad_input = grad_output.new_zeros(grad_output.size(0), weight.size(-1))
        for start in range(0, num_experts, chunk):
            end = min(start + chunk, num_experts)
            lo, hi = ends[start - 1] if start else 0, ends[end - 1]
            if lo == hi:
                continue
            w = _dequantize_fp8_weight(
                weight[start:end], weight_scale_inv[start:end], ctx.block_size, grad_output.dtype
            )
            grad_input[lo:hi] = _grouped_mm(grad_output[lo:hi], w, offs=offsets[start:end] - lo)
            # free this chunk before the next one is dequantized, so only one is alive at a time
            del w
        return grad_input, None, None, None, None, None


def _apply_if_grad(function, direct, input, *args):
    """
    Only goes through the autograd `function` when the input needs a gradient: its dispatch alone costs ~6% of a
    Qwen3-30B-A3B-FP8 decode step
    """
    if torch.is_grad_enabled() and input.requires_grad:
        return function.apply(input, *args)
    return direct(input, *args)


# --- PROTOTYPE (bench only, not for upstream): faster dense FP8 linears, off unless PROTO_FP8_LINEAR is set -----------
# PROTO_FP8_LINEAR=bf16      dequantize the dense FP8 linears to bf16 once at load, then plain `F.linear` (cuBLAS)
# PROTO_FP8_LINEAR=deepgemm  requantize them to power-of-two (UE8M0) block scales at load, as vLLM does, then one Triton
#                            activation-quant launch + one DeepGEMM launch per call, weight scales laid out once
# PROTO_FP8_LINEAR=splitk    keep the checkpoint's weights and fp32 scales, split-K Triton GEMM on pre-quantized activations
_PROTO_FP8_LINEAR = os.environ.get("PROTO_FP8_LINEAR", "")
_proto_deepgemm = None


def _proto_convert_fp8_linears(model: nn.Module) -> None:
    """Load-time half of the `PROTO_FP8_LINEAR` prototypes. Only plain block-128 `FP8Linear` modules with dynamic
    activations are touched; experts and grouped linears keep their path."""
    global _proto_deepgemm
    if _PROTO_FP8_LINEAR not in ("bf16", "deepgemm"):
        return
    from ..distributed.parallel_utils import _use_local_dtensor_params

    if _PROTO_FP8_LINEAR == "deepgemm":
        import importlib

        _proto_deepgemm = importlib.import_module(os.environ.get("PROTO_DEEPGEMM_MODULE", "vllm.third_party.deep_gemm"))
        if os.environ.get("PROTO_DEEPGEMM_PDL", "0") == "1":
            _proto_deepgemm.set_pdl(True)
    converted = 0
    with torch.no_grad():
        for module in model.modules():
            if type(module) is not FP8Linear or module.block_size is None or tuple(module.block_size) != (128, 128):
                continue
            # under tensor parallel the parameters are DTensors: work on this rank's shard. The context wraps the
            # parameters replaced inside it back into DTensors with their placements.
            with _use_local_dtensor_params(module):
                weight, scale = module.weight, module.weight_scale_inv
                if weight.dtype != _FP8_DTYPE or scale.dtype != torch.float32 or module.activation_scale is not None:
                    continue
                if weight.device.type != "cuda":
                    continue
                rows, cols = weight.shape
                if _PROTO_FP8_LINEAR == "bf16":
                    expanded = scale.repeat_interleave(128, dim=0)[:rows].repeat_interleave(128, dim=1)[:, :cols]
                    module.weight = nn.Parameter((weight.float() * expanded).to(torch.bfloat16), requires_grad=False)
                    converted += 1
                    continue
                # DeepGEMM's own limits for the plain fp8 GEMM (same gate as vLLM's `should_use_deepgemm_for_fp8_linear`)
                if not is_sm100() or rows % 64 != 0 or cols % 128 != 0:
                    continue
                # round every block scale up to a power of two and re-round the e4m3 values on the new grid (lossy)
                bits = scale.contiguous().view(torch.int32)
                scale_pow2 = ((bits + ((1 << 23) - 1)) & ~((1 << 23) - 1)).view(torch.float32)
                ratio = (scale / scale_pow2).repeat_interleave(128, dim=0)[:rows].repeat_interleave(128, dim=1)[:, :cols]
                module.weight = nn.Parameter((weight.float() * ratio).to(_FP8_DTYPE), requires_grad=False)
                module.weight_scale_inv = nn.Parameter(scale_pow2.to(_get_ue8m0_dtype()), requires_grad=False)
                # per-row packed UE8M0 scales in DeepGEMM's MN-major TMA-aligned layout, built once
                module._proto_sfb = _proto_deepgemm.transform_sf_into_required_layout(
                    scale_pow2, rows, cols, [1, 128, 128], None, False, False
                )
                converted += 1
    logger.warning(f"PROTO_FP8_LINEAR={_PROTO_FP8_LINEAR}: converted {converted} FP8Linear modules")


def _proto_fp8_linear_forward(module: "FP8Linear", input: torch.Tensor) -> torch.Tensor:
    from ._proto_fp8_kernels import act_quant_packed_ue8m0, triton_splitk_linear

    input_2d = input.reshape(-1, input.shape[-1])
    out_features = module.weight.shape[0]  # this rank's shard under tensor parallel, not `module.out_features`
    if module._proto_sfb is not None:
        quantized, scales = act_quant_packed_ue8m0(input_2d)
        output = torch.empty(input_2d.shape[0], out_features, dtype=input.dtype, device=input.device)
        _proto_deepgemm.fp8_fp4_gemm_nt((quantized, scales), (module.weight, module._proto_sfb), output)
    else:
        output = triton_splitk_linear(input_2d, module.weight, module.weight_scale_inv)
    output = output.view(*input.shape[:-1], out_features)
    if module.bias is not None:
        output = output + module.bias
    return output


class FP8Linear(nn.Linear):
    # Internal, temporary flag — not public API, don't set it directly. `_disable_deepgemm_on_multi_device`
    # flips it True at load when the model spans >1 CUDA device in one process (DeepGEMM's context-bound
    # kernels corrupt across devices); removable once the kernel ships a context-free loader.
    _deepgemm_disabled = False
    _proto_sfb = None  # PROTO_FP8_LINEAR=deepgemm: packed UE8M0 weight scales in DeepGEMM's layout

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        scale_fmt: str = "float",
        has_bias: bool = False,
    ):
        super().__init__(in_features, out_features)

        self.has_bias = has_bias
        self.block_size = block_size
        self.activation_scheme = activation_scheme
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=_FP8_DTYPE))

        if self.block_size is None:
            # If block size is None, it means that we are doing per-tensor quantization
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            sf_dtype = _get_ue8m0_dtype() if scale_fmt == "ue8m0" else torch.float32
            scale_out_features = (out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (in_features + self.block_size[1] - 1) // self.block_size[1]
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

        if (self._proto_sfb is not None and input.dtype == torch.bfloat16) or (
            _PROTO_FP8_LINEAR == "splitk"
            and type(self) is FP8Linear
            and self.block_size is not None
            and tuple(self.block_size) == (128, 128)
            and self.weight_scale_inv.dtype == torch.float32
            and self.activation_scale is None
        ):
            if not (torch.is_grad_enabled() and input.requires_grad):
                return _proto_fp8_linear_forward(self, input)

        return _apply_if_grad(
            FP8LinearFunction,
            fp8_linear,
            input,
            self.weight,
            self.weight_scale_inv,
            self.block_size,
            self.bias,
            self.activation_scale,
            not self._deepgemm_disabled,
        )


class FP8Embedding(nn.Embedding):
    """`nn.Embedding` whose table is stored in FP8 with a single per-tensor `weight_scale`.

    Qwen4-Exp's hashed n-gram table is ~48 GiB in FP8, too big to dequantize at load time, so the
    rescale runs on the rows a lookup gathers instead. Build it on the `meta` device: like
    `FP8Linear`, `__init__` lets `nn.Embedding` allocate a full-precision table first.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=_FP8_DTYPE), requires_grad=False)
        # `(1,)` rather than a scalar: the shape checkpoints serialize per-tensor scales with.
        self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        rows = super().forward(input)
        return rows.to(self.weight_scale.dtype) * self.weight_scale


class FP8GroupedLinear(FP8Linear):
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

        w = self.weight
        scale_inv = self.weight_scale_inv

        w = w.view(self.n_groups, -1, hidden_dim)
        x = x.movedim(-2, 0).reshape(-1, hidden_dim)
        scale_inv = scale_inv.view(self.n_groups, scale_inv.size(0) // self.n_groups, scale_inv.size(1))

        tokens_per_group = x.size(0) // self.n_groups
        tokens_per_expert = torch.full((self.n_groups,), tokens_per_group, device=x.device, dtype=torch.int32)
        offsets = torch.arange(1, self.n_groups + 1, device=x.device, dtype=torch.int32) * tokens_per_group

        y = _apply_if_grad(
            FP8GroupedMatmulFunction, fp8_grouped_matmul, x, w, scale_inv, offsets, tokens_per_expert, self.block_size
        )
        y = y.reshape(self.n_groups, *input_shape, -1).movedim(0, -2)
        if self.has_bias:
            y.add_(self.bias.view(self.n_groups, -1))
        return y


# [gap prototype] HF_FP8_FUSED_EXPERTS=1: at inference, `grouped_mm` and `batched_mm` call the hub kernel's fused MoE
# entry points (routing, gate_up + SiLU + requant, down + weighting, top-k sum in 8 and 3 kernels) instead of doing
# the routing and the combine in torch around two matmul kernels.
_FP8_FUSED_EXPERTS = os.environ.get("HF_FP8_FUSED_EXPERTS", "0") == "1"
# WP prefill (bench only): fused entry points only above this many routed rows per expert on average
# (tokens * top_k > rows * num_experts, the criterion of WP experts' decode schedule, so the two cannot overlap). The
# fused grouped kernel (one persistent program per SM) is 2.3x today's path at ~150 rows per expert (prefill) and no
# better at 2 rows per expert (decode)
_FP8_FUSED_EXPERTS_MIN_ROWS = int(os.environ.get("HF_FP8_FUSED_EXPERTS_MIN_ROWS", "0"))


def _fp8_fused_experts_forward(self, hidden_states, top_k_index, top_k_weights, grouped: bool):
    kernel = lazy_load_kernel("finegrained-fp8")
    fused = kernel.moe_fused_grouped if grouped else kernel.moe_fused_batched
    out = fused(
        hidden_states,
        top_k_index,
        top_k_weights,
        self.gate_up_proj,
        self.down_proj,
        self.gate_up_proj_scale_inv,
        self.down_proj_scale_inv,
        list(self.block_size),
    )
    return out.to(hidden_states.dtype)


def _can_use_fp8_fused_experts(self, hidden_states, top_k_index) -> bool:
    return (
        _FP8_FUSED_EXPERTS
        and top_k_index.numel() > _FP8_FUSED_EXPERTS_MIN_ROWS * self.num_experts
        and not (torch.is_grad_enabled() and hidden_states.requires_grad)
        and self.has_gate
        and self.gate_softcap is None
        and self.swiglu_alpha is None
        and self.limit is None
        and not self._is_expert_parallel
        and not self._proto_trtllm_laid
        and self.gate_up_proj.dtype == _FP8_DTYPE
        and self.block_size is not None
    )


# --- PROTOTYPE (bench only, not for upstream as is): routed-experts forwards for inference, off unless a flag is set ---
# PROTO_EXPERTS_TRTLLM=1  flashinfer's TRT-LLM routed FP8 MoE, as vLLM calls it, on weights re-laid once at load
# PROTO_EXPERTS_DECODE=1  Triton kernel pair with the dispatch fused, one program per (expert, N tile)
# PROTO_EXPERTS_DECODE=fold  the same with one activation scale per (routed row, weight tile) in the first kernel
# Both replace the `grouped_mm` and `batched_mm` forwards of `FP8Experts` when no gradient is needed.
_PROTO_EXPERTS_TRTLLM = os.environ.get("PROTO_EXPERTS_TRTLLM", "0") == "1"
# PROTO_EXPERTS_TRTLLM_INPLACE=1 (with PROTO_EXPERTS_TRTLLM=1): no copy, the parameters are re-laid in place while the
# continuous batching loop runs (`_proto_relayout_fp8_experts`)
_PROTO_EXPERTS_TRTLLM_INPLACE = _PROTO_EXPERTS_TRTLLM and os.environ.get("PROTO_EXPERTS_TRTLLM_INPLACE", "0") == "1"
# PROTO_EXPERTS_TRTLLM_ROUTING=1 (with PROTO_EXPERTS_TRTLLM=1): the MoE block hands the fp32 router logits and the score
# correction bias to the TRT-LLM kernel, which routes inside (DeepSeekV3 routing), instead of the router's own top-k
# (`proto_trtllm_moe_routing`, called from the model's MoE block)
_PROTO_EXPERTS_TRTLLM_ROUTING = _PROTO_EXPERTS_TRTLLM and os.environ.get("PROTO_EXPERTS_TRTLLM_ROUTING", "0") == "1"
# PROTO_EXPERTS_TRTLLM_AUTOTUNE=1 (with PROTO_EXPERTS_TRTLLM=1): flashinfer's autotuner picks the TRT-LLM MoE tactics
# once per process, when the first CB batch processor is created (`_proto_autotune_fp8_experts`), never in band
_PROTO_EXPERTS_TRTLLM_AUTOTUNE = _PROTO_EXPERTS_TRTLLM and os.environ.get("PROTO_EXPERTS_TRTLLM_AUTOTUNE", "0") == "1"
_proto_trtllm_tuned = False
_PROTO_EXPERTS_DECODE = os.environ.get("PROTO_EXPERTS_DECODE", "0") in ("1", "fold")
_PROTO_EXPERTS_DECODE_FOLD = os.environ.get("PROTO_EXPERTS_DECODE", "0") == "fold"


def _proto_experts_supported(module: "FP8Experts") -> bool:
    return (
        module.has_gate
        and module.block_size is not None
        and tuple(module.block_size) == (128, 128)
        and module.activation_scheme == "dynamic"
        and _first_attr(module.config, "hidden_activation", "hidden_act") == "silu"
        and module.gate_softcap is None
        and module.swiglu_alpha is None
        and module.limit is None
        and module.gate_up_proj.dtype == _FP8_DTYPE
        and module.gate_up_proj_scale_inv.dtype == torch.float32
        and module.hidden_dim % 128 == 0
        and module.gate_up_proj.shape[1] % 256 == 0
    )


def _proto_prepare_fp8_experts(model: nn.Module) -> None:
    """Load-time half of `PROTO_EXPERTS_TRTLLM`: every rank re-lays its own shard of the expert weights for the
    TRT-LLM kernels. The original parameters are kept (the default path, saving and training still see them)."""
    if not _PROTO_EXPERTS_TRTLLM or _PROTO_EXPERTS_TRTLLM_INPLACE:
        return
    from ..distributed.parallel_utils import _use_local_dtensor_params
    from . import _proto_fp8_experts as proto

    prepared = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, FP8Experts):
                continue
            with _use_local_dtensor_params(module):
                if not _proto_experts_supported(module) or not proto.trtllm_supported(module.gate_up_proj.device):
                    continue
                module._proto_trtllm = proto.trtllm_prepare(
                    module.gate_up_proj.detach(),
                    module.gate_up_proj_scale_inv.detach(),
                    module.down_proj.detach(),
                    module.down_proj_scale_inv.detach(),
                )
                prepared += 1
    logger.warning(f"PROTO_EXPERTS_TRTLLM: re-laid the weights of {prepared} FP8Experts modules for TRT-LLM")


def _proto_relayout_fp8_experts(model: nn.Module, trtllm: bool) -> bool:
    """`PROTO_EXPERTS_TRTLLM_INPLACE`: re-lays the expert weights of every rank in place, for the TRT-LLM kernels
    (`trtllm=True`) or back to the checkpoint's layout. The continuous batching manager keeps TRT-LLM's layout while its
    generation loop runs and gives the checkpoint's back whenever anything else may read the weights (pause, stop).
    The storage does not move, so CUDA graphs captured on it stay valid. Idempotent. Returns whether a weight changed
    address since the previous re-layout (replaced rather than updated in place): graphs captured before then are stale.
    """
    if not _PROTO_EXPERTS_TRTLLM_INPLACE:
        return False
    from ..distributed.parallel_utils import _use_local_dtensor_params
    from . import _proto_fp8_experts as proto

    moved, count = False, 0
    torch.accelerator.synchronize()  # the generation loop reads the weights on its own stream
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, FP8Experts) or module._proto_trtllm_laid == trtllm or module._is_expert_parallel:
                continue
            if module.config._experts_implementation not in ("grouped_mm", "batched_mm"):
                continue
            with _use_local_dtensor_params(module):
                if not _proto_experts_supported(module) or not proto.trtllm_supported(module.gate_up_proj.device):
                    continue
                gate_up, down = module.gate_up_proj.data, module.down_proj.data
                proto.trtllm_relayout_(gate_up, swap_halves=True, inverse=not trtllm)
                proto.trtllm_relayout_(down, swap_halves=False, inverse=not trtllm)
                module._proto_trtllm_laid = trtllm
                count += 1
                if not trtllm:
                    module._proto_trtllm = None
                    continue
                if module._proto_trtllm_scales is None:
                    module._proto_trtllm_scales = (
                        torch.empty_like(module.gate_up_proj_scale_inv.data),
                        torch.empty_like(module.down_proj_scale_inv.data),
                    )
                s13, s2 = module._proto_trtllm_scales
                proto.trtllm_scales_(s13, s2, module.gate_up_proj_scale_inv.data, module.down_proj_scale_inv.data)
                num_experts, two_i, hidden = gate_up.shape
                ptrs = (gate_up.data_ptr(), down.data_ptr())
                moved |= module._proto_trtllm_ptrs is not None and module._proto_trtllm_ptrs != ptrs
                module._proto_trtllm_ptrs = ptrs
                module._proto_trtllm = (
                    gate_up.view(num_experts, hidden // 128, two_i, 128),
                    s13,
                    down.view(num_experts, two_i // 256, hidden, 128),
                    s2,
                )
    torch.accelerator.synchronize()
    logger.warning_once(f"PROTO_EXPERTS_TRTLLM_INPLACE: {count} FP8Experts modules re-laid in place")
    return moved


def _fuse_trtllm_routed_moe(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    from . import _proto_fp8_experts as proto

    return proto.trtllm_experts_forward((w13, s13, w2, s2), hidden_states, top_k_index, top_k_weights)


def _fuse_trtllm_routed_moe_fake(hidden_states, top_k_index, top_k_weights, w13, s13, w2, s2):
    return torch.empty_like(hidden_states)


# [gap3 compile, bench only] `PROTO_EXPERTS_TRTLLM` inside a torch.compile region (FUSE_COMPILE): the activation quant
# and flashinfer's routed MoE as one op dynamo does not trace into
torch.library.custom_op("transformers::fuse_trtllm_routed_moe", _fuse_trtllm_routed_moe, mutates_args=())
torch.library.register_fake("transformers::fuse_trtllm_routed_moe", _fuse_trtllm_routed_moe_fake)


def _fuse_trtllm_routing_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> torch.Tensor:
    from . import _proto_fp8_experts as proto

    return proto.trtllm_experts_forward_routing(
        (w13, s13, w2, s2), hidden_states, router_logits, routing_bias, top_k, n_group, topk_group,
        routed_scaling_factor, norm_topk_prob,
    )  # fmt: skip


def _fuse_trtllm_routing_moe_fake(hidden_states, router_logits, routing_bias, w13, s13, w2, s2, *args):
    return torch.empty_like(hidden_states)


# [gap3 experts, bench only] PROTO_EXPERTS_TRTLLM_ROUTING inside a torch.compile region (FUSE_COMPILE): one opaque op
torch.library.custom_op("transformers::fuse_trtllm_routing_moe", _fuse_trtllm_routing_moe, mutates_args=())
torch.library.register_fake("transformers::fuse_trtllm_routing_moe", _fuse_trtllm_routing_moe_fake)


def proto_trtllm_moe_routing(experts: nn.Module, router: nn.Module, hidden_states, router_logits):
    """PROTO_EXPERTS_TRTLLM_ROUTING: the routed experts of a MoE block from its fp32 router logits, routed inside the
    TRT-LLM kernel. `router` gives the routing parameters (`top_k`, `num_group`, `topk_group`, `routed_scaling_factor`,
    `norm_topk_prob`, `e_score_correction_bias`). Returns this rank's partial sum, as the experts module does under
    `moe_tp_experts_partial`."""
    if (
        torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and os.environ.get("HF_TP_SINGLE_MLP_ALL_REDUCE", "0") != "1"
        and os.environ.get("HF_TP_ALL_REDUCE", "nccl") != "flashinfer_fused"
    ):
        raise NotImplementedError("PROTO_EXPERTS_TRTLLM_ROUTING under TP needs the experts' partial sums reduced by the MLP block")
    args = (
        router.e_score_correction_bias, *experts._proto_trtllm, router.top_k, router.num_group, router.topk_group,
        float(router.routed_scaling_factor), bool(router.norm_topk_prob),
    )  # fmt: skip
    if torch.compiler.is_compiling():
        return torch.ops.transformers.fuse_trtllm_routing_moe(hidden_states.contiguous(), router_logits, *args)
    return _fuse_trtllm_routing_moe(hidden_states.contiguous(), router_logits, *args)


def _proto_autotune_fp8_experts(model: nn.Module, group=None) -> None:
    """PROTO_EXPERTS_TRTLLM_AUTOTUNE: flashinfer's autotuner, once per process, on the shapes of the first FP8Experts
    module that holds TRT-LLM weights (every MoE layer of a rank has the same shapes). Called when the CB batch
    processor is created, after the re-layout and before the cache is sized and any graph is captured. Every TP rank
    must call it (`group`: the TP ranks' gloo group, so they pick the same tactics)."""
    global _proto_trtllm_tuned
    if not _PROTO_EXPERTS_TRTLLM_AUTOTUNE or _proto_trtllm_tuned:
        return
    from . import _proto_fp8_experts as proto

    experts = next((m for m in model.modules() if isinstance(m, FP8Experts) and m._proto_trtllm is not None), None)
    if experts is None:
        return
    router = next(m for m in model.modules() if "e_score_correction_bias" in m._buffers)
    routing = None
    if _PROTO_EXPERTS_TRTLLM_ROUTING:
        routing = {
            "routing_bias": router.e_score_correction_bias, "n_group": router.num_group,
            "topk_group": router.topk_group, "routed_scaling_factor": float(router.routed_scaling_factor),
            "norm_topk_prob": bool(router.norm_topk_prob),
        }  # fmt: skip
    t = time.perf_counter()
    proto.trtllm_autotune(experts._proto_trtllm, experts.hidden_dim, router.top_k, routing, group)
    _proto_trtllm_tuned = True
    logger.warning(f"PROTO_EXPERTS_TRTLLM_AUTOTUNE: tactics chosen in {time.perf_counter() - t:.1f}s")


def _proto_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """The prototype forward selected by the flags, or `None` when the default path has to run."""
    if self._is_expert_parallel or (torch.is_grad_enabled() and hidden_states.requires_grad):
        if self._proto_trtllm_laid:
            raise RuntimeError(
                "The expert weights hold TRT-LLM's layout while the continuous batching loop runs: use the model "
                "inside `manager.pause()` or after `manager.stop()`."
            )
        return None
    from . import _proto_fp8_experts as proto

    if _PROTO_EXPERTS_TRTLLM and self._proto_trtllm is not None:
        if torch.compiler.is_compiling():  # [gap3 compile] one opaque op inside a compiled region
            return torch.ops.transformers.fuse_trtllm_routed_moe(
                hidden_states.contiguous(), top_k_index.contiguous(), top_k_weights.contiguous(), *self._proto_trtllm
            )
        return proto.trtllm_experts_forward(
            self._proto_trtllm, hidden_states.contiguous(), top_k_index.contiguous(), top_k_weights.contiguous()
        )
    if _PROTO_EXPERTS_DECODE and _proto_experts_supported(self):
        return proto.decode_experts_forward(
            hidden_states.contiguous(),
            top_k_index.contiguous(),
            top_k_weights.contiguous(),
            self.gate_up_proj,
            self.gate_up_proj_scale_inv,
            self.down_proj,
            self.down_proj_scale_inv,
            fold=_PROTO_EXPERTS_DECODE_FOLD,
        )
    return None


def fp8_batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if self.activation_scheme == "static":
        raise NotImplementedError(
            "batched_mm experts dispatch does not support activation_scheme='static'. "
            "Use the default eager dispatch or switch to activation_scheme='dynamic'."
        )
    if _can_use_fp8_fused_experts(self, hidden_states, top_k_index):
        return _fp8_fused_experts_forward(self, hidden_states, top_k_index, top_k_weights, grouped=False)
    if _PROTO_EXPERTS_TRTLLM or _PROTO_EXPERTS_DECODE:
        proto_out = _proto_experts_forward(self, hidden_states, top_k_index, top_k_weights)
        if proto_out is not None:
            return proto_out

    finegrained_fp8 = load_finegrained_fp8_kernel()

    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    # Replicate each token num_top_k times to align with the flattened (S,) routing tensors.
    selected_hidden_states = hidden_states.repeat_interleave(num_top_k, dim=0)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # EP sentinel handling: leave `expert_ids` unclamped — the batched kernel early-returns on
    # `expert_id >= NUM_EXPERTS`, leaving sentinel output rows uninitialized. The post-mask below
    # zeroes them before the per-token reduction so `uninit * 0 = NaN` can't poison the sum.
    sentinel_mask = (expert_ids >= self.num_experts).unsqueeze(-1) if self._is_expert_parallel else None

    weight_up = self.gate_up_proj if self.has_gate else self.up_proj
    weight_scale_up = self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv
    weight_down = self.down_proj
    weight_scale_down = self.down_proj_scale_inv

    # --- Up projection per expert (FP8 batched) ---
    proj_out = finegrained_fp8.batched_matmul(
        selected_hidden_states,
        weight_up,
        weight_scale_up,
        block_size=self.block_size,
        expert_ids=expert_ids,
    )  # (S, 2 * intermediate_dim) or (S, intermediate_dim) depending on gating

    # Apply gating or activation
    if self.has_gate:
        # for gated experts we apply the custom/default gating mechanism
        proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)
    else:
        # for non-gated experts we just apply the activation function
        proj_out = self.act_fn(proj_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 batched) ---
    proj_out = finegrained_fp8.batched_matmul(
        proj_out,
        weight_down,
        weight_scale_down,
        block_size=self.block_size,
        expert_ids=expert_ids,
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights.to(proj_out.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Post-mask sentinel rows: kernel left them uninitialized, so zero them out
    # before the reduction below (uninit may be NaN; NaN * 0 = NaN).
    if sentinel_mask is not None:
        weighted_out.masked_fill_(sentinel_mask, 0.0)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def fp8_grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if self.activation_scheme == "static":
        raise NotImplementedError(
            "grouped_mm experts dispatch does not support activation_scheme='static'. "
            "Use the default eager dispatch or switch to activation_scheme='dynamic'."
        )
    if _can_use_fp8_fused_experts(self, hidden_states, top_k_index):
        return _fp8_fused_experts_forward(self, hidden_states, top_k_index, top_k_weights, grouped=True)
    if _PROTO_EXPERTS_TRTLLM or _PROTO_EXPERTS_DECODE:
        proto_out = _proto_experts_forward(self, hidden_states, top_k_index, top_k_weights)
        if proto_out is not None:
            return proto_out

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Sort by expert for grouped processing
    expert_ids_g, perm = torch.sort(expert_ids)
    selected_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    # Compute offsets for grouped processing.
    # histc instead of bincount avoids cuda-graph issues;
    # CPU requires float input, CUDA requires int input (deterministic mode).
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # EP sentinel handling: leave `expert_ids` unclamped so the sort pushes sentinels to the tail,
    # `histc(max=num_experts-1)` drops them from `tokens_per_expert`, and the grouped matmul skips
    # rows beyond `offsets[-1]`, leaving them uninit (fwd output and bwd `d_input`). The fwd
    # post-mask kills `NaN * 0 = NaN` before the per-token reduction, the bwd pre-mask zeros them
    # in `d_selected_hidden_states_g` before the gather's scatter-add.
    sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1) if self._is_expert_parallel else None

    weight_up = self.gate_up_proj if self.has_gate else self.up_proj
    weight_scale_up = self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv
    weight_down = self.down_proj
    weight_scale_down = self.down_proj_scale_inv

    # Pre-mask (bwd path).
    if sentinel_mask is not None and selected_hidden_states_g.requires_grad:
        selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)

    # --- Up projection per expert (FP8 grouped) ---
    proj_out = _apply_if_grad(
        FP8GroupedMatmulFunction,
        fp8_grouped_matmul,
        selected_hidden_states_g,
        weight_up,
        weight_scale_up,
        offsets,
        tokens_per_expert,
        self.block_size,
    )  # (S, 2 * intermediate_dim)

    # Apply gating or activation
    if self.has_gate:
        # for gated experts we apply the custom/default gating mechanism
        proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)
    else:
        # for non-gated experts we just apply the activation function
        proj_out = self.act_fn(proj_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 grouped) ---
    proj_out = _apply_if_grad(
        FP8GroupedMatmulFunction,
        fp8_grouped_matmul,
        proj_out,
        weight_down,
        weight_scale_down,
        offsets,
        tokens_per_expert,
        self.block_size,
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights_g.to(proj_out.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Post-mask (fwd path).
    if sentinel_mask is not None:
        weighted_out.masked_fill_(sentinel_mask, 0.0)

    # Restore original order
    if _GLUE_EXPERTS:
        # `perm` is a permutation, so scattering row i to row perm[i] is the same reordering, in one kernel
        restored = torch.empty_like(weighted_out)
        restored[perm] = weighted_out
        weighted_out = restored
    else:
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)
        weighted_out = weighted_out[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class FP8Experts(nn.Module):
    # Internal: set True by `_disable_deepgemm_on_multi_device` when the model spans >1 CUDA device.
    _deepgemm_disabled = False
    _proto_trtllm = None  # PROTO_EXPERTS_TRTLLM: this rank's weights in TRT-LLM's layout, built once at load
    # PROTO_EXPERTS_TRTLLM_INPLACE: the parameters hold TRT-LLM's layout (then `_proto_trtllm` views them), the block
    # scale buffers in TRT-LLM's order, and the weights' addresses at the last re-layout
    _proto_trtllm_laid = False
    _proto_trtllm_scales = None
    _proto_trtllm_ptrs = None

    def __init__(
        self,
        config,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        scale_fmt: str = "float",
        has_bias: bool = False,
        has_gate: bool = True,
        expert_dtype: str = "fp8",
    ):
        super().__init__()

        assert has_bias is False, (
            "FP8Experts does not support bias for now, please open an issue if you want this feature"
        )

        self.config = config
        self.has_bias = has_bias
        self.has_gate = has_gate
        self.block_size = block_size
        self.hidden_dim = config.hidden_size
        self.activation_scheme = activation_scheme
        self.num_experts = _first_attr(config, "num_local_experts", "num_experts")
        self.intermediate_dim = _first_attr(config, "moe_intermediate_size", "intermediate_size")
        self.swiglu_alpha = getattr(config, "swiglu_alpha", None)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.act_fn = ACT2FN[_first_attr(config, "hidden_activation", "hidden_act")]
        self.limit = getattr(config, "swiglu_limit", None)
        self.gate_softcap = getattr(config, "mlp_gate_softcap", None)
        self.up_softcap = getattr(config, "mlp_up_softcap", None)

        # Expert weight precision is FP8 by default; DeepSeek V4-style models declare
        # `expert_dtype = "fp4"` for FP4-packed expert weights. FP4 storage:
        #   - weight is `int8`, K dim halved (2 e2m1 values per byte).
        #   - per-row SF at gran_k=32 (no block-wise SF; `block_size` ignored).
        is_fp4 = expert_dtype == "fp4"
        sf_dtype = _get_ue8m0_dtype() if scale_fmt == "ue8m0" else torch.float32
        if is_fp4:
            alloc_kwargs = {
                "weight_dtype": torch.int8,
                "sf_dtype": sf_dtype,
                "weight_k_div": 2,
                "sf_gran_n": 1,
                "sf_gran_k": 32,
            }
        else:
            alloc_kwargs = {
                "weight_dtype": _FP8_DTYPE,
                "sf_dtype": sf_dtype,
                "sf_gran_n": block_size[0] if block_size is not None else None,
                "sf_gran_k": block_size[1] if block_size is not None else None,
            }

        if self.has_gate:
            self.gate_up_proj, self.gate_up_proj_scale_inv = _alloc_expert_proj(
                self.num_experts, 2 * self.intermediate_dim, self.hidden_dim, min_sf_out=2, **alloc_kwargs
            )
            self.register_parameter("gate_up_proj_bias", None)
        else:
            self.up_proj, self.up_proj_scale_inv = _alloc_expert_proj(
                self.num_experts, self.intermediate_dim, self.hidden_dim, **alloc_kwargs
            )
            self.register_parameter("up_proj_bias", None)

        self.down_proj, self.down_proj_scale_inv = _alloc_expert_proj(
            self.num_experts, self.hidden_dim, self.intermediate_dim, **alloc_kwargs
        )
        self.register_parameter("down_proj_bias", None)

        if self.activation_scheme == "static":
            self.gate_up_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))
            self.down_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        if self.gate_softcap is not None:
            # Softcapped SwiGLU (e.g. Inkling2, same math as the model's non-quantized experts)
            input_dtype = gate.dtype
            gate, up = gate.float(), up.float()
            gate_factor = torch.tanh(gate / self.gate_softcap) * self.gate_softcap
            up = torch.tanh(up / self.up_softcap) * self.up_softcap
            return (gate_factor * torch.sigmoid(gate) * up).to(input_dtype)
        if self.swiglu_alpha is not None:
            # Clamped SwiGLU-OAI gate (same math as the model's non-quantized experts).
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
            return (up + 1.0) * glu
        elif self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
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
                self.gate_up_proj[expert_idx] if self.has_gate else self.up_proj[expert_idx],
                self.gate_up_proj_scale_inv[expert_idx] if self.has_gate else self.up_proj_scale_inv[expert_idx],
                activation_scale=gate_up_act_scale,
            )
            proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
            down_act_scale = (
                self.down_proj_activation_scale[expert_idx] if self.activation_scheme == "static" else None
            )
            proj_out = self.linear(
                proj_out,
                self.down_proj[expert_idx],
                self.down_proj_scale_inv[expert_idx],
                activation_scale=down_act_scale,
            )
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            weighted_out = proj_out * routing_weights.to(proj_out.dtype)
            final_hidden_states.index_add_(0, token_idx, weighted_out.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        activation_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weight.element_size() > 1:
            return F.linear(input, weight, None)

        return _apply_if_grad(
            FP8LinearFunction,
            fp8_linear,
            input,
            weight,
            weight_scale_inv,
            self.block_size,
            None,
            activation_scale,
            not self._deepgemm_disabled,
        )


class FP8ExpertsInterface(ExpertsInterface):
    """Interface for registering custom FP8 experts forward functions."""

    _global_mapping = {
        "batched_mm": fp8_batched_mm_experts_forward,
        "grouped_mm": fp8_grouped_mm_experts_forward,
        "deepgemm": deepgemm_fp8_fp4_experts_forward,
        "deepgemm_megamoe": deepgemm_fp8_fp4_megamoe_experts_forward,
    }


ALL_FP8_EXPERTS_FUNCTIONS = FP8ExpertsInterface()


def _disable_deepgemm_on_multi_device(model: nn.Module) -> None:
    """Internal, temporary helper (not public API): flag every FP8 module to skip DeepGEMM when the
    model spans >1 CUDA device in one process.

    DeepGEMM loads each kernel via `cuKernelGetFunction`, which binds the `CUfunction` handle to the
    CUDA context live at load time; driving that cached handle from another device launches it against
    the wrong context and produces garbage. (Build-time fix: compile DeepGEMM with
    `DG_JIT_USE_RUNTIME_API=1` for a context-free `cudaKernel_t` loader; until our wheel picks that up
    we avoid single-process multi-device.) Setting `_deepgemm_disabled` routes both the linear and
    experts paths through Triton/grouped_mm. A model that fits on one device keeps DeepGEMM even with
    other GPUs visible; TP/EP put one device per process, so this is a no-op there.
    """
    fp8_modules = [m for m in model.modules() if isinstance(m, (FP8Linear, FP8Experts))]
    cuda_devices = set()
    for m in fp8_modules:
        param = next(m.parameters(), None)
        if param is not None and param.device.type == "cuda":
            cuda_devices.add(param.device.index)
    if len(cuda_devices) <= 1:
        return
    for m in fp8_modules:
        m._deepgemm_disabled = True
    logger.warning_once(
        "This FP8 model spans multiple CUDA devices in one process; routing its FP8 linear and experts "
        "layers through Triton/grouped_mm instead of DeepGEMM (DeepGEMM's cached kernels are bound to a "
        "single CUDA context and corrupt across devices). Run tensor/expert parallel (one device per "
        "process) to use the faster DeepGEMM path."
    )


def replace_with_fp8_embedding(model, patterns: list[str], modules_to_not_convert: list[str] | None = None):
    """Swap every `nn.Embedding` whose name ends with one of `patterns` for `FP8Embedding`.

    Also runs under `dequantize=True`: a table worth quantizing is one we cannot afford to expand.
    """
    for name, module in list(model.named_modules()):
        if type(module) is nn.Embedding and any(name.endswith(p) for p in patterns):
            if not should_convert_module(name, modules_to_not_convert):
                continue
            with torch.device("meta"):
                new_module = FP8Embedding(module.num_embeddings, module.embedding_dim, module.padding_idx)
            new_module._hf_quantized_needs_local_tp = True
            model.set_submodule(name, new_module)
    return model


def replace_with_fp8_linear(
    model, modules_to_not_convert: list[str] | None = None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `FP8Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `None`):
            Names of the modules to not convert. In practice we keep the `lm_head` in full precision for numerical stability reasons.
        quantization_config (`FineGrainedFP8Config`):
            The quantization config object that contains the quantization parameters.
        pre_quantized (`bool`, defaults to `False`):
            Whether the model is pre-quantized or not
    """

    if quantization_config.dequantize:
        return model

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        if quantization_config.experts_only and not module_name.endswith(".experts"):
            continue

        new_module = None
        with torch.device("meta"):
            if module_name.endswith(".experts"):
                has_gate = getattr(module, "has_gate", True)
                has_bias = getattr(module, "has_bias", False)
                config = getattr(module, "config", model.config.get_text_config())
                new_class = use_experts_implementation(
                    experts_class=FP8Experts,
                    experts_interface=ALL_FP8_EXPERTS_FUNCTIONS,
                    has_bias=has_bias,
                    has_gate=has_gate,
                )
                new_module = new_class(
                    config=config,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    scale_fmt=quantization_config.scale_fmt,
                    has_bias=has_bias,
                    has_gate=has_gate,
                    expert_dtype=quantization_config.expert_dtype,
                )
            elif type(module) is nn.Linear:
                # Vanilla `nn.Linear` → standard FP8Linear swap.
                new_module = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    scale_fmt=quantization_config.scale_fmt,
                    has_bias=module.bias is not None,
                )
            elif isinstance(module, nn.Linear) and "GroupedLinear" in type(module).__name__:
                # Block-diagonal grouped linear (e.g. DSv4's `DeepseekV4GroupedLinear`):
                # one underlying weight conceptually split into `n_groups` independent
                # sub-matmuls fed by disjoint input slices. Vanilla `FP8Linear` would
                # collapse those groups into one giant linear and yield the wrong
                # output dim, so swap to `FP8GroupedLinear` which keeps the per-group
                # bmm contract and runs each block as its own FP8 matmul.
                new_module = FP8GroupedLinear(
                    in_features_per_group=module.in_features,
                    out_features=module.out_features,
                    n_groups=module.n_groups,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    scale_fmt=quantization_config.scale_fmt,
                    has_bias=module.bias is not None,
                )
            if new_module is not None:
                # TP must use local tensors because this quantization path does not support DTensor inputs or weights.
                new_module._hf_quantized_needs_local_tp = True
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )
    return model


class Fp8Quantize(ConversionOps):
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

    def convert(self, input_dict: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # Quantize every (key, tensor) entry in the dict. Single-tensor case (legacy
        # callers that pass one key) and multi-tensor case (reverse of an expert
        # ``MergeModulelist`` that emits one key per expert) are handled the same way.
        result: dict[str, torch.Tensor] = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            result.update(self._quantize_one(key, tensor))
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        return Fp8Dequantize(self.hf_quantizer)


class Fp8Dequantize(ConversionOps):
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
        if "weight" in input_dict:
            # The downstream renamer in `core_model_loading._convert_one_module` uses the
            # output dict's *key*, not its content, to derive prefix/suffix; if `full_layer_name`
            # is unset (direct invocation / tests) fall back to the legacy converter's target.
            target_key = full_layer_name if full_layer_name is not None else "weight"
            quantized = input_dict["weight"]
            quantized = quantized[0] if isinstance(quantized, list) else quantized
            if "weight_scale_inv" in input_dict:
                scales = input_dict["weight_scale_inv"]
                scales = scales[0] if isinstance(scales, list) else scales
                return {target_key: self._dequantize_one(quantized, scales, output_dtype=output_dtype)}
            # A weight the checkpoint stores unquantized (e.g. the embeddings or a norm) still takes the model's dtype
            return {target_key: quantized if output_dtype is None else quantized.to(output_dtype)}

        # Generic chain path: dequantize every weight pattern that has a sibling scale.
        result: dict[str, list[torch.Tensor] | torch.Tensor] = {}
        for key, value in input_dict.items():
            if "activation_scale" in key or "weight_scale_inv" in key:
                continue  # consumed by the dequant; drop from the chain
            scale_key = f"{key}_scale_inv"
            if scale_key not in input_dict:
                # No scale to apply (e.g. unrelated entry) — pass through untouched.
                result[key] = value
                continue
            weights = value if isinstance(value, list) else [value]
            scales = input_dict[scale_key]
            scales = scales if isinstance(scales, list) else [scales]
            if len(weights) != len(scales):
                raise ValueError(
                    f"Fp8Dequantize: weight/scale count mismatch for {key} "
                    f"({len(weights)} weights vs {len(scales)} scales)."
                )
            result[key] = [self._dequantize_one(w, s, output_dtype=output_dtype) for w, s in zip(weights, scales)]
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        # Round-trip: dequantize on load -> re-quantize on save, so the saved
        # checkpoint preserves the FP8 format (weight + per-block ``weight_scale_inv``)
        # whether the in-memory state stayed quantized or was dequantized for compute.
        # With `experts_only`, the dequantized layers save as they are: their config says they're not quantized
        if self.hf_quantizer.quantization_config.experts_only:
            return _IdentityOp()
        return Fp8Quantize(self.hf_quantizer)
