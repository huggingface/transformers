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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name, should_convert_module
from ..utils import logging
from ..utils.import_utils import (
    KERNELS_MAX_VERSION,
    KERNELS_MIN_VERSION,
    is_kernels_available,
    is_torchdynamo_compiling,
)
from .deepgemm import (
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
    deepgemm_fp8_fp4_megamoe_experts_forward,
)
from .hub_kernels import lazy_load_kernel
from .moe import ExpertsInterface, use_experts_implementation
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


@functools.cache
def _get_ue8m0_dtype() -> torch.dtype:
    """Return ``torch.float8_e8m0fnu`` or raise a clear error on torch without FP8 support."""
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


@functools.cache
def _load_finegrained_fp8_kernel() -> FineGrainedFP8:
    """
    Load the finegrained-fp8 Triton kernel once and return its entry points.

    Raises `ImportError` if the `kernels` package is missing, or the kernel or required
    symbols cannot be found.
    """
    if not is_torchdynamo_compiling():
        if not is_kernels_available():
            raise ImportError(
                "finegrained-fp8 kernel requires the `kernels` package. "
                f"Please install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
                f"e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
            )

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
            f"finegrained-fp8 kernel is missing required symbols: {', '.join(missing)}. "
            f"Please install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
            f"e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
        )

    return FineGrainedFP8(
        matmul=matmul,
        batched_matmul=batched_matmul,
        grouped_matmul=grouped_matmul,
    )


@torch._dynamo.allow_in_graph
def _populate_finegrained_fp8_kernel() -> None:
    _ = _load_finegrained_fp8_kernel()
    return None


def load_finegrained_fp8_kernel() -> FineGrainedFP8:
    if is_torchdynamo_compiling():
        _populate_finegrained_fp8_kernel()
    return _load_finegrained_fp8_kernel()


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
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Triton FP8/FP4 linear: fused act-quant + matmul, then optional bias add.

    ``activation_scale=None`` → dynamic per-K-block scales (inline); set it for
    static per-tensor quant. ``weight_scale_inv`` accepts fp32 or UE8M0; the
    dispatcher routes FP4 (``int8``-packed) weights automatically.
    """
    finegrained_fp8 = load_finegrained_fp8_kernel()
    output = finegrained_fp8.matmul(
        input,
        weight,
        weight_scale_inv,
        block_size,
        output_dtype,
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
    output_dtype: torch.dtype | None = None,
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
        output_dtype: desired output dtype.
    """
    # DeepGEMM is CUDA-only, dynamic-only, SM90+ only, FP4/FP8-block-128-only.
    # ``TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`` forces the Triton fallback for this single
    # dispatcher (the experts ``"deepgemm"`` impl is unaffected — use ``set_experts_implementation``
    # for that). Used by the FP8 MoE batched_mm / grouped_mm paths to avoid a still-unexplained
    # DeepGEMM-vs-Triton interaction that degrades end-to-end generation on B200 (per-row kernel
    # outputs still measure bit-perfect, but final tokens drift; not reproducible with the
    # DeepGEMM linear off). Also temporarily skipped under ``torch.compile`` — DeepGEMM's
    # per-token cast calls ``pack_ue8m0_to_int`` which has data-dependent bit-twiddling that
    # dynamo can't guard. TODO: remove the ``is_torchdynamo_compiling`` gate once the upstream
    # ``pack_ue8m0_to_int`` is rewritten to be FakeTensor-friendly; the Triton fallback is
    # dynamo-friendly today via its ``@triton_op`` registration.
    deepgemm_preferred = (
        activation_scale is None
        and weight.device.type == "cuda"
        and torch.cuda.get_device_properties().major >= 9
        and (weight.dtype == torch.int8 or (block_size is not None and block_size[0] == block_size[1] == 128))
        and os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "0") != "1"
        and not is_torchdynamo_compiling()
    )

    if deepgemm_preferred:
        try:
            return deepgemm_fp8_fp4_linear(
                input,
                weight,
                weight_scale_inv,
                block_size=block_size,
                output_dtype=output_dtype,
                activation_scale=activation_scale,
                bias=bias,
            )
        except ImportError as e:
            # Forward the original reason so the user knows whether DeepGEMM is unavailable
            # (env/build issue) or refused this specific input (e.g. multi-device on SM100).
            logger.warning_once(f"DeepGEMM unavailable for this call, falling back to Triton. Reason: {e}")

    return finegrained_fp8_linear(input, weight, weight_scale_inv, block_size, bias, activation_scale, output_dtype)


class FP8Linear(nn.Linear):
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
            self.weight_scale_inv = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=sf_dtype))

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

        return fp8_linear(
            input,
            weight,
            scale_inv,
            block_size=self.block_size,
            activation_scale=self.activation_scale,
            output_dtype=input.dtype,
            bias=self.bias,
        )


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

        w = to_local(self.weight)
        scale_inv = to_local(self.weight_scale_inv)

        w = w.view(self.n_groups, -1, hidden_dim)
        x = x.movedim(-2, 0).reshape(-1, hidden_dim)
        scale_inv = scale_inv.view(self.n_groups, scale_inv.size(0) // self.n_groups, scale_inv.size(1))

        tokens_per_group = x.size(0) // self.n_groups
        tokens_per_expert = torch.full((self.n_groups,), tokens_per_group, device=x.device, dtype=torch.int32)
        offsets = torch.arange(1, self.n_groups + 1, device=x.device, dtype=torch.int32) * tokens_per_group

        finegrained_fp8 = load_finegrained_fp8_kernel()
        y = finegrained_fp8.grouped_matmul(
            x,
            w,
            scale_inv,
            offsets=offsets,
            tokens_per_expert=tokens_per_expert,
            block_size=self.block_size,
        )
        y = y.reshape(self.n_groups, *input_shape, -1).movedim(0, -2)
        if self.has_bias:
            y.add_(self.bias.view(self.n_groups, -1))
        return y


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
    sentinel_mask = (expert_ids >= self.num_experts).unsqueeze(-1)

    weight_up = to_local(self.gate_up_proj if self.has_gate else self.up_proj)
    weight_scale_up = to_local(self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv)
    weight_down = to_local(self.down_proj)
    weight_scale_down = to_local(self.down_proj_scale_inv)

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

    finegrained_fp8 = load_finegrained_fp8_kernel()

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
    # rows beyond `offsets[-1]` — sentinels cost no real GEMM compute. The kernel writes only
    # valid rows, so sentinel-tail `proj_out` rows are uninit; without the post-mask below,
    # `proj_out[sentinel] * 0 = NaN * 0 = NaN` would poison the per-token reduction. FP8
    # quantized weights are inference-only, so no bwd pre-mask is needed.
    sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)

    weight_up = to_local(self.gate_up_proj if self.has_gate else self.up_proj)
    weight_scale_up = to_local(self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv)
    weight_down = to_local(self.down_proj)
    weight_scale_down = to_local(self.down_proj_scale_inv)

    # --- Up projection per expert (FP8 grouped) ---
    proj_out = finegrained_fp8.grouped_matmul(
        selected_hidden_states_g,
        weight_up,
        weight_scale_up,
        offsets=offsets,
        tokens_per_expert=tokens_per_expert,
        block_size=self.block_size,
    )  # (S, 2 * intermediate_dim)

    # Apply gating or activation
    if self.has_gate:
        # for gated experts we apply the custom/default gating mechanism
        proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)
    else:
        # for non-gated experts we just apply the activation function
        proj_out = self.act_fn(proj_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 grouped) ---
    proj_out = finegrained_fp8.grouped_matmul(
        proj_out,
        weight_down,
        weight_scale_down,
        offsets=offsets,
        tokens_per_expert=tokens_per_expert,
        block_size=self.block_size,
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights_g.to(proj_out.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Post-mask (fwd path).
    weighted_out.masked_fill_(sentinel_mask, 0.0)

    # Restore original order
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted_out = weighted_out[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class FP8Experts(nn.Module):
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

        # Expert weight precision is FP8 by default; DeepSeek V4-style models declare
        # `config.expert_dtype = "fp4"` for FP4-packed expert weights. FP4 storage:
        #   - weight is `int8`, K dim halved (2 e2m1 values per byte).
        #   - per-row SF at gran_k=32 (no block-wise SF; `block_size` ignored).
        is_fp4 = getattr(config, "expert_dtype", "fp8") == "fp4"
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

        return fp8_linear(
            input,
            weight,
            weight_scale_inv,
            self.block_size,
            activation_scale=activation_scale,
            output_dtype=input.dtype,
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
        pre_quantized (`book`, defaults to `False`):
            Whether the model is pre-quantized or not
    """

    if quantization_config.dequantize:
        return model

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
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
        # Backward-compatible single-tensor path (the legacy fallback converter declares
        # ``["weight$", "weight_scale_inv", "activation_scale"]`` and produces a single
        # ``weight`` target). Also handles the no-scale case (e.g. RMSNorm weights that
        # match ``weight$`` but ship no ``weight_scale_inv`` alongside).
        if "weight$" in input_dict:
            # The downstream renamer in `core_model_loading._convert_one_module` uses the
            # output dict's *key*, not its content, to derive prefix/suffix; if `full_layer_name`
            # is unset (direct invocation / tests) fall back to the legacy converter's target.
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
        return Fp8Quantize(self.hf_quantizer)


class Fp8DecodeScale(ConversionOps):
    """Decode MXFP8 ``ue8m0`` per-block scales (stored as ``uint8`` exponents) into the
    float32 multiplicative scales the FP8 compute path expects.

    Native MXFP8 loading (``dequantize=False``) keeps weights in ``float8_e4m3fn`` and only
    needs the sibling ``*.weight_scale_inv`` tensors turned from raw E8M0 bytes into real
    scales (``2 ** (byte - 127)``). Prepended to each weight converter, this op runs before
    any merge/concat collapses the per-expert structure: it rewrites only the ``uint8`` scale
    entries and passes weights (and already-float scales) through untouched.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    @staticmethod
    def _decode(tensor: torch.Tensor) -> torch.Tensor:
        # E8M0 stores one exponent byte per block; the real scale is ``2 ** (byte - 127)``.
        return (tensor.to(torch.float32) - 127.0).exp2() if tensor.dtype == torch.uint8 else tensor

    def convert(self, input_dict: dict[str, list[torch.Tensor] | torch.Tensor], **kwargs):
        return {
            key: [self._decode(t) for t in value] if isinstance(value, list) else self._decode(value)
            for key, value in input_dict.items()
        }
