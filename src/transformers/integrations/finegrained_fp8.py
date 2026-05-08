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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import logging
from ..utils.import_utils import get_cuda_runtime_version, is_kernels_available, resolve_internal_import
from .hub_kernels import lazy_load_kernel
from .moe import ExpertsInterface, use_experts_implementation


logger = logging.get_logger(__name__)


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


# DeepGEMM requires M-dimension alignment to 128 for TMA-based contiguous grouped GEMM.
# TMA is an H100 hardware addition that allows applications to asynchronously and
# bi-directionally transfer 1D-5D tensors between GPU global and shared memory.
_DEEPGEMM_M_ALIGNMENT = 128


def _first_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of: {names}")


@dataclass(frozen=True)
class FineGrainedFP8:
    """Entry points exposed by the `kernels-community/finegrained-fp8` Triton kernel."""

    fp8_matmul: Callable
    fp8_act_quant: Callable
    batched_fp8_matmul: Callable
    grouped_fp8_matmul: Callable


@functools.cache
def _load_finegrained_fp8_kernel() -> FineGrainedFP8:
    """
    Load the finegrained-fp8 Triton kernel once and return its entry points.

    Raises `ImportError` if the `kernels` package is missing, or the kernel or required
    symbols cannot be found.
    """
    if not is_kernels_available():
        raise ImportError(
            "finegrained-fp8 kernel requires the `kernels` package. Install it with `pip install -U kernels`."
        )

    kernel = lazy_load_kernel("finegrained-fp8")
    if kernel is None:
        raise ImportError(
            "Failed to load the finegrained-fp8 kernel — check that `kernels-community/finegrained-fp8` "
            "has a build matching the current torch/CUDA."
        )

    fp8_matmul = getattr(kernel, "w8a8_fp8_matmul", None)
    fp8_act_quant = getattr(kernel, "fp8_act_quant", None)
    batched_fp8_matmul = getattr(kernel, "w8a8_fp8_matmul_batched", None)
    grouped_fp8_matmul = getattr(kernel, "w8a8_fp8_matmul_grouped", None)

    missing = [
        name
        for name, attr in [
            ("w8a8_fp8_matmul", fp8_matmul),
            ("fp8_act_quant", fp8_act_quant),
            ("w8a8_fp8_matmul_batched", batched_fp8_matmul),
            ("w8a8_fp8_matmul_grouped", grouped_fp8_matmul),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"finegrained-fp8 kernel is missing required symbols: {', '.join(missing)}. "
            "Please update the `kernels` package (`pip install -U kernels`)."
        )

    return FineGrainedFP8(
        fp8_matmul=fp8_matmul,
        fp8_act_quant=fp8_act_quant,
        batched_fp8_matmul=batched_fp8_matmul,
        grouped_fp8_matmul=grouped_fp8_matmul,
    )


@dataclass(frozen=True)
class DeepGEMM:
    """Entry points exposed by the `kernels-community/deep-gemm` kernel."""

    fp8_matmul: Callable
    grouped_fp8_matmul: Callable
    per_token_cast_to_fp8: Callable


@functools.cache
def _load_deepgemm_kernel() -> DeepGEMM:
    """
    Load DeepGEMM once and return its entry points.

    Raises `ImportError` if CUDA/hardware requirements are not met, or the kernel or
    required symbols are not found.
    """
    if not is_kernels_available():
        raise ImportError("DeepGEMM kernel requires the `kernels` package. Install it with `pip install -U kernels`.")

    if not torch.cuda.is_available():
        raise ImportError(
            "DeepGEMM kernel requires CUDA, but CUDA is not available. Use a different `experts_implementation`."
        )

    # DeepGEMM requires Hopper (SM90) or newer for FP8 WGMMA instructions
    major = torch.cuda.get_device_capability()[0]
    if major < 9:
        raise ImportError(
            f"DeepGEMM requires a Hopper (SM90+) or newer GPU, but the current device "
            f"has compute capability {major}.x. Use a different `experts_implementation`."
        )

    # DeepGEMM requires CUDA runtime >= 12.3
    cuda_major, cuda_minor = get_cuda_runtime_version()
    if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 3):
        raise ImportError(
            f"DeepGEMM requires CUDA runtime 12.3+, but found {cuda_major}.{cuda_minor}. "
            "Please upgrade your CUDA toolkit or use a different `experts_implementation`."
        )

    kernel = lazy_load_kernel("deep-gemm")
    if kernel is None:
        raise ImportError(
            "Failed to load the DeepGEMM kernel — check that `kernels-community/deep-gemm` "
            "has a build matching the current torch/CUDA."
        )

    fp8_matmul = getattr(kernel, "fp8_gemm_nt", None)
    grouped_fp8_matmul = getattr(kernel, "m_grouped_fp8_gemm_nt_contiguous", None)
    per_token_cast_to_fp8 = resolve_internal_import(kernel, chained_path="utils.per_token_cast_to_fp8")

    missing = [
        name
        for name, attr in [
            ("fp8_gemm_nt", fp8_matmul),
            ("m_grouped_fp8_gemm_nt_contiguous", grouped_fp8_matmul),
            ("utils.per_token_cast_to_fp8", per_token_cast_to_fp8),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"DeepGEMM kernel is missing required symbols: {', '.join(missing)}. "
            "Please update the `kernels` package (`pip install -U kernels`)."
        )

    return DeepGEMM(
        fp8_matmul=fp8_matmul,
        grouped_fp8_matmul=grouped_fp8_matmul,
        per_token_cast_to_fp8=per_token_cast_to_fp8,
    )


def _cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def w8a8_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FP8 matmul: C = dequant(A, As) @ dequant(B, Bs)^T.

    Supports both per-tensor and block-wise quantization:
      - block_size=None or block_size=[N, K]: per-tensor mode (As is scalar/per-row, Bs is scalar)
      - block_size=[block_n, block_k]: block-wise mode (As and Bs are per-block scale grids)

    Dispatch order:
      1. DeepGEMM (Hopper+, block_size 128x128) if available
      2. Triton finegrained-fp8 kernel (universal fallback)

    Args:
        A:  (M, K) float8_e4m3fn — quantized activations
        B:  (N, K) float8_e4m3fn — quantized weights
        As: block-wise: (M, K//block_k) float32; per-tensor: (M,) per-row scales
        Bs: block-wise: (N//block_n, K//block_k) float32; per-tensor: scalar or (1,) single weight scale
        block_size: [block_n, block_k] for block-wise quantization, or None/[N, K] for per-tensor
        output_dtype: desired output dtype
    """
    if block_size is not None and block_size[0] == block_size[1] == 128:
        try:
            deepgemm = _load_deepgemm_kernel()
        except ImportError:
            logger.warning_once(
                "DeepGEMM kernel is not available or compatible, falling back to Triton finegrained-fp8 kernel. "
                "To use DeepGEMM FP8 matmul, ensure you have a Hopper (SM90+) or newer GPU with CUDA runtime 12.3+, "
                "and that the `kernels` package is installed and up to date (`pip install -U kernels`)."
            )
        else:
            # 3-6x faster than Triton
            A_2d = A.view(-1, A.shape[-1])
            As_2d = As.view(-1, As.shape[-1])
            output = torch.empty(A_2d.shape[0], B.shape[0], device=A.device, dtype=output_dtype)
            deepgemm.fp8_matmul((A_2d, As_2d.float()), (B, Bs.float()), output)
            return output.view(A.shape[:-1] + (B.shape[0],))

    finegrained_fp8 = _load_finegrained_fp8_kernel()
    return finegrained_fp8.fp8_matmul(A, B, As, Bs, block_size, output_dtype)


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        has_bias: bool = False,
        dtype=_FP8_DTYPE,
    ):
        super().__init__(in_features, out_features)

        self.has_bias = has_bias
        self.block_size = block_size
        self.activation_scheme = activation_scheme
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

        if self.block_size is None:
            # If block size is None, it means that we are doing per-tensor quantization
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = (out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (in_features + self.block_size[1] - 1) // self.block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
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

        weight = self.weight
        scale_inv = self.weight_scale_inv
        if isinstance(weight, torch.distributed.tensor.DTensor):
            weight = weight.to_local()
            scale_inv = scale_inv.to_local()

        if self.activation_scheme == "dynamic":
            finegrained_fp8 = _load_finegrained_fp8_kernel()
            qinput, scale = finegrained_fp8.fp8_act_quant(
                input, self.block_size[1] if self.block_size is not None else input.shape[-1]
            )
        elif self.activation_scheme == "static":
            scale = self.activation_scale.to(torch.float32)
            qinput = (input / scale).clamp(min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        else:
            raise NotImplementedError(f"Unsupported activation scheme: {self.activation_scheme}")

        output = w8a8_fp8_matmul(
            qinput,
            weight,
            scale,
            scale_inv,
            self.block_size,
            output_dtype=input.dtype,
        )

        if self.bias is not None:
            output.add_(self.bias)

        return output.to(dtype=input.dtype)


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

    finegrained_fp8 = _load_finegrained_fp8_kernel()

    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    # Replicate each token num_top_k times to align with the flattened (S,) routing tensors.
    selected_hidden_states = hidden_states.repeat_interleave(num_top_k, dim=0)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Clamp EP sentinels so per-token weight indexing stays in-bounds. Routing weights are already
    # zero at sentinel slots (RouterParallel masks them at dispatch), so the weighted mul drops
    # those contributions — we pay the wasted GEMM compute because batched_mm has no offset to skip.
    expert_ids.clamp_(0, self.num_experts - 1)

    # --- Up projection per expert (FP8 batched) ---
    proj_out = finegrained_fp8.batched_fp8_matmul(
        selected_hidden_states,
        self.gate_up_proj if self.has_gate else self.up_proj,
        self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv,
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
    proj_out = finegrained_fp8.batched_fp8_matmul(
        proj_out,
        self.down_proj,
        self.down_proj_scale_inv,
        block_size=self.block_size,
        expert_ids=expert_ids,
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights.to(proj_out.dtype).unsqueeze(-1)  # (S, hidden_dim)

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

    finegrained_fp8 = _load_finegrained_fp8_kernel()

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

    # FSDP2 / EP wraps weights as DTensors but the kernel takes raw pointers — unwrap to
    # local shards. Inference-only path, so `to_local()` autograd-awareness is moot.
    w_up = self.gate_up_proj if self.has_gate else self.up_proj
    ws_up = self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv
    w_down = self.down_proj
    ws_down = self.down_proj_scale_inv
    if isinstance(w_up, torch.distributed.tensor.DTensor):
        w_up = w_up.to_local()
        ws_up = ws_up.to_local()
        w_down = w_down.to_local()
        ws_down = ws_down.to_local()

    # --- Up projection per expert (FP8 grouped) ---
    proj_out = finegrained_fp8.grouped_fp8_matmul(
        selected_hidden_states_g,
        w_up,
        ws_up,
        tokens_per_expert=tokens_per_expert,
        block_size=self.block_size,
        offsets=offsets,
    )  # (S, 2 * intermediate_dim)

    # Apply gating or activation
    if self.has_gate:
        # for gated experts we apply the custom/default gating mechanism
        proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)
    else:
        # for non-gated experts we just apply the activation function
        proj_out = self.act_fn(proj_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 grouped) ---
    proj_out = finegrained_fp8.grouped_fp8_matmul(
        proj_out,
        w_down,
        ws_down,
        tokens_per_expert=tokens_per_expert,
        block_size=self.block_size,
        offsets=offsets,
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


def _build_deepgemm_contiguous_layout(
    expert_ids_sorted: torch.Tensor, num_experts: int, alignment: int, use_psum_layout: bool
) -> tuple:
    """Build the TMA-aligned layout DeepGEMM's grouped GEMM expects.

    Returns `(sorted_to_padded, grouped_layout, total_padded_rows)`. `grouped_layout` encodes
    expert boundaries as a cumsum of aligned counts on Blackwell (`use_psum_layout=True`) or
    per-row expert ids with -1 for padding on Hopper.

    Accepts EP sentinels: values in `expert_ids_sorted` equal to `num_experts` (unclamped sentinels)
    are routed past the last aligned expert block and marked `-1` in the Hopper layout (and
    excluded from the Blackwell cumsum), so DeepGEMM skips them.
    """
    device = expert_ids_sorted.device
    num_tokens = expert_ids_sorted.size(0)
    # histc drops values > max, so EP sentinels (== num_experts) are excluded from the per-expert count.
    tokens_per_expert = torch.histc(expert_ids_sorted.int(), bins=num_experts, min=0, max=num_experts - 1).long()
    aligned_tokens_per_expert = ((tokens_per_expert + alignment - 1) // alignment) * alignment
    # Upper bound avoids GPU->CPU sync; padding rows are skipped by DeepGEMM.
    total_padded_rows = num_tokens + min(num_tokens, num_experts) * (alignment - 1)

    # Zero-prepended inclusive cumsum of per-expert padding. Indices [0, num_experts) give the
    # exclusive cumsum (padding before expert i) and index `num_experts` gives `sum(padding)`,
    # which routes EP sentinels past all valid aligned expert blocks on Blackwell (where the
    # kernel stops at `aligned_cumsum[-1]`) — so sentinels don't go through the GEMM.
    padding_per_expert = aligned_tokens_per_expert - tokens_per_expert
    cumulative_padding = torch.nn.functional.pad(padding_per_expert.cumsum(0), (1, 0))
    sorted_to_padded = torch.arange(num_tokens, device=device) + cumulative_padding[expert_ids_sorted]

    if use_psum_layout:  # Blackwell (SM100+)
        # psum layout: cumsum of *aligned* per-expert counts — sentinels sit at positions >=
        # `grouped_layout[-1]` (by construction of `cumulative_padding`), so the scheduler
        # stops before them. The kernel's `num_m_blocks = ceil_div(layout[i] - align(layout[i-1], 128), BLOCK_M)`
        # between experts only matches the padded tensor when the stored cumsum is over aligned counts.
        grouped_layout = aligned_tokens_per_expert.cumsum(0).int()
    else:
        # Hopper: per-row expert id, -1 for padding rows and for sentinel slots (kernel skips -1).
        grouped_layout = torch.full((total_padded_rows,), -1, device=device, dtype=torch.int32)
        grouped_layout[sorted_to_padded] = torch.where(expert_ids_sorted < num_experts, expert_ids_sorted.int(), -1)

    return sorted_to_padded, grouped_layout, total_padded_rows


def _pad_to_deepgemm_contiguous_layout(
    hidden_states: torch.Tensor,
    scales: torch.Tensor,
    sorted_to_padded: torch.Tensor,
    total_padded_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sorted hidden states and scales into the TMA-aligned contiguous layout."""
    hidden_padded = torch.zeros(
        total_padded_rows, hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
    )
    hidden_padded[sorted_to_padded] = hidden_states
    scales_padded = torch.zeros(total_padded_rows, scales.shape[1], device=hidden_states.device, dtype=torch.float32)
    scales_padded[sorted_to_padded] = scales
    return hidden_padded, scales_padded


def _unpad_from_deepgemm_contiguous_layout(
    hidden_states_padded: torch.Tensor, sorted_to_padded: torch.Tensor
) -> torch.Tensor:
    """Remove padding rows from the TMA-aligned contiguous layout."""
    return hidden_states_padded[sorted_to_padded]


def fp8_deepgemm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if self.activation_scheme == "static":
        raise NotImplementedError(
            "DeepGEMM experts dispatch does not support activation_scheme='static'. "
            "Use the default eager dispatch or switch to activation_scheme='dynamic'."
        )
    if self.block_size is None:
        raise ValueError(
            "DeepGEMM requires block-wise quantization (block_size=[128, 128]), "
            "but got per-tensor quantization (block_size=None)."
        )
    if self.block_size[0] != 128 or self.block_size[1] != 128:
        raise ValueError(f"DeepGEMM requires block_size=(128, 128), got {self.block_size}")

    deepgemm = _load_deepgemm_kernel()

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

    use_psum_layout = torch.cuda.get_device_capability(device)[0] >= 10
    sorted_to_padded, grouped_layout, total_padded_rows = _build_deepgemm_contiguous_layout(
        expert_ids_g, self.num_experts, alignment=_DEEPGEMM_M_ALIGNMENT, use_psum_layout=use_psum_layout
    )

    # EP sentinel handling: leave `expert_ids` unclamped so the sort pushes sentinels to the tail,
    # `_build_deepgemm_contiguous_layout` marks their positions as skipped (-1 on Hopper, beyond
    # the cumsum on Blackwell), and DeepGEMM skips them — sentinels cost no real GEMM compute.
    # The kernel writes only valid rows, so sentinel-tail `proj_out` rows are uninit; without the
    # post-mask below, `proj_out[sentinel] * 0 = NaN * 0 = NaN` would poison the per-token
    # reduction. DeepGEMM is inference-only, so no bwd pre-mask is needed.
    sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)

    # FSDP2 / EP wraps weights as DTensors but the kernel takes raw pointers — unwrap to
    # local shards. Inference-only path, so `to_local()` autograd-awareness is moot.
    w_up = self.gate_up_proj if self.has_gate else self.up_proj
    ws_up = self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv
    w_down = self.down_proj
    ws_down = self.down_proj_scale_inv
    if isinstance(w_up, torch.distributed.tensor.DTensor):
        w_up = w_up.to_local()
        ws_up = ws_up.to_local()
        w_down = w_down.to_local()
        ws_down = ws_down.to_local()

    # --- Up projection per expert (DeepGEMM grouped contiguous) ---
    act_fp8, act_scales = deepgemm.per_token_cast_to_fp8(selected_hidden_states_g, use_ue8m0=False)
    act_fp8, act_scales = _pad_to_deepgemm_contiguous_layout(act_fp8, act_scales, sorted_to_padded, total_padded_rows)
    proj_out = torch.empty(total_padded_rows, w_up.shape[1], device=device, dtype=torch.bfloat16)
    deepgemm.grouped_fp8_matmul(
        (act_fp8, act_scales), (w_up, ws_up.float()), proj_out, grouped_layout, use_psum_layout=use_psum_layout
    )

    # Apply gating or activation
    if self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # --- Down projection per expert (DeepGEMM grouped contiguous) ---
    proj_fp8, proj_scales = deepgemm.per_token_cast_to_fp8(proj_out, use_ue8m0=False)
    proj_out = torch.empty(total_padded_rows, hidden_dim, device=device, dtype=torch.bfloat16)
    deepgemm.grouped_fp8_matmul(
        (proj_fp8, proj_scales),
        (w_down, ws_down.float()),
        proj_out,
        grouped_layout,
        use_psum_layout=use_psum_layout,
    )

    # Remove padding rows
    proj_out = _unpad_from_deepgemm_contiguous_layout(proj_out, sorted_to_padded)

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
    def __init__(
        self,
        config,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        has_bias: bool = False,
        has_gate: bool = True,
        dtype=_FP8_DTYPE,
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
        self.act_fn = ACT2FN[_first_attr(config, "hidden_activation", "hidden_act")]

        if self.has_gate:
            gu_proj_out, gu_proj_in = 2 * self.intermediate_dim, self.hidden_dim
            self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, gu_proj_out, gu_proj_in, dtype=dtype))
            gu_scale_out = _cdiv(gu_proj_out, self.block_size[0]) if self.block_size is not None else 1
            gu_scale_in = _cdiv(gu_proj_in, self.block_size[1]) if self.block_size is not None else 1
            self.gate_up_proj_scale_inv = nn.Parameter(
                torch.empty(self.num_experts, gu_scale_out, gu_scale_in, dtype=torch.float32)
            )
            self.register_parameter("gate_up_proj_bias", None)
        else:
            u_proj_out, u_proj_in = self.intermediate_dim, self.hidden_dim
            self.up_proj = nn.Parameter(torch.empty(self.num_experts, u_proj_out, u_proj_in, dtype=dtype))
            u_scale_out = _cdiv(u_proj_out, self.block_size[0]) if self.block_size is not None else 1
            u_scale_in = _cdiv(u_proj_in, self.block_size[1]) if self.block_size is not None else 1
            self.up_proj_scale_inv = nn.Parameter(
                torch.empty(self.num_experts, u_scale_out, u_scale_in, dtype=torch.float32)
            )
            self.register_parameter("up_proj_bias", None)

        d_proj_out, d_proj_in = self.hidden_dim, self.intermediate_dim
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, d_proj_out, d_proj_in, dtype=dtype))
        d_scale_out = _cdiv(d_proj_out, self.block_size[0]) if self.block_size is not None else 1
        d_scale_in = _cdiv(d_proj_in, self.block_size[1]) if self.block_size is not None else 1
        self.down_proj_scale_inv = nn.Parameter(
            torch.empty(self.num_experts, d_scale_out, d_scale_in, dtype=torch.float32)
        )
        self.register_parameter("down_proj_bias", None)

        if self.activation_scheme == "static":
            self.gate_up_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))
            self.down_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # index_add_ will accumulate using the dtype of the tensor we write into
        # so we use float32 for the accumulation to avoid numerical issues in bf16/fp16
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
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

        if self.activation_scheme == "static" and activation_scale is not None:
            scale = activation_scale.to(torch.float32)
            qinput = (input / scale).clamp(min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        else:
            finegrained_fp8 = _load_finegrained_fp8_kernel()
            qinput, scale = finegrained_fp8.fp8_act_quant(
                input, self.block_size[1] if self.block_size is not None else input.shape[-1]
            )

        output = w8a8_fp8_matmul(
            qinput,
            weight,
            scale,
            weight_scale_inv,
            self.block_size,
            output_dtype=input.dtype,
        )
        return output.to(dtype=input.dtype)


class FP8ExpertsInterface(ExpertsInterface):
    """Interface for registering custom FP8 experts forward functions."""

    _global_mapping = {
        "batched_mm": fp8_batched_mm_experts_forward,
        "grouped_mm": fp8_grouped_mm_experts_forward,
        "deepgemm": fp8_deepgemm_experts_forward,
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
        quantization_config (`FbgemmFp8Config`):
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

        # we need this to correctly materialize the weights during quantization
        module_kwargs = {} if pre_quantized else {"dtype": None}
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
                    has_bias=has_bias,
                    has_gate=has_gate,
                    **module_kwargs,
                )
            elif isinstance(module, nn.Linear):
                new_module = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    has_bias=module.bias is not None,
                    **module_kwargs,
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
        # Broadcast scales over the block dims and quantize
        scales_broadcast = scales.unsqueeze(-1).unsqueeze(-3)  # (..., rows_tiles, 1, cols_tiles, 1)
        scaled = reshaped * scales_broadcast
        quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        quantized = quantized.reshape(original_shape)
        inv_scales = (1.0 / scales).to(torch.float32)
        scale_key = key.rsplit(".", 1)[0] + ".weight_scale_inv" if key.endswith("weight") else key + "_scale_inv"
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

    def _dequantize_one(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
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
        scale_rows, scale_cols = scales.shape[-2:]
        if rows % scale_rows or cols % scale_cols:
            raise ValueError(
                f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols})."
            )
        block_m = rows // scale_rows
        block_n = cols // scale_cols
        # ``ue8m0`` (``float8_e8m0fnu``) scales have no CUDA ``mul`` kernel, and casting
        # the FP8 weight to that dtype loses precision. Promote both sides to fp32 for
        # the math; emit in the scales' dtype when it's a real float, otherwise bf16.
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        original_shape = quantized_fp32.shape
        q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
        s = scales.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
        return (q * s).to(out_dtype).reshape(original_shape)

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor] | torch.Tensor],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        # Backward-compatible single-tensor path (the legacy fallback converter declares
        # ``["weight$", "weight_scale_inv", "activation_scale"]`` and produces a single
        # ``weight`` target). Also handles the no-scale case (e.g. RMSNorm weights that
        # match ``weight$`` but ship no ``weight_scale_inv`` alongside).
        if "weight$" in input_dict:
            quantized = input_dict["weight$"]
            quantized = quantized[0] if isinstance(quantized, list) else quantized
            if "weight_scale_inv" in input_dict:
                scales = input_dict["weight_scale_inv"]
                scales = scales[0] if isinstance(scales, list) else scales
                return {full_layer_name: self._dequantize_one(quantized, scales)}
            return {full_layer_name: quantized}

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
            result[key] = [self._dequantize_one(w, s) for w, s in zip(weights, scales)]
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        # Round-trip: dequantize on load -> re-quantize on save, so the saved
        # checkpoint preserves the FP8 format (weight + per-block ``weight_scale_inv``)
        # whether the in-memory state stayed quantized or was dequantized for compute.
        return Fp8Quantize(self.hf_quantizer)
