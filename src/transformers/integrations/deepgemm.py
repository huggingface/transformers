# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""DeepGEMM integration: fused grouped GEMM kernels from `kernels-community/deep-gemm`.

Provides:
- `fp8_deepgemm_matmul`: FP8 dense matmul used as a fast path inside the finegrained-fp8 Linear.
- `fp8_deepgemm_experts_forward`: FP8 M-grouped experts forward, registered as "deepgemm" in the FP8 ExpertsInterface.
- `deepgemm_experts_forward`: BF16 M-grouped experts forward, registered as "deepgemm" in the ExpertsInterface.

Requirements: CUDA, Hopper (SM90+), CUDA runtime >= 12.3, `kernels`.
"""

from __future__ import annotations

import functools

import torch

from ..utils import logging
from ..utils.import_utils import get_cuda_runtime_version, is_kernels_available, resolve_internal_import
from .hub_kernels import lazy_load_kernel


logger = logging.get_logger(__name__)

# DeepGEMM requires M-dimension alignment to 128 for TMA-based contiguous grouped GEMM.
# TMA is an H100 hardware addition that allows applications to asynchronously and
# bi-directionally transfer 1D-5D tensors between GPU global and shared memory.
_DEEPGEMM_M_ALIGNMENT = 128


@functools.cache
def _load_deepgemm_kernel():
    """
    Load DeepGEMM once and return its required symbols.

    Raises:
        ImportError if CUDA/hardware requirements are not met, or the kernel or
        required symbols are not found.

    Returns:
        Tuple of (deepgemm_fp8_matmul, deepgemm_grouped_fp8_matmul,
                  deepgemm_grouped_bf16_matmul_nt, deepgemm_grouped_bf16_matmul_nn,
                  deepgemm_per_token_cast_to_fp8) from the DeepGEMM kernel.
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

    deepgemm_fp8_matmul = getattr(kernel, "fp8_gemm_nt", None)
    deepgemm_grouped_fp8_matmul = getattr(kernel, "m_grouped_fp8_gemm_nt_contiguous", None)
    deepgemm_grouped_bf16_matmul_nt = getattr(kernel, "m_grouped_bf16_gemm_nt_contiguous", None)
    deepgemm_grouped_bf16_matmul_nn = getattr(kernel, "m_grouped_bf16_gemm_nn_contiguous", None)
    deepgemm_per_token_cast_to_fp8 = resolve_internal_import(kernel, chained_path="utils.per_token_cast_to_fp8")

    missing = [
        name
        for name, attr in [
            ("fp8_gemm_nt", deepgemm_fp8_matmul),
            ("m_grouped_fp8_gemm_nt_contiguous", deepgemm_grouped_fp8_matmul),
            ("m_grouped_bf16_gemm_nt_contiguous", deepgemm_grouped_bf16_matmul_nt),
            ("m_grouped_bf16_gemm_nn_contiguous", deepgemm_grouped_bf16_matmul_nn),
            ("utils.per_token_cast_to_fp8", deepgemm_per_token_cast_to_fp8),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"DeepGEMM kernel is missing required symbols: {', '.join(missing)}. "
            "Please update the `kernels` package (`pip install -U kernels`)."
        )

    return (
        deepgemm_fp8_matmul,
        deepgemm_grouped_fp8_matmul,
        deepgemm_grouped_bf16_matmul_nt,
        deepgemm_grouped_bf16_matmul_nn,
        deepgemm_per_token_cast_to_fp8,
    )


def fp8_deepgemm_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 dense matmul via DeepGEMM's `fp8_gemm_nt`. Block-wise 128x128 scales expected.

    Args:
        A:  (M, K) float8_e4m3fn — quantized activations
        B:  (N, K) float8_e4m3fn — quantized weights
        As: (M, K//128) float32 — per-block activation scales
        Bs: (N//128, K//128) float32 — per-block weight scales
        output_dtype: desired output dtype.
    """
    deepgemm_fp8_matmul, _, _, _, _ = _load_deepgemm_kernel()
    A_2d = A.view(-1, A.shape[-1])
    As_2d = As.view(-1, As.shape[-1])
    output = torch.empty(A_2d.shape[0], B.shape[0], device=A.device, dtype=output_dtype)
    deepgemm_fp8_matmul((A_2d, As_2d.float()), (B, Bs.float()), output)
    return output.view(A.shape[:-1] + (B.shape[0],))


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


def _pad_for_deepgemm(x: torch.Tensor, sorted_to_padded: torch.Tensor, total_padded_rows: int) -> torch.Tensor:
    """Pad a sorted tensor into the TMA-aligned contiguous layout.

    Padding rows are left uninitialized — the kernel skips them via `grouped_layout=-1` (Hopper)
    or via the psum offsets (Blackwell), so their values never enter the computation.
    """
    padded = torch.empty(total_padded_rows, *x.shape[1:], device=x.device, dtype=x.dtype)
    padded[sorted_to_padded] = x
    return padded


def _unpad_from_deepgemm_contiguous_layout(x_padded: torch.Tensor, sorted_to_padded: torch.Tensor) -> torch.Tensor:
    """Remove padding rows from the TMA-aligned contiguous layout."""
    return x_padded[sorted_to_padded]


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

    _, deepgemm_grouped_fp8_matmul, _, _, deepgemm_per_token_cast_to_fp8 = _load_deepgemm_kernel()

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # EP sentinel handling: leave `expert_ids` unclamped so the sort pushes sentinels to the tail,
    # `_build_deepgemm_contiguous_layout` marks their positions as skipped (-1 on Hopper, beyond the
    # cumsum on Blackwell), and DeepGEMM skips them — so sentinels cost no real GEMM compute.
    # Sentinel rows are zeroed post-weighted-mul (see below), since the kernel leaves them uninitialized.
    expert_ids_g, perm = torch.sort(expert_ids)
    selected_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    use_psum_layout = torch.cuda.get_device_capability(device)[0] >= 10
    sorted_to_padded, grouped_layout, total_padded_rows = _build_deepgemm_contiguous_layout(
        expert_ids_g, self.num_experts, alignment=_DEEPGEMM_M_ALIGNMENT, use_psum_layout=use_psum_layout
    )

    # --- Up projection per expert (DeepGEMM grouped contiguous) ---
    w_up = self.gate_up_proj if self.has_gate else self.up_proj
    ws_up = self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv
    act_fp8, act_scales = deepgemm_per_token_cast_to_fp8(selected_hidden_states_g, use_ue8m0=False)
    act_fp8 = _pad_for_deepgemm(act_fp8, sorted_to_padded, total_padded_rows)
    act_scales = _pad_for_deepgemm(act_scales, sorted_to_padded, total_padded_rows)
    proj_out = torch.empty(total_padded_rows, w_up.shape[1], device=device, dtype=torch.bfloat16)
    deepgemm_grouped_fp8_matmul(
        (act_fp8, act_scales), (w_up, ws_up.float()), proj_out, grouped_layout, use_psum_layout=use_psum_layout
    )

    # Apply gating or activation
    if self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # --- Down projection per expert (DeepGEMM grouped contiguous) ---
    proj_fp8, proj_scales = deepgemm_per_token_cast_to_fp8(proj_out, use_ue8m0=False)
    proj_out = torch.empty(total_padded_rows, hidden_dim, device=device, dtype=torch.bfloat16)
    deepgemm_grouped_fp8_matmul(
        (proj_fp8, proj_scales),
        (self.down_proj, self.down_proj_scale_inv.float()),
        proj_out,
        grouped_layout,
        use_psum_layout=use_psum_layout,
    )

    # Remove padding rows
    proj_out = _unpad_from_deepgemm_contiguous_layout(proj_out, sorted_to_padded)

    # Apply routing weights
    weighted_out = proj_out * sample_weights_g.to(proj_out.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # EP sentinel handling: `proj_out` rows past the valid expert blocks are left uninitialized by the kernel,
    # so `proj_out[sentinel] * 0 = 0 * NaN = NaN` can leak from allocator pool reuse. Zero them here
    # so the downstream reduction stays finite even when the routing weight was already zero.
    weighted_out.masked_fill_((expert_ids_g >= self.num_experts).unsqueeze(-1), 0.0)

    # Restore original order
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted_out = weighted_out[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def deepgemm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"DeepGEMM experts path requires bfloat16 hidden states, got {hidden_states.dtype}")

    # Non-transposed HF experts have weight layout (E, N, K) -> NT kernel.
    # Transposed HF experts have weight layout (E, K, N) -> NN kernel.
    _, _, deepgemm_grouped_bf16_matmul_nt, deepgemm_grouped_bf16_matmul_nn, _ = _load_deepgemm_kernel()
    deepgemm_grouped_bf16_matmul = (
        deepgemm_grouped_bf16_matmul_nn if self.is_transposed else deepgemm_grouped_bf16_matmul_nt
    )

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # EP sentinel handling: leave `expert_ids` unclamped so the sort pushes sentinels to the tail,
    # `_build_deepgemm_contiguous_layout` marks their positions as skipped (-1 on Hopper, beyond the
    # cumsum on Blackwell), and DeepGEMM skips them — so sentinels cost no real GEMM compute.
    # Sentinel rows are zeroed post-weighted-mul (see below), since the kernel leaves them uninitialized.
    expert_ids_g, perm = torch.sort(expert_ids)
    selected_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    use_psum_layout = torch.cuda.get_device_capability(device)[0] >= 10
    sorted_to_padded, grouped_layout, total_padded_rows = _build_deepgemm_contiguous_layout(
        expert_ids_g, self.num_experts, alignment=_DEEPGEMM_M_ALIGNMENT, use_psum_layout=use_psum_layout
    )

    if self.has_bias:
        # Clamp now that the layout has been built — needed for the per-row bias gather below to stay
        # in-bounds. Bias added to sentinel positions falls in rows the kernel skips, so harmless.
        expert_ids_g.clamp_(0, self.num_experts - 1)

    # --- Up projection per expert (DeepGEMM grouped contiguous, bf16) ---
    w_up = self.gate_up_proj if self.has_gate else self.up_proj
    # Output dim is the last weight axis when transposed (E, K, N), second axis when not (E, N, K).
    up_out_dim = w_up.shape[-1] if self.is_transposed else w_up.shape[1]
    act = _pad_for_deepgemm(selected_hidden_states_g, sorted_to_padded, total_padded_rows)
    proj_out = torch.empty(total_padded_rows, up_out_dim, device=device, dtype=hidden_states.dtype)
    deepgemm_grouped_bf16_matmul(act, w_up, proj_out, grouped_layout, use_psum_layout=use_psum_layout)

    # The kernel has no bias input -> add per-expert bias in-place on the unpadded slice;
    # padding rows get discarded at unpad time.
    if self.has_bias:
        up_bias = self.gate_up_proj_bias if self.has_gate else self.up_proj_bias
        proj_out.index_add_(0, sorted_to_padded, up_bias[expert_ids_g])

    # Apply gating or activation
    if self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # --- Down projection per expert (DeepGEMM grouped contiguous, bf16) ---
    out = torch.empty(total_padded_rows, hidden_dim, device=device, dtype=hidden_states.dtype)
    deepgemm_grouped_bf16_matmul(proj_out, self.down_proj, out, grouped_layout, use_psum_layout=use_psum_layout)

    if self.has_bias:
        out.index_add_(0, sorted_to_padded, self.down_proj_bias[expert_ids_g])

    # Remove padding rows
    out = _unpad_from_deepgemm_contiguous_layout(out, sorted_to_padded)

    # Apply routing weights
    weighted_out = out * sample_weights_g.to(out.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # EP sentinel handling: `out` rows past the valid expert blocks are left uninitialized by the kernel,
    # so `out[sentinel] * 0 = 0 * NaN = NaN` can leak from allocator pool reuse. Zero them here
    # so the downstream reduction stays finite even when the routing weight was already zero.
    weighted_out.masked_fill_((expert_ids_g >= self.num_experts).unsqueeze(-1), 0.0)

    # Restore original order
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted_out = weighted_out[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)
