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
- `deepgemm_bf16_experts_forward`: BF16 M-grouped experts forward, registered as "deepgemm" in the ExpertsInterface.
- `deepgemm_fp8_fp4_linear`: end-to-end FP8/FP4 linear (BF16 in, BF16 out) — quantizes activations
  inside, dispatches cast settings on weight dtype, and runs the FP8/FP4 matmul. Used as the
  DeepGEMM fast path inside `fp8_linear`.
- `deepgemm_fp8_fp4_experts_forward`: FP8 (or FP4 on SM100+) M-grouped experts forward, registered as "deepgemm" in the FP8 ExpertsInterface.
- `deepgemm_fp8_fp4_megamoe_experts_forward`: FP8 acts × FP4 weights Mega MoE forward (SM100+,
  fuses EP dispatch + L1 + SwiGLU + L2 + EP combine via a `SymmBuffer`).

Requirements: CUDA, Hopper (SM90+), CUDA runtime >= 12.3, `kernels-community/deep-gemm` (>= 2.5
so the Mega MoE symbols are available — the loader raises if any required symbol is missing).
Mega MoE additionally requires SM100+ at call time.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import torch

from ..utils import logging
from ..utils.import_utils import get_cuda_runtime_version, is_kernels_available, resolve_internal_import
from .hub_kernels import lazy_load_kernel


logger = logging.get_logger(__name__)

# DeepGEMM requires M-dimension alignment to 128 for TMA-based contiguous grouped GEMM.
# TMA is an H100 hardware addition that allows applications to asynchronously and
# bi-directionally transfer 1D-5D tensors between GPU global and shared memory.
_DEEPGEMM_M_ALIGNMENT = 128

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


@functools.cache
def _load_deepgemm_kernel() -> SimpleNamespace:
    """
    Load DeepGEMM once and return its entry points as a `SimpleNamespace`.

    Raises `ImportError` if CUDA/hardware requirements are not met or any required entry
    point is missing.
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

    try:
        import deep_gemm as kernel
    except ImportError:
        if not is_kernels_available():
            raise ImportError(
                "DeepGEMM requires either the `deep_gemm` package (`pip install -U deep-gemm`) or "
                "the `kernels` package (`pip install -U kernels`) for the `kernels-community/deep-gemm` "
                "hub build."
            ) from None
        kernel = lazy_load_kernel("deep-gemm")
        if kernel is None:
            raise ImportError(
                "Failed to load the DeepGEMM kernel — check that `kernels-community/deep-gemm` "
                "has a build matching the current torch/CUDA."
            ) from None

    fp8_fp4_matmul = getattr(kernel, "fp8_fp4_gemm_nt", None)
    grouped_fp8_fp4_matmul = getattr(kernel, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    grouped_bf16_matmul_nt = getattr(kernel, "m_grouped_bf16_gemm_nt_contiguous", None)
    grouped_bf16_matmul_nn = getattr(kernel, "m_grouped_bf16_gemm_nn_contiguous", None)
    per_token_cast_to_fp8 = resolve_internal_import(kernel, chained_path="utils.per_token_cast_to_fp8")
    symm_buffer_cls = getattr(kernel, "SymmBuffer", None)
    fp8_fp4_mega_moe = getattr(kernel, "fp8_fp4_mega_moe", None)
    get_symm_buffer_for_mega_moe = getattr(kernel, "get_symm_buffer_for_mega_moe", None)
    transform_weights_for_mega_moe = getattr(kernel, "transform_weights_for_mega_moe", None)

    missing = [
        name
        for name, attr in [
            ("fp8_fp4_gemm_nt", fp8_fp4_matmul),
            ("m_grouped_fp8_fp4_gemm_nt_contiguous", grouped_fp8_fp4_matmul),
            ("m_grouped_bf16_gemm_nt_contiguous", grouped_bf16_matmul_nt),
            ("m_grouped_bf16_gemm_nn_contiguous", grouped_bf16_matmul_nn),
            ("utils.per_token_cast_to_fp8", per_token_cast_to_fp8),
            ("SymmBuffer", symm_buffer_cls),
            ("fp8_fp4_mega_moe", fp8_fp4_mega_moe),
            ("get_symm_buffer_for_mega_moe", get_symm_buffer_for_mega_moe),
            ("transform_weights_for_mega_moe", transform_weights_for_mega_moe),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"DeepGEMM kernel is missing required symbols: {', '.join(missing)}. "
            "Please update the `kernels` package (`pip install -U kernels`)."
        )

    return SimpleNamespace(
        fp8_fp4_matmul=fp8_fp4_matmul,
        grouped_fp8_fp4_matmul=grouped_fp8_fp4_matmul,
        grouped_bf16_matmul_nt=grouped_bf16_matmul_nt,
        grouped_bf16_matmul_nn=grouped_bf16_matmul_nn,
        per_token_cast_to_fp8=per_token_cast_to_fp8,
        transform_weights_for_mega_moe=transform_weights_for_mega_moe,
        get_symm_buffer_for_mega_moe=get_symm_buffer_for_mega_moe,
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        symm_buffer_cls=symm_buffer_cls,
    )


def _coerce_sf_for_kernel(sf: torch.Tensor) -> torch.Tensor:
    """Normalize a scale-factor tensor for the DeepGEMM kernel boundary.

    Two SF flavors are produced by our path:
      - `float32` (DeepSeek V3-style): pass through; the kernel transforms float→int internally
        on SM100 to feed the 1D1D path.
      - `torch.float8_e8m0fnu` (DeepSeek V4-style, 1 byte per scale): reinterpret 4 contiguous
        bytes as one `int32`. No copy; last-dim shrinks 4×.
    """
    if sf.dtype == torch.float8_e8m0fnu:
        return sf.contiguous().view(torch.int32)
    return sf


def deepgemm_fp8_fp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    activation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-to-end DeepGEMM linear: per-token activation quant + FP8/FP4 matmul.

    Activation cast settings are inferred from the tensor dtypes:
      - FP4 weights (`weight.dtype == torch.int8`): always gran_k=32 with packed-UE8M0 SF. Requires
        SM100+ (Blackwell).
      - FP8 weights + UE8M0 weight SFs (`weight_scale_inv.dtype == torch.float8_e8m0fnu`,
        DeepSeek V4-style): gran_k=128 with packed-UE8M0 SF (skips the kernel-side float→int SF
        transform on SM100).
      - FP8 weights + float weight SFs (DeepSeek V3-style): gran_k=128 with float SF (works on
        Hopper and Blackwell).

    Static (per-tensor) activation quantization is not supported — DeepGEMM's kernel needs per-row
    SFs and rejects scalar SFs at its host-side check. Callers should route static activations
    through the Triton fallback.
    """
    if activation_scale is not None:
        raise NotImplementedError(
            "Static (per-tensor) activation quantization is not supported on the DeepGEMM path. "
            "Use the Triton fallback for static activations."
        )

    is_fp4 = weight.dtype == torch.int8
    if is_fp4 and torch.cuda.get_device_capability(input.device)[0] < 10:
        raise RuntimeError("FP4 weights (int8-packed e2m1) require SM100+ (Blackwell).")

    deepgemm = _load_deepgemm_kernel()

    if is_fp4:
        cast_kwargs = {"use_ue8m0": True, "gran_k": 32, "use_packed_ue8m0": True}
    elif weight_scale_inv.dtype == torch.float8_e8m0fnu:
        cast_kwargs = {"use_ue8m0": True, "gran_k": 128, "use_packed_ue8m0": True}
    else:
        cast_kwargs = {"use_ue8m0": False, "gran_k": 128}
    input_2d = input.view(-1, input.shape[-1])
    qinput_2d, scale_2d = deepgemm.per_token_cast_to_fp8(input_2d, **cast_kwargs)

    output = torch.empty(qinput_2d.shape[0], weight.shape[0], device=input.device, dtype=output_dtype)
    deepgemm.fp8_fp4_matmul(
        (qinput_2d, _coerce_sf_for_kernel(scale_2d)),
        (weight, _coerce_sf_for_kernel(weight_scale_inv)),
        output,
    )
    output = output.view(input.shape[:-1] + (weight.shape[0],))
    if bias is not None:
        output.add_(bias)
    return output


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


def deepgemm_bf16_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"DeepGEMM experts path requires bfloat16 hidden states, got {hidden_states.dtype}")

    # Non-transposed HF experts have weight layout (E, N, K) -> NT kernel.
    # Transposed HF experts have weight layout (E, K, N) -> NN kernel.
    deepgemm = _load_deepgemm_kernel()
    grouped_bf16_matmul = deepgemm.grouped_bf16_matmul_nn if self.is_transposed else deepgemm.grouped_bf16_matmul_nt

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
    grouped_bf16_matmul(act, w_up, proj_out, grouped_layout, use_psum_layout=use_psum_layout)

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
    grouped_bf16_matmul(proj_out, self.down_proj, out, grouped_layout, use_psum_layout=use_psum_layout)

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


def deepgemm_fp8_fp4_experts_forward(
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

    deepgemm = _load_deepgemm_kernel()

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # FP4 weights are int8-packed (2 e2m1 values per byte; `kPackedFP4 == torch::kInt8` in DeepGEMM).
    # `m_grouped_fp8_fp4_gemm_nt_contiguous` accepts both FP8 and FP4 weight dtypes. Activation cast
    # tracks (weight dtype, weight SF dtype), mirroring `deepgemm_fp8_fp4_linear`:
    #   - FP4 weights: gran_k=32 packed-UE8M0 SF (SM100+).
    #   - FP8 weights + UE8M0 SFs: gran_k=128 packed-UE8M0 SF (skips the kernel-side float→int
    #     transform on SM100).
    #   - FP8 weights + float SFs: gran_k=128 float SF (Hopper or Blackwell).
    w_up = self.gate_up_proj if self.has_gate else self.up_proj
    ws_up = self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv
    is_fp4_weights = w_up.dtype == torch.int8

    if is_fp4_weights:
        if torch.cuda.get_device_capability(device)[0] < 10:
            raise RuntimeError(
                "FP4 expert weights (int8-packed e2m1) require SM100+ (Blackwell); use FP8 weights on Hopper."
            )
        cast_kwargs = {"use_ue8m0": True, "gran_k": 32, "use_packed_ue8m0": True}
    else:
        if self.block_size is None:
            raise ValueError(
                "DeepGEMM requires block-wise quantization (block_size=[128, 128]), "
                "but got per-tensor quantization (block_size=None)."
            )
        if self.block_size[0] != 128 or self.block_size[1] != 128:
            raise ValueError(f"DeepGEMM requires block_size=(128, 128), got {self.block_size}")
        if ws_up.dtype == torch.float8_e8m0fnu:
            cast_kwargs = {"use_ue8m0": True, "gran_k": 128, "use_packed_ue8m0": True}
        else:
            cast_kwargs = {"use_ue8m0": False, "gran_k": 128}

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
    act_fp8, act_scales = deepgemm.per_token_cast_to_fp8(selected_hidden_states_g, **cast_kwargs)
    act_fp8 = _pad_for_deepgemm(act_fp8, sorted_to_padded, total_padded_rows)
    act_scales = _pad_for_deepgemm(act_scales, sorted_to_padded, total_padded_rows)
    proj_out = torch.empty(total_padded_rows, w_up.shape[1], device=device, dtype=torch.bfloat16)
    deepgemm.grouped_fp8_fp4_matmul(
        (act_fp8, act_scales),
        (w_up, _coerce_sf_for_kernel(ws_up)),
        proj_out,
        grouped_layout,
        use_psum_layout=use_psum_layout,
    )

    # Apply gating or activation
    if self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # --- Down projection per expert (DeepGEMM grouped contiguous) ---
    proj_fp8, proj_scales = deepgemm.per_token_cast_to_fp8(proj_out, **cast_kwargs)
    proj_out = torch.empty(total_padded_rows, hidden_dim, device=device, dtype=torch.bfloat16)
    deepgemm.grouped_fp8_fp4_matmul(
        (proj_fp8, proj_scales),
        (self.down_proj, _coerce_sf_for_kernel(self.down_proj_scale_inv)),
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


def deepgemm_fp8_fp4_megamoe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """FP8 acts × FP4 weights Mega MoE forward via DeepGEMM.

    Fuses EP dispatch + L1 (FP8×FP4) + SwiGLU + L2 (FP8×FP4) + EP combine into a single
    kernel, overlapping NVLink communication with tensor-core compute. The kernel handles
    the full `(num_tokens, hidden) -> (num_tokens, hidden)` MoE forward including the
    weighted top-k reduction; the caller must NOT all-reduce the output (the EP combine is
    already inside the kernel).

    `process_group` (the EP group) is passed in by `MoeTensorParalellExperts._prepare_input_fn`
    when the module is wrapped for TP — it is required for the symmetric-buffer rendezvous on
    first forward.

    Caller-managed attributes on `self` (this dispatch does no quantization or weight
    transformation — assume they are pre-set on the module):
        - `gate_up_proj`: int8-packed FP4 L1 weight,
          shape `(num_experts_per_rank, intermediate_hidden * 2, hidden // 2)`,
          interleaved gate/up via `transform_weights_for_mega_moe`.
        - `gate_up_proj_scale_inv`: int-packed UE8M0 SF for L1, UTCCP-transposed via
          `transform_weights_for_mega_moe`.
        - `down_proj`, `down_proj_scale_inv`: same conventions for L2.

    The `SymmBuffer` is lazily allocated on first call (and re-allocated if a later call
    has more tokens than the cached buffer). The SwiGLU clamp is read from
    `self.config.swiglu_limit` if present, otherwise the kernel runs unclamped.

    Args:
        hidden_states: bf16 `(num_tokens, hidden)`.
        top_k_index: int `(num_tokens, num_topk)` of GLOBAL expert ids; -1 marks skipped
            slots (the kernel ignores them). Note: this differs from the `RouterParallel`
            output used by the other dispatches, which remaps indices to local + sentinel.
        top_k_weights: float `(num_tokens, num_topk)` routing weights.

    Returns:
        `(num_tokens, hidden)` in `hidden_states.dtype` (already weighted-summed across
        topk and reduced across EP ranks).
    """
    # Mega MoE is Blackwell-only — the impl is `sm100_fp8_fp4_mega_moe.cuh` and there is
    # no SM90 path. Use the regular "deepgemm" dispatch on Hopper.
    if torch.cuda.get_device_capability(hidden_states.device)[0] < 10:
        raise RuntimeError("DeepGEMM Mega MoE requires SM100+ (Blackwell). The 'deepgemm' dispatch supports SM90+.")

    deepgemm = _load_deepgemm_kernel()

    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    num_experts = self.gate_up_proj.size(0)
    intermediate_hidden = self.gate_up_proj.size(1) // 2
    activation_clamp = getattr(getattr(self, "config", None), "swiglu_limit", None)

    # Lazily allocate the symmetric buffer on first call (re-allocate if the cached buffer is
    # too small for this call). `process_group` is threaded in by `MoeTensorParalellExperts`.
    if getattr(self, "symm_buffer", None) is None or self.symm_buffer.num_max_tokens_per_rank < num_tokens:
        if process_group is None:
            raise ValueError(
                "DeepGEMM Mega MoE requires a `process_group` for the EP group; the TP wrapping "
                "(`MoeTensorParalellExperts`) supplies it automatically. If you are calling this "
                "dispatch directly, pass `process_group=...` explicitly."
            )
        # `gate_up_proj.size(0)` is the per-rank expert count after `GroupedGemmParallel`
        # sharding; the buffer needs the GLOBAL count (kernel asserts `num_experts % num_ranks
        # == 0` and computes the per-rank slice itself).
        num_experts_global = num_experts * process_group.size()
        self.symm_buffer = deepgemm.get_symm_buffer_for_mega_moe(
            process_group,
            hidden=hidden_dim,
            num_topk=num_top_k,
            num_experts=num_experts_global,
            num_max_tokens_per_rank=num_tokens,
            intermediate_hidden=intermediate_hidden,
        )

    # Quantize activations to FP8 with packed UE8M0 per-32 SF — the layout the kernel expects.
    x_fp8, x_sf = deepgemm.per_token_cast_to_fp8(hidden_states, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    # Stage inputs into the symmetric buffer; the kernel reads from there during dispatch.
    self.symm_buffer.x[:num_tokens].copy_(x_fp8)
    self.symm_buffer.x_sf[:num_tokens].copy_(x_sf)
    self.symm_buffer.topk_idx[:num_tokens].copy_(top_k_index)
    self.symm_buffer.topk_weights[:num_tokens].copy_(top_k_weights)

    y = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device=hidden_states.device)
    deepgemm.fp8_fp4_mega_moe(
        y,
        (self.gate_up_proj, self.gate_up_proj_scale_inv),
        (self.down_proj, self.down_proj_scale_inv),
        self.symm_buffer,
        activation_clamp=activation_clamp,
    )

    return y.to(hidden_states.dtype)
