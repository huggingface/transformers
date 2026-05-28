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
- `deepgemm_bf16_experts_forward`: BF16 M-grouped experts forward.
- `deepgemm_fp8_fp4_linear`: end-to-end FP8/FP4 linear (BF16 in, BF16 out).
- `deepgemm_fp8_fp4_experts_forward`: FP8 (or FP4 on SM100+) M-grouped experts forward.
- `deepgemm_fp8_fp4_megamoe_experts_forward`: FP8×FP4 Mega MoE forward (SM100+).

Requirements: CUDA, Hopper (SM90+), CUDA runtime ≥ 12.3, kernels-community/deep-gemm
≥ 2.5 (Mega MoE symbols required). Mega MoE additionally needs SM100+ at call time.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..utils import logging
from ..utils.import_utils import get_cuda_runtime_version, is_kernels_available, resolve_internal_import
from .hub_kernels import lazy_load_kernel
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)

# DeepGEMM requires M-dimension alignment to 128 for TMA-based contiguous grouped GEMM.
_DEEPGEMM_M_ALIGNMENT = 128
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


# ── Kernel loading ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeepGEMM:
    """Curated entry points exposed by `kernels-community/deep-gemm`."""

    fp8_fp4_matmul: Callable
    grouped_fp8_fp4_matmul: Callable
    grouped_bf16_matmul_nt: Callable
    grouped_bf16_matmul_nn: Callable
    per_token_cast_to_fp8: Callable
    get_mn_major_tma_aligned_packed_ue8m0_tensor: Callable
    transform_sf_into_required_layout: Callable
    transform_weights_for_mega_moe: Callable
    get_symm_buffer_for_mega_moe: Callable
    fp8_fp4_mega_moe: Callable


@functools.cache
def _load_deepgemm_kernel() -> DeepGEMM:
    """Load DeepGEMM once; raise `ImportError` if env or any required symbol is missing."""
    if not is_kernels_available():
        raise ImportError("DeepGEMM kernel requires the `kernels` package. Install it with `pip install -U kernels`.")
    if not torch.cuda.is_available():
        raise ImportError("DeepGEMM kernel requires CUDA, but CUDA is not available.")

    major = torch.cuda.get_device_capability()[0]
    if major < 9:
        raise ImportError(f"DeepGEMM requires Hopper (SM90+); current device is SM{major}0.")

    cuda_major, cuda_minor = get_cuda_runtime_version()
    if (cuda_major, cuda_minor) < (12, 3):
        raise ImportError(f"DeepGEMM requires CUDA runtime ≥ 12.3, found {cuda_major}.{cuda_minor}.")

    kernel = lazy_load_kernel("deep-gemm")
    if kernel is None:
        raise ImportError(
            "Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA."
        )

    fp8_fp4_matmul = getattr(kernel, "fp8_fp4_gemm_nt", None)
    grouped_fp8_fp4_matmul = getattr(kernel, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    grouped_bf16_matmul_nt = getattr(kernel, "m_grouped_bf16_gemm_nt_contiguous", None)
    grouped_bf16_matmul_nn = getattr(kernel, "m_grouped_bf16_gemm_nn_contiguous", None)
    per_token_cast_to_fp8 = resolve_internal_import(kernel, chained_path="utils.per_token_cast_to_fp8")
    get_mn_major_tma_aligned_packed_ue8m0_tensor = getattr(
        kernel, "get_mn_major_tma_aligned_packed_ue8m0_tensor", None
    )
    transform_sf_into_required_layout = getattr(kernel, "transform_sf_into_required_layout", None)
    transform_weights_for_mega_moe = getattr(kernel, "transform_weights_for_mega_moe", None)
    get_symm_buffer_for_mega_moe = getattr(kernel, "get_symm_buffer_for_mega_moe", None)
    fp8_fp4_mega_moe = getattr(kernel, "fp8_fp4_mega_moe", None)

    missing = [
        name
        for name, attr in [
            ("fp8_fp4_gemm_nt", fp8_fp4_matmul),
            ("m_grouped_fp8_fp4_gemm_nt_contiguous", grouped_fp8_fp4_matmul),
            ("m_grouped_bf16_gemm_nt_contiguous", grouped_bf16_matmul_nt),
            ("m_grouped_bf16_gemm_nn_contiguous", grouped_bf16_matmul_nn),
            ("utils.per_token_cast_to_fp8", per_token_cast_to_fp8),
            ("get_mn_major_tma_aligned_packed_ue8m0_tensor", get_mn_major_tma_aligned_packed_ue8m0_tensor),
            ("transform_sf_into_required_layout", transform_sf_into_required_layout),
            ("transform_weights_for_mega_moe", transform_weights_for_mega_moe),
            ("get_symm_buffer_for_mega_moe", get_symm_buffer_for_mega_moe),
            ("fp8_fp4_mega_moe", fp8_fp4_mega_moe),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"DeepGEMM kernel is missing required symbols: {', '.join(missing)}. Update with `pip install -U kernels`."
        )
    return DeepGEMM(
        fp8_fp4_matmul=fp8_fp4_matmul,
        grouped_fp8_fp4_matmul=grouped_fp8_fp4_matmul,
        grouped_bf16_matmul_nt=grouped_bf16_matmul_nt,
        grouped_bf16_matmul_nn=grouped_bf16_matmul_nn,
        per_token_cast_to_fp8=per_token_cast_to_fp8,
        get_mn_major_tma_aligned_packed_ue8m0_tensor=get_mn_major_tma_aligned_packed_ue8m0_tensor,
        transform_sf_into_required_layout=transform_sf_into_required_layout,
        transform_weights_for_mega_moe=transform_weights_for_mega_moe,
        get_symm_buffer_for_mega_moe=get_symm_buffer_for_mega_moe,
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
    )


# ── Scale-factor helpers ───────────────────────────────────────────────────────


def _ceil_to_ue8m0(sf: torch.Tensor) -> torch.Tensor:
    """Round each fp32 SF up to the nearest power of 2 (zero mantissa).

    Mirrors `deep_gemm.utils.math.ceil_to_ue8m0`. On SM100 the kernel's
    `pack_fp32_into_ue8m0` cleanly extracts the biased exponent only when the
    mantissa is already zero — its inner shifts (`>> 15`, `>> 7`, `<< 1`)
    otherwise leak mantissa bits into adjacent UE8M0 byte slots and silently
    corrupt the SF. SM90 consumes raw fp32 SFs without going through this path.
    """
    int_view = sf.view(torch.int32)
    return (int_view + ((1 << 23) - 1)).bitwise_and_(~((1 << 23) - 1)).view(torch.float)


def _coerce_sf_for_kernel(sf: torch.Tensor, expected_mn: int | None = None) -> torch.Tensor:
    """Lay out `sf` as DeepGEMM's `check_sf_layout` expects: MN-major
    (`stride(-2) == 1`) and TMA-aligned (`stride(-1) == align(mn, 16/esize)`).

    Inputs come in three flavors:
      - `float8_e8m0fnu`: raw UE8M0 bytes — pack 4 K-bytes → int32 (last dim /4).
      - `float32`: per-token / per-block SFs from `per_token_cast_to_fp8` or
        on-disk weights — round to UE8M0 on SM100 (see `_ceil_to_ue8m0`).
      - `int32`: already-packed UE8M0 — pass through.

    When `expected_mn` is set and the SF's M-dim is smaller (block-quantized
    UE8M0, e.g. DSv4-Flash compressor weights with `(N/128, K/128)` SFs), we
    repeat the SF on the M-axis to per-row before packing — the `(INT, 1, gran_k)`
    DeepGEMM kernel branch is the only UE8M0 path on SM100; for `gran_mn > 1`
    the kernel only handles FP32 SFs and would otherwise reject our INT SF here.
    """
    if sf.dtype == torch.float8_e8m0fnu:
        if expected_mn is not None and sf.size(-2) < expected_mn:
            gran_mn = expected_mn // sf.size(-2)
            sf = sf.repeat_interleave(gran_mn, dim=-2)
        sf = sf.contiguous().view(torch.int32)
    elif sf.dtype == torch.float32 and torch.cuda.get_device_capability(sf.device)[0] >= 10:
        sf = _ceil_to_ue8m0(sf)

    if sf.dim() not in (2, 3):
        raise ValueError(f"DeepGEMM SF must be 2D or 3D, got {sf.dim()}D")

    mn, kf = sf.size(-2), sf.size(-1)
    align_to = 16 // sf.element_size()  # `get_tma_aligned_size`: align(mn, 16 / element_size)
    aligned_mn = -(-mn // align_to) * align_to
    target_strides = (1, aligned_mn) if sf.dim() == 2 else (kf * aligned_mn, 1, aligned_mn)

    if tuple(sf.stride()) == target_strides:
        return sf
    out = torch.empty_strided(sf.shape, target_strides, dtype=sf.dtype, device=sf.device)
    out.copy_(sf)
    return out


def _select_fp8_cast_kwargs(
    weight: torch.Tensor, weight_scale_inv: torch.Tensor, block_size: tuple | None, device: torch.device
) -> dict:
    """Pick the `per_token_cast_to_fp8` kwargs from weight dtype + SF dtype.

    Three cases mirror the kernel's recipes:
      - FP4 weights (`int8`): gran_k=32 packed-UE8M0 SF. SM100+ only.
      - FP8 weights + UE8M0 SF: gran_k=128 packed-UE8M0 SF (DSv4).
      - FP8 weights + float SF: gran_k=128 float SF (DSv3).
    """
    if weight.dtype == torch.int8:  # FP4
        if torch.cuda.get_device_capability(device)[0] < 10:
            raise RuntimeError("FP4 weights (int8-packed e2m1) require SM100+ (Blackwell).")
        return {"use_ue8m0": True, "gran_k": 32, "use_packed_ue8m0": True}
    # FP8 weights: validate block_size (informational; kernel infers recipe from SF dtype/shape).
    if block_size is None:
        raise ValueError(
            "DeepGEMM requires block-wise quantized FP8 weights, but the experts have no `block_size` set."
        )
    if block_size not in ((128, 128), (1, 128)):
        raise ValueError(f"DeepGEMM requires `block_size` ∈ {{(128, 128), (1, 128)}}, got {block_size}.")
    if weight_scale_inv.dtype == torch.float8_e8m0fnu:
        return {"use_ue8m0": True, "gran_k": 128, "use_packed_ue8m0": True}
    return {"use_ue8m0": False, "gran_k": 128}


# ── Layout helpers (M-grouped contiguous, TMA-aligned) ─────────────────────────


def _build_deepgemm_contiguous_layout(
    expert_ids_sorted: torch.Tensor, num_experts: int, alignment: int, use_psum_layout: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the TMA-aligned grouped layout DeepGEMM expects.

    Returns `(sorted_to_padded, grouped_layout, total_padded_rows)`:
      - `grouped_layout` is per-row expert id (Hopper, with `-1` for padding /
        sentinels) or a cumsum of aligned per-expert counts (Blackwell).
      - EP sentinels (values == `num_experts`) are routed past the last expert
        block so DeepGEMM skips them.
    """
    device = expert_ids_sorted.device
    num_tokens = expert_ids_sorted.size(0)
    # `histc` drops values > max, so EP sentinels (== num_experts) don't count.
    tokens_per_expert = torch.histc(expert_ids_sorted.int(), bins=num_experts, min=0, max=num_experts - 1).long()
    aligned_tokens_per_expert = ((tokens_per_expert + alignment - 1) // alignment) * alignment
    # Upper bound — avoids GPU→CPU sync; padding rows are skipped.
    total_padded_rows = num_tokens + min(num_tokens, num_experts) * (alignment - 1)

    # Exclusive cumsum of per-expert padding (index `num_experts` = total padding,
    # which routes EP sentinels past all aligned blocks on Blackwell).
    padding_per_expert = aligned_tokens_per_expert - tokens_per_expert
    cumulative_padding = torch.nn.functional.pad(padding_per_expert.cumsum(0), (1, 0))
    sorted_to_padded = torch.arange(num_tokens, device=device) + cumulative_padding[expert_ids_sorted]

    if use_psum_layout:  # SM100+: kernel reads cumsum of aligned counts as expert boundaries.
        grouped_layout = aligned_tokens_per_expert.cumsum(0).int()
    else:  # SM90: per-row expert id, -1 = skip (padding & sentinels).
        grouped_layout = torch.full((total_padded_rows,), -1, device=device, dtype=torch.int32)
        grouped_layout[sorted_to_padded] = torch.where(expert_ids_sorted < num_experts, expert_ids_sorted.int(), -1)

    return sorted_to_padded, grouped_layout, total_padded_rows


def _pad_for_deepgemm(x: torch.Tensor, sorted_to_padded: torch.Tensor, total_padded_rows: int) -> torch.Tensor:
    """Pad a sorted tensor into the TMA-aligned contiguous layout."""
    padded = torch.empty(total_padded_rows, *x.shape[1:], device=x.device, dtype=x.dtype)
    padded[sorted_to_padded] = x
    return padded


def _unpad_from_deepgemm_contiguous_layout(x_padded: torch.Tensor, sorted_to_padded: torch.Tensor) -> torch.Tensor:
    return x_padded[sorted_to_padded]


# ── Routing helpers (sort → matmul → restore) ─────────────────────────────────


def _dispatch_routed_input(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    use_psum_layout: bool,
) -> tuple:
    """Sort tokens by expert id and build the M-grouped padded layout.

    Returns `(sorted_hidden_states_g, sample_weights_g, expert_ids_g,
              sentinel_mask, perm, sorted_to_padded, grouped_layout,
              total_padded_rows)`.
    """
    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    num_top_k = top_k_index.size(-1)
    expert_ids = top_k_index.reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)

    # Sort by expert for grouped processing
    expert_ids_g, perm = torch.sort(expert_ids)
    sorted_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    # Build the M-grouped padded layout (DeepGEMM contract: each expert's rows
    # start on a `_DEEPGEMM_M_ALIGNMENT` boundary, sentinels routed past valid
    # expert blocks).
    sorted_to_padded, grouped_layout, total_padded_rows = _build_deepgemm_contiguous_layout(
        expert_ids_g, num_experts, _DEEPGEMM_M_ALIGNMENT, use_psum_layout
    )

    # EP sentinel mask is captured before the in-place clamp; used by the post-mask in
    # `_combine_routed_output` to zero sentinel rows before the per-token reduction. The clamp
    # keeps any per-row gather (e.g. bias) in-bounds — bias added at sentinel positions falls
    # in rows the kernel skips, so harmless. Safe to mutate now: the layout was built from the
    # unclamped tensor and nothing downstream needs the sentinel info from `expert_ids_g` itself.
    sentinel_mask = (expert_ids_g >= num_experts).unsqueeze(-1)
    expert_ids_g.clamp_(max=num_experts - 1)
    return (
        sorted_hidden_states_g,
        sample_weights_g,
        expert_ids_g,
        sentinel_mask,
        perm,
        sorted_to_padded,
        grouped_layout,
        total_padded_rows,
    )


def _combine_routed_output(
    out_padded: torch.Tensor,
    sorted_weights: torch.Tensor,
    sentinel_mask: torch.Tensor,
    perm: torch.Tensor,
    sorted_to_padded: torch.Tensor,
    num_tokens: int,
    num_top_k: int,
    hidden_dim: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Unpad → weighted multiply → mask sentinels → restore order → top-k reduce."""
    out = _unpad_from_deepgemm_contiguous_layout(out_padded, sorted_to_padded)
    weighted = out * sorted_weights.to(out.dtype).unsqueeze(-1)
    # Sentinel rows past the valid expert blocks may carry NaN from allocator
    # reuse (`0 * NaN = NaN`); zero them so the top-k reduction stays finite.
    weighted.masked_fill_(sentinel_mask, 0.0)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=out.device)
    # Deterministic reshape+sum (index_add_ with duplicates is non-deterministic on CUDA).
    return weighted[inv_perm].view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(out_dtype)


# ── Public dispatches ──────────────────────────────────────────────────────────


def deepgemm_fp8_fp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    activation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-to-end DeepGEMM linear: per-token activation quant + FP8/FP4 matmul.

    Static (per-tensor) activation quantization is rejected — DeepGEMM needs
    per-row SFs. Callers should route static activations through the Triton fallback.
    """
    if activation_scale is not None:
        raise NotImplementedError("Static activation quantization is not supported on the DeepGEMM path.")

    deepgemm = _load_deepgemm_kernel()
    cast_kwargs = _select_fp8_cast_kwargs(weight, weight_scale_inv, block_size=(128, 128), device=input.device)

    input_2d = input.view(-1, input.shape[-1])
    qinput_2d, scale_2d = deepgemm.per_token_cast_to_fp8(input_2d, **cast_kwargs)
    output = torch.empty(qinput_2d.shape[0], weight.shape[0], device=input.device, dtype=output_dtype)

    # Pass `(1, 1, gran_k)` for int-SF paths so the kernel uses the right K granularity
    # (the default `(1, 1, 128)` mismatches FP4's gran_k=32). Float-SF leaves it None.
    sf_recipe = (1, 1, cast_kwargs["gran_k"]) if cast_kwargs.get("use_packed_ue8m0") else None
    deepgemm.fp8_fp4_matmul(
        (qinput_2d, _coerce_sf_for_kernel(scale_2d, expected_mn=qinput_2d.size(0))),
        (weight, _coerce_sf_for_kernel(weight_scale_inv, expected_mn=weight.size(0))),
        output,
        recipe=sf_recipe,
    )
    output = output.view(input.shape[:-1] + (weight.shape[0],))
    if bias is not None:
        output.add_(bias)
    return output


def deepgemm_bf16_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"DeepGEMM experts path requires bfloat16 hidden states, got {hidden_states.dtype}")

    deepgemm = _load_deepgemm_kernel()
    # Non-transposed weights (E, N, K) → NT kernel; transposed (E, K, N) → NN kernel.
    grouped_bf16_matmul = deepgemm.grouped_bf16_matmul_nn if self.is_transposed else deepgemm.grouped_bf16_matmul_nt

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.size(0), hidden_states.size(-1)

    use_psum_layout = torch.cuda.get_device_capability(device)[0] >= 10
    (
        sorted_hidden,
        sorted_weights,
        expert_ids_g,
        sentinel_mask,
        perm,
        sorted_to_padded,
        grouped_layout,
        total_padded_rows,
    ) = _dispatch_routed_input(hidden_states, top_k_index, top_k_weights, self.num_experts, use_psum_layout)

    w_up = to_local(self.gate_up_proj if self.has_gate else self.up_proj)
    w_down = to_local(self.down_proj)
    up_bias = to_local(self.gate_up_proj_bias if self.has_gate else self.up_proj_bias) if self.has_bias else None
    down_bias = to_local(self.down_proj_bias) if self.has_bias else None

    # Up projection.
    up_out_dim = w_up.shape[-1] if self.is_transposed else w_up.shape[1]
    act = _pad_for_deepgemm(sorted_hidden, sorted_to_padded, total_padded_rows)
    proj_out = torch.empty(total_padded_rows, up_out_dim, device=device, dtype=hidden_states.dtype)
    grouped_bf16_matmul(act, w_up, proj_out, grouped_layout, use_psum_layout=use_psum_layout)
    if self.has_bias:
        proj_out.index_add_(0, sorted_to_padded, up_bias[expert_ids_g])

    proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)

    # Down projection.
    out = torch.empty(total_padded_rows, hidden_dim, device=device, dtype=hidden_states.dtype)
    grouped_bf16_matmul(proj_out, w_down, out, grouped_layout, use_psum_layout=use_psum_layout)
    if self.has_bias:
        out.index_add_(0, sorted_to_padded, down_bias[expert_ids_g])

    return _combine_routed_output(
        out,
        sorted_weights,
        sentinel_mask,
        perm,
        sorted_to_padded,
        num_tokens,
        num_top_k,
        hidden_dim,
        hidden_states.dtype,
    )


def deepgemm_fp8_fp4_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if self.activation_scheme == "static":
        raise NotImplementedError(
            "DeepGEMM experts dispatch does not support activation_scheme='static'. Use 'dynamic'."
        )

    deepgemm = _load_deepgemm_kernel()
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.size(0), hidden_states.size(-1)

    w_up = to_local(self.gate_up_proj if self.has_gate else self.up_proj)
    ws_up = to_local(self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv)
    w_down = to_local(self.down_proj)
    ws_down = to_local(self.down_proj_scale_inv)

    cast_kwargs = _select_fp8_cast_kwargs(w_up, ws_up, getattr(self, "block_size", None), device)
    use_psum_layout = torch.cuda.get_device_capability(device)[0] >= 10
    (
        sorted_hidden,
        sorted_weights,
        _expert_ids_g,
        sentinel_mask,
        perm,
        sorted_to_padded,
        grouped_layout,
        total_padded_rows,
    ) = _dispatch_routed_input(hidden_states, top_k_index, top_k_weights, self.num_experts, use_psum_layout)
    sf_recipe = (1, 1, cast_kwargs["gran_k"]) if cast_kwargs.get("use_packed_ue8m0") else None

    # Up projection.
    act_fp8, act_scales = deepgemm.per_token_cast_to_fp8(sorted_hidden, **cast_kwargs)
    act_fp8 = _pad_for_deepgemm(act_fp8, sorted_to_padded, total_padded_rows)
    act_scales = _pad_for_deepgemm(act_scales, sorted_to_padded, total_padded_rows)
    proj_out = torch.empty(total_padded_rows, w_up.shape[1], device=device, dtype=torch.bfloat16)
    deepgemm.grouped_fp8_fp4_matmul(
        (act_fp8, _coerce_sf_for_kernel(act_scales, expected_mn=total_padded_rows)),
        (w_up, _coerce_sf_for_kernel(ws_up, expected_mn=w_up.size(-2))),
        proj_out,
        grouped_layout,
        recipe=sf_recipe,
        use_psum_layout=use_psum_layout,
    )
    proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)

    # Down projection.
    proj_fp8, proj_scales = deepgemm.per_token_cast_to_fp8(proj_out, **cast_kwargs)
    out = torch.empty(total_padded_rows, hidden_dim, device=device, dtype=torch.bfloat16)
    deepgemm.grouped_fp8_fp4_matmul(
        (proj_fp8, _coerce_sf_for_kernel(proj_scales, expected_mn=total_padded_rows)),
        (w_down, _coerce_sf_for_kernel(ws_down, expected_mn=w_down.size(-2))),
        out,
        grouped_layout,
        recipe=sf_recipe,
        use_psum_layout=use_psum_layout,
    )

    return _combine_routed_output(
        out,
        sorted_weights,
        sentinel_mask,
        perm,
        sorted_to_padded,
        num_tokens,
        num_top_k,
        hidden_dim,
        hidden_states.dtype,
    )


def _megamoe_setup_weights(
    self: torch.nn.Module,
    deepgemm: DeepGEMM,
    num_experts: int,
    intermediate_hidden: int,
    hidden_dim: int,
) -> None:
    """One-shot pack + permute of the L1/L2 weights into the Mega MoE UTCCP layout.

    1. Cast UE8M0 SF → FP32 and call `transform_sf_into_required_layout` → packed
       int32 in MN-major TMA-aligned layout.
    2. Run `transform_weights_for_mega_moe`: interleaves gate/up on L1 and transposes
       both SFs for UTCCP.
    3. Overwrite the loader-side parameters in place; the interleave preserves the
       `[E_local, 2*I, *]` leading dims so downstream `.size(...)` reads stay valid.

    Unwraps any `DTensor` wrappers FSDP2/EP may have placed around the loader-side
    Parameters — the kernel takes raw pointers.
    """
    gate_up_sf_raw = to_local(self.gate_up_proj_scale_inv.data)
    down_sf_raw = to_local(self.down_proj_scale_inv.data)
    # `_interleave_l1_weights` does `reshape`/`empty_like`/`copy_` and expects plain int8.
    gate_up_w = to_local(self.gate_up_proj.data).view(torch.int8).contiguous()
    down_w = to_local(self.down_proj.data).view(torch.int8).contiguous()

    gate_up_sf = deepgemm.transform_sf_into_required_layout(
        gate_up_sf_raw.float(),
        gate_up_w.size(1),  # 2 * intermediate
        hidden_dim,
        recipe=(1, 32),
        num_groups=num_experts,
    )
    down_sf = deepgemm.transform_sf_into_required_layout(
        down_sf_raw.float(),
        down_w.size(1),  # hidden
        intermediate_hidden,
        recipe=(1, 32),
        num_groups=num_experts,
    )
    (gate_up, gate_up_sf), (down, down_sf) = deepgemm.transform_weights_for_mega_moe(
        (gate_up_w, gate_up_sf),
        (down_w, down_sf),
    )
    self.gate_up_proj = torch.nn.Parameter(gate_up, requires_grad=False)
    self.gate_up_proj_scale_inv = torch.nn.Parameter(gate_up_sf, requires_grad=False)
    self.down_proj = torch.nn.Parameter(down, requires_grad=False)
    self.down_proj_scale_inv = torch.nn.Parameter(down_sf, requires_grad=False)
    self._megamoe_transformed = True


def deepgemm_fp8_fp4_megamoe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """FP8 acts × FP4 weights Mega MoE forward (SM100+).

    Fuses EP dispatch + L1 + SwiGLU + L2 + EP combine into one kernel,
    overlapping NVLink with tensor-core compute. The kernel handles the full
    `(num_tokens, hidden) → (num_tokens, hidden)` MoE forward including the
    weighted top-k reduction; the caller must NOT all-reduce the output.

    `process_group` is supplied automatically by `MoeTensorParalellExperts._prepare_input_fn`
    when the module is wrapped for TP — it's required for the symm-buffer rendezvous
    on first forward. `top_k_index` is GLOBAL expert ids (`-1` marks skipped slots).

    Caller-managed `self` attributes:
      - `gate_up_proj`, `gate_up_proj_scale_inv`: L1 weight + UE8M0 SF.
      - `down_proj`, `down_proj_scale_inv`: L2 weight + UE8M0 SF.
      Both pairs must be transformed together via
      `transform_weights_for_mega_moe((gate_up, gate_up_sf), (down, down_sf))`.
      - `config.swiglu_limit` (optional): SwiGLU clamp; absent → unclamped.
    """
    if torch.cuda.get_device_capability(hidden_states.device)[0] < 10:
        raise RuntimeError("DeepGEMM Mega MoE requires SM100+ (Blackwell). Use the 'deepgemm' dispatch on Hopper.")

    if process_group is None:
        raise ValueError(
            "DeepGEMM Mega MoE requires a `process_group` for the EP group. The TP wrapping "
            "(MoeTensorParalellExperts) supplies it automatically; pass it explicitly otherwise."
        )

    deepgemm = _load_deepgemm_kernel()
    num_tokens, hidden_dim = hidden_states.size(0), hidden_states.size(-1)
    num_top_k = top_k_index.size(-1)
    num_experts = self.gate_up_proj.size(0)
    intermediate_hidden = self.gate_up_proj.size(1) // 2

    # First-forward one-shot: pack UE8M0 SFs and interleave the L1/L2 weights for UTCCP.
    # The kernel asserts `sf.dtype == torch.int` so the raw loader-side scale_inv (UE8M0)
    # can't be passed directly; setup overwrites the same attributes with transformed views.
    if not getattr(self, "_megamoe_transformed", False):
        _megamoe_setup_weights(self, deepgemm, num_experts, intermediate_hidden, hidden_dim)

    # Lazily (re)allocate the symmetric buffer when the cached one is too small.
    if getattr(self, "symm_buffer", None) is None or self.symm_buffer.num_max_tokens_per_rank < num_tokens:
        self.symm_buffer = deepgemm.get_symm_buffer_for_mega_moe(
            process_group,
            hidden=hidden_dim,
            num_topk=num_top_k,
            num_experts=num_experts * process_group.size(),  # global count
            num_max_tokens_per_rank=num_tokens,
            intermediate_hidden=intermediate_hidden,
        )

    x_fp8, x_sf = deepgemm.per_token_cast_to_fp8(hidden_states, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    self.symm_buffer.x[:num_tokens].copy_(x_fp8)
    self.symm_buffer.x_sf[:num_tokens].copy_(x_sf)
    self.symm_buffer.topk_idx[:num_tokens].copy_(top_k_index)
    self.symm_buffer.topk_weights[:num_tokens].copy_(top_k_weights)

    # `activation_clamp` must match `_apply_gate`'s clamp on the regular path so the kernel's
    # fused SwiGLU sees the same value range the model was calibrated for.
    y = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device=hidden_states.device)
    deepgemm.fp8_fp4_mega_moe(
        y,
        (self.gate_up_proj, self.gate_up_proj_scale_inv),
        (self.down_proj, self.down_proj_scale_inv),
        self.symm_buffer,
        activation_clamp=getattr(getattr(self, "config", None), "swiglu_limit", None),
    )
    return y.to(hidden_states.dtype)
