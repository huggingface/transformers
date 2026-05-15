# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GgufLinear — inference-time replacement for ``nn.Linear`` that runs Q4_0 /
Q8_0 / Q4_K / Q5_K / Q6_K / IQ4_NL / IQ4_XS matmul directly via Metal kernels,
keeping the weight at its native quantization (3.5–4× less RAM than bf16).

The forward path uses ``kernels-community`` Metal kernels (a port of
``llama.cpp`` / ``candle``'s mul_mm and mul_mv templates). On Apple Silicon
this gives **llama.cpp parity** on a single batched command buffer per token
(266 tok/s vs 261 tok/s for Qwen2.5-0.5B Q4_0 on M3 Max).

Activation flow on MPS:
  - Batch-size 1 (decode):     ``mul_mat_vec_<fmt>_f32`` — memory-bound, ~1.27×
                               faster than bf16 mat-vec on MPS.
  - Batch-size > 1 (prefill):  ``mul_mat_<fmt>_f32`` — compute-bound, ~10%
                               slower than bf16 GEMM on MPS but keeps weights
                               at 4–5 bpw so very large models actually fit.

CPU/CUDA: no fast path today; falls back to ``dequantize_gguf_tensor`` +
``torch.nn.functional.linear``. The CUDA matmul kernels are a follow-up
(candle ships them; same shape as the MPS port).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn


if TYPE_CHECKING:
    import gguf


# ---------------------------------------------------------------------------
# Lazy load of the kernels-community Metal kernel package. The kernel repo-id
# is overridable via env var so we can point at staging/personal namespaces
# without code changes. Off when `kernels` isn't installed → CPU/CUDA fallback.
# ---------------------------------------------------------------------------

_GGUF_METAL_KERNELS = None
_GGUF_METAL_LOADED = False


class _DirectKernelHandle:
    """Tiny stand-in for the kernels package's module object — exposes only
    ``_ops`` (a ``torch._C._OpNamespace``) which is the single attribute our
    forward path reads. Used when the ``kernels`` package can't load the repo
    cleanly (its older snapshot_download flow uses a non-recursive ``*`` glob
    that drops the nested ``gguf_dequant/__init__.py`` file, see
    huggingface/kernels#TBD)."""

    def __init__(self, ops):
        self._ops = ops


def _load_kernels_direct(repo: str):
    """Pull the .so for the current torch variant directly from the Hub and
    register its ops via ``torch.ops.load_library``. Returns a stub with a
    ``_ops`` attribute, matching the shape of the ``kernels`` module object.
    """
    import torch
    from huggingface_hub import snapshot_download
    try:
        from kernels.utils import build_variant
    except Exception:
        # Compute the variant string ourselves if kernels isn't installed.
        major, minor = torch.__version__.split(".")[:2]
        build_variant = lambda: f"torch{int(major)}{int(minor)}-metal-aarch64-darwin"  # noqa: E731

    variant = build_variant() if callable(build_variant) else build_variant
    # `**` to grab the nested package files the kernels package itself misses.
    repo_dir = snapshot_download(repo, allow_patterns=[f"build/{variant}/**"])
    import os, re
    var_dir = os.path.join(repo_dir, "build", variant)
    # The canonical .so name lives in ``_ops.py``: ``from . import _<ns>``.
    # Read it to pin the right .so when several builds are shipped side-by-side.
    ops_py = os.path.join(var_dir, "_ops.py")
    with open(ops_py) as f:
        m = re.search(r"from \. import (\w+)", f.read())
    if not m:
        raise FileNotFoundError(f"No `from . import` directive in {ops_py}")
    ns_name = m.group(1)
    so = os.path.join(var_dir, f"{ns_name}.abi3.so")
    if not os.path.exists(so):
        raise FileNotFoundError(f"Canonical .so missing: {so}")
    torch.ops.load_library(so)
    # ``ns_name`` includes a leading underscore from the .so file convention.
    ns = getattr(torch.ops, ns_name)
    return _DirectKernelHandle(ns)


def _ensure_metal_kernels():
    """Idempotent loader for the kernels-community Metal kernel package.

    Tries the ``kernels`` package first; on failure (e.g. when the package's
    snapshot pattern strips the nested module init), falls back to a direct
    ``snapshot_download + torch.ops.load_library`` path that's resilient to
    the kernels lib's layout assumptions.
    """
    global _GGUF_METAL_KERNELS, _GGUF_METAL_LOADED
    if _GGUF_METAL_LOADED:
        return _GGUF_METAL_KERNELS
    _GGUF_METAL_LOADED = True
    if os.environ.get("TRANSFORMERS_GGUF_USE_METAL_KERNELS", "1") == "0":
        return None
    repo = os.environ.get("TRANSFORMERS_GGUF_METAL_KERNELS_REPO", "ArthurZ/gguf-kernels")
    try:
        from kernels import get_kernel  # type: ignore[import-not-found]
        _GGUF_METAL_KERNELS = get_kernel(repo)
    except Exception:
        try:
            _GGUF_METAL_KERNELS = _load_kernels_direct(repo)
        except Exception:
            _GGUF_METAL_KERNELS = None
    return _GGUF_METAL_KERNELS


# Per quant type: (block bytes, block elems). Format names match the suffix of
# the ``mul_mat_<fmt>_f32`` and ``mul_mat_vec_<fmt>_f32`` ops in the kernel.
_QUANT_INFO: dict[str, tuple[int, int]] = {
    "Q4_0":   (18, 32),
    "Q5_0":   (22, 32),
    "Q5_1":   (24, 32),
    "Q8_0":   (34, 32),
    "Q4_K":   (144, 256),
    "Q5_K":   (176, 256),
    "Q6_K":   (210, 256),
    "IQ4_NL": (18, 32),
    "IQ4_XS": (136, 256),
}


def gguf_linear_supports(quant_type) -> bool:
    """Return True if `quant_type` has a matching mul_mat/mul_mat_vec kernel."""
    name = quant_type.name if hasattr(quant_type, "name") else str(quant_type)
    return name in _QUANT_INFO


def _kernel_fmt(quant_type) -> str:
    name = quant_type.name if hasattr(quant_type, "name") else str(quant_type)
    return {
        "Q4_0": "q4_0",
        "Q5_0": "q5_0",
        "Q5_1": "q5_1",
        "Q8_0": "q8_0",
        "Q4_K": "q4_K",
        "Q5_K": "q5_K",
        "Q6_K": "q6_K",
        "IQ4_NL": "iq4_nl",
        "IQ4_XS": "iq4_xs",
    }[name]


class GgufLinear(nn.Module):
    """Linear layer backed by GGUF-quantized weights.

    Stores raw block bytes in ``qweight`` (uint8). Forward picks the matvec
    kernel for batch-size 1 and the matmul kernel otherwise. Inference-only:
    backward raises.

    Args:
        in_features: K dimension (must be a multiple of the format's block size).
        out_features: M dimension (must be a multiple of the format's matvec
            row alignment, and of 64 for matmul to land cleanly on the tile).
        quant_type: GGUF quantization name (string), e.g. ``"Q4_K"``.
        bias: whether to register a learnable fp32 bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: str = "Q4_K",
        bias: bool = False,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        if quant_type not in _QUANT_INFO:
            raise NotImplementedError(
                f"GgufLinear only supports {sorted(_QUANT_INFO)} today, got {quant_type}"
            )
        block_bytes, block_elems = _QUANT_INFO[quant_type]
        if in_features % block_elems != 0:
            raise ValueError(
                f"in_features must be a multiple of {block_elems} for {quant_type}, got {in_features}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self._block_bytes = block_bytes
        self._block_elems = block_elems
        self._fmt = _kernel_fmt(type("X", (), {"name": quant_type})())

        nblocks_per_row = in_features // block_elems
        nbytes = out_features * nblocks_per_row * block_bytes
        self.register_buffer(
            "qweight",
            torch.empty(nbytes, dtype=torch.uint8, device=device or "cpu"),
            persistent=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float32, device=device or "cpu"),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_type={self.quant_type}, bias={self.bias is not None}"
        )

    # ``quant_type`` isn't a tensor, so it doesn't ride along in state_dict() by
    # default. ``get_extra_state`` / ``set_extra_state`` is the standard hook
    # for that — save_pretrained / load_state_dict will carry the quant type so
    # a reloaded module can be validated against the bytes it's about to receive.
    def get_extra_state(self) -> dict:
        return {
            "quant_type": self.quant_type,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }

    def set_extra_state(self, state: dict) -> None:
        qt = state.get("quant_type")
        if qt is not None and qt != self.quant_type:
            raise ValueError(
                f"GgufLinear quant_type mismatch on load: state has {qt!r}, "
                f"module was constructed with {self.quant_type!r}"
            )
        for name in ("in_features", "out_features"):
            v = state.get(name)
            if v is not None and v != getattr(self, name):
                raise ValueError(
                    f"GgufLinear {name} mismatch on load: state has {v}, "
                    f"module has {getattr(self, name)}"
                )

    def _bind_kernels(self) -> None:
        """Cache the matvec/matmul op references at construct time so the hot
        forward path avoids per-call ``_ensure_metal_kernels`` + ``hasattr`` +
        ``getattr`` round-trips. Re-callable; the bench helper invokes this
        again after ``.to(device)`` if needed (the op refs themselves don't
        change on device move)."""
        mod = _ensure_metal_kernels()
        if mod is None:
            self._mv_op = None
            self._mat_op = None
            return
        ops = mod._ops
        self._mv_op  = getattr(ops, f"mul_mat_vec_{self._fmt}_f32", None) \
            if hasattr(ops, f"mul_mat_vec_{self._fmt}_f32") else None
        self._mat_op = getattr(ops, f"mul_mat_{self._fmt}_f32", None) \
            if hasattr(ops, f"mul_mat_{self._fmt}_f32") else None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Last dim of x must be {self.in_features}, got {x.shape[-1]}")

        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features).contiguous().to(torch.float32)
        N = x_flat.shape[0]

        is_mps = x.device.type == "mps"
        if is_mps:
            # ``_bind_kernels`` populates these once per module; lazy-fetch on
            # the first forward if it hasn't run yet (state-dict load path).
            mv_op = getattr(self, "_mv_op", None)
            mat_op = getattr(self, "_mat_op", None)
            if mv_op is None and mat_op is None:
                self._bind_kernels()
                mv_op = self._mv_op
                mat_op = self._mat_op
            if mv_op is None and mat_op is None:
                return self._dequant_forward(x)

            qw = self.qweight  # already on the right device after model.to(...)
            if N == 1 and mv_op is not None:
                y = torch.empty(self.out_features, dtype=torch.float32, device=x.device)
                mv_op(qw, x_flat.reshape(-1), y)
                y = y.reshape(*batch_shape, self.out_features)
            elif N % 32 == 0 and mat_op is not None:
                y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
                mat_op(qw, x_flat, y)
                y = y.reshape(N, self.out_features).reshape(*batch_shape, self.out_features)
            elif mv_op is not None:
                rows = []
                for i in range(N):
                    row_y = torch.empty(self.out_features, dtype=torch.float32, device=x.device)
                    mv_op(qw, x_flat[i].reshape(-1), row_y)
                    rows.append(row_y)
                y = torch.stack(rows, dim=0).reshape(*batch_shape, self.out_features)
            else:
                return self._dequant_forward(x)
        else:
            y = self._dequant_forward(x)

        if self.bias is not None:
            y = y + self.bias.to(y.device).to(y.dtype)
        return y

    def _dequant_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CPU / CUDA fallback: dequantize weight on-the-fly, run torch.linear."""
        import gguf

        from ..integrations.gguf_dequant import dequantize_gguf_tensor

        qt = getattr(gguf.GGMLQuantizationType, self.quant_type)
        w = dequantize_gguf_tensor(self.qweight, qt, device=x.device).reshape(
            self.out_features, self.in_features
        )
        return torch.nn.functional.linear(x.to(w.dtype), w)


# =============================================================================
# GgufQwen2MoeExperts — quantized replacement for ``Qwen2MoeExperts``.
# =============================================================================
#
# ``Qwen2MoeExperts`` holds the *entire* MoE block's expert weights:
#   ``gate_up_proj``: (num_experts, 2*intermediate, hidden)   fp32
#   ``down_proj``   : (num_experts, hidden, intermediate)     fp32
# For Qwen2-MoE-A2.7B that's ~520M params per layer × 24 layers — by far the
# biggest memory line item. Keeping these in fp32 defeats the whole "GGUF
# quantized weights" pitch for MoE models.
#
# This module mirrors the original module's forward pass but stores the gate,
# up and down weights as *flat uint8 quantized buffers* (one per kind) — 3.5×
# smaller for Q4_K. Forward iterates over activated experts (same as the
# original) and per expert dispatches the matching ``mul_mat_vec`` / ``mul_mat``
# Metal kernel against the right byte slice.
#
# Note: the GGUF file ships ``ffn_gate_exps`` and ``ffn_up_exps`` as separate
# tensors. The standard transformers loader *interleaves* them into a single
# ``gate_up_proj`` fp32 tensor during dequant. We keep them separate as
# ``gate_proj_q`` and ``up_proj_q`` because byte-level interleave preserves
# block structure only when we concatenate full rows — separate buffers are
# simpler and equivalent.


class GgufQwen2MoeExperts(nn.Module):
    """Drop-in for ``Qwen2MoeExperts`` with quantized expert weights.

    Q4_K_M GGUF files (and several other mixed-precision .gguf builds) ship
    ``ffn_gate_exps`` / ``ffn_up_exps`` and ``ffn_down_exps`` in **different**
    quant types per layer (e.g. gate/up = Q4_K, down = Q8_0). Each projection
    therefore carries its own quant type + format.
    """

    def __init__(
        self,
        config,
        gate_quant: str,
        up_quant: str,
        down_quant: str,
        device=None,
    ):
        super().__init__()
        for label, qt in (("gate", gate_quant), ("up", up_quant), ("down", down_quant)):
            if qt not in _QUANT_INFO:
                raise NotImplementedError(
                    f"GgufQwen2MoeExperts {label}_quant {qt} not in {sorted(_QUANT_INFO)}"
                )
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_quant = gate_quant
        self.up_quant = up_quant
        self.down_quant = down_quant
        self._gate_fmt = _kernel_fmt(type("X", (), {"name": gate_quant})())
        self._up_fmt = _kernel_fmt(type("X", (), {"name": up_quant})())
        self._down_fmt = _kernel_fmt(type("X", (), {"name": down_quant})())

        def _bytes_per_expert(qt: str, M: int, K: int) -> int:
            bb, be = _QUANT_INFO[qt]
            assert K % be == 0, f"{qt}: K={K} must be a multiple of {be}"
            return M * (K // be) * bb

        # gate / up : (intermediate, hidden);  down : (hidden, intermediate)
        self._gate_bytes_per = _bytes_per_expert(gate_quant, self.intermediate_dim, self.hidden_dim)
        self._up_bytes_per   = _bytes_per_expert(up_quant,   self.intermediate_dim, self.hidden_dim)
        self._down_bytes_per = _bytes_per_expert(down_quant, self.hidden_dim,        self.intermediate_dim)

        # Per-expert byte buffers stored as ``(num_experts, bytes_per_expert)``
        # 2D tensors so the batched-mm forward can ``buf[expert_ids]`` natively
        # into an ``(S, bytes_per_expert)`` gather without manual slicing.
        self.register_buffer(
            "gate_proj_q",
            torch.empty(self.num_experts, self._gate_bytes_per, dtype=torch.uint8, device=device or "cpu"),
            persistent=True,
        )
        self.register_buffer(
            "up_proj_q",
            torch.empty(self.num_experts, self._up_bytes_per, dtype=torch.uint8, device=device or "cpu"),
            persistent=True,
        )
        self.register_buffer(
            "down_proj_q",
            torch.empty(self.num_experts, self._down_bytes_per, dtype=torch.uint8, device=device or "cpu"),
            persistent=True,
        )

        from ..activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]
        # Used by the moe.py ``ExpertsInterface`` dispatcher. Setting it via
        # the parent config (in :func:`replace_qwen2_moe_experts`) is what
        # ultimately routes the forward into ``gguf_bmm_experts_forward``.
        # has_gate / is_transposed / has_bias / is_concatenated are read by
        # the generic dispatchers — none of them apply directly to our layout
        # but we set sane defaults for the registry contract.
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = False

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, "
            f"intermediate_dim={self.intermediate_dim}, "
            f"gate={self.gate_quant}, up={self.up_quant}, down={self.down_quant}"
        )

    def get_extra_state(self) -> dict:
        return {
            "gate_quant": self.gate_quant,
            "up_quant": self.up_quant,
            "down_quant": self.down_quant,
            "num_experts": self.num_experts,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
        }

    def set_extra_state(self, state: dict) -> None:
        for name in ("gate_quant", "up_quant", "down_quant",
                     "num_experts", "hidden_dim", "intermediate_dim"):
            v = state.get(name)
            if v is not None and v != getattr(self, name):
                raise ValueError(
                    f"GgufQwen2MoeExperts {name} mismatch on load: "
                    f"state has {v!r}, module has {getattr(self, name)!r}"
                )

    def _gather_dequant(self, buf_2d: torch.Tensor, expert_ids: torch.Tensor, quant_name: str, fmt: str,
                        M: int, K: int) -> torch.Tensor:
        """Gather + dequant the selected experts in one shot.

        ``buf_2d`` is ``(num_experts, bytes_per_expert)`` uint8; the gather
        produces ``(S, bytes_per_expert)`` then flattens for the per-block
        Metal dequant kernel, which dispatches threadgroups across all S
        experts in a single launch. Returns ``(S, M, K)`` fp32.
        """
        gathered = buf_2d.index_select(0, expert_ids).contiguous()  # (S, bytes_per)
        flat = gathered.view(-1)
        S = expert_ids.shape[0]
        out = torch.empty(S * M * K, dtype=torch.float32, device=flat.device)

        if flat.device.type == "mps":
            mod = _ensure_metal_kernels()
            op_name = f"dequantize_{fmt}"
            if mod is not None and hasattr(mod._ops, op_name):
                getattr(mod._ops, op_name)(flat, out)
                return out.view(S, M, K)

        import gguf
        from ..integrations.gguf_dequant import dequantize_gguf_tensor
        qt = getattr(gguf.GGMLQuantizationType, quant_name)
        return dequantize_gguf_tensor(flat, qt, device=flat.device).view(S, M, K)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch into the ``ExpertsInterface`` registry — same pattern as
        :class:`FP8Experts.forward`.

        Default is ``"gguf_bmm"`` (batched dequant + ``torch.bmm``). Override
        with ``module._experts_implementation = "gguf_grouped_mm"`` or
        ``"eager"`` per instance, or globally via
        ``model.config._experts_implementation``.
        """
        from .moe import ALL_EXPERTS_FUNCTIONS

        impl = getattr(self, "_experts_implementation", None)
        if impl is None:
            impl = "gguf_bmm"
        fwd = ALL_EXPERTS_FUNCTIONS.get_interface(impl, _gguf_eager_experts_forward)
        return fwd(self, hidden_states, top_k_index, top_k_weights)


@torch.no_grad()
def _gguf_eager_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """One-at-a-time fallback. Kept as the safety net for configurations
    where ``torch.bmm`` over batched dequant tensors isn't acceptable (e.g.
    out-of-memory on a giant prefill where ``(S, intermediate, hidden)`` fp32
    won't fit). 90× slower than the bmm path on Apple Silicon for decode.
    """
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
    expert_mask = expert_mask.permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0].item()
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx].to(torch.float32).contiguous()

        ids = torch.tensor([expert_idx], device=current_state.device, dtype=torch.long)
        gate_w = self._gather_dequant(self.gate_proj_q, ids, self.gate_quant, self._gate_fmt,
                                       self.intermediate_dim, self.hidden_dim).squeeze(0)
        up_w   = self._gather_dequant(self.up_proj_q,   ids, self.up_quant,   self._up_fmt,
                                       self.intermediate_dim, self.hidden_dim).squeeze(0)
        down_w = self._gather_dequant(self.down_proj_q, ids, self.down_quant, self._down_fmt,
                                       self.hidden_dim, self.intermediate_dim).squeeze(0)

        gate = torch.nn.functional.linear(current_state, gate_w)
        up   = torch.nn.functional.linear(current_state, up_w)
        inter = (self.act_fn(gate) * up).contiguous()
        out = torch.nn.functional.linear(inter, down_w)

        out = out * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, out.to(final_hidden_states.dtype))

    return final_hidden_states


# =============================================================================
# Fused-expert MoE swap registry — open for extension.
# =============================================================================
#
# Each entry maps the *HF* fused-expert module's class name (matched on the
# live ``nn.Module.__class__.__name__`` so we don't have to import every MoE
# arch upfront) to a "factory" callable that consumes the original module +
# the per-projection {bytes, quant_type} dict and returns the swapped Gguf*
# module. Adding a new MoE architecture is one entry — see
# :func:`_qwen2_moe_factory` below for the contract.
#
# Mixtral / DeepSeek-V3 / and any other arch where experts are stored as a
# ``ModuleList[Linear]`` are already covered by :func:`replace_with_gguf_linear`
# (per-Linear swap) — they don't need a registry entry.


def _qwen2_moe_factory(mod, info, device):
    """Build a :class:`GgufQwen2MoeExperts` from a live ``Qwen2MoeExperts`` /
    ``Qwen3MoeExperts`` / ``MiniMaxM2Experts`` module + a per-projection quant
    info dict. All three share the same fused-expert layout."""

    config = type("C", (), dict(
        num_experts=mod.num_experts,
        hidden_size=mod.hidden_dim,
        moe_intermediate_size=mod.intermediate_dim,
        hidden_act=getattr(mod.act_fn, "_name", None) or "silu",
    ))()
    return GgufQwen2MoeExperts(
        config,
        gate_quant=info["gate_quant"],
        up_quant=info["up_quant"],
        down_quant=info["down_quant"],
        device=device,
    )


# class_name → (factory, has-fused-experts marker).
# Add new entries here as new fused-MoE architectures land. The factory must
# accept ``(mod, info, device)`` and return a module with ``gate_proj_q`` /
# ``up_proj_q`` / ``down_proj_q`` uint8 buffers (the byte-routing pattern).
_FUSED_MOE_REGISTRY: dict[str, "callable"] = {
    "Qwen2MoeExperts":  _qwen2_moe_factory,
    "Qwen3MoeExperts":  _qwen2_moe_factory,
    "MiniMaxM2Experts": _qwen2_moe_factory,
    # Future: "GptOssExperts": _gpt_oss_factory,
}


def register_gguf_moe(class_name: str, factory):
    """Register a swap factory for a new fused-expert MoE class.

    The factory has signature ``(mod, info, device) -> Module`` — see
    :func:`_qwen2_moe_factory` for the contract. Idempotent; later registrations
    overwrite earlier ones, which keeps third-party kernels easy to slot in
    from outside transformers.
    """
    _FUSED_MOE_REGISTRY[class_name] = factory


def replace_qwen2_moe_experts(
    model: nn.Module,
    expert_info_by_layer: dict[str, dict],
) -> int:
    """Walk ``model`` and swap every fused-expert MoE module (any class
    registered in :data:`_FUSED_MOE_REGISTRY`) whose layer name appears in
    ``expert_info_by_layer`` for its quantized counterpart.

    Despite the legacy name, this is **not** Qwen2-specific — it dispatches
    via :data:`_FUSED_MOE_REGISTRY`. The name is kept as a stable entrypoint
    for :class:`GGUFQuantizer._swap_moe_experts`.

    ``expert_info_by_layer`` maps the *parent* path (e.g.
    ``model.layers.0.mlp.experts``) to one entry per projection::

        {
            "gate_quant":  "Q4_K",
            "up_quant":    "Q4_K",
            "down_quant":  "Q8_0",         # mixed quants per layer in Q4_K_M
            "gate_bytes":  GGUFQuantizedTensor,
            "up_bytes":    GGUFQuantizedTensor,
            "down_bytes":  GGUFQuantizedTensor,
        }

    Returns the number of MoE-experts blocks swapped.
    """
    import warnings

    swapped = 0
    for name, mod in list(model.named_modules()):
        factory = _FUSED_MOE_REGISTRY.get(mod.__class__.__name__)
        if factory is None:
            continue
        info = expert_info_by_layer.get(name)
        if info is None:
            continue

        try:
            device = next(mod.parameters()).device
        except StopIteration:
            device = "cpu"
        try:
            new = factory(mod, info, device)
        except Exception as e:
            warnings.warn(f"replace_qwen2_moe_experts: skip {name}: {e}", stacklevel=2)
            continue

        ok = True
        for buf_name, key in [("gate_proj_q", "gate_bytes"),
                              ("up_proj_q",   "up_bytes"),
                              ("down_proj_q", "down_bytes")]:
            src = info[key].detach().contiguous().view(torch.uint8)
            dst = getattr(new, buf_name)
            # 2D buffer is (num_experts, bytes_per_expert) — reshape the source.
            if src.numel() != dst.numel():
                warnings.warn(
                    f"replace_qwen2_moe_experts: {name}.{buf_name} size mismatch "
                    f"({src.numel()} vs {dst.numel()}), skipping",
                    stacklevel=2,
                )
                ok = False
                break
            dst.copy_(src.reshape(dst.shape).to(dst.device))
        if not ok:
            continue

        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, new)
        _prepare_id_kernel_refs(new)
        # Route the new module's forward through the ExpertsInterface dispatcher
        # — same pattern as FP8Experts. ``gguf_bmm`` is the default; users can
        # override via ``model.config._experts_implementation = "gguf_grouped_mm"``.
        parent_config = getattr(model, "config", None)
        if parent_config is not None and not hasattr(parent_config, "_experts_implementation"):
            parent_config._experts_implementation = "gguf_bmm"
        elif parent_config is not None and parent_config._experts_implementation in (None, "eager"):
            parent_config._experts_implementation = "gguf_bmm"
        swapped += 1
    return swapped


def _prepare_id_kernel_refs(mod: nn.Module) -> None:
    """Resolve mul_mat_id ops + precompute flat byte views.

    Called once after a fused-MoE module is swapped in. Caches the three op
    callables and the flat uint8 ``view(-1)`` of each ``<proj>_proj_q`` buffer
    so the hot ``gguf_bmm_experts_forward`` path can do a constant-time
    attribute read instead of an ``_ensure_metal_kernels`` + ``hasattr`` +
    ``getattr(mod._ops, ...)`` per layer. The cache also makes
    ``torch.compile`` see the ops as fixed module attributes.
    """
    metal = _ensure_metal_kernels()
    if metal is None:
        return
    ops = metal._ops
    fmt_to_code = {"q4_K": 0, "q5_0": 1, "q5_1": 2, "q8_0": 3}
    fmt_codes = []
    all_supported = True
    for proj, fmt_attr, cache_attr in (
        ("gate", "_gate_fmt", "_id_op_gate"),
        ("up",   "_up_fmt",   "_id_op_up"),
        ("down", "_down_fmt", "_id_op_down"),
    ):
        fmt = getattr(mod, fmt_attr, None)
        if fmt is None:
            return
        op_name = f"mul_mat_id_{fmt}_f32"
        op = getattr(ops, op_name, None)
        if op is None or not hasattr(ops, op_name):
            return
        setattr(mod, cache_attr, op)
        if fmt in fmt_to_code:
            fmt_codes.append(fmt_to_code[fmt])
        else:
            all_supported = False

    # The fused ``gguf_moe_silu_f32`` op exists but a microbench shows it
    # is ~2.6× slower than the per-projection mul_mat_id path on M3 Max
    # decode shapes (303 µs vs 118 µs / layer). Likely the bundled
    # silu_mul kernel breaks MPSGraph's silu/mul fusion across torch ops.
    # Kept available as ``module._moe_silu_op = ops.gguf_moe_silu_f32`` for
    # callers who want to opt in (e.g. for a future architecture where the
    # activation cost can be properly recouped); off by default.


def replace_with_gguf_linear(
    model: nn.Module,
    weight_info_by_name: dict[str, dict],
    modules_to_not_convert: Optional[set[str]] = None,
) -> int:
    """Walk `model` and swap each ``nn.Linear`` whose weight name appears in
    ``weight_info_by_name`` for a :class:`GgufLinear`.

    ``weight_info_by_name`` maps a HF parameter name (e.g.
    ``model.layers.0.self_attn.v_proj.weight``) to a dict::

        {
            "quant_type": "Q4_0",         # GGUF type name
            "bytes": <uint8 Tensor>,      # original GGUF block bytes (flat)
            "permute": None | "q" | "k",  # llama.cpp Q/K head permute marker
        }

    For non-permuted Linears the **original GGUF bytes are copied verbatim**
    into ``qweight`` — byte-identical to llama.cpp, works for *all* quant
    types (including K-quants where ``gguf.quantize`` isn't implemented in
    Python), and avoids any round-trip precision loss.

    For ``attn_q`` / ``attn_k`` (``permute`` set), the GGUF bytes are stored
    in llama.cpp's permuted layout, so we dequantize → reverse-permute →
    re-quantize through ``gguf-py``. This only works for quant types with a
    Python re-quantizer (today Q4_0 / Q8_0); unsupported permuted layers
    stay as the dequantized ``nn.Linear``.

    Returns the number of layers swapped.
    """
    import gguf

    modules_to_not_convert = modules_to_not_convert or set()
    swapped = 0

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        weight_name = f"{name}.weight"
        info = weight_info_by_name.get(weight_name)
        if info is None:
            continue
        if any(skip in name for skip in modules_to_not_convert):
            continue

        quant_type = info["quant_type"]
        if quant_type not in _QUANT_INFO:
            continue
        block_bytes, block_elems = _QUANT_INFO[quant_type]
        expected_nbytes = mod.out_features * (mod.in_features // block_elems) * block_bytes

        if info.get("permute") is not None:
            # llama.cpp-permuted layout — bytes can't be used as-is. Round-trip
            # through fp32: the loader already produced the un-permuted fp32
            # weight in `mod.weight`, so just re-quantize it back to bytes.
            try:
                qt = getattr(gguf.GGMLQuantizationType, quant_type)
                w_np = mod.weight.detach().to(torch.float32).cpu().numpy()
                qbytes_t = torch.frombuffer(bytearray(gguf.quantize(w_np, qt).tobytes()), dtype=torch.uint8)
            except (NotImplementedError, Exception):
                # K-quants etc. — leave as nn.Linear with the (un-permuted) fp32 weight.
                continue
        else:
            raw = info["bytes"]
            qbytes_t = raw.detach().contiguous().view(torch.uint8).reshape(-1)

        if qbytes_t.numel() != expected_nbytes:
            import warnings
            warnings.warn(
                f"GgufLinear: skip {name} — bytes size mismatch "
                f"({qbytes_t.numel()} vs expected {expected_nbytes})",
                stacklevel=2,
            )
            continue

        new = GgufLinear(
            in_features=mod.in_features,
            out_features=mod.out_features,
            quant_type=quant_type,
            bias=mod.bias is not None,
            device=mod.weight.device,
        )
        new.qweight.copy_(qbytes_t.to(new.qweight.device))
        if mod.bias is not None:
            new.bias.data.copy_(mod.bias.detach().to(torch.float32))

        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, new)
        new._bind_kernels()
        swapped += 1

    return swapped


def is_gguf_linear_enabled() -> bool:
    """``TRANSFORMERS_GGUF_LINEAR=1`` opts into the GgufLinear path. Off by default."""
    return os.environ.get("TRANSFORMERS_GGUF_LINEAR", "0") not in ("0", "", "false", "False")


def setup_for_compile(model, *, mode: str = "reduce-overhead") -> None:
    """Configure a GGUF-loaded model for ``torch.compile``.

    The recipe is the result of profiling ``generate()`` under dynamo on
    Apple Silicon and removing every avoidable recompile:

    1. ``cache_implementation = "static"`` on the model's generation_config.
       The default dynamic KV cache grows on every step, which would force a
       new compiled graph per decode token.
    2. ``torch._dynamo.config.cache_size_limit = 512`` — a 24-layer model
       can have ~170 distinct ``GgufLinear.forward`` specialisations (one
       per layer × per projection); dynamo's default cap of 8 silently
       falls back to eager for the rest.
    3. ``torch._dynamo.config.allow_unspec_int_on_nn_module = True`` so
       the per-layer ``self.layer_idx`` integer attribute on Cache.update
       isn't burned in as a guard (otherwise that recompiles once per layer
       per generate call — 24 useless recompiles).
    4. ``torch.compile(model.forward, mode=mode, dynamic=False)``. The
       default ``reduce-overhead`` mode tries cudagraph replay; that's a
       no-op on MPS but stays consistent with the CUDA recipe.

    With this setup, a 24-layer MoE warms up in ~0.16 s (one trace, static
    shapes) and steady-state decode is consistently 60–65 tok/s on M3 Max
    for ``Qwen1.5-MoE-A2.7B Q4_K_M`` — 2× uncompiled, ~66% of llama.cpp's
    ``llama-bench``. The only remaining recompile is the prefill → decode
    shape change, which fires once per ``generate()`` call.
    """
    import torch as _torch
    import torch._dynamo as _dynamo

    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 512)
    _dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, 512)
    _dynamo.config.allow_unspec_int_on_nn_module = True

    model.generation_config.cache_implementation = "static"
    model.forward = _torch.compile(model.forward, mode=mode, dynamic=False)
