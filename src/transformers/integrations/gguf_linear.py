# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Inference-time GGUF Linear / Experts modules + the meta-time swap helper.

Two building blocks:

* :class:`GgufLinear`: drop-in replacement for ``nn.Linear`` that holds the
  raw GGUF block bytes (uint8) and dispatches to ``mul_mat_<fmt>_f32`` /
  ``mul_mat_vec_<fmt>_f32`` Metal kernels at forward time. There is no slow
  CPU/CUDA fallback — callers that want dequantize-on-load go through the
  ``GGUFDequantize`` conversion op instead.
* :class:`GgufExperts`: drop-in for ``Qwen2MoeExperts`` / ``Qwen3MoeExperts``
  / ``MiniMaxM2Experts`` that stores all expert weights as per-projection
  uint8 buffers and dispatches into the ``mul_mat_id_<fmt>_f32`` Metal kernels.

:func:`replace_with_gguf_linear` walks a meta-device model and performs both
the Linear and the ``.experts`` swap in a single pass — same pattern as
``replace_with_fp8_linear`` in :mod:`integrations.finegrained_fp8`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gguf_kernels import _KERNEL_FMT, ensure_metal_kernels
from .moe import ExpertsInterface


# Per quant type: (block bytes, block elems). The format string side of the
# table (kernel names like mul_mat_q4_K_f32) lives in gguf_kernels._KERNEL_FMT.
_QUANT_INFO: dict[str, tuple[int, int]] = {
    "Q4_0": (18, 32),
    "Q5_0": (22, 32),
    "Q5_1": (24, 32),
    "Q8_0": (34, 32),
    "Q4_K": (144, 256),
    "Q5_K": (176, 256),
    "Q6_K": (210, 256),
    "IQ4_NL": (18, 32),
    "IQ4_XS": (136, 256),
}


def gguf_linear_supports(quant_type) -> bool:
    """Return True if quant_type has a matching mul_mat_<fmt>_f32 kernel."""
    name = quant_type.name if hasattr(quant_type, "name") else str(quant_type)
    return name in _QUANT_INFO


class GgufLinear(nn.Module):
    """Linear layer with GGUF-quantized weights stored as raw block bytes (uint8).

    Forward dispatches to mul_mat_vec_<fmt>_f32 for batch 1 and mul_mat_<fmt>_f32
    for batched inputs. Inference-only — no dequant fallback. Callers that want
    a CPU/CUDA dequant path should let the loader's GGUFDequantize op produce
    fp32 weights into a standard nn.Linear instead.
    """

    def __init__(self, in_features: int, out_features: int, quant_type: str = "Q4_K", bias: bool = False):
        super().__init__()
        if quant_type not in _QUANT_INFO:
            raise NotImplementedError(f"GgufLinear only supports {sorted(_QUANT_INFO)} today, got {quant_type}")
        block_bytes, block_elems = _QUANT_INFO[quant_type]
        if in_features % block_elems != 0:
            raise ValueError(f"in_features must be a multiple of {block_elems} for {quant_type}, got {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        nblocks_per_row = in_features // block_elems
        nbytes = out_features * nblocks_per_row * block_bytes
        self.register_buffer("weight", torch.empty(nbytes, dtype=torch.uint8), persistent=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        # Explicit kernel refs. Both names are deterministic from quant_type so
        # we resolve them once at construction. When kernels aren't loadable
        # (non-MPS / CI) leave the attrs None — forward errors out clearly.
        try:
            ops = ensure_metal_kernels()._ops
            fmt = _KERNEL_FMT[quant_type]
            self._mv_op = getattr(ops, f"mul_mat_vec_{fmt}_f32").default
            self._mat_op = getattr(ops, f"mul_mat_{fmt}_f32").default
        except Exception:
            self._mv_op = self._mat_op = None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_type={self.quant_type}, bias={self.bias is not None}"
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Last dim of x must be {self.in_features}, got {x.shape[-1]}")
        if x.device.type != "mps" or self._mv_op is None:
            raise RuntimeError(
                "GgufLinear runs only on MPS with the metal kernels loaded. Re-load with "
                "`dtype=torch.bfloat16` (or any explicit dtype) to dequantize to a normal nn.Linear."
            )

        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features).contiguous().to(torch.float32)
        N = x_flat.shape[0]
        qw = self.weight
        # Pick the kernel by batch size: matvec when N==1, matmul otherwise. The
        # matmul kernel requires N % 32 == 0; non-aligned batches loop per-row
        # over matvec (rare in practice — most call sites land aligned).
        if N == 1:
            y = torch.empty(self.out_features, dtype=torch.float32, device=x.device)
            self._mv_op(qw, x_flat.view(-1), y)
        elif N % 32 == 0:
            y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
            self._mat_op(qw, x_flat, y)
        else:
            y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
            for i in range(N):
                self._mv_op(qw, x_flat[i].reshape(-1), y[i * self.out_features : (i + 1) * self.out_features])
        y = y.view(N, self.out_features).reshape(*batch_shape, self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(y.device).to(y.dtype)
        # Metal matvec / matmul emit fp32. Cast back to the caller's dtype so the
        # KV cache stays in the model's chosen precision instead of silently widening.
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y


# ----------------------------------------------------------------------------
# GgufExperts — quantized replacement for fused-expert MoE modules
# (Qwen2MoeExperts / Qwen3MoeExperts / MiniMaxM2Experts — same layout).
# ----------------------------------------------------------------------------


class GgufExperts(nn.Module):
    """Drop-in for fused-expert MoE modules (MixtralExperts / Qwen2MoeExperts
    / DeepseekV3NaiveMoe — same layout). Holds the same buffer names as the
    unswapped module (gate_up_proj + down_proj) but as raw uint8 GGUF bytes,
    so the standard GGUF rename pipeline's merge WeightConverter
    (ffn_gate_exps + ffn_up_exps -> gate_up_proj) lands bytes here without
    any per-quantizer rewrite.

    gate_up_quant is assumed equal across gate and up (the merge can only run
    when both halves share a quant type — Q4_K_M ships them both as Q4_K).
    down_quant may differ (Q4_K_M ships down as Q8_0).

    Configs across MoE archs spell these dims differently — Qwen2MoE uses
    num_experts, Mixtral uses num_local_experts, DeepSeek-V3 uses
    n_routed_experts. PretrainedConfig's attribute_map aliases the names so
    config.num_experts / config.moe_intermediate_size resolve consistently;
    archs missing an alias need it added to their config (one-line fix there,
    not here).
    """

    def __init__(self, config, gate_up_quant: str, down_quant: str):
        super().__init__()
        for label, qt in (("gate_up", gate_up_quant), ("down", down_quant)):
            if qt not in _QUANT_INFO:
                raise NotImplementedError(f"GgufExperts {label}_quant {qt} not in {sorted(_QUANT_INFO)}")
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_quant, self.down_quant = gate_up_quant, down_quant

        # Buffer layout matches MixtralExperts.gate_up_proj (3D Parameter):
        # rows = 2*intermediate (gate first, then up — natural output of the
        # merge converter's Concatenate(dim=1)), cols-in-bytes = hidden /
        # block_elems * block_bytes.
        gate_up_rows = 2 * self.intermediate_dim
        gate_up_bytes_per_row = (self.hidden_dim // _QUANT_INFO[gate_up_quant][1]) * _QUANT_INFO[gate_up_quant][0]
        down_bytes_per_row = (self.intermediate_dim // _QUANT_INFO[down_quant][1]) * _QUANT_INFO[down_quant][0]
        self.register_buffer(
            "gate_up_proj",
            torch.empty(self.num_experts, gate_up_rows, gate_up_bytes_per_row, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "down_proj",
            torch.empty(self.num_experts, self.hidden_dim, down_bytes_per_row, dtype=torch.uint8),
            persistent=True,
        )

        from ..activations import ACT2FN

        self.act_fn = ACT2FN[config.hidden_act]
        # ExpertsInterface contract flags.
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True

        # Explicit kernel refs + decode-shape scratch. Same rationale as the
        # GgufLinear case (eager resolution from quant_type, None when kernels
        # are unloadable so non-MPS construction still succeeds). The scratch
        # buffers are registered as non-persistent meta tensors here; ``.to``
        # materialises them when the model lands on its real device. They feed
        # the decode hot path so torch.compile doesn't see fresh ``torch.empty``
        # mutations on the second call (that would force a re-trace).
        try:
            ops = ensure_metal_kernels()._ops
            self._id_op_gate_up = getattr(ops, f"mul_mat_id_{_KERNEL_FMT[gate_up_quant]}_f32").default
            self._id_op_down = getattr(ops, f"mul_mat_id_{_KERNEL_FMT[down_quant]}_f32").default
        except Exception:
            self._id_op_gate_up = self._id_op_down = None
        decode_size = getattr(config, "num_experts_per_tok", None) or 4
        self.register_buffer(
            "_gate_buf", torch.empty(decode_size, self.intermediate_dim, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_up_buf", torch.empty(decode_size, self.intermediate_dim, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_pair_out", torch.empty(decode_size, self.hidden_dim, dtype=torch.float32), persistent=False
        )
        self._decode_size = decode_size

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, "
            f"intermediate_dim={self.intermediate_dim}, "
            f"gate_up={self.gate_up_quant}, down={self.down_quant}"
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch through ALL_GGUF_EXPERTS_FUNCTIONS (GGUF-specific sibling
        of ALL_EXPERTS_FUNCTIONS). Default is the bmm path."""
        impl = getattr(self, "_experts_implementation", None) or "bmm"
        return ALL_GGUF_EXPERTS_FUNCTIONS.get_interface(impl, gguf_bmm_experts_forward)(
            self, hidden_states, top_k_index, top_k_weights
        )


class GgufExpertsTransposed(GgufExperts):
    """Variant for archs that store gate_up_proj transposed —
    (num_experts, hidden_dim, 2*intermediate_dim) instead of
    (num_experts, 2*intermediate_dim, hidden_dim) — and carry biases on both
    projections. Today: gpt_oss (GptOssExperts does
    current_state @ self.gate_up_proj[expert_idx] rather than
    F.linear(..., self.gate_up_proj[expert_idx]), so the parameter is stored
    (in_dim, out_dim) instead of (out_dim, in_dim)).

    TODO: forward kernel. gguf_bmm_experts_forward assumes the
    Mixtral/Qwen2MoE layout (rows = output axis). Loading a gpt_oss GGUF
    populates the buffers correctly via the standard rename pipeline, but
    forward needs a transposed-aware kernel path before it produces correct
    logits.
    """

    def __init__(self, config, gate_up_quant: str, down_quant: str):
        nn.Module.__init__(self)  # skip GgufExperts.__init__: different buffer shapes.
        for label, qt in (("gate_up", gate_up_quant), ("down", down_quant)):
            if qt not in _QUANT_INFO:
                raise NotImplementedError(f"GgufExpertsTransposed {label}_quant {qt} not in {sorted(_QUANT_INFO)}")
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        # gpt_oss uses ``intermediate_size`` for the per-expert MLP width — no
        # moe_intermediate_size distinction since there's no shared dense MLP
        # alongside the experts.
        self.intermediate_dim = config.intermediate_size
        self.gate_up_quant, self.down_quant = gate_up_quant, down_quant

        # Transposed layout: rows = in axis, cols-in-bytes = out / block_elems * block_bytes.
        gate_up_bytes_per_row = ((2 * self.intermediate_dim) // _QUANT_INFO[gate_up_quant][1]) * _QUANT_INFO[
            gate_up_quant
        ][0]
        down_bytes_per_row = (self.hidden_dim // _QUANT_INFO[down_quant][1]) * _QUANT_INFO[down_quant][0]
        self.register_buffer(
            "gate_up_proj",
            torch.empty(self.num_experts, self.hidden_dim, gate_up_bytes_per_row, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "down_proj",
            torch.empty(self.num_experts, self.intermediate_dim, down_bytes_per_row, dtype=torch.uint8),
            persistent=True,
        )
        # gpt_oss has biases on both projections; mirror that here.
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_dim, dtype=torch.float32), requires_grad=False
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_dim, dtype=torch.float32), requires_grad=False
        )

        from ..activations import ACT2FN

        self.act_fn = ACT2FN[config.hidden_act]
        self.has_gate = True
        self.has_bias = True
        self.is_transposed = True
        self.is_concatenated = True

        # Eager kernel refs + decode scratch; same pattern as GgufExperts.
        try:
            ops = ensure_metal_kernels()._ops
            self._id_op_gate_up = getattr(ops, f"mul_mat_id_{_KERNEL_FMT[gate_up_quant]}_f32").default
            self._id_op_down = getattr(ops, f"mul_mat_id_{_KERNEL_FMT[down_quant]}_f32").default
        except Exception:
            self._id_op_gate_up = self._id_op_down = None
        decode_size = getattr(config, "num_experts_per_tok", None) or 4
        self.register_buffer(
            "_gate_buf", torch.empty(decode_size, self.intermediate_dim, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_up_buf", torch.empty(decode_size, self.intermediate_dim, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_pair_out", torch.empty(decode_size, self.hidden_dim, dtype=torch.float32), persistent=False
        )
        self._decode_size = decode_size


# Registry of fused-expert classes keyed by ``model.config.model_type``.
# Covers every MoE arch with GGUF rename rules (``_GGUF_ARCH_CONVERTERS`` in
# ``modeling_gguf_pytorch_utils``) plus the popular MoE archs that share the
# fused-expert layout but don't have GGUF rules yet (Mixtral, DeepSeek-V3 —
# their day-zero rename rules can land bytes here when added).
MODEL_TYPE_TO_GGUF_EXPERTS: dict[str, type[nn.Module]] = {
    # Same layout as MixtralExperts: ``gate_up_proj`` shape
    # ``(num_experts, 2*intermediate, hidden)``.
    "qwen2_moe": GgufExperts,
    "qwen3_moe": GgufExperts,
    "minimax_m2": GgufExperts,
    "mixtral": GgufExperts,
    "deepseek_v3": GgufExperts,
    # Transposed + biased layout: ``gate_up_proj`` shape
    # ``(num_experts, hidden, 2*intermediate)`` plus per-expert biases.
    "gpt_oss": GgufExpertsTransposed,
}


# Fused-expert forward + GgufExpertsInterface — relocated here from moe.py so
# the GGUF integration owns the dispatch end-to-end. Mirrors FP8ExpertsInterface
# in integrations/finegrained_fp8.py.


@torch.no_grad()
def gguf_bmm_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Quantized fused-expert forward for GgufExperts. Each of gate / up / down
    runs as a single mul_mat_id_<fmt>_f32 Metal kernel launch indexed by the
    per-token expert ids — same pattern as llama.cpp's mul_mat_id, never
    materialises the expert weights as fp32.
    """
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = self.hidden_dim
    intermediate_dim = self.intermediate_dim
    device = hidden_states.device
    # S = total token-expert pairs across the batch (each routed token contributes
    # num_top_k work items). The mul_mat_id kernel writes one output row per pair.
    S = num_tokens * num_top_k

    if self._id_op_gate_up is None or self._id_op_down is None:
        raise RuntimeError("GgufExperts requires the GGUF metal kernels (install `kernels`, run on MPS).")

    selected = hidden_states.repeat_interleave(num_top_k, dim=0).to(torch.float32).contiguous()
    ids32 = top_k_index.reshape(-1).to(torch.int32)
    sample_weights = top_k_weights.reshape(-1).to(torch.float32)

    # Decode hot path: reuse the per-module scratch buffers (registered at
    # __init__, materialised by .to(device)). Avoids ``torch.empty`` inside the
    # compiled graph — without this, dynamo would re-trace on the second call
    # when the buffer storage changes. Prefill (num_tokens > 1) allocates per
    # call; that re-trace happens once per shape and dynamo caches it.
    if num_tokens == 1 and self._decode_size == S:
        gate_buf, up_buf, out = self._gate_buf, self._up_buf, self._pair_out
    else:
        gate_buf = torch.empty(S, intermediate_dim, dtype=torch.float32, device=device)
        up_buf = torch.empty(S, intermediate_dim, dtype=torch.float32, device=device)
        out = torch.empty(S, hidden_dim, dtype=torch.float32, device=device)

    # gate_up_proj packs [gate; up] along the row axis (the standard MixtralExperts
    # layout, produced naturally by the GGUF merge converter's Concatenate(dim=1)
    # on uint8 byte tensors). Split into per-projection byte halves at kernel
    # call time — both halves share a quant type so the same op fires twice.
    gate_up_bytes = self.gate_up_proj.view(self.num_experts, -1)
    half = gate_up_bytes.shape[1] // 2
    gate_qw = gate_up_bytes[:, :half].contiguous().view(-1)
    up_qw = gate_up_bytes[:, half:].contiguous().view(-1)
    down_qw = self.down_proj.view(-1)
    self._id_op_gate_up(gate_qw, selected, ids32, gate_buf)
    self._id_op_gate_up(up_qw, selected, ids32, up_buf)
    inter = (self.act_fn(gate_buf) * up_buf).contiguous()
    self._id_op_down(down_qw, inter, ids32, out)

    weighted = out * sample_weights.unsqueeze(-1)
    return weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def _gguf_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Prefill alias — currently delegates to the bmm path (one mul_mat_id kernel
    per projection is the right shape for both decode and prefill on MPS; a
    proper grouped_mm impl would land here when CUDA kernels arrive)."""
    return gguf_bmm_experts_forward(self, hidden_states, top_k_index, top_k_weights)


class GgufExpertsInterface(ExpertsInterface):
    """GGUF-specific experts dispatch — registers the mul_mat_id-backed bmm /
    grouped_mm impls. Mirrors FP8ExpertsInterface in integrations.finegrained_fp8.
    """

    _global_mapping = {
        "bmm": gguf_bmm_experts_forward,
        "grouped_mm": _gguf_grouped_mm_experts_forward,
    }


ALL_GGUF_EXPERTS_FUNCTIONS = GgufExpertsInterface()


# ----------------------------------------------------------------------------
# Meta-time swap helper — FP8-style. Walks ``model`` on the meta device and
# replaces ``nn.Linear`` / fused-expert modules in one pass.
# ----------------------------------------------------------------------------


def _should_convert(name: str, modules_to_not_convert: set[str]) -> bool:
    return not any(skip in name for skip in modules_to_not_convert)


def replace_with_gguf_linear(
    model: nn.Module,
    quant_info_by_target: dict[str, dict],
    modules_to_not_convert: set[str] | None = None,
) -> int:
    """Walk ``model`` on its current device (typically meta) and swap matching
    ``nn.Linear`` / fused-expert modules in place.

    ``quant_info_by_target[name]`` carries the per-module quant info:

    * For an ``nn.Linear`` (key = full module name)::

          {"quant_type": "Q4_K"}

    * For a fused-expert module (key = parent path, e.g. ``model.layers.0.mlp.experts``)::

          {"gate_up_quant": "Q4_K", "down_quant": "Q8_0"}

    The actual bytes flow through the GGUF rename pipeline at weight-load time
    (the merge ``WeightConverter`` for ``gate_up_proj`` + the
    ``ffn_down_exps → down_proj`` rename, both routed through the target-aware
    :class:`~transformers.gguf_conversion_ops.GGUFDequantize`).

    Returns the number of modules swapped.
    """
    modules_to_not_convert = modules_to_not_convert or set()
    swapped = 0
    experts_cls = MODEL_TYPE_TO_GGUF_EXPERTS.get(getattr(model.config, "model_type", None))
    for name, mod in list(model.named_modules()):
        if not _should_convert(name, modules_to_not_convert):
            continue
        info = quant_info_by_target.get(name)
        if info is None:
            continue

        # Construct on meta — same as ``replace_with_fp8_linear``. The buffers
        # come up as meta tensors; the real bytes are filled by the rename
        # pipeline at weight-load time.
        new_module: nn.Module | None = None
        with torch.device("meta"):
            if isinstance(mod, nn.Linear):
                quant_type = info.get("quant_type")
                if quant_type not in _QUANT_INFO:
                    continue
                new_module = GgufLinear(
                    in_features=mod.in_features,
                    out_features=mod.out_features,
                    quant_type=quant_type,
                    bias=mod.bias is not None,
                )
            elif experts_cls is not None and "gate_up_quant" in info:
                new_module = experts_cls(
                    config=getattr(mod, "config", model.config),
                    gate_up_quant=info["gate_up_quant"],
                    down_quant=info["down_quant"],
                )
        if new_module is None:
            continue

        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, new_module)
        swapped += 1
    return swapped


# Public re-exports for callers (quantizer, weight conversion ops, tests).
__all__: list[str] = [
    "ALL_GGUF_EXPERTS_FUNCTIONS",
    "GgufLinear",
    "GgufExperts",
    "GgufExpertsTransposed",
    "MODEL_TYPE_TO_GGUF_EXPERTS",
    "gguf_bmm_experts_forward",
    "gguf_linear_supports",
    "replace_with_gguf_linear",
]
