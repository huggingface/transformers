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

from .gguf_kernels import bind_id_kernel_refs, ensure_metal_kernels


# Per quant type: (block bytes, block elems). Format names match the suffix of
# the ``mul_mat_<fmt>_f32`` and ``mul_mat_vec_<fmt>_f32`` ops in the kernel.
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

_KERNEL_FMT: dict[str, str] = {
    "Q4_0": "q4_0",
    "Q5_0": "q5_0",
    "Q5_1": "q5_1",
    "Q8_0": "q8_0",
    "Q4_K": "q4_K",
    "Q5_K": "q5_K",
    "Q6_K": "q6_K",
    "IQ4_NL": "iq4_nl",
    "IQ4_XS": "iq4_xs",
}


def gguf_linear_supports(quant_type) -> bool:
    """Return True if ``quant_type`` has a matching ``mul_mat_<fmt>_f32`` kernel."""
    name = quant_type.name if hasattr(quant_type, "name") else str(quant_type)
    return name in _QUANT_INFO


def _read_first(config, *attrs: str):
    """Return ``config.<first_attr_that_exists>`` — MoE configs across archs spell
    the same idea differently (``num_experts`` / ``num_local_experts`` /
    ``n_routed_experts``; ``moe_intermediate_size`` / ``intermediate_size``)."""
    for a in attrs:
        v = getattr(config, a, None)
        if v is not None:
            return v
    raise AttributeError(f"GgufExperts: config has none of {attrs}; got {type(config).__name__}")


def _fmt(quant_type) -> str:
    name = quant_type.name if hasattr(quant_type, "name") else str(quant_type)
    return _KERNEL_FMT[name]


class GgufLinear(nn.Module):
    """Linear layer with GGUF-quantized weights stored as raw block bytes (uint8).

    Forward dispatches to ``mul_mat_vec_<fmt>_f32`` for batch 1 and
    ``mul_mat_<fmt>_f32`` for batched / non-padded inputs. Inference-only — no
    dequant fallback. Callers that want a CPU/CUDA dequant path should let the
    loader's :class:`~transformers.gguf_conversion_ops.GGUFDequantize` op
    produce fp32 weights into a standard ``nn.Linear`` instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: str = "Q4_K",
        bias: bool = False,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if quant_type not in _QUANT_INFO:
            raise NotImplementedError(f"GgufLinear only supports {sorted(_QUANT_INFO)} today, got {quant_type}")
        block_bytes, block_elems = _QUANT_INFO[quant_type]
        if in_features % block_elems != 0:
            raise ValueError(f"in_features must be a multiple of {block_elems} for {quant_type}, got {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self._fmt = _KERNEL_FMT[quant_type]
        nblocks_per_row = in_features // block_elems
        nbytes = out_features * nblocks_per_row * block_bytes
        self.register_buffer("weight", torch.empty(nbytes, dtype=torch.uint8, device=device or "cpu"), persistent=True)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float32, device=device or "cpu"), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)
        # Resolved lazily on first forward to keep meta-device construction cheap.
        self._mv_op = None
        self._mat_op = None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_type={self.quant_type}, bias={self.bias is not None}"
        )

    def _bind_kernels(self) -> None:
        """Resolve and cache the matvec / matmul ``OpOverload`` refs (idempotent)."""
        ops = ensure_metal_kernels()._ops
        mv = getattr(ops, f"mul_mat_vec_{self._fmt}_f32", None)
        mat = getattr(ops, f"mul_mat_{self._fmt}_f32", None)
        if mv is None or mat is None:
            raise RuntimeError(
                f"GGUF metal kernels missing required ops for {self.quant_type}: "
                f"mul_mat_vec_{self._fmt}_f32 / mul_mat_{self._fmt}_f32"
            )
        self._mv_op = getattr(mv, "default", mv)
        self._mat_op = getattr(mat, "default", mat)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Last dim of x must be {self.in_features}, got {x.shape[-1]}")
        if x.device.type != "mps":
            raise RuntimeError(
                "GgufLinear runs only on MPS. Re-load with `dtype=torch.bfloat16` (or any explicit dtype) "
                "to dequantize to a normal nn.Linear for non-MPS devices."
            )
        if self._mv_op is None:
            self._bind_kernels()
        mv_op, mat_op = self._mv_op, self._mat_op

        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features).contiguous().to(torch.float32)
        N = x_flat.shape[0]
        qw = self.weight
        # Pick the kernel by batch size: matvec when N==1, matmul otherwise. The
        # matmul kernel requires N % 32 == 0; non-aligned batches loop per-row
        # over matvec (rare in practice — most call sites land aligned).
        if N == 1:
            y = torch.empty(self.out_features, dtype=torch.float32, device=x.device)
            mv_op(qw, x_flat.view(-1), y)
        elif N % 32 == 0:
            y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
            mat_op(qw, x_flat, y)
        else:
            y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
            for i in range(N):
                mv_op(qw, x_flat[i].reshape(-1), y[i * self.out_features : (i + 1) * self.out_features])
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
    """Drop-in for fused-expert MoE modules (``MixtralExperts`` /
    ``Qwen2MoeExperts`` / ``DeepseekV3NaiveMoe`` — same layout). Holds the
    same buffers names as the unswapped module (``gate_up_proj`` + ``down_proj``),
    but as raw uint8 GGUF bytes — so the standard GGUF rename pipeline's merge
    ``WeightConverter`` (``ffn_gate_exps + ffn_up_exps → gate_up_proj``) lands
    bytes here without any per-quantizer rewrite.

    ``gate_quant`` is assumed equal to ``up_quant`` (the merge can only run
    when both halves share a quant type — Q4_K_M ships them both as Q4_K).
    ``down_quant`` may differ (Q4_K_M ships down as Q8_0).
    """

    def __init__(
        self,
        config,
        gate_up_quant: str,
        down_quant: str,
        device=None,
    ):
        super().__init__()
        for label, qt in (("gate_up", gate_up_quant), ("down", down_quant)):
            if qt not in _QUANT_INFO:
                raise NotImplementedError(f"GgufExperts {label}_quant {qt} not in {sorted(_QUANT_INFO)}")
        self.config = config
        # Different MoE archs spell these differently — read every known name.
        self.num_experts = _read_first(config, "num_experts", "num_local_experts", "n_routed_experts")
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = _read_first(config, "moe_intermediate_size", "intermediate_size")
        self.gate_up_quant, self.down_quant = gate_up_quant, down_quant
        self._gate_up_fmt = _KERNEL_FMT[gate_up_quant]
        self._down_fmt = _KERNEL_FMT[down_quant]

        # Same buffer names as MixtralExperts / Qwen2MoeExperts — uint8 bytes.
        # ``gate_up_proj``: rows = 2 * intermediate (gate first, then up — that's the
        # output of the merge converter's ``Concatenate(dim=1)`` over gate+up uint8
        # byte tensors), cols (in bytes) = hidden_dim / block_elems * block_bytes.
        gate_up_rows = 2 * self.intermediate_dim
        gate_up_bytes_per_row = (self.hidden_dim // _QUANT_INFO[gate_up_quant][1]) * _QUANT_INFO[gate_up_quant][0]
        down_bytes_per_row = (self.intermediate_dim // _QUANT_INFO[down_quant][1]) * _QUANT_INFO[down_quant][0]
        self.register_buffer(
            "gate_up_proj",
            torch.empty(
                self.num_experts, gate_up_rows, gate_up_bytes_per_row, dtype=torch.uint8, device=device or "cpu"
            ),
            persistent=True,
        )
        self.register_buffer(
            "down_proj",
            torch.empty(
                self.num_experts, self.hidden_dim, down_bytes_per_row, dtype=torch.uint8, device=device or "cpu"
            ),
            persistent=True,
        )

        from ..activations import ACT2FN

        self.act_fn = ACT2FN[config.hidden_act]
        # ExpertsInterface contract flags.
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True

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
        """Dispatch through :data:`ALL_GGUF_EXPERTS_FUNCTIONS` (GGUF-specific
        sibling of :data:`ALL_EXPERTS_FUNCTIONS`). Default is the bmm path."""
        impl = getattr(self, "_experts_implementation", None) or "bmm"
        return ALL_GGUF_EXPERTS_FUNCTIONS.get_interface(impl, gguf_bmm_experts_forward)(
            self, hidden_states, top_k_index, top_k_weights
        )


class GgufExpertsTransposed(GgufExperts):
    """Variant for arch's that store ``gate_up_proj`` transposed —
    ``(num_experts, hidden_dim, 2*intermediate_dim)`` instead of
    ``(num_experts, 2*intermediate_dim, hidden_dim)`` — and carry biases on
    both projections. Today: ``gpt_oss`` (:class:`GptOssExperts` does
    ``current_state @ self.gate_up_proj[expert_idx]`` rather than
    ``F.linear(..., self.gate_up_proj[expert_idx])``, so the parameter is
    stored ``(in_dim, out_dim)`` instead of ``(out_dim, in_dim)``).

    TODO: forward kernel. The ``gguf_bmm_experts_forward`` path assumes the
    Mixtral/Qwen2MoE layout (rows = output axis). Loading a gpt_oss GGUF
    populates the buffers correctly via the standard rename pipeline, but
    ``forward`` needs a transposed-aware kernel path before it produces
    correct logits.
    """

    def __init__(self, config, gate_up_quant: str, down_quant: str, device=None):
        nn.Module.__init__(self)  # skip GgufExperts.__init__: different buffer shapes.
        for label, qt in (("gate_up", gate_up_quant), ("down", down_quant)):
            if qt not in _QUANT_INFO:
                raise NotImplementedError(f"GgufExpertsTransposed {label}_quant {qt} not in {sorted(_QUANT_INFO)}")
        self.config = config
        self.num_experts = _read_first(config, "num_experts", "num_local_experts", "n_routed_experts")
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = _read_first(config, "moe_intermediate_size", "intermediate_size")
        self.gate_up_quant, self.down_quant = gate_up_quant, down_quant
        self._gate_up_fmt = _KERNEL_FMT[gate_up_quant]
        self._down_fmt = _KERNEL_FMT[down_quant]

        # Transposed layout: rows = in axis, cols (in bytes) = out axis / blocks * bytes.
        gate_up_bytes_per_row = ((2 * self.intermediate_dim) // _QUANT_INFO[gate_up_quant][1]) * _QUANT_INFO[
            gate_up_quant
        ][0]
        down_bytes_per_row = (self.hidden_dim // _QUANT_INFO[down_quant][1]) * _QUANT_INFO[down_quant][0]
        self.register_buffer(
            "gate_up_proj",
            torch.empty(
                self.num_experts, self.hidden_dim, gate_up_bytes_per_row, dtype=torch.uint8, device=device or "cpu"
            ),
            persistent=True,
        )
        self.register_buffer(
            "down_proj",
            torch.empty(
                self.num_experts, self.intermediate_dim, down_bytes_per_row, dtype=torch.uint8, device=device or "cpu"
            ),
            persistent=True,
        )
        # gpt_oss has biases on both projections; mirror that here.
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_dim, dtype=torch.float32, device=device or "cpu"),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_dim, dtype=torch.float32, device=device or "cpu"),
            requires_grad=False,
        )

        from ..activations import ACT2FN

        self.act_fn = ACT2FN[config.hidden_act]
        self.has_gate = True
        self.has_bias = True
        self.is_transposed = True
        self.is_concatenated = True


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


# ----------------------------------------------------------------------------
# Forward kernel for fused-expert MoE — relocated here from moe.py so the GGUF
# integration owns it end-to-end. ``integrations/moe.py`` only imports and
# registers it.
# ----------------------------------------------------------------------------


@torch.no_grad()
def gguf_bmm_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Quantized fused-expert forward for ``GgufExperts``.

    Each of gate / up / down runs as a single ``mul_mat_id_<fmt>_f32`` Metal
    kernel launch indexed by the per-token expert ids (same pattern as
    llama.cpp's ``mul_mat_id`` — never materialises the expert weights as fp32).
    """
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    H = self.hidden_dim
    I_dim = self.intermediate_dim
    device = hidden_states.device
    S = num_tokens * num_top_k

    op_gate_up = getattr(self, "_id_op_gate_up", None)
    op_down = getattr(self, "_id_op_down", None)
    if op_gate_up is None or op_down is None:
        # Bind on first call if the module was constructed without the kernel
        # cache (state-dict load path bypasses the meta-time swap).
        bind_id_kernel_refs(self)
        op_gate_up, op_down = self._id_op_gate_up, self._id_op_down

    selected = hidden_states.repeat_interleave(num_top_k, dim=0).to(torch.float32).contiguous()
    ids32 = top_k_index.reshape(-1).to(torch.int32)
    sample_weights = top_k_weights.reshape(-1).to(torch.float32)

    # Reuse decode-shape scratch buffers if the call matches; otherwise allocate.
    if getattr(self, "_scratch_S", None) == S and getattr(self, "_scratch_T", None) == num_tokens:
        gate_buf, up_buf, out = self._gate_buf, self._up_buf, self._pair_out
    else:
        gate_buf = torch.empty(S, I_dim, dtype=torch.float32, device=device)
        up_buf = torch.empty(S, I_dim, dtype=torch.float32, device=device)
        out = torch.empty(S, H, dtype=torch.float32, device=device)

    # ``gate_up_proj`` packs ``[gate; up]`` along the row axis (the standard
    # MixtralExperts layout, produced naturally by the GGUF merge converter's
    # ``Concatenate(dim=1)`` on uint8 byte tensors). Split into per-projection
    # byte halves at kernel-call time. Each half is per-expert-contiguous
    # (gate0 / gate1 / … gateN, up0 / up1 / … upN — interleaved per expert,
    # since concat is on dim 1 = row axis within each expert's block).
    gate_up_bytes = self.gate_up_proj.view(self.num_experts, -1)
    half = gate_up_bytes.shape[1] // 2
    gate_qw = gate_up_bytes[:, :half].contiguous().view(-1)
    up_qw = gate_up_bytes[:, half:].contiguous().view(-1)
    down_qw = self.down_proj.view(-1)
    op_gate_up(gate_qw, selected, ids32, gate_buf)
    op_gate_up(up_qw, selected, ids32, up_buf)
    inter = (self.act_fn(gate_buf) * up_buf).contiguous()
    op_down(down_qw, inter, ids32, out)

    weighted = out * sample_weights.unsqueeze(-1)
    return weighted.view(num_tokens, num_top_k, H).sum(dim=1).to(hidden_states.dtype)


# GGUF-specific ExpertsInterface — the GGUF integration registers its
# implementations here instead of polluting the base ``ExpertsInterface``.
# Same shape as :class:`FP8ExpertsInterface` in :mod:`integrations.finegrained_fp8`.
def _gguf_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Prefill alias — currently delegates to the bmm path (single ``mul_mat_id``
    kernel per projection is the right shape for both decode and prefill on MPS;
    a real ``grouped_mm`` impl would land here when CUDA kernels arrive)."""
    return gguf_bmm_experts_forward(self, hidden_states, top_k_index, top_k_weights)


def _make_gguf_experts_interface():
    """Build the singleton; deferred so this module's import order doesn't fight
    the ``moe.py`` import order (moe.py defines ``ExpertsInterface``)."""
    from .moe import ExpertsInterface

    class GgufExpertsInterface(ExpertsInterface):
        """GGUF-specific experts dispatch — registers the ``mul_mat_id``-backed
        bmm + grouped_mm impls. Mirrors :class:`FP8ExpertsInterface`."""

        _global_mapping = {
            "bmm": gguf_bmm_experts_forward,
            "grouped_mm": _gguf_grouped_mm_experts_forward,
        }

    return GgufExpertsInterface()


ALL_GGUF_EXPERTS_FUNCTIONS = _make_gguf_experts_interface()


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

        new_module: nn.Module | None = None
        if isinstance(mod, nn.Linear):
            quant_type = info.get("quant_type")
            if quant_type not in _QUANT_INFO:
                continue
            new_module = GgufLinear(
                in_features=mod.in_features,
                out_features=mod.out_features,
                quant_type=quant_type,
                bias=mod.bias is not None,
                device=mod.weight.device,
            )
        elif experts_cls is not None and "gate_up_quant" in info:
            new_module = experts_cls(
                config=getattr(mod, "config", model.config),
                gate_up_quant=info["gate_up_quant"],
                down_quant=info["down_quant"],
                device=_first_device(mod),
            )
        if new_module is None:
            continue

        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, new_module)
        swapped += 1
    return swapped


def _first_device(mod: nn.Module) -> torch.device | str:
    try:
        return next(mod.parameters()).device
    except StopIteration:
        for buf in mod.buffers():
            return buf.device
        return "cpu"


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
