# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tiny loader for the GGUF Metal kernel package.

Owns the ``ensure_metal_kernels`` cache + the kernel-ref helper used by the
fused-expert ``GgufExperts`` path. The kernel itself lives on the Hub and is
loaded via the standard ``kernels`` package (``pip install kernels``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


_GGUF_METAL_KERNELS = None
_GGUF_METAL_LOADED = False


def ensure_metal_kernels(repo: str = "ArthurZ/gguf-kernels"):
    """Return the loaded kernels handle, caching across calls. Raises
    :class:`RuntimeError` on failure — no fallback."""
    global _GGUF_METAL_KERNELS, _GGUF_METAL_LOADED
    if _GGUF_METAL_LOADED:
        if _GGUF_METAL_KERNELS is None:
            raise RuntimeError(
                f"GGUF metal kernels failed to load earlier in this process. Reinstall / pre-fetch {repo!r} and retry."
            )
        return _GGUF_METAL_KERNELS
    _GGUF_METAL_LOADED = True
    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise RuntimeError(
            "The GGUF fast path requires the `kernels` package. Install it with `pip install kernels`."
        ) from exc
    try:
        # ``trust_remote_code=True`` lifts the publisher-allowlist check on the
        # kernels backend for repos outside ``kernels-community/``.
        _GGUF_METAL_KERNELS = get_kernel(repo, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load GGUF metal kernels from {repo!r}: {exc!r}. The kernel repo must be "
            f"registered with the kernels-package backend (typically under ``kernels-community/``); "
            f"once migrated this path Just Works."
        ) from exc
    return _GGUF_METAL_KERNELS


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


def bind_id_kernel_refs(mod: nn.Module) -> None:
    """Resolve mul_mat_id_<fmt>_f32 op overloads + pre-allocate decode scratch
    buffers on a fused-MoE expert module. Called once at swap time so the hot
    forward path can do constant-time attribute reads, and so torch.compile sees
    the ops as fixed module attributes.

    Two op refs to resolve:
      * _id_op_gate_up — fires twice (once for the gate half, once for the up
        half of gate_up_proj). Single ref because the merge converter requires
        gate and up to share a quant type.
      * _id_op_down — fires once for down_proj. May use a different quant type
        (Q4_K_M ships gate/up = Q4_K, down = Q8_0).

    Plus three non-persistent scratch buffers (_gate_buf, _up_buf, _pair_out)
    sized for the decode shape (num_tokens = 1, S = num_experts_per_tok). The
    bmm forward reuses them when shapes match; that's the hot path under
    torch.compile. Pre-allocating here (at swap time, on meta or real device)
    means the compiled graph never executes ``torch.empty`` — fresh tensors of
    matching shape but different storage addresses would trigger dynamo
    recompiles on the second call. Prefill shapes (num_tokens > 1) bypass these
    buffers and pay one-time recompile per new shape; that's the regular dynamo
    contract.
    """
    ops = ensure_metal_kernels()._ops
    for projection, quant_attr, cache_attr in (
        ("gate_up", "gate_up_quant", "_id_op_gate_up"),
        ("down", "down_quant", "_id_op_down"),
    ):
        quant = getattr(mod, quant_attr, None)
        if quant is None:
            return
        op_name = f"mul_mat_id_{_KERNEL_FMT[quant]}_f32"
        op = getattr(ops, op_name, None)
        if op is None:
            raise RuntimeError(f"GGUF metal kernels missing op {op_name!r}; cannot bind {projection} for {mod}")
        # Resolve to the concrete OpOverload — skips one dispatcher level per call.
        setattr(mod, cache_attr, getattr(op, "default", op))

    hidden_dim = mod.hidden_dim
    intermediate_dim = mod.intermediate_dim
    # Decode-shape scratch: per token we route num_experts_per_tok experts, so
    # the bmm output has S = top_k rows. ``num_experts_per_tok`` is a config
    # field (4 / 8 depending on the MoE arch) — fall back to 4 when missing.
    decode_size = getattr(getattr(mod, "config", None), "num_experts_per_tok", None) or 4
    dev = mod.gate_up_proj.device
    for name in ("_gate_buf", "_up_buf", "_pair_out"):
        if name in mod._buffers:
            del mod._buffers[name]
    mod.register_buffer(
        "_gate_buf", torch.empty(decode_size, intermediate_dim, dtype=torch.float32, device=dev), persistent=False
    )
    mod.register_buffer(
        "_up_buf", torch.empty(decode_size, intermediate_dim, dtype=torch.float32, device=dev), persistent=False
    )
    mod.register_buffer(
        "_pair_out", torch.empty(decode_size, hidden_dim, dtype=torch.float32, device=dev), persistent=False
    )
    # ``_scratch_S`` / ``_scratch_T`` are int markers (not tensors) used by the
    # forward to check buffer-reuse eligibility. Assigned via object.__setattr__
    # so nn.Module's typed __setattr__ doesn't reject non-Tensor values.
    object.__setattr__(mod, "_scratch_S", decode_size)
    object.__setattr__(mod, "_scratch_T", 1)
