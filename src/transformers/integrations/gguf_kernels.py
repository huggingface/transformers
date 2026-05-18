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
                f"GGUF metal kernels failed to load earlier in this process. "
                f"Reinstall / pre-fetch {repo!r} and retry."
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


def bind_id_kernel_refs(mod: nn.Module) -> None:
    """Resolve ``mul_mat_id_<fmt>_f32`` op overloads + pre-allocate decode scratch
    buffers on a fused-MoE expert module. Called once at swap time so the hot
    forward path can do constant-time attribute reads + ``torch.compile`` sees the
    ops as fixed module attributes.

    Two op refs to resolve:
      * ``_id_op_gate_up`` — same kernel for both halves of ``gate_up_proj``
        (the merge converter requires gate / up to share a quant type, so we
        only ever need one ref).
      * ``_id_op_down`` — kernel for ``down_proj`` (may differ; Q4_K_M ships
        gate/up = Q4_K but down = Q8_0).
    """
    ops = ensure_metal_kernels()._ops
    for proj, fmt_attr, cache_attr in (
        ("gate_up", "_gate_up_fmt", "_id_op_gate_up"),
        ("down", "_down_fmt", "_id_op_down"),
    ):
        fmt = getattr(mod, fmt_attr, None)
        if fmt is None:
            return
        op_name = f"mul_mat_id_{fmt}_f32"
        op = getattr(ops, op_name, None)
        if op is None:
            raise RuntimeError(f"GGUF metal kernels missing op {op_name!r}; cannot bind {proj} for {mod}")
        # Resolve to the concrete OpOverload to skip one dispatcher level on every call.
        op = getattr(op, "default", op)
        setattr(mod, cache_attr, op)

    # Pre-allocate decode-shape scratch buffers so the forward path never calls
    # ``torch.empty`` from inside dynamo's compiled graph (attribute mutation on
    # first call triggers a full re-trace on the second).
    H = mod.hidden_dim
    I_dim = mod.intermediate_dim
    top_k = getattr(getattr(mod, "config", None), "num_experts_per_tok", None) or 4
    dev = mod.gate_up_proj.device
    S_dec = top_k  # decode-time S (num_tokens=1)
    for name in ("_gate_buf", "_up_buf", "_pair_out"):
        if name in mod._buffers:
            del mod._buffers[name]
    mod.register_buffer("_gate_buf", torch.empty(S_dec, I_dim, dtype=torch.float32, device=dev), persistent=False)
    mod.register_buffer("_up_buf", torch.empty(S_dec, I_dim, dtype=torch.float32, device=dev), persistent=False)
    mod.register_buffer("_pair_out", torch.empty(S_dec, H, dtype=torch.float32, device=dev), persistent=False)
    # ``_scratch_*`` are plain Python attributes (not parameters / buffers) used
    # by the hot forward to skip re-allocation; assign through object to avoid
    # nn.Module's typed __setattr__ rejecting non-Tensor values.
    object.__setattr__(mod, "_scratch_S", S_dec)
    object.__setattr__(mod, "_scratch_T", 1)
