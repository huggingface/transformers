# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tiny loader for the ``ArthurZ/gguf-kernels`` Metal kernel package.

Owns the ``_ensure_metal_kernels`` cache + the kernel-ref helper used by the
fused-expert ``GgufExperts`` path. Lives in its own module so ``gguf_linear``
doesn't have to grow a kernel-loading prelude — the helper will ultimately
move into the kernels package itself.

Contract:
    * ``ensure_metal_kernels()`` returns a handle exposing ``_ops`` (the
      :mod:`torch.ops` namespace) or raises ``RuntimeError``. There is no
      slow fallback: callers that request kernels must get them or fail loudly.
    * ``bind_id_kernel_refs(experts_module)`` caches the per-projection
      ``mul_mat_id_<fmt>_f32`` op overloads + pre-allocates the decode-shape
      scratch buffers on the expert module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_GGUF_METAL_KERNELS = None
_GGUF_METAL_LOADED = False


class _DirectKernelHandle:
    """Stand-in for the kernels-package module — exposes only ``_ops``."""

    def __init__(self, ops):
        self._ops = ops


def _load_kernels_direct(repo: str) -> _DirectKernelHandle:
    """``snapshot_download`` the .so for the current torch variant and load it."""
    import os
    import re

    from huggingface_hub import snapshot_download

    try:
        from kernels.utils import build_variant
    except ImportError:
        major, minor = torch.__version__.split(".")[:2]

        def build_variant():
            return f"torch{int(major)}{int(minor)}-metal-aarch64-darwin"

    variant = build_variant() if callable(build_variant) else build_variant
    repo_dir = snapshot_download(repo, allow_patterns=[f"build/{variant}/**"])
    var_dir = os.path.join(repo_dir, "build", variant)
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
    return _DirectKernelHandle(getattr(torch.ops, ns_name))


def ensure_metal_kernels(repo: str = "ArthurZ/gguf-kernels"):
    """Return the loaded kernels handle, caching across calls.

    Raises :class:`RuntimeError` on failure — there is no slow fallback path.
    """
    global _GGUF_METAL_KERNELS, _GGUF_METAL_LOADED
    if _GGUF_METAL_LOADED:
        if _GGUF_METAL_KERNELS is None:
            raise RuntimeError(
                f"GGUF metal kernels failed to load earlier in this process. Reinstall / pre-fetch {repo!r} and retry."
            )
        return _GGUF_METAL_KERNELS
    _GGUF_METAL_LOADED = True
    last_err: Exception | None = None
    try:
        from kernels import get_kernel

        _GGUF_METAL_KERNELS = get_kernel(repo)
        return _GGUF_METAL_KERNELS
    except Exception as exc:
        last_err = exc
    try:
        _GGUF_METAL_KERNELS = _load_kernels_direct(repo)
        return _GGUF_METAL_KERNELS
    except Exception as exc:
        raise RuntimeError(
            f"Could not load GGUF metal kernels from {repo!r}. The fast GGUF path requires this "
            f"package on Apple Silicon. Either install `kernels` (`pip install kernels`) or make "
            f"sure the .so for your torch build is available locally. Underlying error: {exc!r}"
        ) from last_err or exc


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
