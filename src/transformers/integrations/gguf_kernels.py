# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations


_GGUF_METAL_KERNELS = None
_GGUF_METAL_LOADED = False


# Quant-type → kernel-name suffix table. Kernel names like
# `mul_mat_<fmt>_f32` / `mul_mat_vec_<fmt>_f32` / `mul_mat_id_<fmt>_f32`
# are deterministic from this, so consumers can build the full name once at
# construction.
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


# TODO Change the repo name to kernel community once merged.
def ensure_metal_kernels(repo: str = "ArthurZ/gguf-kernels"):
    """Return the loaded kernels handle, caching across calls. Raises
    `RuntimeError` on failure — no fallback."""
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
        _GGUF_METAL_KERNELS = get_kernel(repo)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load GGUF metal kernels from {repo!r}: {exc!r}. The kernel repo must be "
            f"registered with the kernels-package backend (typically under kernels-community/); "
            f"once migrated this path Just Works."
        ) from exc
    return _GGUF_METAL_KERNELS


def metal_kernels_available() -> bool:
    """Probe whether `ensure_metal_kernels()` would succeed, without raising.
    Used by the quantizer to gate the GgufLinear swap path — if the kernels
    can't load, the safer choice is to fall back to dequant-on-load instead
    of installing modules whose forward will raise.
    """
    try:
        ensure_metal_kernels()
        return True
    except Exception:
        return False
