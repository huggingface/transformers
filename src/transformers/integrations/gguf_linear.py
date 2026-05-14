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


def _ensure_metal_kernels():
    """Idempotent loader for the kernels-community Metal kernel package."""
    global _GGUF_METAL_KERNELS, _GGUF_METAL_LOADED
    if _GGUF_METAL_LOADED:
        return _GGUF_METAL_KERNELS
    _GGUF_METAL_LOADED = True
    if os.environ.get("TRANSFORMERS_GGUF_USE_METAL_KERNELS", "1") == "0":
        return None
    try:
        from kernels import get_kernel  # type: ignore[import-not-found]

        repo = os.environ.get("TRANSFORMERS_GGUF_METAL_KERNELS_REPO", "ArthurZ/gguf-kernels")
        _GGUF_METAL_KERNELS = get_kernel(repo)
    except Exception:
        _GGUF_METAL_KERNELS = None
    return _GGUF_METAL_KERNELS


# Per quant type: (block bytes, block elems). Format names match the suffix of
# the ``mul_mat_<fmt>_f32`` and ``mul_mat_vec_<fmt>_f32`` ops in the kernel.
_QUANT_INFO: dict[str, tuple[int, int]] = {
    "Q4_0":   (18, 32),
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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Last dim of x must be {self.in_features}, got {x.shape[-1]}")

        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features).contiguous().to(torch.float32)
        N = x_flat.shape[0]

        is_mps = x.device.type == "mps"
        if is_mps:
            mod = _ensure_metal_kernels()
            if mod is None:
                # Kernels package not installed — fall back to dequant + linear.
                return self._dequant_forward(x)
            qw = self.qweight.to(x.device)
            if N == 1:
                fn = getattr(mod._ops, f"mul_mat_vec_{self._fmt}_f32")
                y = torch.empty(self.out_features, dtype=torch.float32, device=x.device)
                fn(qw, x_flat.reshape(-1), y)
                y = y.reshape(*batch_shape, self.out_features)
            else:
                fn = getattr(mod._ops, f"mul_mat_{self._fmt}_f32")
                y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
                fn(qw, x_flat, y)
                y = y.reshape(N, self.out_features).reshape(*batch_shape, self.out_features)
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
        swapped += 1

    return swapped


def is_gguf_linear_enabled() -> bool:
    """``TRANSFORMERS_GGUF_LINEAR=1`` opts into the GgufLinear path. Off by default."""
    return os.environ.get("TRANSFORMERS_GGUF_LINEAR", "0") not in ("0", "", "false", "False")
