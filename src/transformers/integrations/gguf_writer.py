# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Thin .gguf writer driven by the quantizer's reverse-rename map.

Replaces the previous 370-line ``gguf_save.py`` with a small driver that
leans on the existing pieces:

* :attr:`GGUFQuantizer.hf_to_gguf` — built at load time, used here in reverse
  to map HF state-dict keys back to gguf names.
* :class:`~transformers.gguf_conversion_ops.GGUFQuantize` — the reverse op of
  :class:`GGUFDequantize`; packs an fp16/bf16/fp32 tensor into GGUF block
  bytes via ``gguf.quants.quantize``.
* :attr:`GGUFQuantizer.gguf_tensors` — the original-source byte map, so the
  round-trip path is just "write the bytes we already have".
* :attr:`GGUFQuantizer.gguf_kv` — replayed verbatim so tokenizer / block_count
  / quantization_version / ... survive a load → save cycle.
"""

from __future__ import annotations

import torch


# GGUF's arch tag mostly matches HF ``model_type``; only the underscore-stripped
# / aliased exceptions need entries here.
_ARCH_OVERRIDES: dict[str, str] = {
    "llama4": "llama",
    "mistral": "llama",
    "qwen2_moe": "qwen2moe",
    "qwen3_moe": "qwen3moe",
    "minimax_m2": "minimax",
}


def _arch_from_config(config) -> str:
    model_type = getattr(config, "model_type", "llama")
    return _ARCH_OVERRIDES.get(model_type, model_type)


def _replay_kv(writer, kv: dict | None) -> None:
    if not kv:
        return
    from gguf import GGUFValueType

    for key, (val, vtype) in kv.items():
        # Skip auto-written / writer-controlled keys (replaying causes "Duplicate" errors).
        if key.startswith("GGUF.") or key in ("general.architecture", "general.alignment"):
            continue
        try:
            if isinstance(val, list):
                writer.add_array(key, val)
            else:
                writer.add_key_value(key, val, GGUFValueType(vtype))
        except Exception:
            continue


def _write_original_bytes(writer, gguf_name: str, tensor) -> None:
    """Round-trip path: write the original ``GGUFQuantizedTensor`` bytes back
    out under the same gguf name, preserving the source quant type."""
    import gguf
    import numpy as np

    qt = getattr(tensor, "quant_type", None)
    if qt is None:
        return
    qt_str = qt.name if hasattr(qt, "name") else str(qt)
    raw = tensor.detach().cpu().numpy()
    raw_dtype = getattr(gguf.GGMLQuantizationType, qt_str)
    shape = tuple(tensor.shape)

    # Float-stored tensors (F32/F16/F64): view raw bytes as that dtype.
    float_dtype = {"F32": np.float32, "F16": np.float16, "F64": np.float64}.get(qt_str)
    if float_dtype is not None:
        arr = np.frombuffer(raw.tobytes(), dtype=float_dtype).reshape(shape)
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        return

    # Quantized: reshape into (rows, row_bytes) so gguf-py recovers logical (rows, cols).
    M = shape[0] if shape else 1
    row_bytes = raw.size // max(M, 1)
    arr = raw.reshape(M, row_bytes) if row_bytes else raw
    writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)


def _write_float_state_tensor(writer, gguf_name: str, tensor: torch.Tensor) -> None:
    """Save norms / embeddings / lm_head as F32 raw bytes — keeps the writer
    free of per-tensor policy decisions."""
    import gguf

    cpu = tensor.detach().to(torch.float32).cpu().numpy()
    writer.add_tensor(gguf_name, cpu, raw_dtype=gguf.GGMLQuantizationType.F32)


def save_pretrained_gguf(model, path: str, *, quant_config=None, quantizer=None) -> str:
    """Write ``model`` back to a ``.gguf`` file at ``path``.

    Round-trip semantics: tensors that the quantizer holds the source bytes for
    are written verbatim (so attn_q / attn_k preserve llama.cpp's permuted
    layout, K-quants stay K-quant, etc.). Remaining floating-point state-dict
    entries are written as F32.

    On-the-fly quantization for save is handled by the loader path
    (``GgufQuantizeConfig`` → :class:`~transformers.gguf_conversion_ops.GGUFQuantize`),
    which packs weights into GGUF bytes at load time so the live model carries
    them. By the time we reach this writer, every quantized weight is already
    a uint8 buffer — we just need to spell it out under the right name.

    ``quant_config`` is reserved for future per-tensor quant policy and is
    ignored today.
    """
    del quant_config  # not honoured yet; kept for API stability with the old saver.
    import gguf

    quantizer = quantizer or getattr(model, "hf_quantizer", None)
    if quantizer is None:
        raise ValueError(
            "save_pretrained_gguf requires a GGUFQuantizer (load the model with `gguf_file=` or "
            "`quantization_config=GgufQuantizeConfig(...)` and re-call from the loaded instance)."
        )

    writer = gguf.GGUFWriter(path, arch=_arch_from_config(model.config))
    _replay_kv(writer, getattr(quantizer, "gguf_kv", None))

    seen: set[str] = set()
    hf_to_gguf: dict[str, str] = dict(getattr(quantizer, "hf_to_gguf", {}) or {})

    # Pass 1 — original-bytes round-trip from the quantizer's source map.
    for gguf_name, tensor in (getattr(quantizer, "gguf_tensors", None) or {}).items():
        _write_original_bytes(writer, gguf_name, tensor)
        seen.add(gguf_name)

    # Pass 2 — remaining state_dict tensors that didn't come from the source .gguf
    # (norms / embeddings / on-the-fly-quantized weights from a non-GGUF checkpoint).
    for hf_name, tensor in model.state_dict().items():
        if not torch.is_tensor(tensor):
            continue
        gguf_name = hf_to_gguf.get(hf_name, hf_name)
        if gguf_name in seen:
            continue
        seen.add(gguf_name)
        if tensor.dtype == torch.uint8:
            # On-the-fly-quantized GgufLinear / GgufExperts buffer. Use the
            # quantizer config's quant_type (which drove the GGUFQuantize op
            # at load time).
            qt_str = getattr(quantizer.quantization_config, "quant_type", "Q4_0")
            raw = tensor.detach().cpu().numpy()
            M = tensor.shape[0] if tensor.dim() > 0 else 1
            row_bytes = raw.size // max(M, 1)
            arr = raw.reshape(M, row_bytes) if row_bytes else raw
            writer.add_tensor(gguf_name, arr, raw_dtype=getattr(gguf.GGMLQuantizationType, qt_str))
        else:
            _write_float_state_tensor(writer, gguf_name, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
