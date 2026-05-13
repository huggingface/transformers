# Copyright 2025 The HuggingFace Inc. team and City96. All rights reserved.
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
"""Pure-PyTorch GGUF dequantization.

Port of city96 / ComfyUI-GGUF (also used by diffusers' GGUF quantizer):
https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py

The reference dequant in ``gguf-py`` is pure NumPy and operates on grouped
row slices, which is ~5–20x slower than the same logic expressed as
``torch`` ops on a ``uint8`` view of the raw bytes (and also avoids the
``__array_finalize__`` overhead from memmap slicing). The ops here run on
CPU or GPU unchanged — pass an already-on-device tensor as input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

# Re-exported lazily so ``import transformers`` does not require gguf installed.
QK_K = 256
K_SCALE_SIZE = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uint32(x):
    import torch

    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _split_block_dims(blocks, *args):
    import torch

    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _get_scale_min(scales):
    import torch

    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


# ---------------------------------------------------------------------------
# Per-format kernels
# ---------------------------------------------------------------------------


def _dq_Q8_0(blocks, block_size, type_size, dtype=None):
    import torch

    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _dq_Q5_1(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return (d * qs) + m


def _dq_Q5_0(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def _dq_Q4_1(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def _dq_Q4_0(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


def _dq_Q6_K(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _dq_Q5_K(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(0, 8, device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 8, 1)
    )
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dq_Q4_K(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dq_Q3_K(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 2, 1)
    )
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.arange(0, 8, device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 8, 1)
    )
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))


def _dq_Q2_K(blocks, block_size, type_size, dtype=None):
    import torch

    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


def _dq_BF16(blocks, block_size, type_size, dtype=None):
    import torch

    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


_IQ4_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _dq_IQ4_NL(blocks, block_size, type_size, dtype=None):
    import torch

    kvalues = torch.tensor(_IQ4_KVALUES, dtype=torch.float32, device=blocks.device)
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs


def _dq_IQ4_XS(blocks, block_size, type_size, dtype=None):
    import torch

    kvalues = torch.tensor(_IQ4_KVALUES, dtype=torch.float32, device=blocks.device)
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = _split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor(
        [2 * i for i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8
    ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)) & 0x03
    scales = (scales_l | (scales_h << 4)) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2], 16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)


def _build_dispatch():
    """Build the {GGMLQuantizationType: kernel} dispatch table lazily so that
    importing this module does not require ``gguf`` to be installed."""
    import gguf

    return {
        gguf.GGMLQuantizationType.IQ4_NL: _dq_IQ4_NL,
        gguf.GGMLQuantizationType.IQ4_XS: _dq_IQ4_XS,
        gguf.GGMLQuantizationType.BF16: _dq_BF16,
        gguf.GGMLQuantizationType.Q8_0: _dq_Q8_0,
        gguf.GGMLQuantizationType.Q5_1: _dq_Q5_1,
        gguf.GGMLQuantizationType.Q5_0: _dq_Q5_0,
        gguf.GGMLQuantizationType.Q4_1: _dq_Q4_1,
        gguf.GGMLQuantizationType.Q4_0: _dq_Q4_0,
        gguf.GGMLQuantizationType.Q6_K: _dq_Q6_K,
        gguf.GGMLQuantizationType.Q5_K: _dq_Q5_K,
        gguf.GGMLQuantizationType.Q4_K: _dq_Q4_K,
        gguf.GGMLQuantizationType.Q3_K: _dq_Q3_K,
        gguf.GGMLQuantizationType.Q2_K: _dq_Q2_K,
    }


_DISPATCH = None


def supported_quant_types():
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = _build_dispatch()
    return set(_DISPATCH.keys())


def dequantize_gguf_tensor(data, quant_type, dtype=None) -> torch.Tensor:
    """Dequantize a GGUF tensor to a ``torch.Tensor`` using torch ops.

    Args:
        data: the ``ReaderTensor.data`` numpy array from ``gguf.GGUFReader``.
            For quantized types this is shaped as the **byte** shape
            (rows × byte-cols, uint8); for F16/F32/F64/I* it is already the
            logical shape with the matching dtype.
        quant_type: ``gguf.GGMLQuantizationType`` enum value.
        dtype: target floating-point dtype for the dequantized output
            (defaults to ``torch.float32``).

    Returns:
        Tensor with the **logical** shape recovered from the byte shape
        (matching what ``gguf.dequantize`` returns). Falls back to gguf-py's
        NumPy dequant for quant types without a torch kernel (uncommon IQ
        formats).
    """
    import gguf
    import numpy as np
    import torch

    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = _build_dispatch()

    if dtype is None:
        dtype = torch.float32

    arr = np.ascontiguousarray(data)  # materialise any mmap view → contiguous RAM

    # Already-float types: ``data`` carries the logical shape with the right dtype.
    if quant_type == gguf.GGMLQuantizationType.F32:
        return torch.from_numpy(arr).to(dtype) if dtype != torch.float32 else torch.from_numpy(arr)
    if quant_type == gguf.GGMLQuantizationType.F16:
        return torch.from_numpy(arr).to(dtype) if dtype != torch.float16 else torch.from_numpy(arr)

    kernel = _DISPATCH.get(quant_type)
    if kernel is None:
        # Fallback: numpy dequant for quant types we have not ported (rare IQ formats).
        return torch.from_numpy(np.ascontiguousarray(gguf.dequantize(arr, quant_type))).to(dtype)

    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    byte_shape = arr.shape  # e.g. (out_features, n_block_cols * type_size)
    logical_shape = (*byte_shape[:-1], byte_shape[-1] // type_size * block_size)

    flat = torch.from_numpy(arr.reshape(-1))
    blocks = flat.reshape((-1, type_size))
    out = kernel(blocks, block_size, type_size, dtype=dtype)
    return out.reshape(logical_shape)
