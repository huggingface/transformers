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

``GGUFQuantizedTensor`` is a ``torch.Tensor`` subclass that carries the
``quant_type`` metadata alongside the raw uint8 bytes, so the standard
loader can move it to the target device with ``.to(device)`` and the
``GGUFDequantize`` op in the weight-conversion chain can dequant on-device
without any GGUF-specific hook in ``core_model_loading``.
"""

from __future__ import annotations

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


class GGUFQuantizedTensor(torch.Tensor):
    """``torch.Tensor`` subclass that carries the GGUF ``quant_type`` alongside
    raw uint8 bytes.

    Three small affordances make the hot path through ``spawn_materialize`` fast
    *without* touching ``core_model_loading``:

    * ``__getitem__(Ellipsis)`` short-circuits — ``tensor[...]`` is a no-op for
      already-loaded torch bytes; the default goes through ``__torch_function__``
      dispatch for no reason.
    * ``.to(...)`` defaults ``non_blocking=True``, letting the loader queue the
      next transfer / dequant on the same MPS/CUDA stream while the previous
      copy is still in flight.
    * ``__torch_function__`` only re-wraps on ``Tensor.to`` — the one place we
      need ``quant_type`` to survive (so the :class:`GGUFDequantize` op in the
      conversion chain can read it on the device side). All other ops return
      plain tensors, which avoids the per-op wrap overhead.

    Inspired by ``GGUFParameter`` in diffusers.
    """

    # Class-level default so subclass instances spawned by torch's default
    # ``__torch_function__`` path (which doesn't invoke our ``__new__``) still
    # have the attribute defined.
    quant_type = None

    @staticmethod
    def __new__(cls, data, quant_type=None):
        data = data if data is not None else torch.empty(0)
        instance = torch.Tensor._make_subclass(cls, data, require_grad=False)
        instance.quant_type = quant_type
        return instance

    def __getitem__(self, key):
        # ``_materialize_copy`` does ``tensor = tensor[...]`` to pull a memmap
        # safetensors slice into RAM. Our bytes are already a torch.Tensor view,
        # so this is a no-op; short-circuit before torch dispatches via
        # ``__torch_function__`` (which would re-wrap into a fresh subclass).
        if key is Ellipsis:
            return self
        return super().__getitem__(key)

    def to(self, *args, **kwargs):
        # The loader queues a dequant op on the destination device right after
        # this transfer (same MPS/CUDA stream), so ``non_blocking=True`` overlaps
        # the bytes copy with the next ``.to(device)`` call from another worker.
        # Skip the inject if ``non_blocking`` was already supplied positionally
        # (3rd positional after device + dtype) — torch's ``nn.Module._apply``
        # calls ``Tensor.to`` with positional args on some torch versions and
        # would otherwise raise a duplicate-kwarg TypeError.
        if "non_blocking" not in kwargs and len(args) < 3:
            kwargs["non_blocking"] = True
        return super().to(*args, **kwargs)

    @staticmethod
    def _extract_quant_type(args):
        for arg in args:
            if isinstance(arg, GGUFQuantizedTensor) and arg.quant_type is not None:
                return arg.quant_type
            if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], GGUFQuantizedTensor):
                if arg[0].quant_type is not None:
                    return arg[0].quant_type
        return None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        # Only re-wrap on ``Tensor.to`` — the GGUFDequantize op in the conversion
        # chain reads ``quant_type`` off the post-transfer tensor. Other ops in
        # the path don't care, so skipping the wrap saves Python overhead per call.
        if func is not torch.Tensor.to:
            return result
        quant_type = cls._extract_quant_type(args)
        if quant_type is None:
            return result
        if isinstance(result, cls):
            result.quant_type = quant_type
            return result
        if isinstance(result, torch.Tensor):
            return cls(result, quant_type=quant_type)
        return result


def dequantize_gguf_tensor(data, quant_type, dtype=None, device=None) -> torch.Tensor:
    """Dequantize a GGUF tensor to a ``torch.Tensor`` using torch ops.

    Args:
        data: the ``ReaderTensor.data`` numpy array from ``gguf.GGUFReader``.
            For quantized types this is shaped as the **byte** shape
            (rows × byte-cols, uint8); for F16/F32 it is already the
            logical shape with the matching dtype.
        quant_type: ``gguf.GGMLQuantizationType`` enum value.
        dtype: target floating-point dtype for the dequantized output
            (defaults to ``torch.float32``).
        device: device to run the dequant kernel on. The raw uint8 input is
            transferred to ``device`` *before* the kernel runs so the big
            float output is produced on-device — this is dramatically faster
            on MPS / CUDA than dequantizing on CPU and copying the
            fully-expanded tensor afterwards.

    Returns:
        Tensor with the **logical** shape recovered from the byte shape
        (matching what ``gguf.dequantize`` returns).

    Raises:
        NotImplementedError: when ``quant_type`` has no torch kernel.
    """
    import gguf
    import torch

    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = _build_dispatch()

    if dtype is None:
        dtype = torch.float32
    target_device = torch.device(device) if device is not None else None

    # Accept either a numpy array (gguf reader) or a torch.Tensor (e.g. a
    # ``GGUFQuantizedTensor`` already moved to the target device). For numpy
    # we materialise the mmap view into RAM and wrap zero-copy; for torch we
    # just take a uint8 view of the raw bytes.
    if isinstance(data, torch.Tensor):
        bytes_view = data.contiguous().view(torch.uint8)
        byte_shape = tuple(bytes_view.shape)
        flat = bytes_view.reshape(-1)
    else:
        import numpy as np

        arr = np.ascontiguousarray(data)
        byte_shape = arr.shape
        flat = torch.from_numpy(arr.reshape(-1))

    if target_device is not None and flat.device != target_device:
        flat = flat.to(target_device, non_blocking=True)

    # Already-float types: just reinterpret the bytes.
    if quant_type == gguf.GGMLQuantizationType.F32:
        out = flat.view(torch.float32).reshape(byte_shape[:-1] + (byte_shape[-1] // 4,))
        return out.to(dtype) if dtype != torch.float32 else out
    if quant_type == gguf.GGMLQuantizationType.F16:
        out = flat.view(torch.float16).reshape(byte_shape[:-1] + (byte_shape[-1] // 2,))
        return out.to(dtype) if dtype != torch.float16 else out

    kernel = _DISPATCH.get(quant_type)
    if kernel is None:
        raise NotImplementedError(
            f"No torch dequant kernel for GGUF quant type {quant_type!r}. "
            f"Supported types: {sorted(t.name for t in _DISPATCH)}"
        )

    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    # logical_shape recovers the per-element shape from the byte shape (last
    # dim shrinks from ``n_blocks * type_size`` bytes → ``n_blocks * block_size`` elements).
    logical_shape = (*byte_shape[:-1], byte_shape[-1] // type_size * block_size)

    blocks = flat.reshape((-1, type_size))
    out = kernel(blocks, block_size, type_size, dtype=dtype)
    return out.reshape(logical_shape)
