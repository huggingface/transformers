# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dequantizing GGUF blocks with torch ops.

Covers the four types a K-quant file actually uses — Q4_K, Q5_K, Q6_K and Q8_0 — which is all of
Q4_K_M/Q4_K_S/Q5_K_*/Q6_K/Q8_0, i.e. the overwhelming majority of GGUF checkpoints in the wild.

Everything is vectorised over blocks, so this runs on whatever device the bytes are on. Block
layouts follow `ggml-common.h`; the arithmetic follows ggml's reference dequantizers, which is what
makes the result bit-identical to them.
"""

import torch


# ggml type ids, as numbered by `enum ggml_type` in ggml.h
GGML_Q8_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 8, 12, 13, 14

# ggml type id -> (elements per block, bytes per block)
GGML_BLOCK = {
    GGML_Q8_0: (32, 34),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
}

# ggml type id -> its name, for messages
GGML_NAME = {GGML_Q8_0: "Q8_0", GGML_Q4_K: "Q4_K", GGML_Q5_K: "Q5_K", GGML_Q6_K: "Q6_K"}


def dequantize(data: torch.Tensor, ggml_type: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Flat `uint8` GGUF bytes -> flat values of `dtype`.

    `dtype` is where the arithmetic happens, not a cast afterwards: the quants are the only large
    tensor here, so widening them to f32 to produce a bf16 weight would double the traffic of every
    unpack. A block's scales stay in f32 — there are two of them per 256 values — and are rounded to
    `dtype` before the multiply.
    """
    if ggml_type not in GGML_BLOCK:
        supported = ", ".join(f"{name} ({type_id})" for type_id, name in sorted(GGML_NAME.items()))
        raise ValueError(f"ggml type {ggml_type} is not supported yet. Supported quantized types: {supported}.")
    block_elems, block_bytes = GGML_BLOCK[ggml_type]
    blocks = data.reshape(-1, block_bytes)
    values = _DEQUANT[ggml_type](blocks, dtype)
    return values.reshape(-1)[: blocks.shape[0] * block_elems]


def _half(blocks: torch.Tensor, start: int) -> torch.Tensor:
    """Read one fp16 scalar per block, as (nb, 1) float32."""
    return blocks[:, start : start + 2].contiguous().view(torch.float16).float()


def _k_scales(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the 12 bytes of 6-bit scales/mins shared by Q4_K and Q5_K (ggml's get_scale_min_k4).

    The first four scale/min pairs are plain 6-bit fields; the last four are split, taking their low
    nibble from bytes 8..11 and their top two bits from the spare high bits of bytes 0..7.
    """
    q = scales.int()
    scale = torch.cat([q[:, :4] & 63, (q[:, 8:12] & 0xF) | ((q[:, 0:4] >> 6) << 4)], dim=1)
    minimum = torch.cat([q[:, 4:8] & 63, (q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)], dim=1)
    return scale.float(), minimum.float()


def _interleave_nibbles(qs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(nb, 128) nibble bytes -> low/high nibbles as (nb, 4, 32) each, still `uint8`.

    ggml walks 64 output values at a time: the 32 low nibbles of a byte group first, then the 32
    high nibbles, so the two halves belong to consecutive 32-element sub-blocks.
    """
    q = qs.reshape(-1, 4, 32)
    return q & 0xF, q >> 4


def _dequant_q8_0(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d = _half(blocks, 0).to(dtype)
    qs = blocks[:, 2:34].contiguous().view(torch.int8).to(dtype)
    return d * qs


def _dequant_q4_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d, dmin = _half(blocks, 0), _half(blocks, 2)
    scale, minimum = _k_scales(blocks[:, 4:16])
    low, high = _interleave_nibbles(blocks[:, 16:144])
    q = torch.stack([low, high], dim=2).reshape(-1, 8, 32).to(dtype)
    return (d * scale).to(dtype)[..., None] * q - (dmin * minimum).to(dtype)[..., None]


def _dequant_q5_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d, dmin = _half(blocks, 0), _half(blocks, 2)
    scale, minimum = _k_scales(blocks[:, 4:16])
    qh = blocks[:, 16:48].unsqueeze(1)  # (nb, 1, 32), one extra bit per value
    low, high = _interleave_nibbles(blocks[:, 48:176])
    shift = torch.arange(4, device=blocks.device, dtype=torch.uint8).reshape(1, 4, 1) * 2
    low = low + ((qh >> shift) & 1) * 16
    high = high + ((qh >> (shift + 1)) & 1) * 16
    q = torch.stack([low, high], dim=2).reshape(-1, 8, 32).to(dtype)
    return (d * scale).to(dtype)[..., None] * q - (dmin * minimum).to(dtype)[..., None]


def _dequant_q6_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d = _half(blocks, 208)
    ql, qh = blocks[:, 0:128], blocks[:, 128:192]
    scales = blocks[:, 192:208].contiguous().view(torch.int8).float()
    # 16 values share a scale; the four quarters of each 128-element half use scales is+0/2/4/6
    which = torch.arange(32, device=blocks.device) // 16
    out = []
    for half in range(2):
        lo, hi = ql[:, half * 64 : half * 64 + 32], ql[:, half * 64 + 32 : (half + 1) * 64]
        h, sc = qh[:, half * 32 : (half + 1) * 32], scales[:, half * 8 : (half + 1) * 8]
        quants = [
            (lo & 0xF) | ((h & 3) << 4),
            (hi & 0xF) | (((h >> 2) & 3) << 4),
            (lo >> 4) | (((h >> 4) & 3) << 4),
            (hi >> 4) | (((h >> 6) & 3) << 4),
        ]
        for quarter, q in enumerate(quants):
            scale = (d * sc[:, which + 2 * quarter]).to(dtype)
            out.append(scale * (q.to(dtype) - 32))
    return torch.cat(out, dim=1)


_DEQUANT = {
    GGML_Q8_0: _dequant_q8_0,
    GGML_Q4_K: _dequant_q4_k,
    GGML_Q5_K: _dequant_q5_k,
    GGML_Q6_K: _dequant_q6_k,
}
