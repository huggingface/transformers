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
"""Dequantizing GGUF blocks with torch ops."""

import torch


# ggml type ids, as numbered by `enum ggml_type` in ggml.h
GGML_Q8_0, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 8, 11, 12, 13, 14
GGML_IQ4_NL, GGML_IQ3_S, GGML_IQ4_XS = 20, 21, 23

# ggml type id -> (elements per block, bytes per block)
GGML_BLOCK = {
    GGML_Q8_0: (32, 34),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ3_S: (256, 110),
    GGML_IQ4_XS: (256, 136),
}

# ggml type id -> its name, for messages
GGML_NAME = {
    GGML_Q8_0: "Q8_0",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ4_NL: "IQ4_NL",
    GGML_IQ3_S: "IQ3_S",
    GGML_IQ4_XS: "IQ4_XS",
}

# The 16 levels an IQ4 nibble indexes, shared by IQ4_NL and IQ4_XS (ggml's `kvalues_iq4nl`).
_IQ4_LEVELS = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def dequantize(data: torch.Tensor, ggml_type: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Flat `uint8` GGUF bytes -> flat values of `dtype`."""
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
    """Unpack the 12 bytes of 6-bit scales/mins shared by Q4_K and Q5_K (ggml's get_scale_min_k4)."""
    q = scales.int()
    scale = torch.cat([q[:, :4] & 63, (q[:, 8:12] & 0xF) | ((q[:, 0:4] >> 6) << 4)], dim=1)
    minimum = torch.cat([q[:, 4:8] & 63, (q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)], dim=1)
    return scale.float(), minimum.float()


def _interleave_nibbles(qs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(nb, 128) nibble bytes -> low/high nibbles as (nb, 4, 32) each, still `uint8`."""
    q = qs.reshape(-1, 4, 32)
    return q & 0xF, q >> 4


def _shifted(data: torch.Tensor, shifts: tuple[int, ...], width: int) -> torch.Tensor:
    """`data` read as fields of `len(shifts)` per byte: (nb, n, 1, width) >> shifts -> (nb, -1, width)."""
    shift = torch.tensor(shifts, device=data.device, dtype=torch.uint8).reshape(1, 1, -1, 1)
    return (data.reshape(data.shape[0], -1, 1, width) >> shift).reshape(data.shape[0], -1, width)


def _iq4_levels(nibbles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Nibbles -> the levels they index."""
    levels = torch.tensor(_IQ4_LEVELS, device=nibbles.device, dtype=torch.int8)
    return levels[nibbles.long()].to(dtype)


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


def _dequant_q3_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d = _half(blocks, 108)
    hmask, qs, scales = blocks[:, 0:32], blocks[:, 32:96], blocks[:, 96:108]
    # 16 six-bit scales, low nibbles in the first 8 bytes and high pairs in the last 4
    low = _shifted(scales[:, :8], (0, 4), 8).reshape(-1, 16)
    high = _shifted(scales[:, 8:12], (0, 2, 4, 6), 4).reshape(-1, 16)
    scale = (((low & 0xF) | ((high & 3) << 4)).to(torch.int8).float() - 32).to(dtype)

    ql = _shifted(qs, (0, 2, 4, 6), 32).reshape(-1, 16, 16) & 3
    # the high bit is an inverted borrow: the offset applies where the mask bit is clear
    qh = (_shifted(hmask, tuple(range(8)), 32).reshape(-1, 16, 16) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8)).to(dtype)
    return ((d.to(dtype) * scale)[..., None] * q).reshape(blocks.shape[0], -1)


def _dequant_iq4_nl(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d = _half(blocks, 0).to(dtype)
    nibbles = _shifted(blocks[:, 2:18], (0, 4), 16).reshape(-1, 32) & 0xF
    return d * _iq4_levels(nibbles, dtype)


def _dequant_iq4_xs(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d = _half(blocks, 0)
    scales_h = blocks[:, 2:4].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    # eight six-bit scales: four bytes of low nibbles here, low then high *within* each byte, with
    # their top two bits spread across one uint16
    shift = torch.tensor((0, 4), device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2)
    low = (blocks[:, 4:8].reshape(-1, 4, 1) >> shift).reshape(-1, 8) & 0xF
    shift = torch.arange(0, 16, 2, device=blocks.device, dtype=torch.int32).reshape(1, 8)
    high = ((scales_h >> shift) & 3).to(torch.uint8)
    scale = ((low | (high << 4)).to(torch.int8).float() - 32).to(dtype)

    nibbles = _shifted(blocks[:, 8:136], (0, 4), 16).reshape(-1, 8, 32) & 0xF
    return ((d.to(dtype) * scale)[..., None] * _iq4_levels(nibbles, dtype)).reshape(blocks.shape[0], -1)


# IQ3_S indexes a fixed codebook of 512 four-value points, packed as ggml ships it: two hex digits per
# byte, three bits per value, indexing `_IQ3S_LEVELS`.
_IQ3S_LEVELS = (1, 3, 5, 7, 9, 11, 13, 15)
_IQ3S_GRID_HEX = (
    b"0000010002000500070010001100120014001600200021002500330040004200"
    b"4500470051005300600062007100740077000001010102010401100111011501"
    b"2001230127013101350144016101650172010002010205020702100213021602"
    b"2102250230023402420245024702510253027002730203031103150320032203"
    b"3103330336034403500352036703710375030004130417042104240432044004"
    b"4304510470040205040520052205260533054105450547056605730506061106"
    b"1306310652067106000702070407200722072607330750075407001001100210"
    b"0410101011101310151017102010221031103410361054105610611072100011"
    b"0111031106111011141121113011331141115011521170117611001212121512"
    b"1712201224123212401243125512601272120113041307131013131321132713"
    b"3013341341136213701303140514121414143114331442144614501454140115"
    b"1015131521153015321551152016241627164416461601170317101712172117"
    b"3517411762177017002001200320052007201020122014201620212023202720"
    b"3020322041204320452050205220672070207320752000210221102113211721"
    b"2221252131213421422151210122042207222122232230223722412253225722"
    b"7122742200230223052311232223242331233323422350236623012407242024"
    b"2324322435244124722475240425112522253725402553257025002602260726"
    b"2126552661260527112726273027432750270230113013301530173022303130"
    b"3330353042304430473051306330713001310331053114312131233140316031"
    b"7231763100321232203232323432503201331033143321332333273330334133"
    b"4333473355337333033411341634223431345234603464340135103512352535"
    b"3235443556357335163641360137033720372237353700400440124020402440"
    b"2740324041405040704002410741114113412241304135414341514155410142"
    b"0342104215422142334240425742624270420443114313432043224331433543"
    b"0044024424443744404471440545074521456245134634466046104715473047"
    b"4347514702501050145022504050445047505250665074500151035105511251"
    b"2151325172510052115223523052365253520253075310532753445351536553"
    b"7353015404542054325446541255265551555355425602570457225711601360"
    b"1560316033606060006120612761646112623462426255626262706200631463"
    b"2163406325644364626400650365346560650566406611671367007004700770"
    b"2070227036704070547062700271117124714371457101720472107216722172"
    b"3072517202733273357353730174057413742074507422754275027631760077"
)
_IQ3S_GRIDS: dict[torch.device, torch.Tensor] = {}


def _iq3s_grid(device: torch.device) -> torch.Tensor:
    """The codebook as (512, 4) float32, built once per device."""
    if device not in _IQ3S_GRIDS:
        digits = torch.tensor(list(_IQ3S_GRID_HEX), dtype=torch.uint8, device=device).reshape(-1, 2)
        nibbles = torch.where(digits > 0x40, digits + 9, digits) & 0x0F
        packed = (nibbles[:, 0] << 4) | nibbles[:, 1]
        shift = torch.tensor((0, 4), dtype=torch.uint8, device=device).reshape(1, 2)
        levels = torch.tensor(_IQ3S_LEVELS, dtype=torch.float32, device=device)
        _IQ3S_GRIDS[device] = levels[((packed.reshape(-1, 1) >> shift) & 7).reshape(-1).long()].reshape(512, 4)
    return _IQ3S_GRIDS[device]


def _dequant_iq3_s(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    nb = blocks.shape[0]
    d = _half(blocks, 0)
    qs, qh, signs, scales = blocks[:, 2:66], blocks[:, 66:74], blocks[:, 74:106], blocks[:, 106:110]

    # four bytes hold eight four-bit scales, low then high nibble within each byte
    shift4 = torch.tensor((0, 4), device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2)
    scale = ((scales.reshape(nb, -1, 1) >> shift4) & 0xF).reshape(nb, -1).float()
    db = (d * (1 + 2 * scale)).to(dtype).reshape(nb, -1, 1, 1)

    bit = torch.arange(8, device=blocks.device, dtype=torch.uint8).reshape(1, 1, 8)
    sign = torch.where((signs.reshape(nb, -1, 1) >> bit) & 1 == 0, 1.0, -1.0).to(dtype).reshape(nb, -1, 4, 8)

    high = ((qh.reshape(nb, -1, 1) >> bit) & 1).reshape(nb, -1).int()
    index = qs.int() | (high << 8)
    points = _iq3s_grid(blocks.device)[index.reshape(-1).long()].to(dtype).reshape(nb, -1, 4, 8)
    return (db * points * sign).reshape(nb, -1)


_DEQUANT = {
    GGML_Q8_0: _dequant_q8_0,
    GGML_Q3_K: _dequant_q3_k,
    GGML_Q4_K: _dequant_q4_k,
    GGML_Q5_K: _dequant_q5_k,
    GGML_Q6_K: _dequant_q6_k,
    GGML_IQ4_NL: _dequant_iq4_nl,
    GGML_IQ3_S: _dequant_iq3_s,
    GGML_IQ4_XS: _dequant_iq4_xs,
}
