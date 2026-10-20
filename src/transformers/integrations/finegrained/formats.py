# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Storage of the block-scaled formats.

Both the modules in `layers` and the ops in `conversions` read it.
"""

import functools
from dataclasses import dataclass

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MIN = torch.finfo(FP8_DTYPE).min
FP8_MAX = torch.finfo(FP8_DTYPE).max


@functools.cache
def ue8m0_dtype() -> torch.dtype:
    """Return the native UE8M0 dtype when torch has it, else `uint8` for the biased exponent bytes."""
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def cdiv(a: int, b: int) -> int:
    """Return the ceiling of `a / b`."""
    return (a + b - 1) // b


@dataclass(frozen=True)
class WeightFormat:
    """Storage of one weight format.

    `scale_dtype` and `scale_group` are `None` for block-FP8.
    The config's `scale_fmt` and `weight_block_size` set them instead.
    `global_scale_dtype` is the dtype of NVFP4's per-matrix global, and `None` for other formats.
    """

    weight_dtype: torch.dtype
    values_per_byte: int = 1
    scale_dtype: torch.dtype | None = None
    scale_group: tuple[int, int] | None = None
    global_scale_dtype: torch.dtype | None = None


def get_weight_format(weight_format: str) -> WeightFormat:
    """Return the storage of `weight_format`.

    Only the requested format is built, so FP8 and NVFP4 do not need UE8M0 support.
    """
    if weight_format == "fp8":
        return WeightFormat(weight_dtype=FP8_DTYPE)
    if weight_format in ("mxfp8", "mxfp4"):
        packed = weight_format == "mxfp4"
        return WeightFormat(
            weight_dtype=torch.int8 if packed else FP8_DTYPE,
            values_per_byte=2 if packed else 1,
            scale_dtype=ue8m0_dtype(),
            scale_group=(1, 32),
        )
    if weight_format == "nvfp4":
        return WeightFormat(
            weight_dtype=torch.int8,
            values_per_byte=2,
            scale_dtype=FP8_DTYPE,
            scale_group=(1, 16),
            global_scale_dtype=torch.float32,
        )
    raise ValueError(f"unknown weight_format {weight_format!r}; expected fp8, mxfp8, mxfp4 or nvfp4")


def resolve_weight_format(
    weight_format: str,
    scale_fmt: str = "float",
    block_size: tuple[int, int] | None = None,
) -> tuple[WeightFormat, torch.dtype, tuple[int, int] | None]:
    """Return `(format, scale_dtype, scale_group)` for a format name.

    The config's `scale_fmt` and `weight_block_size` fill what the format leaves open.
    """
    format_spec = get_weight_format(weight_format)
    scale_dtype = format_spec.scale_dtype
    if scale_dtype is None:
        scale_dtype = ue8m0_dtype() if scale_fmt == "ue8m0" else torch.float32
    return format_spec, scale_dtype, format_spec.scale_group if format_spec.scale_group is not None else block_size


# the E2M1 value of each 4-bit code, with the sign in the high bit
# FP4 packs two codes per byte, low nibble first
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 bytes to float32 values. The last dim doubles."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = packed.contiguous().view(torch.uint8)
    unpacked = torch.stack([table[(codes & 0xF).long()], table[(codes >> 4).long()]], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])


def slot_name(weight: str, slot: str) -> str:
    """Return the parameter name a module gives `slot` of `weight`.

    `slot` is `"scale_inv"`, `"weight_global_scale"` or `"activation_scale"`.
    A dense `weight` gets `weight_scale_inv`. An experts stack gets names like `gate_up_proj_scale_inv`.
    """
    if weight != "weight":
        return f"{weight}_{slot}"
    return "weight_scale_inv" if slot == "scale_inv" else slot
