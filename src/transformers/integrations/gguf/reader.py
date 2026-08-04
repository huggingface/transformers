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
"""Reading a GGUF file: architecture, tensor types, and tensors keyed by their GGUF names.

Nothing is read eagerly: each name maps to a `LazyGgufTensor` that the loading pipeline materializes
when it gets to that parameter. Quantized tensors are dequantized there (see `dequant.py`), so
everything downstream — the per-arch renamings and converters — works on dense tensors and does not
care how the file stored them.

Renaming and any conversion are *not* done here — they are `WeightConverter`s in
`gguf_conversion_mapping.py`, run by the normal loading pipeline. This module only turns a file into
`{gguf_name: tensor}`.

The header is parsed here rather than with `gguf.GGUFReader`, which builds a Python object per
metadata element and so spends seconds on the tokenizer vocabulary — most of the load time for a
model whose weights we then read in under a second. All we need from it is the architecture and the
tensor table.
"""

import re
import struct
from functools import lru_cache
from math import prod
from typing import NamedTuple

import numpy as np
import torch

from .dequant import GGML_BLOCK, GGML_NAME, dequantize


# ggml type ids for the types a non-quantized GGUF can hold, and their width in bytes
_GGML_F32, _GGML_F16, _GGML_BF16 = 0, 1, 30
_TORCH_DTYPE = {_GGML_F32: torch.float32, _GGML_F16: torch.float16, _GGML_BF16: torch.bfloat16}
_FLOAT_WIDTH = {_GGML_F32: 4, _GGML_F16: 2, _GGML_BF16: 2}

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSIONS = (2, 3)  # v1 counted tensors in 32 bits; no file in the wild still uses it
_DEFAULT_ALIGNMENT = 32

# width of the fixed-size metadata value types: u8/i8/bool, u16/i16, u32/i32/f32, u64/i64/f64
_KV_WIDTH = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}
_KV_STRING, _KV_ARRAY = 8, 9


class TensorInfo(NamedTuple):
    """Where one tensor lives in the file, and how it is stored."""

    name: str
    shape: tuple[int, ...]  # in torch order
    ggml_type: int
    offset: int  # from the start of the data section
    nbytes: int


def _mapped(gguf_path: str) -> np.ndarray:
    """The file as a `uint8` memory map. Cheap: pages are only read when a tensor is materialized."""
    return np.memmap(gguf_path, mode="r", dtype=np.uint8)


def _read_string(blob: np.ndarray, pos: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", blob, pos)
    pos += 8
    return bytes(blob[pos : pos + length]).decode("utf-8"), pos + length


def _skip_value(blob: np.ndarray, pos: int, value_type: int) -> int:
    """Advance past one metadata value without building it."""
    if value_type in _KV_WIDTH:
        return pos + _KV_WIDTH[value_type]
    if value_type == _KV_STRING:
        (length,) = struct.unpack_from("<Q", blob, pos)
        return pos + 8 + length
    if value_type == _KV_ARRAY:
        element_type, count = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        if element_type in _KV_WIDTH:
            return pos + count * _KV_WIDTH[element_type]
        if element_type != _KV_STRING:
            raise ValueError(f"GGUF metadata holds an array of type {element_type}, which this reader cannot skip.")
        for _ in range(count):  # the vocabulary: variable-length, so there is nothing to do but walk it
            (length,) = struct.unpack_from("<Q", blob, pos)
            pos += 8 + length
        return pos
    raise ValueError(f"GGUF metadata holds a value of type {value_type}, which this reader cannot skip.")


@lru_cache(maxsize=1)
def _header(gguf_path: str) -> tuple[str, tuple[TensorInfo, ...], int]:
    """`(architecture, tensor infos, offset of the data section)`, read without touching tensor data.

    Cached because a single load asks for it several times: to pick a loader, to plan which weights
    stay packed, to build the conversion mapping, and to read the tensors.
    """
    blob = _mapped(gguf_path)
    if bytes(blob[:4]) != _GGUF_MAGIC:
        raise ValueError(f"{gguf_path} does not start with the GGUF magic bytes, so it is not a GGUF file.")
    version, tensor_count, metadata_count = struct.unpack_from("<IQQ", blob, 4)
    if version not in _GGUF_VERSIONS:
        raise ValueError(
            f"{gguf_path} is GGUF v{version}; this reader handles v{' and v'.join(map(str, _GGUF_VERSIONS))}."
        )
    pos = 24

    architecture, alignment = None, _DEFAULT_ALIGNMENT
    for _ in range(metadata_count):
        key, pos = _read_string(blob, pos)
        (value_type,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        if key == "general.architecture":
            architecture, pos = _read_string(blob, pos)
        elif key == "general.alignment":
            (alignment,) = struct.unpack_from("<I", blob, pos)
            pos += 4
        else:
            pos = _skip_value(blob, pos, value_type)
    if architecture is None:
        raise ValueError(f"{gguf_path} has no `general.architecture` in its metadata.")

    infos = []
    for _ in range(tensor_count):
        name, pos = _read_string(blob, pos)
        (dim_count,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        dims = struct.unpack_from(f"<{dim_count}Q", blob, pos)
        pos += 8 * dim_count
        ggml_type, offset = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        # ggml stores dimensions fastest-moving first, torch the other way round
        shape = tuple(reversed(dims))
        infos.append(TensorInfo(name, shape, ggml_type, offset, _byte_count(ggml_type, prod(shape), gguf_path)))

    # the data section starts at the next alignment boundary after the tensor table
    return architecture, tuple(infos), (pos + alignment - 1) // alignment * alignment


def _byte_count(ggml_type: int, elements: int, gguf_path: str) -> int:
    """How many bytes `elements` of this type occupy in the file."""
    if ggml_type in _FLOAT_WIDTH:
        return elements * _FLOAT_WIDTH[ggml_type]
    if ggml_type not in GGML_BLOCK:
        supported = ", ".join(f"{name} ({type_id})" for type_id, name in sorted(GGML_NAME.items()))
        raise ValueError(
            f"{gguf_path} holds tensors of ggml type {ggml_type}, which is not supported yet. "
            f"Supported quantized types: {supported}."
        )
    block_elements, block_bytes = GGML_BLOCK[ggml_type]
    return elements // block_elements * block_bytes


def read_gguf_architecture(gguf_path: str) -> str:
    """Read only the `general.architecture` field, without touching tensor data."""
    return _header(gguf_path)[0]


def read_gguf_tensor_types(gguf_path: str) -> dict[str, int]:
    """`{gguf_name: ggml_type}` from the header, without reading tensor data."""
    return {info.name: info.ggml_type for info in _header(gguf_path)[1]}


def unused_gguf_tensors(tensor_names, config) -> dict[str, str]:
    """`{gguf_name: reason}` for tensors this model has no parameter for.

    A GGUF file can hold blocks beyond the model's decoder stack — Qwen3.5 ships a
    multi-token-prediction block, for instance. Nothing has to be done about them at load time: they
    are never renamed onto a parameter, so they are never read. This declares *which* ones those are,
    so that a genuinely missing renaming — indistinguishable from the outside — is still an error.
    """
    n_layers = config.get_text_config().num_hidden_layers
    unused = {}
    for name in tensor_names:
        match = re.match(r"^blk\.(\d+)\.", name)
        if match is not None and int(match.group(1)) >= n_layers:
            unused[name] = (
                f"block {match.group(1)} is beyond the model's {n_layers} layers (e.g. a multi-token-prediction head)"
            )
    return unused


class LazyGgufTensor:
    """One tensor of the file, read only when the loading pipeline asks for it.

    `from_pretrained` materializes a state dict one parameter at a time, from a thread pool, and reads
    each entry as `tensor[...]` — the same interface a safetensors slice offers. Deferring to that
    point spreads the copy and the dequantization over the workers and keeps a handful of tensors in
    host memory instead of the whole file.
    """

    def __init__(self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...], packed: bool, dtype=None):
        self._data = data  # a read-only mmap view, untouched until materialized
        self._ggml_type = ggml_type
        self._shape = shape
        self._packed = packed
        self._dtype = dtype

    def is_floating_point(self) -> bool:
        # asked by the pipeline so it does not cast a checkpoint dtype it should leave alone; packed
        # blocks are bytes, anything else materializes as a float tensor
        return not self._packed

    def __getitem__(self, _) -> torch.Tensor:
        # The mmap is read-only: copy before wrapping, or a later `.to(device, dtype)` may treat that
        # storage as scratch.
        raw = torch.from_numpy(np.copy(self._data))
        if self._packed:
            return raw.reshape(self._shape)
        if self._ggml_type in _TORCH_DTYPE:
            dense = raw.view(_TORCH_DTYPE[self._ggml_type])
        else:
            dense = dequantize(raw, self._ggml_type)
        if self._dtype is not None and self._dtype.is_floating_point:
            # Here rather than downstream: dequantization produces fp32, so casting on the spot keeps
            # the transient down, and a GGUF mixes dtypes — norms are stored fp32 next to bf16 weights
            # — where the model wants one dtype throughout.
            dense = dense.to(self._dtype)
        return dense.reshape(self._shape)


def load_gguf_state_dict(gguf_path: str, dtype=None, keep_packed=()) -> dict[str, LazyGgufTensor]:
    """`{gguf_name: LazyGgufTensor}` — the file's tensors, none of them read yet.

    Names in `keep_packed` keep their raw `(rows, bytes_per_row)` uint8 blocks instead of being
    dequantized — that is how a `GgufLinear` gets its weight.

    `dtype` is the dtype every float tensor is cast to, whatever the file stored it as.
    """
    _, infos, data_start = _header(gguf_path)
    blob = _mapped(gguf_path)

    state_dict = {}
    for info in infos:
        shape = info.shape
        packed = info.name in keep_packed
        if packed:
            block_elements, block_bytes = GGML_BLOCK[info.ggml_type]
            shape = (shape[0], shape[1] // block_elements * block_bytes)
        start = data_start + info.offset
        state_dict[info.name] = LazyGgufTensor(blob[start : start + info.nbytes], info.ggml_type, shape, packed, dtype)

    return state_dict
