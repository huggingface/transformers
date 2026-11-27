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
"""Reading a GGUF file: architecture, tensor types, and tensors keyed by their GGUF names."""

import struct
from collections.abc import Container
from math import prod
from os import PathLike
from typing import NamedTuple

import numpy as np
import torch

from .dequant import GGML_BLOCK, GGML_NAME


# the ggml type ids a GGUF holds values rather than blocks under, and what they are
_GGML_F32, _GGML_F16, _GGML_BF16 = 0, 1, 30
_TORCH_DTYPE = {_GGML_F32: torch.float32, _GGML_F16: torch.float16, _GGML_BF16: torch.bfloat16}

_GGUF_VERSIONS = (2, 3)  # v1 counted tensors in 32 bits; no file in the wild still uses it

# the fixed-size metadata value types: u8/i8/bool, u16/i16, u32/i32/f32, u64/i64/f64. The two that are
# not fixed-size are 8 (string) and 9 (array), handled in `_read_value`.
_KV_FORMAT = {0: "<B", 1: "<b", 7: "<?", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
_KV_WIDTH = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}


class TensorInfo(NamedTuple):
    """Where one tensor lives in the file, and how it is stored."""

    name: str
    shape: tuple[int, ...]  # in torch order
    ggml_type: int
    offset: int  # from the start of the data section
    nbytes: int


class GgufHeader(NamedTuple):
    """A GGUF file's metadata: everything the loading path needs except the tensor data."""

    path: str
    architecture: str
    tensors: tuple[TensorInfo, ...]
    data_start: int  # where the tensor data begins, after the aligned metadata
    shards: tuple["GgufHeader", ...] = ()

    @property
    def ggml_types(self) -> dict[str, int]:
        """`{gguf_name: ggml_type}`."""
        return {info.name: info.ggml_type for info in self.tensors}

    @property
    def has_quantized_weights(self) -> bool:
        """Whether any tensor is stored as GGUF blocks rather than as plain floats."""
        return any(ggml_type in GGML_BLOCK for ggml_type in self.ggml_types.values())

    @property
    def dtype(self) -> "torch.dtype | None":
        """The float type this file was written in, or `None` if it holds quantized blocks."""
        types = set(self.ggml_types.values())
        if not types <= set(_TORCH_DTYPE):  # blocks, not values, somewhere in the file
            return None
        # a file's F32 tensors sit alongside its half ones — the norms — so the half type is the model's
        if _GGML_BF16 in types:
            return torch.bfloat16
        if _GGML_F16 in types:
            return torch.float16
        return torch.float32

    @classmethod
    def from_file(cls, gguf_path: str) -> "GgufHeader":
        """Parse a file's metadata and tensor table, without touching its tensor data."""
        return cls.from_files(gguf_path)

    @classmethod
    def from_files(cls, gguf_paths: str | PathLike | list[str]) -> "GgufHeader":
        """Read one file, or a complete checkpoint with per-file tensor offsets."""
        metadata, files = _read_gguf_files(gguf_paths)
        shards = []
        for path, entries, data_start in files:
            infos = tuple(
                TensorInfo(name, shape, ggml_type, offset, _byte_count(ggml_type, prod(shape), path))
                for name, shape, ggml_type, offset in entries
            )
            shards.append(cls(path, metadata["general.architecture"], infos, data_start))
        first = shards[0]
        return first._replace(
            tensors=tuple(info for shard in shards for info in shard.tensors),
            shards=tuple(shards) if len(shards) > 1 else (),
        )


def read_gguf_metadata(
    gguf_path: str | PathLike | list[str], string_arrays: "Container[str]" = ()
) -> tuple[dict, tuple[str, ...]]:
    """Read one file's metadata, or the first metadata and complete tensor inventory of a list of shards."""
    metadata, files = _read_gguf_files(gguf_path, string_arrays)
    return metadata, tuple(name for _, entries, _ in files for name, *_ in entries)


def _read_gguf_files(
    gguf_paths: str | PathLike | list[str], string_arrays: "Container[str]" = ()
) -> tuple[dict, list[tuple[str, list[tuple], int]]]:
    complete = not isinstance(gguf_paths, (str, PathLike))
    if not complete:
        gguf_paths = [gguf_paths]
    if not gguf_paths:
        raise ValueError("A GGUF checkpoint must contain at least one file.")

    files = []
    first_metadata = {}
    tensor_names = set()
    total_tensors = None
    for index, path in enumerate(gguf_paths):
        blob = _mapped(path)
        metadata, tensor_count, pos = _read_metadata(blob, path, string_arrays)
        architecture = metadata.get("general.architecture", first_metadata.get("general.architecture"))
        if architecture is None:
            raise ValueError(f"{path} has no `general.architecture` in its metadata.")
        if index == 0:
            first_metadata = metadata
        elif architecture != first_metadata["general.architecture"]:
            raise ValueError(f"{path} has a conflicting `general.architecture` for this GGUF checkpoint.")

        # llama.cpp's gguf-split --merge retains split metadata with split.count set to zero.
        is_merged = len(gguf_paths) == 1 and metadata.get("split.count") == 0
        expected_split = {
            "split.no": index,
            "split.count": len(gguf_paths),
            "split.tensors.count": first_metadata.get("split.tensors.count"),
        }
        if complete and not is_merged and (len(gguf_paths) > 1 or any(key in metadata for key in expected_split)):
            for key, expected in expected_split.items():
                if key not in metadata:
                    raise ValueError(f"{path} is missing required GGUF shard metadata `{key}`.")
                if metadata[key] != expected:
                    raise ValueError(f"{path} has `{key}` {metadata[key]}, but expected {expected}.")
            total_tensors = metadata["split.tensors.count"]

        entries, pos = _read_tensor_table(blob, tensor_count, pos)
        alignment = metadata.get("general.alignment", 32)
        data_start = (pos + alignment - 1) // alignment * alignment
        for name, *_ in entries:
            if complete and name in tensor_names:
                raise ValueError(f"Duplicate GGUF tensor name `{name}` in {path}.")
            tensor_names.add(name)
        files.append((path, entries, data_start))

    if total_tensors is not None and len(tensor_names) != total_tensors:
        raise ValueError(
            f"The GGUF checkpoint contains {len(tensor_names)} tensors, but `split.tensors.count` is {total_tensors}."
        )
    return first_metadata, files


def _read_tensor_table(blob: np.ndarray, tensor_count: int, pos: int) -> tuple[list[tuple], int]:
    """`([(name, shape, ggml_type, offset), ...], offset just past the table)`."""
    entries = []
    for _ in range(tensor_count):
        name, pos = _read_string(blob, pos)
        (dim_count,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        dims = struct.unpack_from(f"<{dim_count}Q", blob, pos)
        pos += 8 * dim_count
        ggml_type, offset = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        # ggml stores dimensions fastest-moving first, torch the other way round
        entries.append((name, tuple(reversed(dims)), ggml_type, offset))
    return entries, pos


def _read_metadata(blob: np.ndarray, gguf_path: str, string_arrays: "Container[str]" = ()) -> tuple[dict, int, int]:
    """`(metadata, tensor_count, offset of the tensor table)`."""
    if bytes(blob[:4]) != b"GGUF":
        raise ValueError(f"{gguf_path} does not start with the GGUF magic bytes, so it is not a GGUF file.")
    version, tensor_count, metadata_count = struct.unpack_from("<IQQ", blob, 4)
    if version not in _GGUF_VERSIONS:
        raise ValueError(
            f"{gguf_path} is GGUF v{version}; this reader handles v{' and v'.join(map(str, _GGUF_VERSIONS))}."
        )

    pos = 24
    metadata = {}
    for _ in range(metadata_count):
        key, pos = _read_string(blob, pos)
        (value_type,) = struct.unpack_from("<I", blob, pos)
        metadata[key], pos = _read_value(blob, pos + 4, value_type, key in string_arrays)
    return metadata, tensor_count, pos


def _read_value(blob: np.ndarray, pos: int, value_type: int, keep_strings: bool = False):
    """One metadata value, and the offset just past it."""
    if value_type in _KV_WIDTH:
        (value,) = struct.unpack_from(_KV_FORMAT[value_type], blob, pos)
        return value, pos + _KV_WIDTH[value_type]
    if value_type == 8:  # string
        return _read_string(blob, pos)
    if value_type == 9:  # array
        element_type, count = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        if element_type in _KV_WIDTH:  # small, and a config can want them: the mrope sections
            values = struct.unpack_from(f"<{count}{_KV_FORMAT[element_type][1]}", blob, pos)
            return list(values), pos + count * _KV_WIDTH[element_type]
        if element_type != 8:
            raise ValueError(f"GGUF metadata holds an array of type {element_type}, which this reader cannot read.")
        if keep_strings:  # asked for: a vocabulary or a merge table
            values = []
            for _ in range(count):
                value, pos = _read_string(blob, pos)
                values.append(value)
            return values, pos
        for _ in range(count):  # variable-length, so there is nothing to do but walk it
            (length,) = struct.unpack_from("<Q", blob, pos)
            pos += 8 + length
        return count, pos
    raise ValueError(f"GGUF metadata holds a value of type {value_type}, which this reader cannot read.")


def _mapped(gguf_path: str) -> np.ndarray:
    """The file as a `uint8` memory map. Cheap: pages are only read when a tensor is materialized."""
    return np.memmap(gguf_path, mode="r", dtype=np.uint8)


def _read_string(blob: np.ndarray, pos: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", blob, pos)
    pos += 8
    return bytes(blob[pos : pos + length]).decode("utf-8"), pos + length


def _byte_count(ggml_type: int, elements: int, gguf_path: str) -> int:
    """How many bytes `elements` of this type occupy in the file."""
    if ggml_type in _TORCH_DTYPE:
        return elements * _TORCH_DTYPE[ggml_type].itemsize
    if ggml_type not in GGML_BLOCK:
        supported = ", ".join(f"{name} ({type_id})" for type_id, name in sorted(GGML_NAME.items()))
        raise ValueError(
            f"{gguf_path} holds tensors of ggml type {ggml_type}, which is not supported yet. "
            f"Supported quantized types: {supported}."
        )
    block_elements, block_bytes = GGML_BLOCK[ggml_type]
    return elements // block_elements * block_bytes


class LazyGgufTensor:
    """One tensor of the file, read only when the loading pipeline asks for it."""

    def __init__(self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...]):
        self.data = data  # a read-only mmap view, untouched until materialized
        self.ggml_type = ggml_type
        self.shape = shape

    def __getitem__(self, _) -> torch.Tensor:
        raw = torch.from_numpy(np.copy(self.data))
        if self.ggml_type not in _TORCH_DTYPE:
            return raw.reshape(self.shape)
        # The file is mapped as bytes, since numpy has no bfloat16, so the values are reinterpreted here.
        # Left in the type the file wrote: the transforms need it -- a norm is stored as `w + 1`, and
        # rounding before the subtraction spends the precision available near 1.0 on a much smaller
        # weight. `Cast` is the last op of every chain, so this lands in the model's dtype anyway.
        return raw.view(_TORCH_DTYPE[self.ggml_type]).reshape(self.shape)


def load_gguf_state_dict(header: GgufHeader) -> dict[str, LazyGgufTensor]:
    """`{gguf_name: LazyGgufTensor}` — the file's tensors, none of them read yet."""
    if header.shards:
        state_dict = {}
        for shard in header.shards:
            state_dict.update(load_gguf_state_dict(shard))
        return state_dict
    blob = _mapped(header.path)

    state_dict = {}
    for info in header.tensors:
        shape = info.shape
        if info.ggml_type not in _TORCH_DTYPE:
            block_elements, block_bytes = GGML_BLOCK[info.ggml_type]
            shape = (*shape[:-1], shape[-1] // block_elements * block_bytes)
        start = header.data_start + info.offset
        state_dict[info.name] = LazyGgufTensor(blob[start : start + info.nbytes], info.ggml_type, shape)

    return state_dict
