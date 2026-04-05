# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
GPU Direct Storage (GDS) utilities for safetensors loading.
"""

import json
import os
import struct
from typing import Any

import torch
from safetensors import safe_open

from .import_utils import is_env_variable_true, is_torch_greater_or_equal


# Tensors below this use safe_open instead of cuFile (per-call overhead).
_GDS_MIN_BYTES = 1 * 1024 * 1024

_gds_available: bool | None = None


def is_gds_available() -> bool:
    """Check if ``torch.cuda.gds.GdsFile`` is usable. Requires PyTorch >= 2.10, CUDA >= 12.6."""
    global _gds_available
    if _gds_available is not None:
        return _gds_available
    _gds_available = False
    if not is_torch_greater_or_equal("2.10"):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        if hasattr(torch._C, "_gds_is_available"):
            _gds_available = torch._C._gds_is_available()
        else:
            from torch.cuda.gds import GdsFile  # noqa: F401

            _gds_available = True
    except (ImportError, AttributeError, RuntimeError):
        _gds_available = False
    return _gds_available


def should_use_gds() -> bool:
    """``True`` when GDS is available and opted-in via ``HF_ENABLE_GDS=1``."""
    return is_env_variable_true("HF_ENABLE_GDS") and is_gds_available()


# Resolve once at class init, not per-tensor
_str_to_torch_dtype: dict[str, torch.dtype] | None = None


def _get_dtype_map() -> dict[str, torch.dtype]:
    global _str_to_torch_dtype
    if _str_to_torch_dtype is None:
        from ..modeling_utils import str_to_torch_dtype

        _str_to_torch_dtype = str_to_torch_dtype
    return _str_to_torch_dtype


class GdsSafetensorsFile:
    """GDS-backed safetensors reader — drop-in replacement for ``safe_open()``."""

    def __init__(self, filename: str):
        self.filename = str(filename)
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        data_offset = 8 + header_size

        self._tensor_meta: dict[str, dict[str, Any]] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            self._tensor_meta[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "file_offset": data_offset + start,
                "nbytes": end - start,
            }

        from torch.cuda.gds import GdsFile

        self._gds_file = GdsFile(self.filename, os.O_RDONLY)
        self._safe_fp = safe_open(self.filename, framework="pt", device="cpu")
        self._dtype_map = _get_dtype_map()

    def keys(self):
        return self._tensor_meta.keys()

    def get_slice(self, name: str) -> "GdsSlice":
        return GdsSlice(self, name)

    def get_tensor(self, name: str, device: torch.device | None = None) -> torch.Tensor:
        meta = self._tensor_meta[name]
        if device is not None and device.type == "cuda" and meta["nbytes"] >= _GDS_MIN_BYTES:
            tensor = torch.empty(meta["shape"], dtype=self._dtype_map[meta["dtype"]], device=device)
            self._gds_file.load_storage(tensor.untyped_storage(), meta["file_offset"])
            return tensor
        return self._safe_fp.get_tensor(name)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._close()

    def __del__(self):
        self._close()

    def _close(self):
        gds = self.__dict__.pop("_gds_file", None)
        safe = self.__dict__.pop("_safe_fp", None)
        del gds
        if safe is not None:
            safe.__exit__(None, None, None)


class GdsSlice:
    """Lazy tensor reference compatible with ``PySafeSlice``."""

    __slots__ = ("_gds_file", "_name", "_dtype", "_shape", "_target_device")

    def __init__(self, gds_file: GdsSafetensorsFile, name: str):
        meta = gds_file._tensor_meta[name]
        self._gds_file = gds_file
        self._name = name
        self._dtype = meta["dtype"]
        self._shape = meta["shape"]
        self._target_device: torch.device | None = None

    def _set_target_device(self, device) -> None:
        self._target_device = torch.device(device) if device is not None else None

    def get_dtype(self) -> str:
        return self._dtype

    def get_shape(self) -> list[int]:
        return self._shape

    def __getitem__(self, slices):
        tensor = self._gds_file.get_tensor(self._name, self._target_device)
        if slices is Ellipsis or slices == (Ellipsis,):
            return tensor
        return tensor[slices]
