# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Core helpers for loading model checkpoints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch

from .quantizers.quantizers_utils import get_module_from_name


"""
For mixtral, the fp8 quantizer should add the "quantization" op.

Quantizer says wether we need all weights or not.

TP probably does not need?


model.layers.0.block_sparse_moe.experts.1.w1.input_scale	[]
model.layers.0.block_sparse_moe.experts.1.w1.weight	[14 336, 4 096]
model.layers.0.block_sparse_moe.experts.1.w1.weight_scale	[]
model.layers.0.block_sparse_moe.experts.1.w2.input_scale	[]
model.layers.0.block_sparse_moe.experts.1.w2.weight	[4 096, 14 336]
model.layers.0.block_sparse_moe.experts.1.w2.weight_scale	[]
model.layers.0.block_sparse_moe.experts.1.w3.input_scale	[]
model.layers.0.block_sparse_moe.experts.1.w3.weight	[14 336, 4 096]
model.layers.0.block_sparse_moe.experts.1.w3.weight_scale	[]
"""


class ConversionOps:
    """
    Base class with a reusable buffer to avoid repeated allocations.
    Subclasses implement `convert(collected_tensors) -> torch.Tensor` and
    write results into a view of `self._buffer`.
    """

    target_tensor_shape: torch.Tensor
    can_be_quantized: bool = True
    can_be_distributed: bool = False

    # Lazily created on first use; no __init__ needed.
    _buffer: Optional[torch.Tensor] = None

    def _ensure_buffer(
        self, required_shape: torch.Size, *, dtype: torch.dtype, device: torch.device, growth_factor: float = 1.5
    ) -> torch.Tensor:
        """
        Ensure we have a buffer with enough capacity (and correct dtype/device).
        Returns a *view* of the buffer shaped as `required_shape` without new allocation.
        """
        required_elems = int(torch.tensor(required_shape).prod().item()) if len(required_shape) else 1

        need_new = (
            self._buffer is None
            or self._buffer.dtype != dtype
            or self._buffer.device != device
            or self._buffer.numel() < required_elems
        )

        if need_new:
            # grow capacity to reduce future reallocations
            capacity = max(required_elems, int(required_elems * growth_factor))
            self._buffer = torch.empty(capacity, dtype=dtype, device=device)

        # return a view with the requested shape using only the needed slice
        return self._buffer[:required_elems].view(required_shape)

    def clear_cache(self):
        """Free the cached buffer (optional)."""
        self._buffer = None

    def convert(self, collected_tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Fuse(ConversionOps):
    """
    Concatenate along `dim` without allocating a fresh output each call:
    copies into a preallocated buffer slice-by-slice.
    """

    dim: int = 0  # adjust if you want a different default

    def convert(self, collected_tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        tensors = tuple(collected_tensors)
        if not tensors:
            # Return a zero-size view on an empty buffer on CPU by default
            self._buffer = None
            return torch.empty(0)

        # Basic checks & canonical attrs
        first = tensors[0]
        dtype, device = first.dtype, first.device
        dim = self.dim

        # Validate shapes/dtypes/devices
        base_shape = list(first.shape)
        for t in tensors:
            if t.dtype != dtype or t.device != device:
                raise TypeError("All tensors must share dtype and device for Fuse.")
            if len(t.shape) != len(base_shape):
                raise ValueError("All tensors must have the same rank for Fuse.")
            for d, (a, b) in enumerate(zip(base_shape, t.shape)):
                if d == dim:
                    continue
                if a != b:
                    raise ValueError(f"Non-concat dims must match; got {a} vs {b} at dim {d}.")

        # Compute fused shape
        total_along_dim = sum(t.shape[dim] for t in tensors)
        out_shape = list(base_shape)
        out_shape[dim] = total_along_dim
        out_shape = torch.Size(out_shape)

        with torch.no_grad():
            out = self._ensure_buffer(out_shape, dtype=dtype, device=device)

            # Copy into preallocated buffer without creating a new result tensor
            # We slice along `dim` and copy each piece.
            idx = 0
            for t in tensors:
                slc = [slice(None)] * t.ndim
                slc[dim] = slice(idx, idx + t.shape[dim])
                out[tuple(slc)].copy_(t)
                idx += t.shape[dim]

        return out


class MergeModuleList(ConversionOps):
    """
    Stack tensors along a new leading dimension without allocating a new tensor:
    writes each tensor into a preallocated [N, ...] buffer.
    """

    stack_dim: int = 0  # new dimension index in the *output*

    def convert(self, collected_tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        tensors = tuple(collected_tensors)
        if not tensors:
            self._buffer = None
            return torch.empty(0)

        first = tensors[0]
        dtype, device = first.dtype, first.device
        base_shape = first.shape

        # Validate consistency
        for t in tensors:
            if t.dtype != dtype or t.device != device:
                raise TypeError("All tensors must share dtype and device for MergeModuleList.")
            if t.shape != base_shape:
                raise ValueError("All tensors must have identical shapes to stack.")

        N = len(tensors)
        # Normalize stack_dim (allow negative)
        stack_dim = self.stack_dim % (first.ndim + 1)

        # Output shape: insert N at stack_dim
        out_shape = list(base_shape)
        out_shape.insert(stack_dim, N)
        out_shape = torch.Size(out_shape)

        with torch.no_grad():
            out = self._ensure_buffer(out_shape, dtype=dtype, device=device)

            # Write each tensor into the appropriate slice
            for i, t in enumerate(tensors):
                slc = [slice(None)] * out.ndim
                slc[stack_dim] = i
                out[tuple(slc)].copy_(t)

        return out

class Fp8Quantize(ConversionOps):
    def convert(self, collected_tensors):
        from .quantizers.quantizers_finegrained_fp8 import FineGrainedFP8HfQuantizer
        return FineGrainedFP8HfQuantizer.create_quantized_param(collected_tensors)


class Slice(ConversionOps):
    # TODO: implement slicing for tp
    def convert(self, inputs):
        return inputs

class ConversionType(Enum):
    FUSE = Fuse()
    MERGE_MODULE_LIST = MergeModuleList()
    FP8_QUANTIZE = Fp8Quantize()
    SLICE = Slice()
    def __call__(self, *args, **kwargs):
        # Call enum member as a constructor: ConversionType.FUSE() -> Fuse()
        return self.value(*args, **kwargs) @ dataclass(frozen=True)


globals().update({member.name: member for member in ConversionType})


class WeightConversion:
    """

    Specification for applying renaming and other operations.

    Most probably take the tp_plan here, the quantization_config, and call all the different ops 
    """

    new_key_name: str
    operations: Optional[list[ConversionType]]  # if TP or quantization, some ops like "slicing" will be added?S

    def __init__(self, new_key_name, operations: Optional[Union[ConversionType, list[ConversionType]]]):
        self.new_key_name
        self.operations = list(operations) if not isinstance(operations, list) else operations

    # Ex rank1 for w1,w3 -> gate_up_proj:
    # 1. read the weights
    # 2. rename
    # 3. MergeModuleList, but dim=0, and there is tp_plan on gate_up_proj -> slice to only experts of this rank
    # 4. cat(cat(gate_4, gate_5, gate_6, gate_7), cat(up_4, up_5, up_6, up_7))
    # 5. quantize? -> A new ConversionType op

    # We want the quantizers to have:
    # -


__all__ = ["WeightConversion", "ConversionType"]
