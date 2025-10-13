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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch

from .quantizers.quantizers_utils import get_module_from_name


@dataclass(frozen=True)
class WeightConversion:
    """Specification for applying a post-rename weight transformation."""

    new_key: str
    function: str
    dim: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, resolved_key: str) -> "ResolvedWeightConversion":
        return ResolvedWeightConversion(
            target_key=resolved_key,
            function=self.function,
            dim=self.dim,
            kwargs=dict(self.kwargs),
        )


@dataclass
class ResolvedWeightConversion:
    target_key: str
    function: str
    dim: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightConversionPlan:
    conversion: ResolvedWeightConversion
    source_keys: Tuple[str, ...]

    def __post_init__(self):
        self.source_index = {key: idx for idx, key in enumerate(self.source_keys)}

    @property
    def num_parts(self) -> int:
        return len(self.source_keys)


class ConversionAccumulator:
    """Runtime helper that assembles tensors according to a conversion plan."""

    def __init__(self, plan: WeightConversionPlan, model: Any):
        self.plan = plan
        module, tensor_name = get_module_from_name(model, plan.conversion.target_key)
        self._target_template = getattr(module, tensor_name)
        self._buffer: Optional[torch.Tensor] = None
        self._filled = set()
        self._parts_seen = 0

    @property
    def is_complete(self) -> bool:
        return self._parts_seen >= self.plan.num_parts

    def _allocate_buffer(self, reference: torch.Tensor) -> torch.Tensor:
        if self._buffer is not None:
            return self._buffer

        target_shape = tuple(self._target_template.shape)
        target_dtype = getattr(self._target_template, "dtype", reference.dtype)
        target_device = reference.device
        if target_dtype is None:
            target_dtype = reference.dtype

        self._buffer = torch.empty(target_shape, dtype=target_dtype, device=target_device)
        return self._buffer

    def add(self, source_index: int, tensor: torch.Tensor):
        if source_index in self._filled:
            raise ValueError(
                f"Weight conversion for {self.plan.conversion.target_key} received duplicate source index {source_index}."
            )

        buffer = self._allocate_buffer(tensor)
        conversion = self.plan.conversion
        if conversion.function == "merge_module_list":
            dim = 0 if conversion.dim is None else conversion.dim
            indexer: List[slice] = [slice(None)] * buffer.ndim
            indexer[dim] = source_index
            buffer[tuple(indexer)].copy_(tensor.to(buffer.dtype))
        else:
            raise NotImplementedError(f"Unsupported weight conversion function: {conversion.function}")

        self._filled.add(source_index)
        self._parts_seen += 1

    def materialize(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError(
                f"Attempted to materialize conversion result for {self.plan.conversion.target_key} before any data was added."
            )
        return self._buffer


def build_weight_conversion_plans(
    conversion_specs: Dict[str, WeightConversion], conversion_sources: Dict[str, Iterable[str]]
) -> Dict[str, WeightConversionPlan]:
    """Instantiate `WeightConversionPlan` objects for each converted key."""

    plans: Dict[str, WeightConversionPlan] = {}
    for target, source_list in conversion_sources.items():
        plans[target] = WeightConversionPlan(
            conversion=conversion_specs[target].instantiate(target),
            source_keys=tuple(source_list),
        )
    return plans


def collate_converted_state_dict(
    state_dict: Dict[str, torch.Tensor], key_renaming_mapping: Dict[str, str]
) -> Dict[str, List[Tuple[str, torch.Tensor]]]:
    """Group tensors that map to the same resolved key.

    The returned mapping keeps track of the original serialized key for each tensor so safetensors slices can be
    retrieved lazily when needed.
    """

    converted_state_dict: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
    for original_key, value in state_dict.items():
        target_key = key_renaming_mapping.get(original_key)
        if target_key is None:
            continue
        converted_state_dict[target_key].append((original_key, value))
    return dict(converted_state_dict)


def materialize_param_from_contributions(
    model: Any,
    param_name: str,
    contributions: List[Tuple[str, torch.Tensor]],
    plan: Optional[WeightConversionPlan],
    conversion_runtime: Dict[str, ConversionAccumulator],
    file_pointer: Optional[Any],
    tensor_device: Union[str, torch.device],
) -> Optional[torch.Tensor]:
    """Return a tensor ready to load into the model, or `None` if more shards are required."""

    if not contributions:
        return None

    if plan is None:
        original_key, tensor_value = contributions[0]
        if file_pointer is not None:
            return file_pointer.get_slice(original_key)
        return tensor_value.to(tensor_device)

    accumulator = conversion_runtime.get(param_name)
    if accumulator is None:
        accumulator = ConversionAccumulator(plan, model)
        conversion_runtime[param_name] = accumulator

    for original_key, tensor_value in contributions:
        if file_pointer is not None:
            tensor_slice = file_pointer.get_slice(original_key)
        else:
            tensor_slice = tensor_value.to(tensor_device)
        source_index = plan.source_index[original_key]
        accumulator.add(source_index, tensor_slice)

    if not accumulator.is_complete:
        return None

    conversion_runtime.pop(param_name, None)
    return accumulator.materialize()


__all__ = [
    "WeightConversion",
    "ResolvedWeightConversion",
    "WeightConversionPlan",
    "ConversionAccumulator",
    "build_weight_conversion_plans",
    "collate_converted_state_dict",
    "materialize_param_from_contributions",
]
