# coding=utf-8
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
"""Core helpers for loading model checkpoints."""

from __future__ import annotations

import itertools
import os
import re
from abc import abstractmethod
from collections import defaultdict
from collections.abc import MutableMapping, MutableSet, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from .integrations.tensor_parallel import ALL_PARALLEL_STYLES, DTensor, Replicate, TensorParallelLayer
from .utils import is_torch_greater_or_equal, logging


_torch_distributed_available = torch.distributed.is_available()
_is_dtensor_available = _torch_distributed_available and is_torch_greater_or_equal("2.5")
if _is_dtensor_available:
    from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .quantizers import HfQuantizer


logger = logging.get_logger(__name__)

str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


logger = logging.get_logger(__name__)


def compile_glob_rule(source_glob: str, target_glob: str) -> Tuple[re.Pattern, str]:
    """
    Convert a glob-style source + target into a full regex + replacement.

    Rules:
      - '*' in source_glob  →  (.*) capture group
      - '*' in target_glob  →  \\1, \\2, ... backrefs
    """
    pattern = re.escape(source_glob).replace(r"\*", "(.*)")
    regex = re.compile(pattern)

    counter = 0

    def _star_to_backref(_: re.Match) -> str:
        nonlocal counter
        counter += 1
        return rf"\{counter}"

    replacement = re.sub(r"\*", _star_to_backref, target_glob)
    return regex, replacement


def build_glob_alternation(globs: List[str]) -> Tuple[re.Pattern, Dict[str, str]]:
    """
    Build a single alternation regex with one named group per glob.
    """
    if not globs:
        return re.compile(r"(?!x)"), {}
    group_to_glob: Dict[str, str] = {}
    branches: List[str] = []

    for i, glob in enumerate(globs):
        group_name = f"g{i}"
        group_to_glob[group_name] = glob
        body = re.escape(glob).replace(r"\*", ".*")
        branches.append(f"(?P<{group_name}>.*{body}.*)")

    alternation = re.compile("|".join(branches))
    return alternation, group_to_glob


def sub_key(
    key: str,
    alternation: re.Pattern,
    group_to_glob: Dict[str, str],
    compiled_rules: Dict[str, Tuple[re.Pattern, str]],
) -> str:
    """
    Apply glob-based rewrite rules to a single key.
    """
    match = alternation.match(key)
    if not match:
        return key

    matched_globs = [
        group_to_glob[group_name]
        for group_name, value in match.groupdict().items()
        if value is not None
    ]

    result: str | None = None

    for source_glob in matched_globs:
        regex, replacement = compiled_rules[source_glob]

        if not regex.search(key):
            continue

        candidate = regex.sub(replacement, key, count=1)

        if result is None:
            result = candidate
        elif candidate != result:
            raise ValueError(
                f"Contradictory rules for key {key!r}: {result!r} vs {candidate!r}"
            )

    return result if result is not None else key


def match_glob(key: str, alt: re.Pattern, name_map: dict[str, str]) -> Optional[str]:
    """
    Match the key against the alternation; return the original glob string that matched.
    """
    m = alt.match(key)
    if not m or m.lastgroup is None:
        return None
    return name_map.get(m.lastgroup)


class ConversionOps:
    """Base class for weight conversion operations."""

    # The inverse operation class, will be used when saving the checkpoint
    reverse_op: type[ConversionOps]

    @abstractmethod
    def convert(
        self,
        value: dict[str, Any],
        *,
        source_keys: list[str],
        target_keys: list[str],
        concrete_target_keys: list[str],
        config,
    ) -> dict[str, list[torch.Tensor]]:
        raise NotImplementedError


class Chunk(ConversionOps):
    """Split a tensor along ``dim`` into equally sized chunks or using explicit ``sizes``."""

    reverse_op: type[ConversionOps]

    def __init__(self, dim: int = 0, chunks: Optional[int] = None, sizes: Optional[Sequence[int]] = None):
        if chunks is None and sizes is None:
            raise ValueError("`chunks` or `sizes` must be provided for Chunk operations.")
        if chunks is not None and chunks <= 0:
            raise ValueError("`chunks` must be a strictly positive integer.")
        self.dim = dim
        self.chunks = chunks
        self.sizes = list(sizes) if sizes is not None else None
        self.reverse_op = Concatenate

    def convert(
        self,
        value: dict[str, list[torch.Tensor]],
        *,
        source_keys: list[str],
        target_keys: list[str],
        concrete_target_keys: list[str],
        config,
    ) -> dict[str, list[torch.Tensor]]:
        if len(value) != 1:
            raise ValueError("Chunk operation expects a single source tensor.")
        tensors = next(iter(value.values()))
        if len(tensors) != 1:
            raise ValueError("Chunk operation received unexpected multiple tensors for a single source.")
        tensor = tensors[0]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Chunk operation requires torch.Tensor inputs.")
        if self.sizes is not None:
            chunks = torch.split(tensor, self.sizes, dim=self.dim)
        else:
            chunks = torch.chunk(tensor, self.chunks, dim=self.dim)
        if len(chunks) != len(concrete_target_keys):
            raise ValueError("Number of produced chunks does not match the expected targets.")
        return {target: [chunk] for target, chunk in zip(concrete_target_keys, chunks)}


class Concatenate(ConversionOps):
    """Concatenate tensors along `dim` using a reusable buffer."""

    reverse_op: type[ConversionOps]

    def __init__(self, dim: int = 0):
        self.dim = dim
        self.reverse_op = Chunk

    @torch.no_grad
    def convert(
        self,
        value: dict[str, list[torch.Tensor]],
        *,
        source_keys: list[str],
        target_keys: list[str],
        concrete_target_keys: list[str],
        config,
    ) -> dict[str, list[torch.Tensor]]:
        if len(concrete_target_keys) != 1:
            raise ValueError("Concatenate expects a single target key.")
        tensors: list[torch.Tensor] = []
        for key in source_keys:
            tensor = value.get(key)
            if tensor is None:
                raise ValueError(f"Missing tensor for source pattern {key}.")
            if len(tensor) != 1:
                raise ValueError(f"Concatenate expected exactly one tensor per source, got {len(tensor)} for {key}.")
            current = tensor[0]
            if not isinstance(current, torch.Tensor):
                raise TypeError("Concatenate can only operate on torch.Tensor instances.")
            tensors.append(current)
        if not tensors:
            raise ValueError("Concatenate requires at least one tensor.")
        return {concrete_target_keys[0]: [torch.cat(tuple(tensors), dim=self.dim)]}


class MergeModulelist(Concatenate):
    """
    Merge a list of tensors into a single tensor along the first dimension.
    We explicitly define this because for EP or TP you want to make sure you know what you are doing!

    """

    def __init__(self, dim: int = 0):
        super().__init__(dim=dim)
        self.reverse_op = SplitModulelist

    @torch.no_grad
    def convert(
        self,
        value: dict[str, list[torch.Tensor]],
        *,
        source_keys: list[str],
        target_keys: list[str],
        concrete_target_keys: list[str],
        config,
    ) -> dict[str, list[torch.Tensor]]:
        merged: dict[str, list[torch.Tensor]] = {}
        rename_to_targets = len(target_keys) == len(source_keys) == len(concrete_target_keys)
        for idx, key in enumerate(source_keys):
            tensors = value.get(key, [])
            if not tensors:
                raise ValueError(f"MergeModulelist requires non-empty tensors for {key}.")
            stacked = torch.stack(tensors, dim=self.dim)
            if rename_to_targets:
                merged[concrete_target_keys[idx]] = [stacked]
            else:
                merged[key] = [stacked]
        return merged


class SplitModulelist(ConversionOps):
    """Inverse of :class:`MergeModulelist` using explicit split sizes per group."""

    def __init__(self, sizes: Sequence[Sequence[int]], dim: int = 0):
        if not isinstance(sizes, Sequence) or not all(isinstance(sub, Sequence) and sub for sub in sizes):
            raise ValueError("`sizes` must be a sequence of non-empty sequences of integers.")
        self.sizes = [list(sub) for sub in sizes]
        self.dim = dim
        self.reverse_op = MergeModulelist

    @torch.no_grad
    def convert(
        self,
        value: dict[str, list[torch.Tensor]],
        *,
        source_keys: list[str],
        target_keys: list[str],
        concrete_target_keys: list[str],
        config,
    ) -> dict[str, list[torch.Tensor]]:
        if len(value) != len(self.sizes):
            raise ValueError("SplitModulelist received an unexpected number of tensors.")
        result: dict[str, list[torch.Tensor]] = {}
        for (key, tensors), split_sizes in zip(value.items(), self.sizes):
            if len(tensors) != 1:
                raise ValueError("SplitModulelist expects exactly one tensor per key.")
            current_tensor = tensors[0]
            if not isinstance(current_tensor, torch.Tensor):
                raise TypeError("SplitModulelist can only split torch.Tensor instances.")
            result[key] = list(torch.split(current_tensor, split_sizes, dim=self.dim))
        return result


class PermuteForRope(ConversionOps):
    """
    Applies the permutation required to convert complex RoPE weights to the split sin/cos format.
    """

    def __init__(self):
        pass

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        dim1, dim2 = tensor.shape
        n_heads = self.config.getattr("num_attention_heads", 1)

        tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        tensor = tensor.transpose(1, 2).reshape(dim1, dim2)
        return tensor

    @torch.no_grad
    def convert(
        self,
        value: dict[str, list[torch.Tensor]],
        *,
        source_keys: list[str],
        target_keys: list[str],
        concrete_target_keys: list[str],
        config,
    ) -> dict[str, list[torch.Tensor]]:
        self.config = config
        output: dict[str, list[torch.Tensor]] = {}
        for key, tensors in value.items():
            if len(tensors) != 1:
                raise ValueError("PermuteForRope expects a single tensor per key.")
            output[key] = [self._apply(tensors[0])]
        return output


@dataclass(slots=True)
class WeightConverter:
    r"""
    A weight convert that acts on a pattern of source keys.
    The keys need to be collected based on the target keys.

    With wild card, glob patterns are matched, so you have to be detailed with what to match. If you match:
    `model.layers.*.experts.*` -> it will act on all of them
    {"model.layers.*.experts.*": []}
    but
    `experts.*.mlp` will be layer specific.
    {"model.layers.1.experts.*": [], }
    - source_keys: str | list[str] (wildcards '*' match digits)
    - target_keys: str | list[str] | None
    - distributed_operation / operations / quantization_operations are ALWAYS lists.

    TODO: for BNB we need to collect model.weight.quant_state_keys
    """

    source_keys: Union[str, list[str]]
    target_keys: Optional[Union[str, list[str]]] = None
    operations: list[ConversionOps] = field(default_factory=list, repr=False)

    distributed_operation: Optional[TensorParallelLayer] = None
    quantization_operation: Optional[ConversionOps] = None
    collected_tensors: dict[str, defaultdict[str, list[Future]]] = field(default_factory=dict, init=False)
    layer_targets: dict[str, list[str]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if not isinstance(self.source_keys, list):
            object.__setattr__(self, "source_keys", [self.source_keys])
        targets_were_none = False
        if not isinstance(self.target_keys, list):
            if self.target_keys is None:
                object.__setattr__(self, "target_keys", list(self.source_keys))
                targets_were_none = True
            else:
                object.__setattr__(self, "target_keys", [self.target_keys])

        if not targets_were_none and bool(len(self.source_keys) - 1) + bool(len(self.target_keys) - 1) >= 2:
            raise ValueError(
                f"source keys={self.source_keys}, target_keys={self.target_keys} but you can only have one to many, one to one or many to one."
            )

        if not self.operations:
            raise ValueError("WeightConverter requires at least one operation.")

    def add_tensor(self, layer_key: str, source_pattern: str, future: Future, resolved_targets: list[str]):
        bucket = self.collected_tensors.setdefault(layer_key, defaultdict(list))
        bucket[source_pattern].append(future)
        self.layer_targets.setdefault(layer_key, resolved_targets)


@dataclass(slots=True)
class WeightRenaming:
    source_key: str
    target_key: str
    operations: list[ConversionOps] = field(default_factory=list, repr=False)
    distributed_operation: Optional[TensorParallelLayer] = None
    quantization_operation: Optional[ConversionOps] = None
    collected_tensors: dict[str, defaultdict[str, list[Future]]] = field(default_factory=dict, init=False)
    layer_targets: dict[str, list[str]] = field(default_factory=dict, init=False)
    source_keys: list[str] = field(init=False)
    target_keys: list[str] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "source_keys", [self.source_key])
        object.__setattr__(self, "target_keys", [self.target_key])

    def add_tensor(self, layer_key: str, source_pattern: str, future: Future, resolved_targets: list[str]):
        bucket = self.collected_tensors.setdefault(layer_key, defaultdict(list))
        bucket[source_pattern].append(future)
        self.layer_targets.setdefault(layer_key, resolved_targets)

    def __post_init__(self):
        self.source_keys = [self.source_key]
        self.target_keys = [self.target_key]


GLOBAL_WORKERS = min(16, (os.cpu_count() or 8) * 2)  # NVMe: 8-16; HDD/NFS: 2-4


def _materialize_copy(tensor, dtype=None):
    tensor = tensor[...]
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def spawn_materialize(thread_pool, tensor, dtype=None) -> Future:
    def _job():
        return _materialize_copy(tensor, dtype)

    return thread_pool.submit(_job)


def spawn_tp_materialize(thread_pool, tensor, sharding_method, tensor_idx, dtype=None) -> Future:
    def _job():
        return sharding_method.shard_tensor(tensor, param_casting_dtype=dtype, tensor_idx=tensor_idx)[0]

    return thread_pool.submit(_job)


def dot_natural_key(s: str):
    parts = s.split(".")
    for i, p in enumerate(parts):
        # whole-segment digits -> int; otherwise leave as str
        if p.isdigit():
            parts[i] = int(p)
    return parts


@contextmanager
def log_to_misc(
    layer_name: str,
    misc: MutableMapping[str, str],
    extras: Any = None,
    op: Union[list[ConversionOps], ConversionOps, None] = None,
):
    # A simple helper to handle errors with contextual messages.
    try:
        yield
    except Exception as e:

        def _format_op_name(curr_op: Union[list[ConversionOps], ConversionOps, None]) -> Optional[str]:
            if curr_op is None:
                return None
            if isinstance(curr_op, (list, tuple, set)):
                names = [o.__class__.__name__ for o in curr_op if o is not None]
                if not names:
                    return None
                return ", ".join(names)
            return curr_op.__class__.__name__

        op_name = _format_op_name(op)
        if isinstance(extras, tuple) and len(extras) == 2:
            values, target_keys = extras
            descriptor = f"{op_name} " if op_name else ""
            misc[layer_name] = (
                f"{e}\nError: {descriptor}on tensors destined for {target_keys}. Ckpt contains: {len(values[0])}"
            )
        elif isinstance(extras, str):
            suffix = f" via {op_name}" if op_name else ""
            misc[layer_name] = f"{e}\nError{suffix} when processing parameter {extras}"
        elif extras is None and op_name:
            misc[layer_name] = f"{op_name}: {e}"
        else:
            misc[layer_name] = f"{extras} |Error: {e}"
        raise SkipLayer()


def set_param_for_module(
    model: PreTrainedModel,
    layer_name: str,
    param_value: torch.Tensor,
    mismatch_keys: MutableSet[tuple[str, torch.Size, torch.Size]],
    missing_keys: MutableSet[str],
    misc: MutableMapping[str, Any],
    unexpected_keys: MutableSet[str],
    distributed_operation: Optional[TensorParallelLayer],
):
    with log_to_misc(layer_name, misc, layer_name):
        module_path, _, param_name = layer_name.rpartition(".")
        module_obj = model.get_submodule(module_path) if module_path else model
        param_value = param_value[0] if isinstance(param_value, list) else param_value[...]
        ref = getattr(module_obj, param_name)
        if ref is None:
            unexpected_keys.add(t)
            continue

        use_dtensor = hasattr(distributed_operation, "use_dtensor") and distributed_operation.use_dtensor
        if not isinstance(param_value, torch.nn.Parameter):
            if distributed_operation is not None:
                param_value = DTensor.from_local(
                    param_value,
                    distributed_operation.device_mesh,
                    getattr(distributed_operation, "shard", Replicate()),
                    run_check=False,
                    shape=ref.size(),
                    stride=ref.stride(),
                )
                if not use_dtensor:
                    # we convert to local
                    param_value = param_value.to_local()
            if param_name not in module_obj._buffers:
                param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())

        # Remove from missing keys (it's either mismatched, or all good)
        missing_keys.discard(layer_name)
        if ref is not None and ref.shape != param_value.shape:
            mismatch_keys.add((layer_name, param_value.shape, ref.shape))
            module_obj.param_name._is_hf_initialized = False  # Needs to be initialized
        else:
            param_value._is_hf_initialized = True  # super important otherwise _init_weight re-initi if bias is missing
            setattr(module_obj, param_name, param_value)


class SkipLayer(Exception):
    """Control-flow sentinel: abort processing of the current layer only."""

    pass


def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    weight_mapping: list[WeightConverter | WeightRenaming] | None,
    tp_plan: dict[str, str] | None,
    quantizer: HfQuantizer | None,
    dtype: torch.dtype | None = None,
    device_map: dict | None = None,
    dtype_plan: dict | None = None,
    device_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
):
    """
    Convert a state dict according to a weight mapping (one WeightConverter per glob pattern),
    collecting tensors per *layer instance* (the concrete indices captured from '*').
    """

    prefix = model.base_model_prefix
    tp_plan = tp_plan or {}  # {glob_pattern: plan_obj_or_key}
    device_map = device_map or {}  # {exact_target_key: device}
    dtype_plan = dtype_plan or {}  # {glob_pattern: dtype}
    weight_mapping = weight_mapping or []
    meta_model_state_dict = model.state_dict()
    missing_keys = set(meta_model_state_dict.keys())

    misc = {}
    mismatch_keys = set()
    unexpected_keys = set()
    # Global thread_pool
    thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)

    renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
    all_mappings: list[WeightRenaming | WeightConverter] = renamings + converters
    passthrough_renamings: dict[str, WeightRenaming] = {}

    rename_patterns = [entry.source_key for entry in renamings]
    rename_alt, rename_by_group = build_glob_alternation(rename_patterns)
    rename_rules = {entry.source_key: compile_glob_rule(entry.source_key, entry.target_key) for entry in renamings}
    rename_map = {entry.source_key: entry for entry in renamings}

    converter_patterns = list(itertools.chain.from_iterable(converter.source_keys for converter in converters))
    pattern_to_converter = {pattern: converter for converter in converters for pattern in converter.source_keys}
    weight_pattern_alt, weight_pattern_by_group_name = build_glob_alternation(converter_patterns)
    tp_plan = tp_plan or {}
    dtype_plan = dtype_plan or {}
    tp_plan_alt, tp_plan_by_group_name = build_glob_alternation(list(tp_plan.keys()))
    dtype_policy_alt, dtype_policy_by_group_name = build_glob_alternation(list(dtype_plan.keys()))

    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    for original_key, tensor in state_dict:
        renamed_key = original_key
        applied_rename: WeightRenaming | None = None
        rename_match = match_glob(original_key, rename_alt, rename_by_group)
        if rename_match is not None:
            rule = rename_rules.get(rename_match)
            if rule is not None:
                regex, replacement = rule
                renamed_key = regex.sub(replacement, original_key, count=1)
                applied_rename = rename_map.get(rename_match)

        matched_pattern = match_glob(renamed_key, weight_pattern_alt, weight_pattern_by_group_name)
        if matched_pattern is not None:
            converter = pattern_to_converter[matched_pattern]
            extractor_pattern = matched_pattern.replace("*", r"(\d+)")
            sub_with_extractor = partial(re.sub, extractor_pattern, string=renamed_key)
            resolved_targets = list(map(sub_with_extractor, [k.replace("*", "\\1") for k in converter.target_keys]))
            source_pattern = matched_pattern
        else:
            converter = None
            resolved_targets = [renamed_key]
            source_pattern = rename_match or renamed_key

        _dtype = dtype
        new_target_key: list[str] = []
        pending_quantize_op = None
        for t in resolved_targets:
            if t.startswith(prefix) and meta_model_state_dict.get(re.sub(f"^{prefix}.", "", t, count=1)) is not None:
                t = re.sub(f"^{prefix}.", "", t, count=1)
            elif meta_model_state_dict.get(f"{prefix}.{t}") is not None:
                t = f"{prefix}.{t}"
            empty_param = meta_model_state_dict.get(t)
            new_target_key.append(t)

            if quantizer is not None and quantizer.param_needs_quantization(model, t):
                if quantizer.__class__.__name__ == "FineGrainedFP8HfQuantizer":
                    from .integrations.finegrained_fp8 import Fp8Quantize

                    if converter is not None:
                        converter.quantization_operation = Fp8Quantize()
                    else:
                        pending_quantize_op = Fp8Quantize()
                else:
                    raise ValueError("This quantization method is gonna be supported SOOOON")
            else:
                _dtype = dtype
                matched_dtype_pattern = match_glob(t, dtype_policy_alt, dtype_policy_by_group_name)
                if matched_dtype_pattern is not None:
                    _dtype = dtype_plan[matched_dtype_pattern]
                elif empty_param.dtype != _dtype:
                    _dtype = empty_param.dtype

        if not new_target_key:
            continue

        first_target_key = new_target_key[0]
        if converter is not None:
            mapping: WeightRenaming | WeightConverter = converter
        elif applied_rename is not None:
            mapping = applied_rename
        else:
            mapping = passthrough_renamings.get(renamed_key)
            if mapping is None:
                mapping = WeightRenaming(renamed_key, renamed_key)
                passthrough_renamings[renamed_key] = mapping
                all_mappings.append(mapping)

        if pending_quantize_op is not None and getattr(mapping, "quantization_operation", None) is None:
            mapping.quantization_operation = pending_quantize_op

        future = None
        if device_mesh:
            if matched_tp_pattern := match_glob(first_target_key, tp_plan_alt, tp_plan_by_group_name):
                empty_param = meta_model_state_dict.get(first_target_key)
                if getattr(mapping, "distributed_operation", None) is None:
                    tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]].__class__
                    mapping.distributed_operation = tp_layer(
                        device_mesh=device_mesh, rank=device_map[""].index, empty_param=empty_param.clone()
                    )
                layer_key = "|".join(new_target_key)
                shard_index = len(mapping.collected_tensors.get(layer_key, {}).get(source_pattern, []))
                future = spawn_tp_materialize(
                    thread_pool,
                    tensor,
                    _dtype,
                    mapping.distributed_operation,
                    shard_index,
                )

        if future is None:
            future = spawn_materialize(thread_pool, tensor, _dtype)

        layer_key = "|".join(new_target_key)
        mapping.add_tensor(layer_key, source_pattern, future, new_target_key)

    total_entries = sum(len(mapping.collected_tensors) for mapping in all_mappings)

    with logging.tqdm(total=total_entries, desc="Loading weights") as pbar:
        for mapping in all_mappings:
            operations = mapping.operations if isinstance(mapping.operations, list) else [mapping.operations]
            for layer_name, tensors_for_this_layer in mapping.collected_tensors.items():
                pbar.update(1)
                pbar.set_postfix({"Materializing param": layer_name})
                pbar.refresh()
                concrete_target_keys = mapping.layer_targets.get(layer_name, [])
                if not bool(set(concrete_target_keys) - unexpected_keys):
                    continue
                try:
                    with log_to_misc(layer_name, misc):
                        values = {
                            pattern: [future.result() for future in futures]
                            for pattern, futures in tensors_for_this_layer.items()
                        }

                    if operations:
                        payload = values
                        for op in operations:
                            with log_to_misc(layer_name, misc, (payload, concrete_target_keys), operations):
                                payload = op.convert(
                                    payload,
                                    source_keys=mapping.source_keys,
                                    target_keys=mapping.target_keys,
                                    concrete_target_keys=concrete_target_keys,
                                    config=model.config,
                                )
                    else:
                        if len(values) != len(concrete_target_keys):
                            raise ValueError("Renaming expects one tensor per target key.")
                        payload = {}
                        for src_key, target in zip(mapping.source_keys, concrete_target_keys):
                            tensors = values.get(src_key, [])
                            if not tensors:
                                raise ValueError(f"Missing tensors for source pattern {src_key}.")
                            payload[target] = tensors

                    realized_value = {}
                    for key, tensors in payload.items():
                        if key in unexpected_keys or not tensors:
                            continue
                        realized_value[key] = tensors[0]

                    if mapping.quantization_operation is not None and quantizer is not None:
                        with log_to_misc(layer_name, misc, op=mapping.quantization_operation):
                            realized_value = mapping.quantization_operation.convert(
                                realized_value, quant_config=quantizer.quantization_config
                            )

                    for k, output_value in realized_value.items():
                        set_param_for_module(
                            model,
                            k,
                            output_value,
                            mismatch_keys,
                            missing_keys,
                            misc,
                            unexpected_keys,
                            mapping.distributed_operation,
                        )
                except SkipLayer:
                    continue
    thread_pool.shutdown(wait=False)
    return missing_keys, unexpected_keys, mismatch_keys, misc


# TODO this is not done yet!
def revert_weight_conversion(model, state_dict):
    mapping = getattr(model, "_checkpoint_conversion_mapping", {})  # IDK why but setting this will fail all llava.
    reverse_key_mapping = [(v, k) for k, v in mapping.items()]
    original_state_dict = {}
    for key, value in state_dict.items():
        for pattern, inverse_converter in reverse_key_mapping:
            # TODO FIXME you name it
            replacement = inverse_converter.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break
        original_state_dict[key] = value
    state_dict = original_state_dict
    return state_dict
