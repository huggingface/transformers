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

import os
import re
from abc import abstractmethod
from collections.abc import MutableMapping, MutableSet, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from .integrations.accelerate import offload_weight
from .integrations.tensor_parallel import ALL_PARALLEL_STYLES
from .utils import is_torch_greater_or_equal, logging


_torch_distributed_available = torch.distributed.is_available()
_is_dtensor_available = _torch_distributed_available and is_torch_greater_or_equal("2.5")
if _is_dtensor_available:
    from torch.distributed.tensor import DTensor, Replicate

if TYPE_CHECKING:
    from .integrations.tensor_parallel import TensorParallelLayer
    from .modeling_utils import PreTrainedModel
    from .quantizers import HfQuantizer


logger = logging.get_logger(__name__)

logger = logging.get_logger(__name__)


def compile_glob_rule(source_glob: str, target_glob: str) -> tuple[re.Pattern, str]:
    """
    Convert a glob-style source + target into a full regex + replacement.

    Rules:
      - '*' in source_glob  →  (.*) capture group
      - '*' in target_glob  →  \\1, \\2, ... backrefs
    """
    regex = re.compile(source_glob)

    counter = 0

    def _star_to_backref(_: re.Match) -> str:
        nonlocal counter
        counter += 1
        return rf"\{counter}"

    replacement = re.sub(r"\*", _star_to_backref, target_glob)
    return regex, replacement


def build_glob_alternation(
    globs: list[Union[WeightRenaming, WeightConverter, str]],
) -> tuple[re.Pattern, dict[str, str], dict[str, str]]:
    """
    Build a single alternation regex with one named group per glob.
    """
    src_group_to_glob: dict[str, str] = {}
    tgt_group_to_glob: dict[str, str] = {}
    branches: list[str] = []
    i = 0
    for glob in globs:
        if isinstance(glob, (WeightRenaming, WeightConverter)):
            for src in glob.source_keys:
                group_name = f"g{i}"
                src_group_to_glob[group_name] = src
                i += 1
                body = src.replace("*", r".*")
                branches.append(f"(?P<{group_name}>{body})")
                tgt_group_to_glob[group_name] = glob.target_keys[0]  # we index witht the first target
        else:
            group_name = f"g{i}"
            src_group_to_glob[group_name] = glob
            i += 1
            body = glob
            body = body.replace("*", r".*")
            branches.append(f"(?P<{group_name}>{body})")
            tgt_group_to_glob[group_name] = glob

    alternation = re.compile("|".join(branches))
    return alternation, src_group_to_glob, tgt_group_to_glob


class ConversionOps:
    """Base class for weight conversion operations."""

    # The inverse operation class, will be used when saving the checkpoint
    reverse_op: type[ConversionOps]

    @abstractmethod
    def convert(
        self,
        value: dict[str, Any],
        source_keys: list[str],
        target_keys: list[str],
        full_layer_name: str,
        model,
        missing_keys,
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        raise NotImplementedError


class Chunk(ConversionOps):
    """Split a tensor along ``dim`` into equally sized chunks or using explicit ``sizes``."""

    reverse_op: type[ConversionOps]

    def __init__(self, dim: int = 0, chunks: Optional[int] = None, sizes: Optional[Sequence[int]] = None):
        self.dim = dim
        self.chunks = chunks
        self.sizes = list(sizes) if sizes is not None else None
        self.reverse_op = Concatenate

    def convert(
        self,
        value: dict[str, list[torch.Tensor]],
        source_keys: list[str],
        target_keys: list[str],
        full_layer_name: str,
        model,
        missing_keys,
        config,
    ) -> dict[str, list[torch.Tensor]]:
        tensors = next(iter(value.values()))
        tensor = tensors[0]
        sizes = len(target_keys)
        chunks = torch.chunk(tensor, sizes, dim=self.dim)
        return {full_layer_name.replace(target_keys[0], target): [chunk] for target, chunk in zip(target_keys, chunks)}


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
        source_keys: list[str],
        target_keys: list[str],
        full_layer_name: str,
        model,
        missing_keys,
        config,
    ) -> dict[str, torch.Tensor]:
        if len(target_keys) != 1:
            raise ValueError("Concatenate expects a single target key.")
        if len(value) != len(source_keys):
            raise ValueError("Concatenate received an unexpected number of tensors compared to source keys.")

        return {full_layer_name: torch.cat(tuple(value.values()), dim=self.dim)}


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
        source_keys: list[str],
        target_keys: list[str],
        full_layer_name: str,
        model,
        missing_keys,
        config,
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        for idx, key in enumerate(value.keys()):
            tensors = value.get(key, [])
            if len(source_keys) == 1:
                key = full_layer_name
            stacked = torch.stack(tensors, dim=self.dim)
            merged[key] = stacked
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
        source_keys: list[str],
        target_keys: list[str],
        full_layer_name: str,
        model,
        missing_keys,
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
        source_keys: list[str],
        target_keys: list[str],
        full_layer_name: str,
        model,
        missing_keys,
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
class WeightTransform:
    source_keys: Union[str, list[str]] = field(init=True)
    target_keys: Union[str, list[str]] = field(init=True)

    distributed_operation: Optional[TensorParallelLayer] = None
    quantization_operation: Optional[ConversionOps] = None

    collected_tensors: dict[str, list[Future]] = field(default_factory=dict, init=False)
    layer_targets: dict[str, set[str]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if isinstance(self.source_keys, str):
            self.source_keys = [self.source_keys]
        if isinstance(self.target_keys, str):
            self.target_keys = [self.target_keys]

    def add_tensor(self, target_key: str, source_key: str, source_pattern: str, future: Future):
        bucket = self.collected_tensors.setdefault(source_pattern, [])
        bucket += [future]

        bucket = self.layer_targets.setdefault(target_key, set())
        bucket.add(source_key)


@dataclass(slots=True)
class WeightRenaming(WeightTransform):
    # Special case of WeightTransform that only renames keys without any conversion.

    def convert(
        self,
        layer_name: str,
        model=None,
        config=None,
        hf_quantizer=None,
        missing_keys: Optional[MutableSet[str]] = None,
        misc: Optional[MutableMapping[str, str]] = None,
    ):
        for pattern, futures in self.collected_tensors.items():
            self.collected_tensors[pattern] = [future.result() for future in futures]

        collected_tensors = self.collected_tensors
        if hf_quantizer is not None and self.quantization_operation is not None:
            with log_to_misc(layer_name, misc, (self.collected_tensors, layer_name), self.quantization_operation):
                collected_tensors = self.quantization_operation.convert(
                    self.collected_tensors,
                    source_keys=self.source_keys,
                    target_keys=self.target_keys,
                    full_layer_name=layer_name,
                    model=model,
                    config=config,
                    missing_keys=missing_keys,
                )

        return collected_tensors, misc


@dataclass(slots=True)
class WeightConverter(WeightTransform):
    operations: list[ConversionOps] = field(default_factory=list, repr=False)

    def __post_init__(self):
        WeightTransform.__post_init__(self)
        if bool(len(self.source_keys) - 1) + bool(len(self.target_keys) - 1) >= 2:
            raise ValueError(
                f"source keys={self.source_keys}, target_keys={self.target_keys} but you can only have one to many, one to one or many to one."
            )
        if not self.operations:
            raise ValueError("WeightConverter requires at least one operation.")

    def convert(
        self,
        layer_name: str,
        model=None,
        config=None,
        hf_quantizer=None,
        missing_keys: Optional[MutableSet[str]] = None,
        misc: Optional[MutableMapping[str, str]] = None,
    ):
        for pattern, futures in self.collected_tensors.items():
            self.collected_tensors[pattern] = [future.result() for future in futures]

        collected_tensors = self.collected_tensors
        for op in self.operations:
            with log_to_misc(layer_name, misc, (collected_tensors, layer_name), op):
                collected_tensors = op.convert(
                    collected_tensors,
                    source_keys=self.source_keys,
                    target_keys=self.target_keys,
                    full_layer_name=layer_name,
                    model=model,
                    config=config,
                    missing_keys=missing_keys,
                )
        if hf_quantizer is not None and self.quantization_operation is not None:
            with log_to_misc(layer_name, misc, (collected_tensors, layer_name), self.quantization_operation):
                collected_tensors = self.quantization_operation.convert(
                    collected_tensors,
                    source_keys=self.source_keys,
                    target_keys=self.target_keys,
                    full_layer_name=layer_name,
                    config=config,
                    model=model,
                    missing_keys=missing_keys,
                )
        return collected_tensors, misc


# For I/O bound operations (i.e. here reading files), it is better to have fewer threads, e.g. 4 is a good default.
# Having too many is actually harming performances quite a lot, i.e. using 16 can sometimes lead to taking TWICE
# as much time to load the same model
GLOBAL_WORKERS = min(4, os.cpu_count() or 4)


def _materialize_copy(tensor, device=None, dtype=None):
    tensor = tensor[...]
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def spawn_materialize(thread_pool, tensor, device=None, dtype=None) -> Future:
    def _job():
        return _materialize_copy(tensor, device, dtype)

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
    first_target_key: str,
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
            misc[first_target_key] = (
                f"{e}\nError: {descriptor}on tensors destined for {target_keys}. Ckpt contains: {len(values)}"
            )
        elif isinstance(extras, str):
            suffix = f" via {op_name}" if op_name else ""
            misc[first_target_key] = f"{e}\nError{suffix} when processing parameter {extras}"
        elif extras is None and op_name:
            misc[first_target_key] = f"{op_name}: {e}"
        else:
            misc[first_target_key] = f"{extras} |Error: {e}"
        raise SkipLayer()


def set_param_for_module(
    model: PreTrainedModel,
    target_name: str,
    param_value: torch.Tensor,
    mismatch_keys: MutableSet[tuple[str, torch.Size, torch.Size]],
    missing_keys: MutableSet[str],
    misc: MutableMapping[str, Any],
    unexpected_keys: MutableSet[str],
    distributed_operation: Optional[TensorParallelLayer],
    hf_quantizer: HfQuantizer,
):
    with log_to_misc(target_name, misc, target_name):
        module_path, _, param_name = target_name.rpartition(".")
        module_obj = model.get_submodule(module_path) if module_path else model

        ref = getattr(module_obj, param_name)
        if ref is None:
            unexpected_keys.add(target_name)
        else:
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
            missing_keys.discard(target_name)
            if ref is not None and ref.shape != param_value.shape and hf_quantizer is None:
                mismatch_keys.add((target_name, param_value.shape, ref.shape))
                module_obj.param_name._is_hf_initialized = False  # Needs to be initialized
            else:
                # super important otherwise _init_weight will re-init the param
                param_value._is_hf_initialized = True
                setattr(module_obj, param_name, param_value)


def offload_and_maybe_resave_param(
    target_name: str,
    param: torch.Tensor,
    missing_keys: MutableSet[str],
    disk_offload_folder: str,
    disk_offload_index: dict,
    applied_ops: WeightConverter | WeightRenaming,
) -> dict:
    """Takes care of correctly offloading `param`. If it's not already present in the `disk_offload_index`, or if any
    WeightConverter operations have been applied, it will resave the new parameter. Otherwise, it will use the original
    `disk_offload_index` for this given param."""
    # We need to remove from missing keys
    missing_keys.discard(target_name)
    # If not already offloaded, or if we applied any special Operation except Renaming, we need to re-save
    if target_name not in disk_offload_index or isinstance(applied_ops, WeightConverter):
        disk_offload_index = offload_weight(param, target_name, disk_offload_folder, disk_offload_index)
    return disk_offload_index


class SkipLayer(Exception):
    """Control-flow sentinel: abort processing of the current layer only."""

    pass


def repl(m, repl_map: dict[str, str]) -> str:
    # Collect all groups that matched
    matched_groups = [name for name, val in m.groupdict().items() if val]

    if len(matched_groups) == 0:
        # Should never happen
        return m.group(0)

    if len(matched_groups) > 1:
        raise ValueError(
            "only a single match should happen, your regex patterns are tangled: "
            f"groups matched = {matched_groups} for the patternsL {repl_map.keys()}"
        )

    # Exactly one match => return replacement
    name = matched_groups[0]
    return repl_map[name]


def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    weight_mapping: list[WeightConverter | WeightRenaming] | None,
    tp_plan: dict[str, str] | None,
    hf_quantizer: HfQuantizer | None,
    dtype: torch.dtype | None = None,
    device_map: dict | None = None,
    dtype_plan: dict | None = None,
    device_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
    disk_offload_index: dict | None = None,
    disk_offload_folder: str | None = None,
):
    r"""
    We build a mapping from the keys obtained by renaming each of the checkpoint keys according to the weight_mapping rules.
    Then we load the tensors into the model, applying any conversion operations as needed.

    The `param_name_to_load` will look like this:
    {
        "model.layers.0.attention.q.weight": # Notice here there is only the first key of the target keys
            WeightConverter(
                source_keys=["qkv"],
                target_keys=["q", "k","v"],
                operations=[Chunk(dim=0, chunks=3)]),
                collected_tensors={
                    "qkv": [Future, Future, Future]},
                layer_targets={
                    "model.layers.0.attention.q.weight": {"model.layers.0.attention.qkv.weight"},
                    "model.layers.0.attention.k.weight": {"model.layers.0.attention.qkv.weight"},
                    "model.layers.0.attention.v.weight": {"model.layers.0.attention.qkv.weight"},
                }
            ),
        ...
    }

    We make sure that the keys are the full keys. The only "nit" here is that 1 key can map to multiple target keys (e.g. qkv -> q, k, v).
    In that case the weight converter will take care of doing the appropriate renaming.

    For example for:
    ```python
    WeightConverter(
        source_keys=["mlp.experts.*.gate_proj.weight","mlp.experts.*.up_proj.weight"],
        target_keys="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    )
    ```
    we would have the following collected tensors:
    ```python
    collected_tensors = {
        "mlp.experts.*.gate_proj.weight": [Future, Future, Future, Future, Future, Future, Future, Future],
        "mlp.experts.*.up_proj.weight": [Future, Future, Future, Future, Future, Future, Future, Future],
    }
    ```
    The first op, `MergeModulelist`, would stack the 8 tensors of each source but will not "rename" them into the fused target name.
    The second op, `Concatenate`, would then rename the fused tensor into the final target name.

    If we want to split `qkv` we would have:
    ```python
    collected_tensors = {
        "attention.qkv.weight": [Future], # here its the full SOURCE keys.
    }
    ```
    The `Chunk` operation would then split the single tensor into 3 and rename them accordingly and update the collected tensors to:
    ```python
    realized_values = {
        "attention.q.weight": [Tensor],
        "attention.k.weight": [Tensor],
        "attention.v.weight": [Tensor],
    }
    ```

    Now that this is done, we can quantize / dequantize accordingly the collected_tensors.

    For some quantization methods, we need to gather different tensors:

    ```python
    # for "medmekk/llama-3.2-1b-float8-torchao"
    WeightConverter(
        source_keys=[":qdata", ":scale"],
        target_keys="",
        operations=[TorchaoDeserialize()],
    )
    ```
    This will collect all tensors that have the same prefix, but end with `:qdata` or `:scale`. This will give us:
    ```python
    all_weight_mapping = {
        "model.layers.13.self_attn.o_proj.weight": WeightConverter(
            source_keys=[":qdata", ":scale"],
            target_keys="",
            operations=[TorchaoDeserialize()],
            collected_tensors={
                ":qdata": [Future],
                ":scale": [Future],
            },
        ...
    }
    ```

    """
    prefix = model.base_model_prefix
    tp_plan = tp_plan or {}
    device_map = device_map or {"": "cpu"}
    # Here, we first sort by number of submodules, then length of the full string, to make sure to match correctly
    device_map_regex = re.compile(
        "|".join(rf"({k})" for k in sorted(device_map.keys(), key=lambda x: (x.count("."), len(x)), reverse=True))
    )
    dtype_plan = dtype_plan or {}
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
    if hf_quantizer:
        # We will add the quantizer's deserialization WeightConverter here.
        pass

    param_name_to_load: dict[str, Union[WeightRenaming | WeightConverter]] = {}

    # build '(?P<g0>.*.*\\.block_sparse_moe\\..*)' and group to source {'g0': '*.block_sparse_moe.'}
    # and target to source {'g0': '*.mlp.'}. This allows us to quickly find which pattern matched.
    rename_alt, _, rename_by_group = build_glob_alternation(renamings)
    if converters != []:
        weight_pattern_alt, src_group_to_glob, tgt_group_to_glob = build_glob_alternation(converters)
    if tp_plan != {}:
        tp_plan_alt, tp_plan_by_group_name, _ = build_glob_alternation(list(tp_plan.keys()))
    if dtype_plan != {}:
        dtype_policy_alt, dtype_policy_by_group_name, _ = build_glob_alternation(list(dtype_plan.keys()))

    pattern_to_converter = {k: converter for converter in converters for k in converter.source_keys}

    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    for original_key, tensor in state_dict:
        # 1. apply all renamings
        renamed_key = rename_alt.sub(lambda m: repl(m, rename_by_group), original_key).replace("\\", "")

        # 2. apply 1 weight conversion on the key
        matched_pattern = weight_pattern_alt.search(renamed_key) if converters != [] else None
        if matched_pattern is not None:  # we have a converter to apply
            renamed_key = weight_pattern_alt.sub(lambda m: repl(m, tgt_group_to_glob), renamed_key).replace("\\", "")

        # 3. check if we need to add or remove prefix
        if (
            renamed_key.startswith(prefix)
            and meta_model_state_dict.get(re.sub(f"^{prefix}.", "", renamed_key, count=1)) is not None
        ):
            renamed_key = re.sub(f"^{prefix}.", "", renamed_key, count=1)
        elif meta_model_state_dict.get(f"{prefix}.{renamed_key}") is not None:
            renamed_key = f"{prefix}.{renamed_key}"

        # 4. finally, collect the tensor into the proper converter
        if renamed_key in missing_keys:
            empty_param = meta_model_state_dict.get(renamed_key)
            if matched_pattern:
                new_converter = deepcopy(pattern_to_converter[src_group_to_glob[matched_pattern.lastgroup]])
                # each target key gets its own converter instance
                mapping = param_name_to_load.setdefault(renamed_key, new_converter)
                source_pattern = src_group_to_glob[matched_pattern.lastgroup]
            else:
                mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(renamed_key, renamed_key))
                source_pattern = renamed_key

            # 5. Handle dtype casting
            if hf_quantizer is not None and hf_quantizer.param_needs_quantization(model, renamed_key):
                mapping.quantization_operation = hf_quantizer.get_quantize_ops()

            _dtype = dtype
            if dtype_plan != {} and dtype_policy_alt.search(renamed_key):
                matched_dtype_pattern = dtype_policy_alt.search(renamed_key)
                if matched_dtype_pattern is not None:
                    _dtype = dtype_plan[matched_dtype_pattern.group()]
            elif empty_param is not None and empty_param.dtype != _dtype:
                _dtype = empty_param.dtype  # usually correct when initializing

            # 6. Handle TP sharding or device_map placement -> scheduled materialization
            future = None
            if device_mesh:
                if matched_tp_pattern := tp_plan_alt.search(renamed_key):
                    matched_tp_pattern = tp_plan_by_group_name[matched_tp_pattern.lastgroup]
                    if getattr(mapping, "distributed_operation", None) is None:
                        tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]].__class__
                        mapping.distributed_operation = tp_layer(
                            device_mesh=device_mesh, rank=device_map[""].index, empty_param=empty_param.clone()
                        )
                    shard_index = len(mapping.collected_tensors.get(original_key, []))
                    future = spawn_tp_materialize(
                        thread_pool,
                        tensor,
                        mapping.distributed_operation,
                        shard_index,
                        _dtype,
                    )

            if future is None:
                device_match = device_map_regex.match(renamed_key)
                param_device = device_map[device_match.group()] if device_match else device_map.get("", "cpu")
                # If disk, we need to materialize on cpu first
                param_device = "cpu" if param_device == "disk" else param_device
                future = spawn_materialize(thread_pool, tensor, param_device, _dtype)

            mapping.add_tensor(renamed_key, original_key, source_pattern, future)
        elif matched_pattern:  # add all target keys as unexpected
            mapping = pattern_to_converter[src_group_to_glob[matched_pattern.lastgroup]]
            for k in mapping.target_keys:
                unexpected_keys.add(renamed_key.replace(mapping.target_keys[0], k))
        else:
            unexpected_keys.add(renamed_key)

    total_entries = len(param_name_to_load)
    with logging.tqdm(total=total_entries, desc="Loading weights") as pbar:
        for first_param_name, mapping in param_name_to_load.items():
            pbar.update(1)
            pbar.set_postfix({"Materializing param": first_param_name})
            pbar.refresh()
            try:
                realized_value, misc = mapping.convert(
                    first_param_name,
                    model=model,
                    config=model.config,
                    hf_quantizer=hf_quantizer,
                    missing_keys=missing_keys,
                    misc=misc,
                )
                for target_name, param in realized_value.items():
                    param = param[0] if isinstance(param, list) else param
                    device_match = device_map_regex.match(target_name)
                    param_device = device_map[device_match.group()] if device_match else device_map.get("", "cpu")
                    # Offloading support
                    if param_device == "disk":
                        disk_offload_index = offload_and_maybe_resave_param(
                            target_name, param, missing_keys, disk_offload_folder, disk_offload_index, mapping
                        )
                    else:
                        set_param_for_module(
                            model,
                            target_name,
                            param,
                            mismatch_keys,
                            missing_keys,
                            misc,
                            unexpected_keys,
                            mapping.distributed_operation,
                            hf_quantizer,
                        )
            except SkipLayer:
                continue

    thread_pool.shutdown(wait=False)
    return missing_keys, unexpected_keys, mismatch_keys, disk_offload_index, misc


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
