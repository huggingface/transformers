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
import threading
from abc import abstractmethod
from collections import defaultdict
from collections.abc import MutableMapping, MutableSet, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Union

import torch
from torch.distributed.tensor import DTensor

from .integrations.tensor_parallel import ALL_PARALLEL_STYLES, TensorParallelLayer
from .utils import logging


logger = logging.get_logger(__name__)


def _glob_to_regex_src(glob: str, *, digits_only: bool = True) -> str:
    """
    Convert a glob with '*' into a regex *source* string. We don't use `glob.translate`
    '*' matches (\\d+) if digits_only else (.+). Inner groups are non-capturing.
    """
    star = r"(\d+)" if digits_only else r"(.+)"
    return re.escape(glob).replace(r"\*", star)


def build_glob_alt(
    globs: list[str],
) -> tuple[re.Pattern, dict[str, str]]:
    r"""
    Build one compiled regex alternation with a named group per glob. This allows to run a single
    re.match and get the correct group name to finally get which pattern matched.
    Returns (compiled_regex, name->glob map).

    Example:

    ```py
    >>> reg, map_ = build_glob_alt(["mlp.*.w1", "mlp.*.w2"])
    >>> print(reg)
    (re.compile(r'(?P<g0>.*mlp\.(\d+)\.w1)|(?P<g1>.*mlp\.(\d+)\.w2)', re.UNICODE),
    >>> print(map_)
    {'g0': 'mlp.*.w1', 'g1': 'mlp.*.w2'})
    >>> match_ = reg.match("model.layers.0.mlp.0.w1.weight")
    >>> print(match_.lastgroup)
    'g0'
    >>> print(map_[match_.lastgroup])
    mlp.*.w1
    ```
    """
    name_map: dict[str, str] = {}
    parts: list[str] = []
    prefix_src = r".*"

    for i, g in enumerate(globs):
        name = f"g{i}"
        name_map[name] = g
        pat_src = _glob_to_regex_src(g)
        parts.append(f"(?P<{name}>{prefix_src}{pat_src})")

    alt_src = "|".join(parts)
    return re.compile(alt_src), name_map


def match_glob(key: str, alt: re.Pattern, name_map: dict[str, str]) -> Optional[str]:
    """
    Match the key against the alternation; return the original glob string that matched.
    """
    m = alt.match(key)
    if not m:
        return None
    return name_map.get(m.lastgroup)


class ConversionOps:
    """Base class for weight conversion operations."""

    # Reusable staging/scratch buffer to avoid reallocations.
    _buffer: Optional[torch.Tensor] = None
    # The inverse operation class, will be used when saving the checkpoint
    reverse_op: type[ConversionOps]

    def _ensure_buffer(
        self,
        required_shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
        growth_factor: float = 1.5,
    ) -> torch.Tensor:
        """Ensure a pre-allocated buffer large enough for ``required_shape`` exists."""

        required_elems = 1
        for dim in required_shape:
            required_elems *= int(dim)

        need_new = (
            self._buffer is None
            or self._buffer.dtype != dtype
            or self._buffer.device != device
            or self._buffer.numel() < required_elems
        )

        if need_new:
            capacity = max(required_elems, int(required_elems * growth_factor))
            self._buffer = torch.empty(capacity, dtype=dtype, device=device)

        return self._buffer[:required_elems].view(required_shape)

    def clear_cache(self) -> None:
        """Free any cached buffers."""
        self._buffer = None

    @abstractmethod
    def convert(
        self, value: Union[dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
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

    def convert(self, value: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        if not isinstance(value, torch.Tensor):
            raise TypeError("Chunk expects a torch.Tensor as input.")
        if self.sizes is not None:
            return list(torch.split(value, self.sizes, dim=self.dim))
        return list(torch.chunk(value, self.chunks, dim=self.dim))


class Concatenate(ConversionOps):
    """Concatenate tensors along `dim` using a reusable buffer."""

    reverse_op: type[ConversionOps]

    def __init__(self, dim: int = 0):
        self.dim = dim
        self.reverse_op = Chunk

    @torch.no_grad
    def convert(self, value: Sequence[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        if isinstance(value[0], list):
            value = [v[0] for v in value]
        tensors = value
        if not tensors:
            raise ValueError("Fuse requires at least one tensor to concatenate.")

        out_shape = list(tensors[0].shape)
        out_shape[self.dim] = sum([t.size(self.dim) for t in tensors])

        with torch.no_grad():  # we use staging buffers
            out = self._ensure_buffer(torch.Size(out_shape), dtype=tensors[0].dtype, device=tensors[0].device)
            torch.cat(tuple(tensors), dim=self.dim, out=out)
            # offset = 0
            # for tensor in tensors:
            #     index = [slice(None)] * tensor.ndim
            #     index[self.dim] = slice(offset, offset + tensor.shape[self.dim])
            #     out[tuple(index)].copy_(tensor, non_blocking=tensor.is_cuda)
            #     offset += tensor.shape[self.dim]
        return out.clone()  # need to say I can overwrite this storage now


class MergeModulelist(Concatenate):
    """
    Merge a list of tensors into a single tensor along the first dimension.
    We explicitly define this because for EP or TP you want to make sure you know what you are doing!

    """

    def __init__(self, dim: int = 0):
        super().__init__(dim=dim)
        self.reverse_op = SplitModulelist

    def convert(self, value: Sequence[torch.Tensor], *args, **kwargs) -> list[torch.Tensor]:
        merged = []
        with torch.no_grad():  # we use staging buffers
            for group in value:
                if not isinstance(group, Sequence) or len(group) == 0:
                    raise ValueError("MergeModulelist requires non-empty sub-sequences.")
                group = [k for k in group if k.ndim]
                out_shape = list(group[0].shape)
                out_shape.insert(self.dim, len(group))
                out = self._ensure_buffer(torch.Size(out_shape), dtype=group[0].dtype, device=group[0].device)
                torch.stack(tuple(group), dim=self.dim, out=out)
                # for off, tensor in enumerate(group):
                #     out[off].copy_(tensor, non_blocking=tensor.is_cuda)
                # torch.as_tensor(numpy.stack(batch))
                merged.append(out.clone())  # TODO have a single staging tensor here as well!
        return merged


class SplitModulelist(ConversionOps):
    """Inverse of :class:`MergeModulelist` using explicit split sizes per group."""

    def __init__(self, sizes: Sequence[Sequence[int]], dim: int = 0):
        if not isinstance(sizes, Sequence) or not all(isinstance(sub, Sequence) and sub for sub in sizes):
            raise ValueError("`sizes` must be a sequence of non-empty sequences of integers.")
        self.sizes = [list(sub) for sub in sizes]
        self.dim = dim
        self.reverse_op = MergeModulelist

    def convert(self, value: Sequence[torch.Tensor], *, context: dict[str, Any]) -> list[list[torch.Tensor]]:
        if not isinstance(value, Sequence):
            raise TypeError("SplitModulelist expects a sequence of tensors.")
        if len(value) != len(self.sizes):
            raise ValueError("Number of tensors does not match the provided split specifications.")

        result: list[list[torch.Tensor]] = []
        for tensor, split_sizes in zip(value, self.sizes):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("SplitModulelist can only split torch.Tensor instances.")
            splits = torch.split(tensor, split_sizes, dim=self.dim)
            result.append(list(splits))
        return result


class Cast(ConversionOps):
    """
    Casts the tensor to a given dtype
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def convert(self, value, *args, **kwargs):
        out = [
            [x.to(self.dtype) for x in inner] if isinstance(inner, list) else inner.to(self.dtype) for inner in value
        ]
        return out


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

    def convert(
        self, value: Union[dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor], config
    ) -> Union[dict[str, torch.Tensor], list[torch.Tensor], torch.Tensor]:
        self.config = config
        out = [[self._apply(x) for x in inner] if isinstance(inner, list) else self._apply(inner) for inner in value]
        return out


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
    """

    source_keys: Union[str, list[str]]
    target_keys: Optional[Union[str, list[str]]] = None
    operations: list[ConversionOps] = field(default_factory=list, repr=False)

    distributed_operation: Optional[TensorParallelLayer] = None
    quantization_operation: Optional[ConversionOps] = None

    def __post_init__(self):
        if not isinstance(self.source_keys, list):
            self.source_keys = [self.source_keys]
        targets_were_none = False
        if not isinstance(self.target_keys, list):
            if self.target_keys is None:
                self.target_keys = list(self.source_keys)
                targets_were_none = True
            else:
                self.target_keys = [self.target_keys]

        if not targets_were_none and bool(len(self.source_keys) - 1) + bool(len(self.target_keys) - 1) >= 2:
            raise ValueError(
                f"source keys={self.source_keys}, target_keys={self.target_keys} but you can only have one to many, one to one or many to one."
            )

        for pattern in self.source_keys:
            if any(ch in pattern for ch in set("^$+?{}[]|()")):
                raise AssertionError(f"'{pattern}' is not glob")
        for pattern in self.target_keys:
            if any(ch in pattern for ch in set("^$+?{}[]|()")):
                raise AssertionError(f"'{pattern}' is not glob")


@dataclass(slots=True)
class ConversionEntry:
    weight_converter: WeightConverter
    collected_tensors: dict = field(default_factory=lambda: defaultdict(dict))


GLOBAL_WORKERS = min(16, (os.cpu_count() or 8) * 2)  # NVMe: 8-16; HDD/NFS: 2-4
PER_FILE_LIMIT = 4  # concurrent reads per file


def _materialize_copy(x):
    # PyTorch: this runs in C and releases the GIL; good for threads.
    return x[...]


def spawn_materialize(thread_pool, _file_semaphore, file_id, t) -> Future:
    sem = _file_semaphore[file_id]

    def _job():
        with sem:
            return _materialize_copy(t)

    return thread_pool.submit(_job)


def spawn_tp_materialize(thread_pool, _file_semaphore, file_id, t, sharding_method, tensor_idx) -> Future:
    sem = _file_semaphore[file_id]

    def _job():
        with sem:
            return sharding_method.shard_tensor(t, tensor_idx=tensor_idx)[0]

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
    model: torch.nn.Module,
    layer_name: str,
    param_value: torch.Tensor,
    meta_model_state_dict: MutableMapping[str, Any],
    empty_param: torch.Tensor,
    mismatch_keys: MutableSet[tuple[str, torch.Size, torch.Size]],
    missing_keys: MutableSet[str],
    misc: MutableMapping[str, Any],
    distributed_operation: Optional[TensorParallelLayer],
):
    with log_to_misc(layer_name, misc, layer_name):
        module_path, _, param_name = layer_name.rpartition(".")
        module_obj = model.get_submodule(module_path) if module_path else model
        param_value = param_value[0] if isinstance(param_value, list) else param_value[...]
        ref = meta_model_state_dict.get(layer_name, empty_param)
        use_dtensor = hasattr(distributed_operation, "use_dtensor") and distributed_operation.use_dtensor
        if not isinstance(param_value, torch.nn.Parameter):
            if distributed_operation is not None and use_dtensor:
                param_value = DTensor.from_local(
                    param_value,
                    distributed_operation.device_mesh,
                    distributed_operation.shard,
                    run_check=False,
                    shape=ref.size(),
                    stride=ref.stride(),
                )
            else:
                pass  # TODO for "local" stuff, it will trigger missmatched no?
            param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())

        if ref is not None and ref.shape != param_value.shape:
            mismatch_keys.add((layer_name, param_value.shape, ref.shape))
        missing_keys.discard(layer_name)
        setattr(module_obj, param_name, param_value)


class SkipLayer(Exception):
    """Control-flow sentinel: abort processing of the current layer only."""

    pass


def convert_and_load_state_dict_in_model(
    model,
    state_dict,
    weight_mapping,
    tp_plan,
    quantizer,
    dtype=torch.float32,
    device_map=None,
    keep_in_dtype=None,
    device_mesh=None,
    profile: bool = False,
):
    """
    Convert a state dict according to a weight mapping (one WeightConverter per glob pattern),
    collecting tensors per *layer instance* (the concrete indices captured from '*').
    """
    from .modeling_utils import str_to_torch_dtype

    prefix = model.base_model_prefix
    tp_plan = tp_plan or {}  # {glob_pattern: plan_obj_or_key}
    device_map = device_map or {}  # {exact_target_key: device}
    keep_in_dtype = keep_in_dtype or {}  # {glob_pattern: dtype}
    weight_mapping = weight_mapping or {}  # {glob_pattern: WeightConverter}
    meta_model_state_dict = model.state_dict()
    missing_keys = set(meta_model_state_dict.keys())

    if isinstance(model._tied_weights_keys, list):
        for k in model._tied_weights_keys:
            missing_keys.discard(k)

    misc = {}
    mismatch_keys = set()
    unexpected_keys = set()
    # Global thread_poolutor + per-file semaphores: allow lock only upon 4 file access? Should be tensor get_shape dependant?
    thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)
    _file_semaphore = defaultdict(lambda: threading.Semaphore(PER_FILE_LIMIT))

    _patterns = list(itertools.chain.from_iterable([k.source_keys for k in weight_mapping]))
    source_to_target = {sk: k for k in weight_mapping for sk in k.source_keys}
    weight_pattern_alt, weight_pattern_by_group_name = build_glob_alt(_patterns)
    tp_plan_alt, tp_plan_by_group_name = build_glob_alt(list(tp_plan.keys()))
    dtype_policy_alt, dtype_policy_by_group_name = build_glob_alt(list(keep_in_dtype.keys()))

    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    # 1. Create the conversion entries
    by_conversion_pattern: dict[str, ConversionEntry] = {}
    for original_key, (file_id, tensor) in state_dict:
        matched_pattern = match_glob(original_key, weight_pattern_alt, weight_pattern_by_group_name)
        if matched_pattern is not None:
            converter = source_to_target[matched_pattern]  # TODO make sure its the ref
            sub_with_extractor = partial(re.sub, _glob_to_regex_src(matched_pattern), string=original_key)
            entry_key = "|".join(converter.target_keys)
            target_key = "|".join(map(sub_with_extractor, [k.replace("*", "\\1") for k in converter.target_keys]))
            entry: ConversionEntry = by_conversion_pattern.setdefault(entry_key, ConversionEntry(converter))
            converter_key = sub_with_extractor(matched_pattern)
        else:
            converter = WeightConverter(original_key)
            converter_key = entry_key = target_key = original_key
            entry = by_conversion_pattern.setdefault(converter_key, ConversionEntry(converter))

        new_target_key = []
        for t in target_key.split("|"):  # let's correct the keys
            if t.startswith(prefix) and meta_model_state_dict.get(t.replace(f"{prefix}.", "")) is not None:
                t = t.replace(f"{prefix}.", "")
            elif meta_model_state_dict.get(f"{prefix}.{t}") is not None:
                t = f"{prefix}.{t}"
            new_target_key.append(t)
        target_key = "|".join(new_target_key)

        for t in target_key.split("|"):
            empty_param = meta_model_state_dict.get(t)
            if empty_param is None:
                unexpected_keys.add(t)
                continue
            if quantizer is not None and quantizer.param_needs_quantization(model, t):
                if quantizer.__class__.__name__ == "FineGrainedFP8HfQuantizer":
                    from .integrations.finegrained_fp8 import Fp8Quantize

                    converter.quantization_operation = Fp8Quantize()  # TODO support other methods
                else:
                    raise ValueError("This quantization method is gonna be supported SOOOON")
            else:
                matched_dtype_pattern = match_glob(t, dtype_policy_alt, dtype_policy_by_group_name)
                if matched_dtype_pattern is not None:
                    dtype = keep_in_dtype[matched_dtype_pattern]
                tensor_dtype = (
                    tensor.dtype if isinstance(tensor, torch.Tensor) else str_to_torch_dtype[tensor.get_dtype()]
                )
                if dtype != tensor_dtype and dtype is not None:
                    converter.operations.append(Cast(dtype))

        first_target_key = target_key.split("|")[0]
        future = None
        if device_mesh:
            if matched_tp_pattern := match_glob(first_target_key, tp_plan_alt, tp_plan_by_group_name):
                empty_param = meta_model_state_dict.get(first_target_key)
                if getattr(converter, "distributed_operation", {}) is None:
                    tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]].__class__
                    converter.distributed_operation = tp_layer(
                        device_mesh=device_mesh, rank=device_map[""].index, empty_param=empty_param.clone()
                    )
                    # VERY IMPORTANT: this tells us wether we collected stuffs or not.
                shard_index = len(entry.collected_tensors[target_key].get(converter_key, []))
                future = spawn_tp_materialize(
                    thread_pool,
                    _file_semaphore,
                    file_id,
                    tensor,
                    converter.distributed_operation,
                    shard_index,
                )

        if future is None:  # If not TP, async materialize the tensors. TODO probably need a check for To() op.
            future = spawn_materialize(thread_pool, _file_semaphore, file_id, tensor)
        entry.collected_tensors[target_key].setdefault(converter_key, []).append(future)

    # 2. Actually convert the ckpt
    inverse_converters = {}
    keys = list(by_conversion_pattern.keys())
    total_layers = sum(len(by_conversion_pattern[key].collected_tensors) for key in keys)
    progress_bar = logging.tqdm(total=total_layers, desc="Converting weights", leave=False) if total_layers else None

    for key in keys[::-1]:  # revert to process simple keys first
        group = by_conversion_pattern.pop(key)
        converter = group.weight_converter
        operations = converter.operations if isinstance(converter.operations, list) else [converter.operations]
        for layer_name, tensors_for_this_layer in group.collected_tensors.items():
            concrete_target_keys = layer_name.split("|")
            try:
                if bool(set(concrete_target_keys) - unexpected_keys):
                    with log_to_misc(layer_name, misc):
                        values = [[k.result() for k in inner] for inner in tensors_for_this_layer.values()]

                    for op in operations:
                        with log_to_misc(layer_name, misc, (values, concrete_target_keys), operations):
                            values = op.convert(values, model.config)

                    values = [values] if not isinstance(values, list) else values
                    with log_to_misc(layer_name, misc, (values, concrete_target_keys), operations):
                        realized_value = {
                            k: t for k, t in zip(concrete_target_keys, values) if k not in unexpected_keys
                        }

                    for k in list(realized_value.keys()).copy():
                        if op := converter.quantization_operation:
                            with log_to_misc(layer_name, misc, op=op):
                                realized_value.update(
                                    op.convert({k: realized_value.pop(k)}, quant_config=quantizer.quantization_config)
                                )

                    if progress_bar is not None:
                        progress_bar.set_postfix_str(layer_name, refresh=False)
                        progress_bar.update()

                    for k, output_value in realized_value.items():
                        for src in converter.source_keys:  # what should happen to k when we meet k at saving
                            inverse_converters[k] = {src: converter}
                        set_param_for_module(
                            model,
                            k,
                            output_value,
                            meta_model_state_dict,
                            empty_param,
                            mismatch_keys,
                            missing_keys,
                            misc,
                            converter.distributed_operation,
                        )
            except SkipLayer:
                continue
        del group
        for op in operations:
            op.clear_cache()
    if progress_bar is not None:
        progress_bar.close()
    model.inverse_converters = inverse_converters
    thread_pool.shutdown(wait=True)
    return missing_keys, unexpected_keys, mismatch_keys, misc


# TODO this is not done yet!
def revert_weight_conversion(model, state_dict):
    mapping = getattr(model, "", {})  # IDK why but setting this will fail all llava.
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
