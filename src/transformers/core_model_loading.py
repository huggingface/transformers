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
from typing import TYPE_CHECKING, Any, Optional, Union

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


def _glob_to_regex_src(glob: str, *, digits_only: bool = True) -> str:
    """
    Convert a glob with '*' into a regex *source* string. We don't use `glob.translate`
    '*' matches (\\d+) if digits_only else (.+). Inner groups are non-capturing.
    """
    star = r"(\d+)" if digits_only else r"(.+)"
    return glob.replace(r"\*", star)


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

    for i, g in enumerate(globs):
        name = f"g{i}"
        name_map[name] = g
        pat_src = _glob_to_regex_src(g)
        prefix_src = ""
        if pat_src.startswith("*"):
            prefix_src = "."
        elif not pat_src.startswith(r"\^") and not pat_src.startswith(r".*"):
            prefix_src = ".*"

        parts.append(f"(?P<{name}>{prefix_src}{pat_src}.*)")

    alt_src = "|".join(parts).replace("\\^", "^").replace("\\.", r"\.")
    try:
        reg = re.compile(alt_src)
    except re.error as e:
        logger.error(f"Error compiling regex for alternation: {alt_src}")
        raise e

    return reg, name_map


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

    # The inverse operation class, will be used when saving the checkpoint
    reverse_op: type[ConversionOps]

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
        # chunk requires a single tensor input
        if len(value) != 1 or len(value[0]) != 1:
            raise ValueError("Chunk operation requires a single tensor input.")
        return list(torch.chunk(value[0][0], self.chunks, dim=self.dim))


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

        return torch.cat(tuple(tensors), dim=self.dim)


class MergeModulelist(Concatenate):
    """
    Merge a list of tensors into a single tensor along the first dimension.
    We explicitly define this because for EP or TP you want to make sure you know what you are doing!

    """

    def __init__(self, dim: int = 0):
        super().__init__(dim=dim)
        self.reverse_op = SplitModulelist

    @torch.no_grad
    def convert(self, value: Sequence[torch.Tensor], *args, **kwargs) -> list[torch.Tensor]:
        merged = []
        for group in value:
            if not isinstance(group, Sequence) or len(group) == 0:
                raise ValueError("MergeModulelist requires non-empty sub-sequences.")
            group = [k for k in group if k.ndim]
            merged.append(torch.stack(group, dim=self.dim))
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

    TODO: for BNB we need to collect model.weight.quant_state_keys
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


@dataclass(slots=True)
class ConversionEntry:
    weight_converter: WeightConverter
    collected_tensors: dict = field(default_factory=lambda: defaultdict(dict))


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
    distributed_operation: Optional[TensorParallelLayer],
):
    with log_to_misc(layer_name, misc, layer_name):
        module_path, _, param_name = layer_name.rpartition(".")
        module_obj = model.get_submodule(module_path) if module_path else model
        param_value = param_value[0] if isinstance(param_value, list) else param_value[...]
        ref = getattr(module_obj, param_name)

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
    weight_mapping: dict[str, WeightConverter] | None,
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
    weight_mapping = weight_mapping or {}  # {glob_pattern: WeightConverter}
    meta_model_state_dict = model.state_dict()
    missing_keys = set(meta_model_state_dict.keys())

    misc = {}
    mismatch_keys = set()
    unexpected_keys = set()
    # Global thread_pool
    thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)

    _patterns = list(itertools.chain.from_iterable([k.source_keys for k in weight_mapping]))
    source_to_target = {sk: k for k in weight_mapping for sk in k.source_keys}
    weight_pattern_alt, weight_pattern_by_group_name = build_glob_alt(_patterns)
    tp_plan_alt, tp_plan_by_group_name = build_glob_alt(list(tp_plan.keys()))
    dtype_policy_alt, dtype_policy_by_group_name = build_glob_alt(list(dtype_plan.keys()))

    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    # 1. Create the conversion entries
    by_conversion_pattern: dict[str, ConversionEntry] = {}
    for original_key, tensor in state_dict:
        matched_pattern = match_glob(original_key, weight_pattern_alt, weight_pattern_by_group_name)
        if matched_pattern is not None:
            converter = source_to_target[matched_pattern]  # TODO make sure its the ref
            sub_with_extractor = partial(re.sub, matched_pattern.replace("*", r"(\d+)"), string=original_key)
            entry_key = "|".join(converter.target_keys)
            target_key = "|".join(map(sub_with_extractor, [k.replace("*", "\\1") for k in converter.target_keys]))
            entry: ConversionEntry = by_conversion_pattern.setdefault(entry_key, ConversionEntry(converter))
            converter_key = sub_with_extractor(matched_pattern)
        else:
            converter = WeightConverter(original_key)
            converter_key = entry_key = target_key = original_key
            entry = by_conversion_pattern.setdefault(converter_key, ConversionEntry(converter))

        _dtype = dtype
        new_target_key = []  # test_load_with_mismatched_shapes for AutoModel.from_pretrained(AutoForCausal, vocab=10)
        for t in target_key.split("|"):
            if t.startswith(prefix) and meta_model_state_dict.get(re.sub(f"^{prefix}.", "", t, count=1)) is not None:
                t = re.sub(f"^{prefix}.", "", t, count=1)
            elif meta_model_state_dict.get(f"{prefix}.{t}") is not None:
                t = f"{prefix}.{t}"
            new_target_key.append(t)
            empty_param = meta_model_state_dict.get(t)
            # If it does not exist, it's unexpected
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
                _dtype = dtype
                matched_dtype_pattern = match_glob(t, dtype_policy_alt, dtype_policy_by_group_name)
                if matched_dtype_pattern is not None:
                    _dtype = dtype_plan[matched_dtype_pattern]
                elif empty_param.dtype != _dtype:
                    _dtype = empty_param.dtype

        first_target_key = new_target_key[0]
        target_key = "|".join(new_target_key)

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
                    tensor,
                    _dtype,
                    converter.distributed_operation,
                    shard_index,
                )

        if future is None:  # If not TP, async materialize the tensors. TODO handle disk offload?
            future = spawn_materialize(thread_pool, tensor, _dtype)
        entry.collected_tensors[target_key].setdefault(converter_key, []).append(future)

    # 2. Actually convert the ckpt
    inverse_converters = {}
    keys = list(by_conversion_pattern.keys())

    with logging.tqdm(total=len(keys), desc="Loading weights") as pbar:
        for key in keys[::-1]:  # revert to process simple keys first
            group = by_conversion_pattern.pop(key)
            converter = group.weight_converter
            operations = converter.operations if isinstance(converter.operations, list) else [converter.operations]
            for layer_name, tensors_for_this_layer in group.collected_tensors.items():
                pbar.update(1)
                pbar.set_postfix({"Materializing param": layer_name})
                pbar.refresh()
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
                                        op.convert(
                                            {k: realized_value.pop(k)}, quant_config=quantizer.quantization_config
                                        )
                                    )

                        for k, output_value in realized_value.items():
                            for src in converter.source_keys:  # what should happen to k when we meet k at saving
                                inverse_converters[k] = {src: converter}
                            set_param_for_module(
                                model,
                                k,
                                output_value,
                                mismatch_keys,
                                missing_keys,
                                misc,
                                converter.distributed_operation,
                            )

                except SkipLayer:
                    continue
            del group

    model.inverse_converters = inverse_converters
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
