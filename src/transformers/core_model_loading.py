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

import itertools
import math
import os
import re
import threading
import time
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Union
from torch.distributed.tensor import DTensor

import torch
from torch import Tensor

from .integrations.tensor_parallel import ALL_PARALLEL_STYLES
from .utils import logging


logger = logging.get_logger(__name__)

try:
    _FP8_DTYPE = torch.float8_e4m3fn
    _FP8_MIN = torch.finfo(_FP8_DTYPE).min
    _FP8_MAX = torch.finfo(_FP8_DTYPE).max
    _FP8_IS_INT = False
except AttributeError:
    _FP8_DTYPE = torch.int8
    _FP8_MIN, _FP8_MAX = -127, 127
    _FP8_IS_INT = True
    logger.warning_once(
        "torch.float8_e4m3fn not available; falling back to int8 emulation for Fp8Quantize operations."
    )

try:
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile
except (ImportError, AttributeError):
    ProfilerActivity = None
    torch_profile = None


def _glob_to_regex_src(glob: str, *, digits_only: bool = True) -> str:
    """
    Convert a glob with '*' into a regex *source* string.
    '*' matches (\\d+) if digits_only else (.+). Inner groups are non-capturing.
    """
    star = r"(\d+)" if digits_only else r"(.+)"
    return re.escape(glob).replace(r"\*", star)


def build_glob_alt(
    globs: list[str],
    *,
    digits_only: bool = True,
    allow_prefix: bool = True,
) -> tuple[re.Pattern, dict[str, str]]:
    """
    Build one compiled regex alternation with a named group per glob.
    - digits_only: '*' => digits only (\\d+) if True, else any chars (.+)
    - allow_prefix: if True, allow arbitrary prefix before the pattern
                    (keeps '$' so we still require a full suffix match)
    Returns (compiled_regex, name->glob map).
    """
    name_map: dict[str, str] = {}
    parts: list[str] = []

    # If we keep using .match(), we must handle prefix allowance in the pattern itself.
    prefix_src = r".*" if allow_prefix else r"^"

    for i, g in enumerate(globs):
        name = f"g{i}"
        name_map[name] = g
        pat_src = _glob_to_regex_src(g, digits_only=digits_only)
        # Each branch is fully wrapped and uniquely named.
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


def glob_to_re(glob: str, *, digits_only: bool = True, allow_prefix: bool = True) -> str:
    """
    Build a regex for a single glob that captures each '*' so we can extract per-layer identifiers.
    """
    star = r"\d+" if digits_only else r".+"
    src = glob.replace("*", star)
    return rf"{src}"


def _apply_star_subst(pattern: str, star_values: list[str]) -> str:
    """
    Replace each '*' in 'pattern' with the next value from 'star_values' (in order).
    """
    it = iter(star_values)
    out = []
    for ch in pattern:
        if ch == "*":
            out.append(str(next(it)))
        else:
            out.append(ch)
    return "".join(out)


class ConversionOps:
    """Base class for weight conversion operations.

    If you chain operations, they need to be ordered properly.
    Some flags will help. Probably "typing" them ( TP op, Quant OP, Other OP)?

    Tricky part is you can go from

    model.layers.0.a                      -> [model.layers.0.a | model.layers.0.b]  # ex: chunk when saving, or quantization
    [model.layers.0.a | model.layers.0.b] ->          model.layers.0.a
    model.layers.0.a                      ->          model.layers.0.b

    and before everything, you have to do the renaming!
    1. weight rename (because the tp plan will be defined only for the renamed weights)
      -> you get many keys with the same tensor
      -> use default dict list
    """

    # Reusable scratch buffer to avoid reallocations.
    _buffer: Optional[torch.Tensor] = None
    # The inverse operation class, will be used when saving the checkpoint
    _inverse_op: type[ConversionOps]
    # Latest runtime/profiling information for introspection.
    last_runtime_seconds: Optional[float] = None
    last_profile_summary: Optional[str] = None

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
    def convert(self, value: Union[Sequence[torch.Tensor], torch.Tensor], *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        value: Union[Sequence[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]],
        *,
        profile: bool = False,
    ) -> Any:
        """
        Execute the conversion while measuring runtime and optionally profiling the call.
        """
        start = time.perf_counter()
        result = self.convert(value)
        elapsed = time.perf_counter() - start
        if profile:
            print(elapsed)
        return result


class Chunk(ConversionOps):
    """Split a tensor along ``dim`` into equally sized chunks or using explicit ``sizes``."""

    _inverse_op: type[ConversionOps]

    def __init__(self, dim: int = 0, chunks: Optional[int] = None, sizes: Optional[Sequence[int]] = None):
        if chunks is None and sizes is None:
            raise ValueError("`chunks` or `sizes` must be provided for Chunk operations.")
        if chunks is not None and chunks <= 0:
            raise ValueError("`chunks` must be a strictly positive integer.")
        self.dim = dim
        self.chunks = chunks
        self.sizes = list(sizes) if sizes is not None else None
        self._inverse_op = Concatenate

    def convert(self, value: torch.Tensor) -> list[torch.Tensor]:
        if not isinstance(value, torch.Tensor):
            raise TypeError("Chunk expects a torch.Tensor as input.")
        if self.sizes is not None:
            return list(torch.split(value, self.sizes, dim=self.dim))
        return list(torch.chunk(value, self.chunks, dim=self.dim))


class Concatenate(ConversionOps):
    """Concatenate tensors along `dim` using a reusable buffer."""

    _inverse_op: type[ConversionOps]

    def __init__(self, dim: int = 0):
        self.dim = dim
        self._inverse_op = Chunk

    @torch.no_grad
    def convert(self, value: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(value[0], list):
            value = [v[0] for v in value]
        tensors = value
        if not tensors:
            raise ValueError("Fuse requires at least one tensor to concatenate.")

        out_shape = list(tensors[0].shape)
        out_shape[self.dim] = sum([t.size(self.dim) for t in tensors])

        with torch.no_grad():  # we use staging buffers
            out = self._ensure_buffer(torch.Size(out_shape), dtype=tensors[0].dtype, device=tensors[0].device)
            torch.cat(tuple(tensors),dim =self.dim, out=out)
            # offset = 0
            # for tensor in tensors:
            #     index = [slice(None)] * tensor.ndim
            #     index[self.dim] = slice(offset, offset + tensor.shape[self.dim])
            #     out[tuple(index)].copy_(tensor, non_blocking=tensor.is_cuda)
            #     offset += tensor.shape[self.dim]
        # torch.testing.assert_close(out, torch.cat(value, dim=self.dim))
        return out.clone()  # need to say I can overwrite this storage now


class MergeModulelist(Concatenate):
    """
    Merge a list of tensors into a single tensor along the first dimension.
    We explicitly define this because for EP or TP you want to make sure you know what you are doing!

    """

    def __init__(self, dim: int = 0):
        super().__init__(dim=dim)
        self._inverse_op = SplitModulelist

    def convert(self, value: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        merged = []
        with torch.no_grad():  # we use staging buffers
            for group in value:
                if not isinstance(group, Sequence) or len(group) == 0:
                    raise ValueError("MergeModulelist requires non-empty sub-sequences.")
                group = [k for k in group if k.ndim]
                out_shape = list(group[0].shape)
                out_shape.insert(self.dim, len(group))
                out = self._ensure_buffer(torch.Size(out_shape), dtype=group[0].dtype, device=group[0].device)
                # torch.stack(tuple(group), dim=self.dim, out=out)
                for off, tensor in enumerate(group):
                    out[off].copy_(tensor, non_blocking=tensor.is_cuda)
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
        self._inverse_op = MergeModulelist

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

    def convert(self, realized_value):
        return realized_value.to(self.dtype)


class To(ConversionOps):
    """
    Transfers the tensor to the provided device potentially using a stream?

    if param_device == "disk":
        if not is_safetensors:
            disk_offload_index = offload_weight(param, param_name, disk_offload_folder, disk_offload_index)
    elif not is_quantized or not hf_quantizer.param_needs_quantization(model, param_name):
        if is_fsdp_enabled():
            param_device = "cpu" if is_local_dist_rank_0() else "meta"
    """

    def __init__(self, device):
        self.device = device

    def convert(self, realized_value: list[list[PySafeSlice]]):
        with torch.device(self.device):
            out = [[x[:] for x in inner] if isinstance(inner, list) else inner[:] for inner in realized_value]
        return out


class DistributedOp(ConversionOps):  # all `distributed_operation` need to respect this
    pass


class Shard(DistributedOp):
    """Shard tensors along a specific dimension.

    The operation supports two modes:

    - ``return_all=False`` (default): behaves like classical tensor parallel sharding and returns only the shard for the
      current ``rank``.
    - ``return_all=True``: returns a list containing the shards for all ranks. This mode is handy when the conversion
      needs to materialize every shard in a single pass (for instance when round-tripping in tests).
    """

    _inverse_op: type[ConversionOps] = Concatenate

    def __init__(
        self,
        dim: int,
        *,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        return_all: bool = False,
    ):
        self.dim = dim
        self.world_size = world_size
        self.rank = rank
        self.return_all = return_all

    def convert(self, value: Union[Tensor, Sequence], *, context: dict[str, Any]) -> Union[Tensor, list[Tensor]]:
        """
        This is akin to a normal sharding, BUT we handle a list of tensor inputs (which are gonna be merged later on)
        """

        def _shard_tensor(tensor: Tensor, rank: int) -> Tensor:
            dim_size = tensor.shape[self.dim]
            local_world_size = max(world_size, 1)
            slice_size = math.ceil(dim_size / local_world_size)
            start = min(rank * slice_size, dim_size)
            end = min(start + slice_size, dim_size)
            index = [slice(None)] * tensor.ndim
            index[self.dim] = slice(start, end)
            return tensor[tuple(index)]

        world_size = self.world_size or context.get("tp_world_size") or 1
        rank = self.rank if self.rank is not None else context.get("tp_rank", 0)

        if isinstance(value, torch.Tensor):
            if self.return_all and world_size > 1:
                return [_shard_tensor(value, r) for r in range(world_size)]
            return _shard_tensor(value, rank)

        if isinstance(value, (list, tuple)):
            return [self.convert(item, context=context) for item in value]

        if isinstance(value, dict):
            return {k: self.convert(v, context=context) for k, v in value.items()}

        raise TypeError("Shard only supports tensors, sequences of tensors or dicts of tensors.")


class QuantizationOp(ConversionOps):
    """Base class for quantization operations."""

    pass


class Fp8Quantize(QuantizationOp):
    """
    A quantization operation that creates two tensors, weight and scale out of a weight.
    """

    _inverse_op: type[ConversionOps]

    def __init__(self, block_size: Optional[tuple[int, int]] = None):
        self.block_size = block_size
        self._inverse_op = Fp8Dequantize

    def convert(self, input_dict: torch.Tensor, *, quant_config: dict[str, Any]) -> dict[str, torch.Tensor]:
        # Unpack single key/value (value may be wrapped in a list)
        target_keys, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        # Resolve block size (support dict-like or attr-like quant_config)
        block_size = None
        if quant_config is not None:
            if isinstance(quant_config, dict):
                block_size = quant_config.get("weight_block_size")
            else:
                block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is None:
            block_size = (value.shape[-2], value.shape[-1])

        block_m, block_n = block_size
        rows, cols = value.shape[-2], value.shape[-1]

        # Enforce exact tiling like your original
        if rows % block_m != 0 or cols % block_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n}). for {target_keys}"
            )

        # Leading dims can be empty (2D) or include num_experts/... (3D+)
        leading_shape = value.shape[:-2]
        rows_tiles = rows // block_m
        cols_tiles = cols // block_n

        original_shape = value.shape
        value_fp32 = value.to(torch.float32)

        # Reshape to (..., rows_tiles, block_m, cols_tiles, block_n)
        reshaped = value_fp32.reshape(*leading_shape, rows_tiles, block_m, cols_tiles, block_n)

        # Per-tile max-abs over the block dims
        # dims: block_m is at -3, block_n is at -1 after the reshape
        max_abs = reshaped.abs().amax(dim=(-3, -1))
        safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))

        # Tile scale (we store inverse scale like your Linear: weight_scale_inv)
        scales = _FP8_MAX / safe_max_abs
        scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))  # keep zeros stable

        # Broadcast scales back over the block dims and quantize
        # max_abs/scales shape: (..., rows_tiles, cols_tiles)
        scales_broadcast = scales.unsqueeze(-1).unsqueeze(-3)  # -> (..., rows_tiles, 1, cols_tiles, 1)
        scaled = reshaped * scales_broadcast

        if _FP8_IS_INT:
            quantized = torch.clamp(scaled.round(), min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        else:
            quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)

        quantized = quantized.reshape(original_shape)

        inv_scales = (1.0 / scales).to(torch.float32)  # shape: (*leading, rows_tiles, cols_tiles)
        if target_keys.endswith("weight"):
            scale_key = target_keys.rsplit(".", 1)[0] + ".weight_scale_inv"
        else:
            scale_key = target_keys + "_scales_inv"

        # Return both quantized weights and per-tile inverse scales (keeps leading dims, e.g., num_experts)
        return {
            target_keys: quantized,
            scale_key: inv_scales,
        }


class Fp8Dequantize(QuantizationOp):
    """Inverse operation of :class:`Fp8Quantize`. Takes a pair (weight, scale) and reconstructs the fp32 tensor."""

    def __init__(self, block_size: Optional[tuple[int, int]] = None):
        self.block_size = block_size
        self._inverse_op = Fp8Quantize

    def convert(
        self,
        value: Union[Sequence[torch.Tensor], dict[str, torch.Tensor]],
        *,
        context: dict[str, Any],
    ) -> torch.Tensor:
        if isinstance(value, dict):
            tensors = list(value.values())
        else:
            tensors = list(value) if isinstance(value, Sequence) else [value]
        if len(tensors) != 2:
            raise ValueError("Fp8Dequantize expects exactly two tensors: quantized weights and scales.")
        quantized, scales = tensors
        if not isinstance(quantized, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise TypeError("Fp8Dequantize expects tensors as inputs.")

        quantized_fp32 = quantized.to(torch.float32)
        rows, cols = quantized_fp32.shape[-2:]
        block_size = self.block_size
        if block_size is None:
            quant_config = context.get("quantization_config")
            block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is None:
            block_size = (rows, cols)
        block_m, block_n = block_size
        if rows % block_m != 0 or cols % block_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n})."
            )

        reshaped = quantized_fp32.reshape(-1, rows // block_m, block_m, cols // block_n, block_n)
        expanded_scales = scales.to(torch.float32).reshape(-1, rows // block_m, cols // block_n)
        expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)
        dequantized = reshaped * expanded_scales
        return dequantized.reshape(quantized_fp32.shape)


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

    distributed_operation: dict[str, ConversionOps] = field(default_factory=dict, compare=False, repr=False)
    quantization_operation: dict[str, ConversionOps] = field(default_factory=dict, compare=False, repr=False)
    _compiled: tuple[tuple[str, re.Pattern], ...] = field(default_factory=tuple, compare=False, repr=False)
    _regex_pat: tuple[re.Pattern, dict[str, str]] = field(default_factory=tuple, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.source_keys, list):
            self.source_keys = [self.source_keys]
        if not isinstance(self.target_keys, list):
            if self.target_keys is None:
                self.target_keys = self.source_keys
            else:
                self.target_keys = [self.target_keys]
        self._regex_pat = build_glob_alt(self.source_keys)


def set_param_for_module(model, k, v, meta_model_state_dict, empty_tensor, mismatch_keys, missing_keys, misc, distributed_operation):
    try:
        module_path, _, param_name = k.rpartition(".")
        module_obj = model.get_submodule(module_path) if module_path else model
        param_value = v[0] if isinstance(v, list) else v[:]
        ref = meta_model_state_dict.get(k, empty_tensor)
        use_dtensor = hasattr(distributed_operation, "use_dtensor") and distributed_operation.use_dtensor
        if not isinstance(param_value, torch.nn.Parameter):
            if distributed_operation != {} and use_dtensor:
                param_value = DTensor.from_local(
                    param_value, distributed_operation.device_mesh, distributed_operation.shard, run_check=False, shape=ref.size(), stride=ref.stride()
                )
            else:
                pass # TODO for "local" stuff, it will trigger missmatched no?
            param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())

        if ref is not None and ref.shape != param_value.shape:
            mismatch_keys.add((k, param_value.shape, ref.shape))
        if k in missing_keys:
            missing_keys.remove(k)

        setattr(module_obj, param_name, param_value)
    except Exception as e:
        misc[k] = f"{e} for {k} on {list(module_obj.state_dict().keys())}"


@dataclass(slots=True)
class ConversionEntry:
    weight_converter: WeightConverter
    collected_tensors: dict = field(default_factory=lambda: defaultdict(dict))


# Tune these to your storage:
GLOBAL_WORKERS = min(32, (os.cpu_count() or 8) * 2)  # NVMe: 8-16; HDD/NFS: 2-4
PER_FILE_LIMIT = 4  # concurrent reads per file



def _materialize_copy(x):
    # PyTorch: this runs in C and releases the GIL; good for threads.
    return x[:] #.contiguous()  needed????

def spawn_materialize(EXEC, _file_sems, file_id, t) -> Future:
    sem = _file_sems[file_id]
    def _job():
        with sem:
            return _materialize_copy(t)

    return EXEC.submit(_job)


def spawn_tp_materialize(EXEC, _file_sems, file_id, t, sharding_method, empty_tensor, tensor_idx) -> Future:
    sem = _file_sems[file_id]

    def _job():
        with sem:
            return sharding_method.shard_tensor(t, empty_tensor, tensor_idx=tensor_idx)[0]

    return EXEC.submit(_job)


def convert_and_load_state_dict_in_model(
    model,
    state_dict,
    weight_mapping,
    tp_plan,
    quantizer,
    device_map=None,
    keep_in_dtype=None,
    device_mesh=None,
    profile: bool = False,
):
    """
    Convert a state dict according to a weight mapping (one WeightConverter per glob pattern),
    collecting tensors per *layer instance* (the concrete indices captured from '*').
    """
    tp_plan = tp_plan or {}  # {glob_pattern: plan_obj_or_key}
    device_map = device_map or {}  # {exact_target_key: device}
    keep_in_dtype = keep_in_dtype or {}  # {glob_pattern: dtype}
    weight_mapping = weight_mapping or {}  # {glob_pattern: WeightConverter}
    meta_model_state_dict = model.state_dict()
    missing_keys = set(meta_model_state_dict.keys())
    if model.config.tie_word_embeddings:
        missing_keys.remove("lm_head.weight")

    misc = {}
    mismatch_keys = set()
    unexpected_keys = set()
    # Global executor + per-file semaphores
    EXEC = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)
    _file_sems = defaultdict(lambda: threading.Semaphore(PER_FILE_LIMIT))

    _patterns = list(itertools.chain.from_iterable([k.source_keys for k in weight_mapping]))
    source_to_target = {sk: k for k in weight_mapping for sk in k.source_keys}
    weight_pattern_alt, weight_pattern_by_group_name = build_glob_alt(_patterns)
    tp_plan_alt, tp_plan_by_group_name = build_glob_alt(list(tp_plan.keys()))
    dtype_policy_alt, dtype_policy_by_group_name = build_glob_alt(list(keep_in_dtype.keys()))

    # 1. Create the conversion entries
    by_conversion_pattern: dict[str, ConversionEntry] = {}
    for original_key, (file_id, tensor) in state_dict.items():
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

        first_target_key = target_key.split("|")[0]
        fut = None
        if device_mesh:
            if matched_tp_pattern := match_glob(first_target_key, tp_plan_alt, tp_plan_by_group_name):
                empty_tensor = meta_model_state_dict.get(first_target_key)
                if getattr(converter, "distributed_operation", {}) == {}:
                    converter.distributed_operation = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]]
                    converter.distributed_operation.device_mesh = device_mesh
                    converter.distributed_operation.rank = device_map[""].index
                    converter.distributed_operation.empty_tensor = empty_tensor.clone()
                shard_index=len(entry.collected_tensors[target_key].get(converter_key, []))
                fut = spawn_tp_materialize(EXEC, _file_sems, file_id, tensor, converter.distributed_operation, empty_tensor, shard_index)

        if fut is None:  # If not TP, async move tensors
            fut = spawn_materialize(EXEC, _file_sems, file_id, tensor)

        entry.collected_tensors[target_key].setdefault(converter_key, []).append(fut)
        for t in target_key.split("|"):
            empty_tensor = meta_model_state_dict.get(t)
            if empty_tensor is None:
                unexpected_keys.add(t)
                continue
            if quantizer is not None and quantizer.param_needs_quantization(model, t):
                # converter.quantization_operation[target_key] = quantizer.quantize_tensor
                converter.quantization_operation[t] = Fp8Quantize()

    # 2. Actually convert the ckpt
    inverse_converters = {}
    keys = list(by_conversion_pattern.keys())
    total_layers = sum(len(by_conversion_pattern[key].collected_tensors) for key in keys)
    progress_bar = logging.tqdm(total=total_layers, desc="Converting weights", leave=False) if total_layers else None

    try:
        for key in keys:
            group = by_conversion_pattern.pop(key)
            converter = group.weight_converter
            operations = converter.operations if isinstance(converter.operations, list) else [converter.operations]
            for layer_name, tensors_for_this_layer in group.collected_tensors.items():
                concrete_target_keys = layer_name.split("|")
                if bool(set(concrete_target_keys) - unexpected_keys):
                    values = [[k.result() for k in inner] for inner in tensors_for_this_layer.values()]

                    if op := converter.distributed_operation:
                        try:
                            values = op(values)
                        except Exception as e:
                            misc[layer_name] = f"Failed to apply {converter.distributed_operation.__class__.__name__}: {e}"
                            continue

                    for op in operations:
                        try:
                            values = op(values)
                        except Exception as e:
                            misc[layer_name] = (
                                f"{e}\nError: {op.__class__.__name__} on tensors collected from {converter.source_keys}. Ckpt contains: {values}"
                            )

                    values = [values] if not isinstance(values, list) else values
                    realized_value = {k: t for k, t in zip(concrete_target_keys, values) if k not in unexpected_keys}

                    for k in list(realized_value.keys()).copy():
                        if op := converter.quantization_operation.get(k):
                            try:
                                realized_value.update(
                                    op.convert({k: realized_value.pop(k)}, quant_config=quantizer.quantization_config)
                                )
                            except Exception as e:
                                misc[layer_name] = f"{op.__class__.__name__}: {e}"

                    if progress_bar is not None:
                        progress_bar.set_postfix_str(layer_name, refresh=False)
                        progress_bar.update()

                    for k, output_value in realized_value.items():
                        matched_dtype_pattern = match_glob(k, dtype_policy_alt, dtype_policy_by_group_name)
                        if matched_dtype_pattern is not None:
                            op = Cast(keep_in_dtype[matched_dtype_pattern])
                            output_value = op(output_value)

                        for src in converter.source_keys:  # what should happen to k when we meet k at saving
                            inverse_converters[k] = {src: converter}
                        set_param_for_module(
                            model, k, output_value, meta_model_state_dict, empty_tensor, mismatch_keys, missing_keys, misc, converter.distributed_operation
                        )

            del group
            for op in operations:
                op.clear_cache()
    finally:
        pass
        # if progress_bar is not None:
        #     progress_bar.close()
    model.inverse_converters = inverse_converters
    # EXEC.shutdown(wait=True)
    return missing_keys, unexpected_keys, mismatch_keys, misc


# TODO this is not done yet!
def revert_weight_conversion(model, state_dict):
    reverse_key_mapping = model.inverse_converters
    original_state_dict = {}
    for key, value in state_dict.items():
        for pattern, inverse_converter in reverse_key_mapping.items():
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
