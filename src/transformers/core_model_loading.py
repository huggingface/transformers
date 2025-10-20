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

import math
import re
import time
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import itertools
from typing import Any, Optional, Union

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
    star = r"(?:\d+)" if digits_only else r"(?:.+)"
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
        parts.append(f"(?P<{name}>{prefix_src}{pat_src}$)")

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


def _compile_single_glob_for_extract(glob: str, *, digits_only: bool = True, allow_prefix: bool = True) -> str:
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
        context: dict[str, Any],
        profile: bool = False,
    ) -> Any:
        """
        Execute the conversion while measuring runtime and optionally profiling the call.
        """

        profiling_enabled = bool(profile)
        profiler_ctx = nullcontext()

        if profiling_enabled:
            if torch_profile is None or ProfilerActivity is None:
                logger.warning_once(
                    "torch.profiler is unavailable; skipping profiling for %s operations.",
                    self.__class__.__name__,
                )
                profiling_enabled = False
            else:
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                profiler_ctx = torch_profile(activities=activities, record_shapes=True, profile_memory=True)

        start = time.perf_counter()
        with profiler_ctx as prof:
            result = self.convert(value, context=context)
        elapsed = time.perf_counter() - start

        # Store the latest runtime for downstream consumers.
        self.last_runtime_seconds = elapsed

        logger.info("%s convert() finished in %.2f ms", self.__class__.__name__, elapsed * 1000)

        if profiling_enabled and prof is not None:
            try:
                summary = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
            except Exception as error:
                logger.warning(
                    "Failed to render profiler summary for %s due to %s.",
                    self.__class__.__name__,
                    error,
                )
            else:
                self.last_profile_summary = summary
                logger.info("Profiler summary for %s:\n%s", self.__class__.__name__, summary)

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

    def convert(self, value: torch.Tensor, *, context: dict[str, Any]) -> list[torch.Tensor]:
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

    def convert(self, value: Sequence[torch.Tensor], *, context: dict[str, Any]) -> torch.Tensor:
        tensors = tuple(value)
        if not tensors:
            raise ValueError("Fuse requires at least one tensor to concatenate.")

        out_shape = list(tensors[0].shape)
        out_shape[self.dim] *= len(tensors)

        with torch.no_grad():
            out = self._ensure_buffer(torch.Size(out_shape), dtype=tensors[0].dtype, device=tensors[0].device)
            offset = 0
            for tensor in tensors:
                index = [slice(None)] * tensor.ndim
                index[self.dim] = slice(offset, offset + tensor.shape[self.dim])
                out[tuple(index)].copy_(tensor, non_blocking=tensor.is_cuda)
                offset += tensor.shape[self.dim]
        return out


class MergeModulelist(Concatenate):
    """
    Merge a list of tensors into a single tensor along the first dimension.
    We explicitly define this because for EP or TP you want to make sure you know what you are doing!

    """

    def __init__(self, dim: int = 0):
        super().__init__(dim=dim)
        self._inverse_op = SplitModulelist

    def convert(self, value: Sequence[Sequence[torch.Tensor]], *, context: dict[str, Any]) -> list[torch.Tensor]:
        if not isinstance(value, Sequence):
            raise TypeError("MergeModulelist expects a sequence of sequences of tensors.")
        merged: list[torch.Tensor] = []
        for group in value:
            if not isinstance(group, Sequence) or len(group) == 0:
                raise ValueError("MergeModulelist requires non-empty sub-sequences.")
            merged.append(torch.cat(tuple(group), dim=self.dim))
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
            shards = [self.convert(item, context=context) for item in value]
            return list(shards) if isinstance(value, list) else tuple(shards)

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

    def convert(self, value: torch.Tensor, *, context: dict[str, Any]) -> dict[str, torch.Tensor]:
        if not isinstance(value, torch.Tensor):
            raise TypeError("Fp8Quantize expects a tensor as input.")

        target_keys = context.get("target_keys")
        if not isinstance(target_keys, str):
            raise ValueError("Fp8Quantize requires a single string target key.")

        quant_config = context.get("quantization_config")
        block_size = self.block_size
        if block_size is None and quant_config is not None:
            block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is None:
            block_size = (value.shape[-2], value.shape[-1])

        block_m, block_n = block_size
        rows, cols = value.shape[-2:]
        if rows % block_m != 0 or cols % block_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n})."
            )

        original_shape = value.shape
        value_fp32 = value.to(torch.float32)
        reshaped = value_fp32.reshape(-1, rows // block_m, block_m, cols // block_n, block_n)
        max_abs = reshaped.abs().amax(dim=(2, 4))
        safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        scales = _FP8_MAX / safe_max_abs
        scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
        scales_reshaped = scales.unsqueeze(-1).unsqueeze(2)
        scaled = reshaped * scales_reshaped
        if _FP8_IS_INT:
            quantized = torch.clamp(scaled.round(), min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        else:
            quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        quantized = quantized.reshape(original_shape)
        inv_scales = (1.0 / scales).reshape(-1, rows // block_m, cols // block_n).to(torch.float32)

        scale_key = target_keys.rsplit(".", 1)[0] + ".scale"
        return {target_keys: quantized, scale_key: inv_scales}


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


@dataclass
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

    distributed_operation: Optional[ConversionOps] = None
    quantization_operation: Optional[ConversionOps] = None
    _operations: list[ConversionOps] = field(default_factory=list, repr=False)
    operations: list[ConversionOps] = field(default_factory=list, repr=False)

    _compiled: tuple[tuple[str, re.Pattern], ...] = field(default_factory=tuple, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.source_keys, list):
            self.source_keys = [self.source_keys]
        if not isinstance(self.target_keys, list):
            if self.target_keys is None:
                self.target_keys = self.source_keys
            else:
                self.target_keys = [self.target_keys]
        self._regex_pat = build_glob_alt(self.source_keys)
        self.operations = self._operations

    @property
    def operations(self) -> list[ConversionOps]:
        return self._operations

    @operations.setter
    def operations(self, v: Union[None, ConversionOps, list[ConversionOps]]):
        if v is None:
            self._operations = []
        elif isinstance(v, list):
            self._operations = v
        else:
            self._operations = [v]


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
    # Inputs defaulting
    tp_plan = tp_plan or {}  # {glob_pattern: plan_obj_or_key}
    device_map = device_map or {}  # {exact_target_key: device}
    keep_in_dtype = keep_in_dtype or {}  # {glob_pattern: dtype}
    weight_mapping = weight_mapping or {}  # {glob_pattern: WeightConverter}
    meta_model_state_dict = model.state_dict()

    # Fast alternations; allow prefixes (e.g., "model.model.layers..." should match "model.layers.*...")
    _patterns = list(itertools.chain.from_iterable( [k.source_keys for k in weight_mapping]))
    source_to_target = {sk: k for k in weight_mapping for sk in k.source_keys}
    weight_pattern_alt, weight_pattern_by_group_name = build_glob_alt(_patterns)
    tp_plan_alt, tp_plan_by_group_name = build_glob_alt(list(tp_plan.keys()))
    dtype_policy_alt, dtype_policy_by_group_name = build_glob_alt(list(keep_in_dtype.keys()))

    used_operations: list[ConversionOps] = []

    # We organize tensors by the conversion pattern, then by layer (captured '*' tuple)
    # by_conversion_pattern[glob_pattern] = {
    #   "conversion": WeightConverter, -> usually a single conversion needed for all layers
    #   "tensors_per_layer": { layer_indices_tuple: [tensors...] }
    # }
    by_conversion_pattern: dict[str, dict] = {}
    # ------------ First pass: decide the conversion pattern and layer indices for each key ------------
    for original_key, tensor in state_dict.items():
        matched_pattern = match_glob(original_key, weight_pattern_alt, weight_pattern_by_group_name)
        # FINE UP UNTIL HERE
        if matched_pattern is not None:
            conversion: WeightConverter = source_to_target[matched_pattern]
            extractor = _compile_single_glob_for_extract(matched_pattern)
            converter_key = re.sub(extractor, matched_pattern, original_key)
            entry = by_conversion_pattern.setdefault(
                matched_pattern, {"conversion": conversion, "tensors_per_layer": defaultdict(list)}
            )
            entry["tensors_per_layer"][converter_key].append(tensor)
        else:
            # No pattern matched -> identity conversion keyed by the exact key (no '*', single "layer" = empty tuple)
            conversion = WeightConverter(original_key)
            entry = by_conversion_pattern.setdefault(
                original_key, {"conversion": conversion, "tensors_per_layer": defaultdict(list)}
            )
            entry["tensors_per_layer"][()].append(tensor)

    missing_keys = set(meta_model_state_dict.keys())
    mismatch_keys = []
    unexpected_keys = []

    # ------------ Second pass: for each conversion pattern and each layer instance, realize outputs ------------
    for conversion_pattern, group in by_conversion_pattern.items():
        conversion: WeightConverter = group["conversion"]
        tensors_per_layer: dict[str, list[torch.Tensor]] = group["tensors_per_layer"]

        for layer_name, tensors_for_this_layer in tensors_per_layer.items():
            # Materialize concrete target keys for this specific layer instance
            target_patterns = conversion.target_keys
            concrete_target_keys = [re.sub(conversion_pattern, p, layer_name) for p in target_patterns]

            for target_key in concrete_target_keys:
                empty_tensor = meta_model_state_dict.get(target_key)
                if empty_tensor is None:
                    unexpected_keys.append(target_key)
                    continue

                # Tensor-parallel plan matching on the *concrete* target key
                matched_tp_pattern = match_glob(target_key, tp_plan_alt, tp_plan_by_group_name)
                if matched_tp_pattern is not None:
                    if getattr(conversion, "distributed_operation", None) is None:
                        conversion.distributed_operation = ALL_PARALLEL_STYLES[matched_tp_pattern].shard_tensor
                    rank = device_mesh.get_local_rank() if device_mesh is not None else 0
                    realized_value = conversion.distributed_operation(
                        tensors_for_this_layer[0],
                        context={"tp_world_size": None, "tp_rank": rank},
                    )
                else:
                    realized_value = [t[:] for t in tensors_for_this_layer]

                for op in conversion.operations:
                    realized_value = op.convert(realized_value, context={})
                    used_operations.append(op)

                # Quantization (may produce a dict of tensors)
                if quantizer is not None:
                    if getattr(conversion, "quantization_operation", None) is None:
                        conversion.quantization_operation = Fp8Quantize()
                    realized_value = conversion.quantization_operation(
                        realized_value if isinstance(realized_value, torch.Tensor) else realized_value[0],
                        context={},
                    )
                    used_operations.append(conversion.quantization_operation)

                # Device & dtype policies
                output_value = realized_value
                if target_key in device_map:
                    op = To(device_map[target_key])
                    conversion.operations.append(op)
                    output_value = op.convert(output_value, context={})
                    used_operations.append(op)

                matched_dtype_pattern = match_glob(target_key, dtype_policy_alt, dtype_policy_by_group_name)
                if matched_dtype_pattern is not None:
                    op = Cast(keep_in_dtype[matched_dtype_pattern])
                    conversion.operations.append(op)
                    output_value = op.convert(output_value, context={})
                    used_operations.append(op)

                # Install into the module
                to_install = output_value.items() if isinstance(output_value, dict) else [(target_key, output_value)]
                for install_key, value_like in to_install:
                    module_path, _, param_name = install_key.rpartition(".")
                    module_obj = model.get_submodule(module_path) if module_path else model

                    param_value = value_like
                    if not isinstance(param_value, torch.nn.Parameter):
                        param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())

                    ref = meta_model_state_dict.get(install_key, empty_tensor if install_key == target_key else None)
                    if ref is not None and ref.shape != param_value.shape:
                        mismatch_keys.append((install_key, param_value.shape, ref.shape))

                    if install_key in missing_keys:
                        missing_keys.remove(install_key)

                    setattr(module_obj, param_name, param_value)

    # Clear any cached buffers on unique ops
    for op in {op for op in used_operations if hasattr(op, "clear_cache")}:
        op.clear_cache()

    return used_operations, missing_keys, unexpected_keys, mismatch_keys
