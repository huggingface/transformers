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
import re

import math
import time
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from fnmatch import fnmatchcase
from itertools import chain
from typing import Any, Optional, Union

import torch
from torch import Tensor

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
    from torch.profiler import ProfilerActivity, profile as torch_profile
except (ImportError, AttributeError):
    ProfilerActivity = None
    torch_profile = None


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

    Case 1: Sequence[ Fuse nn list, Fuse gate and up]
    ---------------------------------------------------------------------------------
      "model.layers.0.block_sparse_moe.experts.(0, 1, ..., 7).w1.weight"
      +
      "model.layers.0.block_sparse_moe.experts.(0, 1, ..., 7).w3.weight"
      =>
      "model.layers.0.block_sparse_moe.experts.gate_up_proj.weight": [0.w1, 0.w2, ..., 7.w1, 7.w2]  if 8 experts -> Final name and tensors
    ---------------------------------------------------------------------------------

    Case 2: fuse qkv
    ---------------------------------------------------------------------------------
      "model.layers.0.self_attn.q_proj.weight"
      +
      "model.layers.0.self_attn.k_proj.weight"
      +
      "model.layers.0.self_attn.v_proj.weight"
      =>
      "model.layers.0.self_attn.qkv_proj.weight": [q, k, v]
    ---------------------------------------------------------------------------------

    Case 3: chunk
    ---------------------------------------------------------------------------------
      "model.layers.0.mlp.gate_up_proj.weight"
      =>
      "model.layers.0.mlp.gate_proj.weight"
      +
      "model.layers.0.mlp.up_proj.weight"
    ---------------------------------------------------------------------------------

    Case 4: Quantize
    ---------------------------------------------------------------------------------
      "model.layers.0.mlp.gate_up_proj.weight"
      =>
      "model.layers.0.mlp.gate_proj.blocks"
      +
      "model.layers.0.mlp.up_proj.scales"
    ---------------------------------------------------------------------------------



    ALWAYS TP FIRST !!! If we compute we compute fast locally -> communicate async.
    The set of operations that we need to support is actually not that big:

    https://github.com/cchen1436/NeMo/blob/eb5426e6d00b0d0225442d4b8ced1185dbc9a2ff/nemo/lightning/io/state.py#L511
    I am taking a bit of inspiration from this, as it looks fairly similar appart from not having embedded quantization
    and the TP sharding.

    rename region
    -------------------
        collect region
        --------------
            here we have a list of un-materialized weights! (merge module list, or fuse. Any "cat" operation will give us a list.

            BUT IF WE TP 0.w1[rank], 0.w3[rank] then we need to slice the tensor and not the list of tensors!

            which we always TP first (shard) then we apply the ops (merging)
            TP REGION
            ---------
                Materialize only the correct shards
                Concat, Chunk. If you need to split a layer into 2 here, then each split is potentially quantizable

                    Quantization region
                    -------------------
                     Can produce 2 "weights" from 1 (blocks and scales)
            Based on quant_layout, we might need to all reduce the scales -> the quantization op tells us to do it or not
            ---------
            torch.distributed.all_reduce(max_abs, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        ----------------
    -------------------------
    Say we want to quantize:




    We are probably gonna be reading from left to right -> FuseGateUp and MergeModuleList and FuseQkv are prob the only
    ops we currently need. With potentially RotateQkv.



    3. a. If not quantization, or can be quantized independently (EP or som quantization) -> Shard
    3. b. If needs full tensor for quantize: materialize the tensor on cpu, quantize -> Shard
    4.
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
    def convert(self, value: Union[Sequence[torch.Tensor], torch.Tensor], *, context: dict[str, Any]) -> torch.Tensor:
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


class MergeModuleList(Concatenate):
    """
    Merge a list of tensors into a single tensor along the first dimension.
    We explicitly define this because for EP or TP you want to make sure you know what you are doing!

    """

    def __init__(self, dim: int = 0):
        super().__init__(dim=dim)
        self._inverse_op = SplitModuleList

    def convert(self, value: Sequence[Sequence[torch.Tensor]], *, context: dict[str, Any]) -> list[torch.Tensor]:
        if not isinstance(value, Sequence):
            raise TypeError("MergeModuleList expects a sequence of sequences of tensors.")
        merged: list[torch.Tensor] = []
        for group in value:
            if not isinstance(group, Sequence) or len(group) == 0:
                raise ValueError("MergeModuleList requires non-empty sub-sequences.")
            merged.append(torch.cat(tuple(group), dim=self.dim))
        return merged


class SplitModuleList(ConversionOps):
    """Inverse of :class:`MergeModuleList` using explicit split sizes per group."""

    def __init__(self, sizes: Sequence[Sequence[int]], dim: int = 0):
        if not isinstance(sizes, Sequence) or not all(isinstance(sub, Sequence) and sub for sub in sizes):
            raise ValueError("`sizes` must be a sequence of non-empty sequences of integers.")
        self.sizes = [list(sub) for sub in sizes]
        self.dim = dim
        self._inverse_op = MergeModuleList

    def convert(self, value: Sequence[torch.Tensor], *, context: dict[str, Any]) -> list[list[torch.Tensor]]:
        if not isinstance(value, Sequence):
            raise TypeError("SplitModuleList expects a sequence of tensors.")
        if len(value) != len(self.sizes):
            raise ValueError("Number of tensors does not match the provided split specifications.")

        result: list[list[torch.Tensor]] = []
        for tensor, split_sizes in zip(value, self.sizes):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("SplitModuleList can only split torch.Tensor instances.")
            splits = torch.split(tensor, split_sizes, dim=self.dim)
            result.append(list(splits))
        return result


class Shard(ConversionOps):
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


class Fp8Dequantize(ConversionOps):
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


@dataclass(frozen=True)
class WeightConversion:
    """Describe how a serialized weight maps to a model parameter.
    if people need to use a custom op, they just have to make it inherit from ConversionOps
    We need to allow going from a list of keys to a unique key and vice versa.
    This will also allow us to write quantization as WeightConversion("weight", ["weight_blocks", "weight_scales"], Fp8Quantize)
    potentially with filtering?

    YES because we can check nn.
    And sharding written as WeightConversion("weight", operations = Shard)?
    This way we explicit the full operations

    The operation can be "instantiated" this way we pass potential arguments.
    """

    source_keys: Union[str, list[str]]
    target_keys: Optional[Union[str, list[str]]] = None
    operations: Optional[
        Union[Union[type[ConversionOps], ConversionOps], list[Union[type[ConversionOps], ConversionOps]]]
    ] = None


def convert_state_dict(model, state_dict, weight_mapping, tp_plan, quantization_config, profile: bool = False):
    """Convert a state dict according to a weight mapping.

    Given that the model might be sharded, and that some patterns might fuse experts, there will
    be small edgecases to handle.

    If q,k and v need to be merged, but they are on a different state dict, we need to make sure
    we collected all of the keys.


    There is an ordered collection. so experts.*.w1.weight will collect all keys that match first.

    Given that the tensors are mmaped, its fine if we read all safetensors.json files first! We
    can load directly any tensors that does not match the mapping, but for those that do, we need to
    collect them first.

    Args:
        model (`torch.nn.Module`):
            The model to load the converted state dict into. We need this to get the type
            of the layer. TODO not used yet
        state_dict (`dict`):
            A state dict containing the weights to convert.
        weight_mapping (`List[WeightConversion]`):
            A list of `WeightConversion` objects describing how to convert the weights.
        tp_plan:
            The tensor parallelism plan for this model. Used to shard the weights correctly.
        quantization_config:
            The quantization configuration for this model. Used to quantize the weights correctly.
        profile (`bool`, *optional*, defaults to `False`):
            If set, wraps each conversion operation in a ``torch.profiler`` context (when available) and logs per-op
            execution time and profiling summaries.

    Returns:
        - `dict`: The converted state dict.
        - list[ConversionOps]: The list of operations used during the conversion. This is useful if the model needs to be saved
          in its legacy format later on.
    """
    if state_dict is None:
        raise ValueError("`state_dict` must be provided for conversion.")

    if isinstance(state_dict, OrderedDict):
        working_state = OrderedDict(state_dict)
    else:
        working_state = dict(state_dict)

    if hasattr(torch, "distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
        default_world_size = torch.distributed.get_world_size()
        default_rank = torch.distributed.get_rank()
    else:
        default_world_size = 1
        default_rank = 0
    from collections import defaultdict
    collected_keys: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    # 1. we need to find which key we have (so we keep track of which pattern was matched)
    converted_state_dict: dict[str, torch.Tensor] = {}
    used_operations: list[ConversionOps] = []
    keys_to_convert = [ rf"{ '|'.join(k.source_keys) if isinstance(k.source_keys, list) else k.source_keys}" for k in weight_mapping ]
    # tensor parallel is also a conversion scheme! So add it to the keys to convert!
    # quantization as well! But for quantization we would need to get the module, check if its a linear?

    for k,v in state_dict.items():
        if re.sub(rf"^({ '|'.join(keys_to_convert) })$", "", k) == k:
            converted_state_dict[k] = v
        else:
            # we replace the whole key by the matched pattern so that we can find it later
            pattern = re.sub(rf"^({ '|'.join(keys_to_convert) })$", r"\1", k)
            collected_keys[pattern][k] += [v] # we collect all tensors that match the pattern
        if pattern in tp_plan: # If we want this to work conversion needs to be explicit no?
            # TODO: for now just shard but we should create the op based on the TP plan
            # TODO: don't add sharding or tp ops if such ops are already present?
            weight_mapping[pattern].operations = Shard(0) +  weight_mapping[pattern].operation
        if pattern in quantization_config.conversion_mapping:
            # TODO: here again we need to check for other quantization. Maybe these are two
            # keys that we want to have explicit
            weight_mapping[pattern].operations.append(Fp8Quantize)

    # 2. now that we collectedd the tensors, we iterate over the "patterns" that were matched
    # Cuz remember we have to add TP and QUANT to the ops of some keys. but we do it on the renamed!
    for mapping in weight_mapping or []:
        source_patterns = _ensure_list(mapping.source_keys)
        matched_keys, collected_values = _collect_source_values(working_state, source_patterns)
        if not any(matched_keys):
            logger.debug("No keys matched pattern(s) %s; skipping conversion.", source_patterns)
            continue
        if any(len(group) == 0 for group in matched_keys):
            logger.debug(
                "At least one pattern in %s had no matches (%s); skipping conversion.",
                source_patterns,
                matched_keys,
            )
            continue

        operations = _prepare_operations(mapping.operations)
        operations = _order_operations(operations)

        target_spec = mapping.target_keys
        if isinstance(target_spec, Sequence) and not isinstance(target_spec, str) and len(target_spec) == 1:
            target_for_ops: Union[str, Sequence[str], None] = target_spec[0]
        else:
            target_for_ops = target_spec

        context = {
            "model": model,
            "tp_plan": tp_plan,
            "quantization_config": quantization_config,
            "target_keys": target_for_ops,
            "source_keys": source_patterns,
            "matched_keys": matched_keys,
            "tp_world_size": default_world_size,
            "tp_rank": default_rank,
        }

        current_value: Any = collected_values
        for operation in operations:
            used_operations.append(operation)
            current_value = operation(current_value, context=context, profile=profile)

        assignments = _assign_to_targets(current_value, target_spec, matched_keys)

        # Remove consumed keys from the intermediate dict so they do not leak in the output.
        for keys_group in matched_keys:
            for key in keys_group:
                working_state.pop(key, None)

        converted_state_dict.update(assignments)
        working_state.update(assignments)

    # Add all leftover keys that were never converted.
    for key, tensor in working_state.items():
        if key not in converted_state_dict:
            converted_state_dict[key] = tensor

    # Clear cached buffers in unique operations
    for op in {op for op in used_operations if hasattr(op, "clear_cache")}:
        op.clear_cache()

    return converted_state_dict, used_operations


def _ensure_list(value: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _prepare_operations(
    operations: Optional[Union[ConversionOps, type[ConversionOps], Sequence]],
) -> list[ConversionOps]:
    if operations is None:
        return []
    if isinstance(operations, (ConversionOps, type)):
        operations = [operations]
    prepared: list[ConversionOps] = []
    for op in operations:  # type: ignore[assignment]
        if isinstance(op, ConversionOps):
            prepared.append(op)
        elif isinstance(op, type) and issubclass(op, ConversionOps):
            prepared.append(op())
        else:
            raise TypeError(f"Unsupported operation specification: {op!r}")
    return prepared


def _order_operations(operations: list[ConversionOps]) -> list[ConversionOps]:
    if not operations:
        return []
    tp_ops = [op for op in operations if isinstance(op, Shard)]
    quant_ops = [op for op in operations if isinstance(op, QuantizationOp)]
    middle_ops = [op for op in operations if op not in tp_ops and op not in quant_ops]
    return tp_ops + middle_ops + quant_ops


def _collect_source_values(
    state_dict: dict[str, torch.Tensor], patterns: list[str]
) -> tuple[list[list[str]], list[Any]]:
    matched_keys: list[list[str]] = []
    collected: list[Any] = []
    for pattern in patterns:
        keys = sorted(_match_pattern(state_dict, pattern))
        matched_keys.append(keys)
        collected.append([state_dict[key] for key in keys])

    simplified = [_simplify_singletons(bucket) for bucket in collected]
    return matched_keys, _simplify_singletons(simplified)


def _match_pattern(state_dict: dict[str, torch.Tensor], pattern: str) -> list[str]:
    if pattern in state_dict:
        return [pattern]
    matched = [key for key in state_dict if fnmatchcase(key, pattern)]
    if not matched:
        logger.debug("Pattern %s did not match any key.", pattern)
    return matched


def _simplify_singletons(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        inner = value[0]
        simplified_inner = _simplify_singletons(inner)
        return simplified_inner
    if isinstance(value, list) and all(isinstance(elem, list) and len(elem) == 1 for elem in value):
        return [elem[0] for elem in value]
    return value


def _assign_to_targets(
    value: Any,
    target_spec: Optional[Union[str, Sequence[str]]],
    matched_keys: list[list[str]],
) -> dict[str, torch.Tensor]:
    assignments: dict[str, torch.Tensor] = {}
    target_keys = target_spec

    if isinstance(value, dict):
        assignments.update(value)
        return assignments

    if target_keys is None:
        flattened = list(chain.from_iterable(matched_keys))
        if isinstance(value, (list, tuple)):
            if len(flattened) != len(value):
                raise ValueError(
                    f"Cannot assign {len(value)} tensors to {len(flattened)} targets (patterns {matched_keys})."
                )
            for key, tensor in zip(flattened, value):
                assignments[key] = tensor
        elif len(flattened) == 1:
            assignments[flattened[0]] = value
        else:
            raise ValueError("Ambiguous assignment with multiple matched keys and scalar value.")
        return assignments

    if isinstance(target_keys, str):
        assignments[target_keys] = value
        return assignments

    if isinstance(target_keys, Sequence):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Expected a sequence of tensors to match multiple target keys.")
        if len(target_keys) != len(value):
            raise ValueError(
                f"Expected {len(target_keys)} tensors but received {len(value)} for targets {target_keys}."
            )
        for key, tensor in zip(target_keys, value):
            assignments[key] = tensor
        return assignments
    raise TypeError(f"Unsupported target key specification: {target_keys!r}")
