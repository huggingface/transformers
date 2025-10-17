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
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile
except (ImportError, AttributeError):
    ProfilerActivity = None
    torch_profile = None


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

class DistributedOp(ConversionOps): # all `distributed_operations` need to respect this
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

    collected_keys: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    # 1. we need to find which key we have (so we keep track of which pattern was matched)
    converted_state_dict: dict[str, torch.Tensor] = {}
    used_operations: list[ConversionOps] = []
    keys_to_convert = [
        rf"{'|'.join(k.source_keys) if isinstance(k.source_keys, list) else k.source_keys}" for k in weight_mapping
    ]
    # tensor parallel is also a conversion scheme! So add it to the keys to convert!
    # quantization as well! But for quantization we would need to get the module, check if its a linear?

    for k, v in state_dict.items():
        if re.sub(rf"^({'|'.join(keys_to_convert)})$", "", k) == k:
            converted_state_dict[k] = v
        else:
            # we replace the whole key by the matched pattern so that we can find it later
            pattern = re.sub(rf"^({'|'.join(keys_to_convert)})$", r"\1", k)
            collected_keys[pattern][k] += [v]  # we collect all tensors that match the pattern
        converter = weight_mapping[pattern]
        if pattern in tp_plan:  # If we want this to work conversion needs to be explicit no?
            if converter.distributed_operation is None:
                converter.distributed_operation = Shard(0)  # for now
        # TODO: use `param_needs_quantization` !
        if pattern in quantization_config.conversion_mapping:
            if converter.quantize_operations is None:
                converter.quantize_operations = Fp8Quantize()
        # if pattern in device_map:
        #     converter.operations.append(To(device_map[pattern]))
        # TODO: always call .contiguous()
        # TODO: the only missing part now is to update the TP plan for quantized weights
        # TODO: AND quantization that updates the keys (adds some). THIS IS FOR THE HOOKS
        # NOT FOR THE WEIGHTS

    # 2. now that we collectedd the tensors, we iterate over the "patterns" that were matched
    # Cuz remember we have to add TP and QUANT to the ops of some keys. but we do it on the renamed!
    for key, current_value in collected_keys:
        # 1. Distributed, equivalent to our `shard_and_distribute_module`
        used_operations.append(weight_mapping[key].distributed_operation)
        current_value = weight_mapping[key].distributed_operation(current_value)

        # 2. Other opérations
        for operation in weight_mapping[key].operations:
            used_operations.append(operation)
            current_value = operation(current_value, profile=profile)

        # 3. Quantization equivalent to `create_quantized_param`
        used_operations.append(weight_mapping[key].quantization_operation)
        current_value = weight_mapping[key].quantization_operation(current_value)
        converted_state_dict[key] = current_value

        # Clear cached buffers in unique operations
    for op in {op for op in used_operations if hasattr(op, "clear_cache")}:
        op.clear_cache()

    return converted_state_dict, used_operations

