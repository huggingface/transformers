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
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Iterable
from collections import defaultdict
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
from .integrations.tensor_parallel import ALL_PARALLEL_STYLES

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

class DistributedOp(ConversionOps): # all `distributed_operation` need to respect this
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


@dataclass
class WeightConversion:
    """
    - source_keys: str | list[str] (wildcards '*' match digits)
    - target_keys: str | list[str] | None
    - distributed_operation / operations / quantization_operations are ALWAYS lists.
    """
    source_keys: Union[str, list[str]]
    target_keys: Optional[Union[str, list[str]]] = None

    distributed_operation: Optional[ConversionOps] = None
    quantization_operation: Optional[ConversionOps] = None
    _operations: list[ConversionOps] = field(default_factory=list, repr=False)

    _compiled: tuple[tuple[str, re.Pattern], ...] = field(default_factory=tuple, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.source_keys, list):
            self.source_keys = [self.source_keys]
        if not isinstance(self.target_keys, list):
            if self.target_keys is None:
                self.target_keys = self.source_keys
            else:
                self.target_keys = [self.target_keys]

        regex_pat = r""
        for p in self.source_keys:
            pat = re.escape(p).replace(r"\*", r"\d+")
            regex_pat += f"({re.compile(fr'^{pat}$')})|"
        self._regex_pat = regex_pat[:-1]
        self.operations = self._operations

    @property
    def operations(self) -> list[ConversionOps]:
        return self._operations
    @operations.setter
    def operations(self, v: Union[None, ConversionOps, list[ConversionOps]]):
        if v is None: self._operations = []
        elif isinstance(v, list): self._operations = v
        else: self._operations = [v]



def convert_and_load_state_dict_in_model(model, state_dict, weight_mapping, tp_plan, quantizer, device_map=None, keep_in_dtype=None, device_mesh=None, profile: bool = False):
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
    tp_plan = tp_plan or {} # keys are * patterns, exact match with model.state_dict().keys()
    device_map = device_map or {} # keys are the `target` obtained from the model
    keep_in_dtype = keep_in_dtype or {} # keys are * pattern model.state_dict().keys()
    weight_mapping = weight_mapping or {} # keys are * patterns model.state_dict().keys()

    tp_regex_pattern = f"""({')|()'.join(tp_plan.keys()).replace("*", "d+")})"""
    keep_in_dtype_pattern = f"""({')|()'.join(keep_in_dtype.keys()).replace("*", "d+")})"""
    weight_mapping_pattern = weight_mapping._regex_pat
    # Store which ops were applied for saving
    used_operations: list[ConversionOps] = []
    # Let's create a mapping from the keys we will read -> the operations to perform
    # tensor parallel is also a conversion scheme! So add it to the keys to convert!
    # quantization as well! But for quantization we would need to get the module, check if its a linear?

    # 1. We figure out whatever needs to happen to each weights!
    #   - we need to take care of `device_map="auto"` -> add To(device_map[layer_name])
    #   - we need to take care of `tp_plan`           -> add Shard() and etc automatically
    #   - we need to take care of the `keep_in_dtype` -> add Cast(keep_in_dtype[layer_name])
    #   - we need to take care of `quantization`      -> add target keys created by the method + update TP plan?
    #   - we need to take care of lora later on.
    collected_target_keys = defaultdict(list)
    for original_key, tensor in state_dict.items():
        default_op = re.sub(weight_mapping_pattern, r"\1", original_key)
        if default_op is not None:
            converter: WeightConversion =  weight_mapping[default_op] # forget about this
        else:
            converter : WeightConversion = WeightConversion(default_op) # source and target are the same!
            weight_mapping[default_op] = converter

        current_key = converter.target_keys if isinstance(converter.target_keys, str) else "|".join(converter.target_keys)
        collected_target_keys[current_key] += [tensor]

    for collected_keys, collected_tensors in collected_target_keys.items(): # a single key indexes many target keys
        target_keys = collected_keys.split('|')
        for target_key in target_keys: # some of these can be newly created by quantizer / merge or chunk op
            if plan:=re.sub(target_key, r"\1", tp_regex_pattern):
                if converter.distributed_operation is None:
                    converter.distributed_operation = ALL_PARALLEL_STYLES[plan].distributed_op
                # TODO: here we need to translate the sharding as we have a collection of tensors
                # so shard[0] would mean we split the list of tensor, shard(1) we split each tensor along dim 1
                # but that's only if we collected more than 1 key
                rank = device_mesh.get_local_rank()
                final_target = converter.distributed_operation.convert(tensor, empty_tensor, tensor_type, rank, device_mesh)
            else:
                final_target = [ k[:] for k in collected_tensors] # we materialize the weights on device?

            # Now we need to add the standard operations
            for op in converter.operations:
                final_target = op.convert(final_target)

            # Finaly the quantizer comes into play!
            if quantizer is not None:
                if converter.quantize_operation is None:
                    converter.quantize_operation = quantizer.quantize_op
                final_target = converter.quantize_operation(final_target, ...)


            # Finally, once we have the final keys, some might be new -> we move them to the operation's device
            # and we cast to the correct dype if provided.
            if target_key in device_map:
                op = To(device_map[target_key])
                converter.operations.append(op)
                for k,v in final_target.items():op.convert(final_target)
            if match:= re.sub(keep_in_dtype_pattern, "\1", target_key):
                op = Cast(keep_in_dtype[match])
                converter.operations.append(op)
                for k,v in final_target.items():op.convert(final_target)

            for k,v in final_target.items():
                module_to_tp = model.get_submodule(k)
                param_type = k.rsplit('.')[:-1]
                if not isinstance(tensor, torch.nn.Parameter):
                    param = torch.nn.Parameter(k, requires_grad=k.is_floating_point())
                setattr(module_to_tp, param_type, param)

        # Clear cached buffers in unique operations
    for op in {op for op in used_operations if hasattr(op, "clear_cache")}:
        op.clear_cache()

    return used_operations
