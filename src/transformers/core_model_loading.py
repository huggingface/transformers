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
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any, Optional, Union
from abc import abstractmethod
import torch


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


class Chunk(ConversionOps):
    pass


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

        out_shape = tensors[0].shape
        out_shape[self.dim] *= len(tensors)

        with torch.no_grad():
            out = self._ensure_buffer(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
            offset = 0
            for tensor in tensors:
                index = [slice(None)] * tensor.ndim
                index[self.dim] = slice(offset, offset + tensor.shape[self.dim])
                out[tuple(index)].copy_(tensor, async_op=True)
                offset += tensor.shape[self.dim]
        return out


class MergeModuleList(ConversionOps):
    """Stack tensors along a new leading dimension."""

    def __init__(self, stack_dim: int = 0):
        self.stack_dim = stack_dim

    def convert(self, value: Sequence[torch.Tensor], *, context: dict[str, Any]) -> torch.Tensor:
        tensors = tuple(value)
        if not tensors:
            raise ValueError("MergeModuleList requires at least one tensor to merge.")

        first = tensors[0]
        dtype, device = first.dtype, first.device
        out_shape = tensors[0].shape
        out_shape[0] *= len(tensors)

        with torch.no_grad():
            out = self._ensure_buffer(out_shape, dtype=dtype, device=device)
            for index, tensor in enumerate(tensors):
                slice = slice(index, index + 1)
                out[slice].copy_(tensor)
        return out


class Shard(ConversionOps):
    def __init__(self, device_mesh, rank, dim):
        self.dim = dim
        self.device_mesh = device_mesh
        self.rank = rank

    def convert(self, param, empty_param):
        param_dim = empty_param.dim()
        # Flatten the mesh to get the total number of devices
        mesh_shape = self.device_mesh.shape
        world_size = reduce(operator.mul, mesh_shape)

        if self.rank >= world_size:
            raise ValueError(f"Rank {self.rank} is out of bounds for mesh size {world_size}")

        shard_size = math.ceil(empty_param.shape[self.dim] / world_size)
        start = self.rank * shard_size

        # Construct slicing index dynamically
        end = min(start + shard_size, empty_param.shape[self.dim])
        slice_indices = [slice(None)] * param_dim
        if start < empty_param.shape[self.dim]:
            slice_indices[self.dim] = slice(start, end)
            return param[tuple(slice_indices)]
        dimensions = list(param.shape)
        dimensions[self.dim] = 0
        return torch.empty(tuple(dimensions), dtype=torch.int64)


class Fp8Quantize(ConversionOps):
    """
    A quantization operation that creates two tensors, weight and scale out of a weight.
    """

    def convert(self, param_value, param_name: str) -> dict[str, torch.Tensor]:
        param_value = param_value.to(target_device)

        # Get FP8 min/max values
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        block_size_m, block_size_n = self.quantization_config.weight_block_size

        rows, cols = param_value.shape[-2:]

        if rows % block_size_m != 0 or cols % block_size_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
            )
        param_value_orig_shape = param_value.shape
        param_value = param_value.reshape(-1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n)

        # Calculate scaling factor for each block
        max_abs = torch.amax(torch.abs(param_value), dim=(2, 4))
        scale = fp8_max / max_abs
        scale_orig_shape = scale.shape
        scale = scale.unsqueeze(-1).unsqueeze(-1)

        quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        quantized_param = quantized_param.reshape(param_value_orig_shape)
        scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()

        return {param_name: quantized_param, param_name.rsplit(".")[0] + ".scale": scale}


@dataclass(frozen=True)
class WeightConversion:
    """Describe how a serialized weight maps to a model parameter.
    if people need to use a custom op, they just have to make it inherit from ConversionOps
    """

    target_key: str
    source_key: str
    operations: tuple[Union[type[ConversionOps], type[ConversionOps]]]


def convert_state_dict(model, state_dict, weight_mapping, tp_plan, quantization_config):
    """Convert a state dict according to a weight mapping.

    Given that the model might be sharded, and that some patterns might fuse experts, there will
    be small edgecases to handle.

    If q,k and v need to be merged, but they are on a different state dict, we need to make sure
    we collected all of the keys.


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

    Returns:
        - `dict`: The converted state dict.
        - list[ConversionOps]: The list of operations used during the conversion. This is useful if the model needs to be saved
          in its legacy format later on.
    """
    converted_state_dict = {}
    ops_cache = {}
    # 1. We need to rename / collect all the weights (iterate once through all the state dict)

    # 2. Now that we have all the weights, we can apply the operations

    # Clear cached buffers in all operations
    for op in ops_cache.values():
        op.clear_cache()

    return converted_state_dict
