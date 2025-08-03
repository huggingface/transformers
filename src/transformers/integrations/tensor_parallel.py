# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import math
import operator
import os
import re
from functools import partial, reduce

import torch
import torch.distributed as dist
from torch import nn

from ..distributed import DistributedConfig
from ..utils import is_torch_greater_or_equal, logging
from ..utils.generic import GeneralInterface


logger = logging.get_logger(__name__)

# Cache this result has it's a C FFI call which can be pretty time-consuming
_torch_distributed_available = torch.distributed.is_available()


if is_torch_greater_or_equal("2.5") and _torch_distributed_available:
    from torch.distributed.tensor import DTensor, Placement, Replicate, Shard


def initialize_tensor_parallelism(tp_plan, tp_size=None):
    r"""
    Sets up the device mesh and initilized the backend for tensor parallelism.
    This function is called when the model is loaded and the TP plan is set to 'auto'.
    """
    if tp_plan is None:
        return None, None, None

    if not is_torch_greater_or_equal("2.5"):
        raise OSError("Tensor parallel is only supported for `torch>=2.5`.")

    # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
    device_type = torch._C._get_accelerator().type
    current_device = getattr(torch, device_type)
    if not torch.distributed.is_initialized():
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
            backend = backend_map.get(device_type)
            if device_type == "cpu" and int(os.environ.get("CCL_WORKER_COUNT", 0)):
                backend = "ccl"
            if device_type == "xpu" and not is_torch_greater_or_equal("2.8", accept_dev=True):
                backend = "ccl"

            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
            current_device = getattr(torch, device_type)
            if device_type != "cpu":
                current_device.set_device(local_rank)

        except Exception as e:
            raise OSError(
                "We tried to initialize torch.distributed for you, but it failed. Make "
                "sure you init torch distributed in your script to use `tp_plan='auto'`."
            ) from e

    if device_type != "cpu":
        current_device.set_device(int(os.environ["LOCAL_RANK"]))
    index = current_device.current_device() if device_type != "cpu" else None
    tp_device = torch.device(device_type, index)

    # Silence output for non-primary ranks
    if index is not None and index > 0:
        import sys

        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    device_map = tp_device
    tp_size = tp_size if tp_size is not None else torch.distributed.get_world_size()
    device_mesh = torch.distributed.init_device_mesh(tp_device.type, (tp_size,))
    return tp_device, device_map, device_mesh, tp_size


def _blocks_to_block_sizes(total_size: int, blocks: int | list[int]) -> list[int]:
    """
    Convert block count or proportions to block sizes.

    This function accepts

    - The number of blocks (int), in which case the block size is
      total_size//blocks; or
    - A list of block sizes (list[int]).

    In the second case, if sum(blocks) < total_size, the ratios between
    the block sizes will be preserved. For instance, if blocks is
    [2, 1, 1] and total_size is 1024, the returned block sizes are
    [512, 256, 256].
    """
    if isinstance(blocks, list):
        total_blocks = sum(blocks)
        assert total_size % total_blocks == 0, f"Cannot split {total_size} in proportional blocks: {blocks}"
        part_size = total_size // total_blocks
        return [part_size * block for block in blocks]
    else:
        assert total_size % blocks == 0, f"Prepacked is not divisible by {blocks}"
        single_size = total_size // blocks
        return [single_size] * blocks


def _get_parameter_tp_plan(parameter_name: str, tp_plan: dict[str, str], is_weight=True) -> str | None:
    """
    Get the TP style for a parameter from the TP plan.

    The TP plan is a dictionary that maps parameter names to TP styles.
    The parameter name can be a generic name with wildcards (e.g. "*.weight") or a specific name (e.g. "layer_1.weight").

    The `is_weight` is important because for weights, we want to support `.weights` and `.bias` cases seamlessly! but
    not parrent classes for `post_init` calls
    """
    generic_param_name = re.sub(r"\d+", "*", parameter_name)
    if generic_param_name in tp_plan:
        return tp_plan[generic_param_name]
    elif "." in generic_param_name and generic_param_name.rsplit(".", 1)[0] in tp_plan and is_weight:
        return tp_plan[generic_param_name.rsplit(".", 1)[0]]
    return None


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
}


def get_packed_weights(param, empty_param, device_mesh, rank, dim):
    """
    When weights are packed (gate_up_proj), we need to make sure each shard gets its correct share.
    So if you have: gate_proj       ( 16, 5120, 8190)
    and             up_proj         ( 16, 5120, 8190)
    packed as       gate_up_proj    ( 16, 5120, 2 * 8190)
    And you shard along the last dimension, you need to interleave the gate and up values:

    Now, if we shard along the last dimension across TP_size (Tensor Parallelism size), we must interleave the values from gate and up projections correctly.

    Let's take TP_size = 4 for an example:

    Packed tensor `gate_up_proj`
    ---------------------------------------------------------------
    [ G0  G1  G2  G3 | G4  G5  G6  G7 | ... | U0  U1  U2  U3 | U4  U5  U6  U7 | ... ]
     ↑─────────────↑   ↑─────────────↑        ↑─────────────↑  ↑─────────────↑
       Gate Slice 0      Gate Slice 1            Up Slice 0       Up Slice 1

    Explanation:
    - The first half of the tensor (left of the center) holds the gate_proj values.
    - The second half (right of the center) holds the up_proj values.
    - For TP=4, we divide each half into 4 slices. In this example, we show two slices for brevity.
    - Each shard receives one slice from the gate part and the corresponding slice from the up part.

    For instance:
    • Shard 0 gets: [ Gate Slice 0, Up Slice 0 ] = [ G0, G1, G2, G3, U0, U1, U2, U3 ]
    • Shard 1 gets: [ Gate Slice 1, Up Slice 1 ] = [ G4, G5, G6, G7, U4, U5, U6, U7 ]
    • … and so on.

    This ensures that each shard receives an equal portion of both gate and up projections, maintaining consistency across tensor parallelism.
    """
    slice_ = param
    total_size = empty_param.shape[dim]
    world_size = device_mesh.size()
    block_sizes = _blocks_to_block_sizes(total_size=total_size, blocks=2)

    tensors_slices = []
    block_offset = 0
    for block_size in block_sizes:
        shard_block_size = block_size // world_size
        start = rank * shard_block_size
        stop = (rank + 1) * shard_block_size
        tensors_slices += range(block_offset + start, block_offset + stop)
        block_offset += block_size

    slice_dtype = slice_.get_dtype()
    # Handle F8_E4M3 dtype by converting to float16 before slicing
    # Without upcasting, the slicing causes : RuntimeError: "index_cpu" not implemented for 'Float8_e4m3fn'
    casted = False
    if slice_dtype == "F8_E4M3" or slice_dtype == "F8_E5M2":
        slice_ = slice_[...].to(torch.float16)
        casted = True

    if dim == 0:
        tensor = slice_[tensors_slices, ...]
    elif dim == 1 or dim == -2:
        tensor = slice_[:, tensors_slices, ...]
    elif dim == 2 or dim == -1:
        tensor = slice_[..., tensors_slices]
    else:
        raise ValueError(f"Unsupported dim {dim}, only dim 0, 1 or 2 are supported")

    if casted:
        return tensor
    else:
        return tensor.to(str_to_torch_dtype[slice_dtype])


def repack_weights(
    packed_parameter: torch.Tensor,
    sharded_dim: int,  # The dimension index in the global tensor that was sharded
    world_size: int,
    num_blocks: int = 2,
) -> torch.Tensor:
    """
    Reorders a tensor that was reconstructed from sharded packed weights into its canonical packed format.

    For example, if a weight was packed (e.g., gate_proj and up_proj) and then sharded,
    DTensor.full_tensor() might produce an interleaved layout like [G0, U0, G1, U1, ...]
    along the sharded dimension. This function reorders it to [G0, G1, ..., U0, U1, ...].
    This is an inverse operation to get_packed_weights.

    Args:
        reconstructed_tensor: The tensor reconstructed from DTensor (e.g., via .full_tensor().contiguous()).
        sharded_dim: The dimension index in the reconstructed_tensor that was originally sharded.
        world_size: The tensor parallel world size.
        num_packed_projs: The number of projections that were packed together (e.g., 2 for gate_up_proj).

    Returns:
        The reordered tensor in canonical packed format.
    """

    if num_blocks != 2:
        raise ValueError(
            "Num blocks different from 2 is not supported yet. This is most likely a bug in your implementation as we only pack gate and up projections together."
        )

    actual_sharded_dim = sharded_dim if sharded_dim >= 0 else sharded_dim + packed_parameter.ndim
    total_size_on_sharded_dim = packed_parameter.shape[actual_sharded_dim]
    original_block_size_on_dim = total_size_on_sharded_dim // num_blocks
    shard_chunk_size = original_block_size_on_dim // world_size

    prefix_shape = packed_parameter.shape[:actual_sharded_dim]
    suffix_shape = packed_parameter.shape[actual_sharded_dim + 1 :]

    tensor_view = packed_parameter.view(
        *prefix_shape,
        world_size,
        num_blocks,
        shard_chunk_size,
        *suffix_shape,
    )

    # Permute to bring num_packed_projs first, then world_size, then shard_chunk_size
    # This groups all chunks of G together, then all chunks of U together.
    # Target order of these middle dimensions: (num_packed_projs, world_size, shard_chunk_size)
    # Current order of view's middle dimensions: (world_size, num_packed_projs, shard_chunk_size)
    # Absolute indices of the dimensions to be permuted (world_size, num_packed_projs)
    axis_ws_abs = len(prefix_shape)
    axis_npp_abs = len(prefix_shape) + 1

    permute_order = list(range(tensor_view.ndim))
    permute_order[axis_ws_abs], permute_order[axis_npp_abs] = permute_order[axis_npp_abs], permute_order[axis_ws_abs]

    tensor_permuted = tensor_view.permute(*permute_order)

    # Reshape back to the original tensor's ndim, with the sharded dimension now correctly ordered as [G_all, U_all].
    # The final shape should be the same as reconstructed_tensor.
    final_ordered_tensor = tensor_permuted.reshape_as(packed_parameter)

    return final_ordered_tensor


def get_tensor_shard(param, empty_param, device_mesh, rank, dim):
    """
    Generalized tensor sharding across a multi-dimensional device mesh.
    Extract only the fraction of the parameter owned by the given `rank` when the parameter would have gone sharding at provided `dim`.
    Extraction follows the pytorch `Shard` placement so that sharding and materializing back to full tensor follows `Shard` semantics.
    `Shard` follows torch.chunk style sharding of the tensor. We demonstrate some cases below on how sharding happens including some edge cases
    such as some ranks having an empty tensor as shard. Below implementation is robut to all these cases.

    Case (1)
    empty_param                 (16, 5120, 8190)
    dim                         0
    device_mesh.size()          4
    rank 0 gets					(4, 5120, 8190)			 (0 ... 4, 5120, 8190)
    rank 1 gets					(4, 5120, 8190)			 (4 ... 8, 5120, 8190)
    rank 2 gets					(4, 5120, 8190)			 (8 ... 12, 5120, 8190)
    rank 3 gets					(4, 5120, 8190)			 (12 ... 16, 5120, 8190)

    Case (2)
    empty_param                 (16, 5120, 8190)
    dim                         0
    device_mesh.size()          14
    rank 0 gets					(2, 5120, 8190)			 (0 ... 2, 5120, 8190)
    rank 1 gets					(2, 5120, 8190)			 (2 ... 4, 5120, 8190)
    rank 2 gets					(2, 5120, 8190)			 (4 ... 6, 5120, 8190)
    rank 3 gets					(2, 5120, 8190)			 (6 ... 8, 5120, 8190)
    rank 4 gets					(2, 5120, 8190)			 (8 ... 10, 5120, 8190)
    rank 5 gets					(2, 5120, 8190)			 (10 ... 12, 5120, 8190)
    rank 6 gets					(2, 5120, 8190)			 (12 ... 14, 5120, 8190)
    rank 7 gets					(2, 5120, 8190)			 (14 ... 16, 5120, 8190)
    rank 8 gets					(0, 5120, 8190)
    rank 9 gets					(0, 5120, 8190)
    rank 10 gets			    (0, 5120, 8190)
    rank 11 gets				(0, 5120, 8190)
    rank 12 gets				(0, 5120, 8190)
    rank 13 gets				(0, 5120, 8190)

    Case (3)
    empty_param                 (16, 5120, 8190)
    dim                         0
    device_mesh.size()          3
    rank 0 gets					(6, 5120, 8190)			 (0 ... 6, 5120, 8190)
    rank 1 gets					(6, 5120, 8190)			 (6 ... 12, 5120, 8190)
    rank 2 gets					(4, 5120, 8190)			 (12 ... 16, 5120, 8190)

    In case (2), empty shards are returned with appropriate dimension to allow for operations to work smoothly.
    Args:
        param (torch.Tensor): The tensor to shard.
        empty_param (torch.Tensor): A tensor used for shape reference.
        device_mesh (torch.Tensor): Shape [d_0, ..., d_n] representing the mesh.
        rank (int): Global rank of the current process/device.
        dim (int): Dimension along which to shard the tensor.
    """
    param_dim = empty_param.dim()

    if dim < 0:
        dim = param_dim + dim
    if dim >= param_dim:
        raise ValueError(f"dim {dim} is out of bounds for tensor of dimension {param_dim}")

    # Flatten the mesh to get the total number of devices
    mesh_shape = device_mesh.shape
    world_size = reduce(operator.mul, mesh_shape)

    if rank >= world_size:
        raise ValueError(f"Rank {rank} is out of bounds for mesh size {world_size}")

    shard_size = math.ceil(empty_param.shape[dim] / world_size)
    start = rank * shard_size

    # Construct slicing index dynamically
    end = min(start + shard_size, empty_param.shape[dim])
    slice_indices = [slice(None)] * param_dim
    if start < empty_param.shape[dim]:
        slice_indices[dim] = slice(start, end)
        return param[tuple(slice_indices)]
    dimensions = list(param.shape)
    dimensions[dim] = 0
    return torch.empty(tuple(dimensions), dtype=torch.int64)


def distribute_module(
    module: nn.Module,
    device_mesh=None,
    input_fn=None,
    output_fn=None,
) -> nn.Module:
    """
    Copy pasted from torch's function but we remove the communications (partitioning)
    as well as buffer registering that is similarly not efficient.
    """
    if len(module._forward_pre_hooks) == 0:
        if input_fn is not None:
            module.register_forward_pre_hook(lambda mod, inputs: input_fn(mod, inputs, device_mesh))
        if output_fn is not None:
            module.register_forward_hook(lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh))
    return module


class TensorParallelLayer:
    """
    General tensor parallel layer for transformers.
    """

    use_dtensor = True

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh): ...

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh): ...

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        raise NotImplementedError

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        if self.use_dtensor:
            distribute_module(
                module,
                device_mesh,
                partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
                partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
            )


# use_dtensor needs to be set to false for nn.Parameter when you want to view, chunk, slice
# you name it. Whatever you want to do that is a bit unconventional, you need local tensors
class GatherParallel(TensorParallelLayer):
    """
    Simple class used to define the hooks to add to a layer when we just want to gather the outputs
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = output_layouts
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        mod.expert_parallel_group = device_mesh.get_group()
        if inputs and isinstance(inputs[0], DTensor):
            inputs = inputs[0].to_local()
        return inputs

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, torch.Tensor):
            dist.all_reduce(outputs, op=dist.ReduceOp.SUM, async_op=False)
        else:
            dist.all_reduce(outputs[0], op=dist.ReduceOp.SUM, async_op=False)
        return outputs

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            partial(self._prepare_input_fn, None, None),
            partial(self._prepare_output_fn, None, None),
        )


class IsolatedParallel(TensorParallelLayer):
    """
    This class is used to isolate computation in a TP layer from the rest of the world.
    Parameters need to be LOCAL, so not dtensors
    """

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh=None):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            input_tensor = input_tensor.to_local()
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh=None):
        # TODO: figure out dynamo support for instance method and switch this to instance method
        return outputs

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        param = param[...].to(param_casting_dtype)
        if to_contiguous:
            param = param.contiguous()
        param = param / device_mesh.size()  # TODO should be optionable
        # TODO: assumes parent module will allreduce the output afterwards (e.g rowlinear bias is IsolatedParallel and parent module is GatherParallel)
        return param

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            partial(self._prepare_input_fn, None, None),
            partial(self._prepare_output_fn, None, None),
        )


class ReplicateParallel(TensorParallelLayer):
    """
    This class is used to replicate computation in a TP layer (used in SP regions when we don't use sequence parallelism for example)
    """

    def __init__(self, *, use_dtensor=True, use_local_output=True):
        super().__init__()
        self.input_layouts = (Replicate(),)
        self.output_layouts = (Replicate(),)
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output
        self.use_dtensor = use_dtensor

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # TODO: figure out dynamo support for instance method and switch this to instance method
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)

        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output and isinstance(outputs, DTensor) else outputs

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        param = param[...].to(param_casting_dtype)
        if to_contiguous:
            param = param.contiguous()
        param = DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
        return param


class ColwiseParallel(TensorParallelLayer):
    """
    General tensor parallel layer for transformers.
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
        use_dtensor=True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output
        self.use_dtensor = use_dtensor

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # TODO: figure out dynamo support for instance method and switch this to instance method
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts, async_op=False)
        return input_tensor

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(-2) (0 if you have 1 dim only)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        if param_type == "bias":
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -1)
            shard = [Shard(-1)]
        else:
            shard = [Shard(-2)]
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -2)

        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        if self.use_dtensor:
            parameter = DTensor.from_local(
                parameter, device_mesh, shard, run_check=False, shape=empty_param.size(), stride=empty_param.stride()
            )
        return nn.Parameter(parameter, requires_grad=parameter.is_floating_point())

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=False)
        # back to local tensor
        return outputs.to_local() if use_local_output and isinstance(outputs, DTensor) else outputs


class PackedColwiseParallel(ColwiseParallel):
    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(-2) (0 if you have 1 dim only)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        parameter = get_packed_weights(param, empty_param, device_mesh, rank, -2)
        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        if self.use_dtensor:
            parameter = DTensor.from_local(parameter, device_mesh, [Shard(-2)], run_check=False)
        return nn.Parameter(parameter, requires_grad=parameter.is_floating_point())


class RowwiseParallel(TensorParallelLayer):
    """
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
        use_dtensor=True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output
        self.use_dtensor = use_dtensor

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        if param_type != "bias":
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -1)
            shard = [Shard(-1)]
        else:
            shard = [Replicate()]
            parameter = param[:]

        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        if self.use_dtensor:
            parameter = DTensor.from_local(
                parameter, device_mesh, shard, run_check=False, shape=empty_param.size(), stride=empty_param.stride()
            )
        return nn.Parameter(parameter, requires_grad=parameter.is_floating_point())

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        if hasattr(mod, "bias") and mod.bias is not None:
            mod._bias = mod.bias
            mod.bias = None

        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts, async_op=True)
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        if hasattr(mod, "_bias"):
            outputs += mod._bias
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output and isinstance(outputs, DTensor) else outputs

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        module._distribute_module_applied = True
        if self.use_dtensor:
            if isinstance(module, nn.Linear):
                # rowwise linear runtime sharding requires input tensor shard on last dim
                self.desired_input_layouts: tuple[Placement, ...] = (Shard(-1),)
            elif isinstance(module, nn.Embedding):
                # rowwise embedding runtime sharding requires input tensor replicated
                self.desired_input_layouts = (Replicate(),)
            elif isinstance(module, nn.Parameter):
                # rowwise embedding runtime sharding requires input tensor replicated
                self.desired_input_layouts = (Shard(-1),)
            else:
                raise NotImplementedError("RowwiseParallel currently only support nn.Linear and nn.Embedding!")

            distribute_module(
                module,
                device_mesh,
                partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
                partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
            )


class PackedRowwiseParallel(RowwiseParallel):
    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(-2) (0 if you have 1 dim only)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        parameter = get_packed_weights(param, empty_param, device_mesh, rank, -1)
        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        if self.use_dtensor:
            parameter = DTensor.from_local(parameter, device_mesh, [Shard(-1)], run_check=False)
        return nn.Parameter(parameter, requires_grad=parameter.is_floating_point())


class SequenceParallel(TensorParallelLayer):
    """
    SequenceParallel replicates a compatible ``nn.Module`` parameters and runs the sharded computation with
    input sharded on the sequence dimension. This currently supports ``nn.LayerNorm``, ``nn.Dropout``, and the
    `RMSNorm python implementation <https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34>`__

    This style implements the operation that is described in the paper
    `Reducing Activation Recomputation in Large Transformer Models <https://huggingface.co/papers/2205.05198>`__

    If the input passed in to this ``nn.Module`` is a :class:`torch.Tensor`, it assumes that the input is already sharded
    on the sequence dimension and converts the input to a :class:`DTensor` sharded on the sequence dimension. If the input
    passed in to this ``nn.Module`` is already a :class:`DTensor` but is not sharded on the sequence dimension, it would
    redistribute the input to be sharded on the sequence dimension.

    The output of the ``nn.Module`` will be sharded on the sequence dimension.

    Keyword Args:
        sequence_dim (int, optional):
            The sequence dimension of the input tensor for the ``nn.Module``, this is used to annotate the input tensor to
            become a DTensor that is sharded on the sequence dimension, default: 1.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: False.
    Returns:
        A :class:`ParallelStyle` object that represents Sequence Parallel of the ``nn.Module``.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "norm" nn.LayerNorm submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "norm" will be converted to DTensor that shards on the sequence dim
        >>> # and the output of "norm" will return a sharded on sequence dimension :class:`DTensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"norm": SequenceParallel()}),
        >>> ...

    .. note:: SequenceParallel style assumes ones initialization if there are weights in the nn.Module (i.e.
        ``nn.LayerNorm`` or ``RMSNorm``, and they by default have ones initialization). If you have custom
        inits for the weights on those modules, you need to broadcast the weights before/after parallelizing
        to ensure that they are replicated.
    """

    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False, use_dtensor=False):
        super().__init__()
        self.input_layouts = (Replicate(),)
        self.desired_input_layouts = (Shard(1),)
        self.output_layouts = (Replicate(),)
        self.use_local_output = use_local_output
        self.use_dtensor = True
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts, async_op=True)
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        outputs = outputs.redistribute(
            placements=(Replicate(),), async_op=True
        )  # maybe we have to replicate ? because next layer is not sharded
        return outputs.to_local()  # if use_local_output else outputs

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(-2) (0 if you have 1 dim only)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        parameter = param[...]
        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        if self.use_dtensor:
            parameter = DTensor.from_local(parameter, device_mesh, [Replicate()], run_check=False)
        return nn.Parameter(parameter, requires_grad=parameter.is_floating_point())


class GroupedGemmParallel(TensorParallelLayer):
    """
    Applies Expert Parallelism to MoE experts by loading the correct experts on each device.
    """

    def __init__(self):
        super().__init__()
        self.use_dtensor = False

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        ep_rank = rank
        global_num_experts = empty_param.shape[0]
        if global_num_experts % device_mesh.size() != 0:
            raise ValueError(
                f"Global number of experts must be divisible by number of devices: {global_num_experts} % {device_mesh.size()} != 0"
            )
        local_num_experts = global_num_experts // device_mesh.size()
        param = param[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts].to(param_casting_dtype)
        if to_contiguous:
            param = param.contiguous()
        return param


class RouterParallel(TensorParallelLayer):
    """
    Allows to reshape the router scores to support running expert parallel.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.use_dtensor = False

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            raise NotImplementedError("RouterParallel does not support DTensor input for now")
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        """
        Imagine if you had 4 tokens, top_k = 4, and 128experts.
        With EP = 8.
        Imagine router_indices being:
        [ 52,  42, 119,  67],
        [102,  89,  61,  40],
        [ 82, 103,   4,  34],
        [ 93,  23, 109,  11],

        then you can map which rank should be getting which values

        [3, 2, 7, 4],
        [6, 5, 3, 2],
        [5, 6, 0, 2],
        [5, 1, 6, 0],

        Thus for say rank 0, you fill with 0 the index tensor

        [ 0, 0, 0, 0],
        [ 0, 0, 0, 0],
        [ 0, 0, 4, 0],
        [ 0, 0, 0, 11],

        This works well. For another rank you need to make sure you round to num_local_expert
        because the next operation will one hot encode the router index vector.

        This allows us to know directly which local expert is hit.
        Similarly the scores are indexed with something created form
        router_indices.

        The kinda naive training loop that we use for device_map "auto" uses a similar logic.
        Here we are just making each rank believe that he is alone, and he computes his part of the hiddenstates.
        """
        ep_rank, ep_size = device_mesh.get_local_rank(), device_mesh.size()
        num_local_experts = mod.num_experts // ep_size
        router_scores, router_indices = outputs
        router_scores = router_scores[:, ep_rank * num_local_experts : (ep_rank + 1) * num_local_experts]
        router_indices = router_indices.masked_fill((router_indices // num_local_experts) != ep_rank, 0)
        router_indices = router_indices % num_local_experts
        return router_scores, router_indices

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        # TODO: i'd like for this to be the default
        param = param[...].to(param_casting_dtype)
        if to_contiguous:
            param = param.contiguous()
        return param

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        # TODO: need an abstract Parallel class that is different from TensorParallelLayer
        distribute_module(
            module,
            device_mesh,
            partial(self._prepare_input_fn, None, None),
            partial(self._prepare_output_fn, None, None),
        )


class ParallelInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given entry)
    _global_mapping = (
        {
            "colwise": ColwiseParallel(),
            "rowwise": RowwiseParallel(),
            "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
            "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
            "local_colwise": ColwiseParallel(use_dtensor=False),
            "local_rowwise": RowwiseParallel(use_dtensor=False),
            "local": IsolatedParallel(),
            "gather": GatherParallel(),
            "local_packed_rowwise": PackedRowwiseParallel(use_dtensor=False),
            "sequence_parallel": SequenceParallel(),
            "replicate": ReplicateParallel(),
            "grouped_gemm": GroupedGemmParallel(),
            "ep_router": RouterParallel(),
        }
        if is_torch_greater_or_equal("2.5") and _torch_distributed_available
        else {}
    )


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


def convert_local_tensor_to_dtensor(
    parameter: torch.Tensor, parameter_name: str, device_mesh, tp_plan: dict[str, str]
) -> DTensor:
    """
    Converts a local variant of weights to a DTensor with corresponding placements. Shouldn't be done ever except of before saving the model.
    """
    _, param_type = parameter_name.rsplit(".", 1) if "." in parameter_name else parameter_name
    tp_style = _get_parameter_tp_plan(parameter_name, tp_plan)
    if not tp_style:
        return parameter

    if tp_style not in ["local_packed_rowwise", "local_rowwise", "local_colwise"]:
        return parameter
    # TODO: this logic should be wrapped in a function, this is copied from corresponding tp classes.
    if tp_style == "local_packed_rowwise":
        placements = [Shard(-1)]
    elif tp_style == "local_rowwise":
        if param_type == "bias":
            placements = [Replicate()]
        else:
            placements = [Shard(-1)]
    elif tp_style == "local_colwise":
        if param_type == "bias":
            placements = [Shard(-1)]
        else:
            placements = [Shard(-2)]
    return DTensor.from_local(parameter, device_mesh, placements, run_check=False)


def replace_state_dict_local_with_dtensor(
    state_dict: dict[str, torch.Tensor],
    tp_plan: dict[str, str],
    device_mesh,
) -> dict[str, torch.Tensor]:
    """
    Replaces all tensors that were sharded with `local_*` strategy with DTensor to make determining their proper size possible.
    """
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and not isinstance(value, DTensor):
            state_dict[key] = convert_local_tensor_to_dtensor(value, key, device_mesh, tp_plan)
    return state_dict


def add_tensor_parallel_hooks_to_module(
    model, module, tp_plan, layer_name, current_module_plan, device_mesh, parameter_name=None
):
    r"""
    This function is called in `PretrainedModel.post_init()`. It is responsible of adding hooks
    to the modules of the `model`, based on the `PretrainedModel._tp_plan`.

    This is the place where we add the `pre_forward` and `post_forwards` hooks. These are defined
    for each `TensorParallelLayer` as `_prepare_input_fn` and `_prepare_output_fn`.

    """
    if current_module_plan is not None:
        tp_layer = ALL_PARALLEL_STYLES[current_module_plan]
        try:
            tp_layer.prepare_module_tp(module, device_mesh)
        except NotImplementedError as e:
            print(
                f"Trying to prepare {layer_name}, but it's not supported. Corresponding module: {module} Fix it's TP plan: {e}"
            )

        module._hf_tp_plan = current_module_plan
        module.__repr__ = lambda: f"{module.__repr__()}\nTP Plan: {current_module_plan}"


def shard_and_distribute_module(
    model, param, empty_param, parameter_name, param_casting_dtype, is_contiguous, rank, device_mesh
):  # TODO: rename to shard_and_distribute_param
    r"""
    This function is called in `from_pretrained` when loading a model's checkpoints.
    It receives the pointer to the parameter (or the parameter itself) and takes care of "sharding".
    All process run this function, so they just load the partition of the tensor that they require.

    Main uses cases:
    - column / rowise parallelism, you just shard all the weights of the layer (weight and bias)
    - packed layers: you slice the weights, then shard like above
    - custom operation:
        - you want to add an all-gather at the end of a local layer.
        - you want to have a layer that is isolated from the rest of the world (because torch.DTensor does not work well with `.view` for instance)

    """
    param_name, param_type = parameter_name.rsplit(".", 1) if "." in parameter_name else parameter_name
    tp_plan = model._tp_plan or {}
    tp_plan.update(getattr(type(model), "_tp_plan", {}))
    module_to_tp = model.get_submodule(param_name)  # TODO: can i loop over modules?
    rank = int(rank)
    current_shard_plan = _get_parameter_tp_plan(parameter_name, tp_plan)

    if dist.get_rank() == 0:
        if current_shard_plan is None:
            logger.info(f"Tensor sharding plan for {param_name} not found, using default 'replicate' plan.")
        else:
            logger.info(f"Tensor sharding plan for {param_name}: {current_shard_plan}")

    if current_shard_plan is not None:
        try:
            tp_layer = ALL_PARALLEL_STYLES[current_shard_plan]
            param = tp_layer.partition_tensor(
                param, empty_param, param_type, param_casting_dtype, is_contiguous, rank, device_mesh
            )
        except NotImplementedError as e:
            print(
                f"Trying to prepare {parameter_name}, but it's not supported. Corresponding module: {module_to_tp} Fix it's TP plan, current layer: {tp_layer} : {e}"
            )
    else:
        param = param[:].to(param_casting_dtype)

    # SUPER IMPORTANT we have to use setattr
    # otherwise loading is crazy slow
    if not isinstance(param, torch.nn.Parameter):
        param = torch.nn.Parameter(param, requires_grad=empty_param.is_floating_point())
    setattr(module_to_tp, param_type, param)
    # module_to_tp.load_state_dict({param_type: param}, strict=False, assign=True)
    return param


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.
    """

    if tp_plan is None:
        return

    generic_keys = {re.sub(r"\d+", "*", key) for key in expected_keys}
    unsharded_layers = set(generic_keys)
    unused_rules = tp_plan

    for key in generic_keys:
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = re.sub(r"\d+", "*", param_name)

        if generic_param_name in tp_plan:
            unused_rules.pop(generic_param_name)
            unsharded_layers.discard(key)
        elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
            unused_rules.pop(parent_param_name)
            unsharded_layers.discard(key)
        else:
            pass  # we couldn't find the rule for this parameter, so it's not sharded

    if len(unused_rules) > 0:
        logger.warning(f"The following TP rules were not applied on any of the layers: {unused_rules}")
    if len(unsharded_layers) > 0:
        logger.warning(f"The following layers were not sharded: {', '.join(unsharded_layers)}")


def distribute_model(model, distributed_config, device_mesh, tp_size):
    _plan = "_tp_plan"
    tp_plan = getattr(model, "_tp_plan", {}).copy()
    model._tp_plan = getattr(model.config, "base_model_tp_plan").copy()
    model._tp_plan.update(tp_plan)
    model._tp_size = tp_size
    model._device_mesh = device_mesh
    if distributed_config is not None:
        if isinstance(distributed_config, dict):
            distributed_config = DistributedConfig.from_dict(distributed_config)
        if distributed_config.enable_expert_parallel:
            _plan = "_ep_plan"
            model._tp_plan = getattr(model.config, "base_model_ep_plan", model._tp_plan).copy()

    # now fetch my childrens
    for name, module in model.named_children():
        if plan := getattr(module, _plan, getattr(module, "tp_plan", None)):
            model._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
        if hasattr(module, "config"):
            plan = getattr(module.config, f"base_model{_plan}", {})
            if plan == {}:
                plan = getattr(module.config, "base_model_tp_plan", {})
            model._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})

    if model._tp_plan is not None and is_torch_greater_or_equal("2.5") and _torch_distributed_available:
        for v in model._tp_plan.values():
            if v not in ALL_PARALLEL_STYLES:
                raise ValueError(f"Unsupported tensor parallel style {v}. Supported styles are {ALL_PARALLEL_STYLES}")
        for name, module in model.named_modules():
            if not getattr(module, "_is_hooked", False):
                from transformers.integrations.tensor_parallel import add_tensor_parallel_hooks_to_module

                plan = _get_parameter_tp_plan(parameter_name=name, tp_plan=model._tp_plan, is_weight=False)
                add_tensor_parallel_hooks_to_module(
                    model=model,
                    module=module,
                    tp_plan=model._tp_plan,
                    layer_name="",
                    current_module_plan=plan,
                    device_mesh=device_mesh,
                )
            module._is_hooked = True
    return model
