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
from functools import reduce

from ..distributed import DistributedConfig
from ..utils import is_torch_greater_or_equal, logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    from torch import nn

    # Cache this result has it's a C FFI call which can be pretty time-consuming
    _torch_distributed_available = torch.distributed.is_available()


logger = logging.get_logger(__name__)


def initialize_tensor_parallelism(
    tp_plan: str | dict[str, str] | None, tp_size: int | None = None, device_mesh=None, device_map=None
):
    r"""
    Sets up the device mesh and initialized the backend for tensor parallelism.
    This function is called when the model is loaded and the TP plan is set to 'auto'.
    """
    if tp_size is not None and tp_plan is None:
        raise ValueError("tp_plan has to be set when tp_size is passed.")
    if tp_plan is not None and device_map is not None:
        raise ValueError("`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization.")
    if device_mesh is None:
        if not is_torch_greater_or_equal("2.5"):
            raise OSError("Tensor parallel is only supported for `torch>=2.5`.")

        # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
        device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            device_type = "cpu"  # fallback
        current_device = getattr(torch, device_type)
        if not torch.distributed.is_initialized():
            try:
                rank = int(os.environ["RANK"])
                local_rank = int(os.environ["LOCAL_RANK"])
                world_size = int(os.environ["WORLD_SIZE"])

                backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
                backend = backend_map.get(device_type)
                if device_type == "cpu" and int(os.environ.get("CCL_WORKER_COUNT", "0")):
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
                    "sure you init torch distributed in your script to use `tp_plan`."
                ) from e

        if device_type != "cpu":
            current_device.set_device(int(os.environ["LOCAL_RANK"]))
            index = current_device.current_device()
            tp_device = torch.device(device_type, index)
            device_map = tp_device
            # Silence output for non-primary ranks
            if index > 0:
                import sys

                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

        else:
            tp_device = torch.device(device_type)
            device_map = device_type or {}

        tp_size = tp_size if tp_size is not None else torch.distributed.get_world_size()
        device_mesh = torch.distributed.init_device_mesh(tp_device.type, (tp_size,))
    else:
        if device_mesh.ndim > 1:
            if "tp" not in device_mesh.mesh_dim_names:
                raise ValueError(
                    "When using `tp_plan` and n-d `device_mesh`, it must contain a 'tp' dimension. "
                    "Please provide a valid `device_mesh`."
                )
            device_mesh = device_mesh["tp"]
        tp_size = device_mesh.size()
        device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")

    return device_map, device_mesh, tp_size


def replace_layer_number_by_wildcard(name: str) -> str:
    """
    Replace the numbers in the `name` by wildcards, only if they are in-between dots (`.`) or if they are between
    a dot (`.`) and the end of the string.
    This matches how modules are named/numbered when using a nn.ModuleList or nn.Sequential, but will NOT match
    numbers in a parameter name itself, e.g. if the param is named `"w1"` or `"w2"`.
    """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def _get_parameter_tp_plan(parameter_name: str, tp_plan: dict[str, str], is_weight=True) -> str | None:
    """
    Get the TP style for a parameter from the TP plan.

    The TP plan is a dictionary that maps parameter names to TP styles.
    The parameter name can be a generic name with wildcards (e.g. "*.weight") or a specific name (e.g. "layer_1.weight").

    The `is_weight` is important because for weights, we want to support `.weights` and `.bias` cases seamlessly! but
    not parent classes for `post_init` calls
    """
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in tp_plan:
        return tp_plan[generic_param_name]
    elif is_weight and "." in generic_param_name and (module_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
        return tp_plan[module_name]
    return None


# =============================================================================
# Tensor Sharding Utilities
# =============================================================================


if is_torch_available():
    str_to_dtype = {
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
        return tensor.to(str_to_dtype[slice_dtype])


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


def get_tensor_shard(param, empty_param, device_mesh, rank, dim, tensor_idx: int | None = None):
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
    param_dim = empty_param.ndim
    # Flatten the mesh to get the total number of devices
    mesh_shape = device_mesh.shape
    world_size = reduce(operator.mul, mesh_shape)
    if dim < 0:
        dim = param_dim + dim
    if empty_param.dim() == 3 and dim == 1 and len(param.get_shape()) == 2:
        dim = 0
    elif empty_param.dim() == 3 and dim == 2 and len(param.get_shape()) == 2:
        dim = 0

    shard_size = math.ceil(empty_param.size(dim) / world_size)
    start = rank * shard_size
    end = min(start + shard_size, empty_param.size(dim))

    if dim >= param_dim:
        raise ValueError(f"dim {dim} is out of bounds for tensor of dimension {param_dim}")

    if rank >= world_size:
        raise ValueError(f"Rank {rank} is out of bounds for mesh size {world_size}")

    # we have the full tensor not 1 part of it.
    # in that case, we just assume that the weight was properly saved
    # and thus because we TP if the layer is colwise it should not use this. Layer should be packed_colwise
    # to inform that it needs to read form a packed tensor. It will also take care of the module list thingy.
    # here we take care of potential chunking / layer split / layer chunking.
    # The only "hard" case is? if we collect q,k,v -> merge it into qkv. In that case
    # actually we still shard dim=0 does not change
    # so only case is if the dim of the empty param is 3 and the shard dim is 0 -> we put the
    # tensor on a certain device (with the input tensor_index)
    dimensions = param.get_shape()

    if empty_param.dim() == 3 and dim == 0 and len(param.get_shape()) == 2:
        # special case we don't "shard" just send this entire tensor to the correct rank.
        if start <= tensor_idx < end:
            # this tensor does need to be materialized on this device:
            return param[:]
        else:
            return torch.empty([], dtype=torch.int64, device=rank)

    slice_indices = [slice(None)] * len(param.get_shape())

    if start < param.get_shape()[dim]:
        slice_indices[dim] = slice(start, end)
        param = param[tuple(slice_indices)]
        if isinstance(param, list):  # TODO handle the modulelist case!
            param = [p[:] for p in param]
        return param

    dimensions[dim] = 0
    return torch.empty(tuple(dimensions), dtype=torch.int64)  # empty allocates memory....


def _split_along_last_dim(x, world_size):
    """Split tensor along last dimension into world_size chunks."""
    return torch.chunk(x, world_size, dim=-1)


# =============================================================================
# Distributed Communication
# =============================================================================


class _AllReduceBackward(torch.autograd.Function):
    """Identity forward, all-reduce backward. Used before colwise layers (f in Megatron)."""

    @staticmethod
    def forward(ctx, x, device_mesh):
        ctx.device_mesh = device_mesh
        return x

    @staticmethod
    def backward(ctx, grad_output):
        device_mesh = ctx.device_mesh
        if device_mesh.size() == 1:
            return grad_output, None
        #TODO(3outeille): do it for other reduce ops as well
        grad_output = grad_output.clone()  # Clone to avoid in-place mutation (compile-compatible)
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=device_mesh.get_group())
        return grad_output, None


class _AllReduceForward(torch.autograd.Function):
    """All-reduce forward, identity backward. Used after rowwise layers (g in Megatron)."""

    @staticmethod
    def forward(ctx, x, device_mesh):
        if device_mesh.size() == 1:
            return x
        x = x.clone()  # Clone to avoid in-place mutation (compile-compatible)
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=device_mesh.get_group())
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _AllGather(torch.autograd.Function):
    """All-gather forward, split backward. Gathers sharded outputs."""

    @staticmethod
    def forward(ctx, x, device_mesh):
        ctx.device_mesh = device_mesh
        world_size = device_mesh.size()

        if world_size == 1:
            return x

        last_dim = x.dim() - 1
        rank = device_mesh.get_local_rank()
        group = device_mesh.get_group()

        x = x.contiguous()
        tensor_list = [torch.empty_like(x) for _ in range(world_size)]
        tensor_list[rank] = x
        dist.all_gather(tensor_list, x, group=group)
        return torch.cat(tensor_list, dim=last_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        device_mesh = ctx.device_mesh
        world_size = device_mesh.size()

        if world_size == 1:
            return grad_output, None

        rank = device_mesh.get_local_rank()
        chunks = _split_along_last_dim(grad_output, world_size)
        return chunks[rank].contiguous(), None


class _Split(torch.autograd.Function):
    """Split forward, all-gather backward. Scatters replicated input."""

    @staticmethod
    def forward(ctx, x, device_mesh):
        ctx.device_mesh = device_mesh
        world_size = device_mesh.size()

        if world_size == 1:
            return x

        rank = device_mesh.get_local_rank()
        chunks = _split_along_last_dim(x, world_size)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        device_mesh = ctx.device_mesh
        world_size = device_mesh.size()

        if world_size == 1:
            return grad_output, None

        last_dim = grad_output.dim() - 1
        rank = device_mesh.get_local_rank()
        group = device_mesh.get_group()

        grad_output = grad_output.contiguous()
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        tensor_list[rank] = grad_output
        dist.all_gather(tensor_list, grad_output, group=group)
        return torch.cat(tensor_list, dim=last_dim).contiguous(), None


class _ReduceScatter(torch.autograd.Function):
    """Reduce-scatter forward, all-gather backward. For sequence parallel."""

    @staticmethod
    def forward(ctx, x, device_mesh):
        ctx.device_mesh = device_mesh
        world_size = device_mesh.size()

        if world_size == 1:
            return x

        last_dim = x.dim() - 1
        group = device_mesh.get_group()

        input_chunks = list(x.chunk(world_size, dim=last_dim))
        output_shape = list(x.shape)
        output_shape[last_dim] //= world_size
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        dist.reduce_scatter(output, input_chunks, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        device_mesh = ctx.device_mesh
        world_size = device_mesh.size()

        if world_size == 1:
            return grad_output, None

        last_dim = grad_output.dim() - 1
        rank = device_mesh.get_local_rank()
        group = device_mesh.get_group()

        grad_output = grad_output.contiguous()
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        tensor_list[rank] = grad_output
        dist.all_gather(tensor_list, grad_output, group=group)
        return torch.cat(tensor_list, dim=last_dim).contiguous(), None


# =============================================================================
# Convenience wrappers
# =============================================================================


def all_reduce_backward(x, device_mesh):
    """Identity forward, all-reduce backward. Use before colwise layers."""
    return _AllReduceBackward.apply(x, device_mesh)


def all_reduce_forward(x, device_mesh):
    """All-reduce forward, identity backward. Use after rowwise layers."""
    return _AllReduceForward.apply(x, device_mesh)


def all_gather(x, device_mesh):
    """All-gather forward, split backward."""
    return _AllGather.apply(x, device_mesh)


def split(x, device_mesh):
    """Split forward, all-gather backward."""
    return _Split.apply(x, device_mesh)


def reduce_scatter(x, device_mesh):
    """Reduce-scatter forward, all-gather backward."""
    return _ReduceScatter.apply(x, device_mesh)


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
    if input_fn is not None:
        module.register_forward_pre_hook(lambda mod, inputs: input_fn(mod, inputs, device_mesh))
    if output_fn is not None:
        module.register_forward_hook(lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh))
    return module


class TensorParallelLayer:
    """General tensor parallel layer for transformers"""

    device_mesh = None
    rank = None
    empty_param = None

    def __init__(self, device_mesh=None, rank=None, empty_param=None):
        self.rank = rank
        self.device_mesh = device_mesh
        self.empty_param = empty_param

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh): ...

    @staticmethod
    def _prepare_output_fn(mod, outputs, device_mesh): ...

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        raise NotImplementedError

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )


class ColwiseParallel(TensorParallelLayer):
    """
    Column-wise parallel: weight is sharded on dim -2 (output features).
    Forward: input replicated -> output sharded on last dim.
    If gather_output=True, output is all-gathered to produce full tensor.
    """

    def __init__(self, gather_output: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gather_output = gather_output

    def _prepare_input_fn(self, mod, inputs, device_mesh):
        input_tensor = inputs[0] if inputs else inputs
        return all_reduce_backward(input_tensor, device_mesh)

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        if self.gather_output:
            return all_gather(outputs, device_mesh)
        return outputs

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        device_mesh = self.device_mesh
        empty_param = self.empty_param
        rank = self.rank
        if param_type == "bias":
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -1, tensor_idx)
        else:
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -2, tensor_idx)
        parameter = parameter.to(param_casting_dtype)
        return parameter, None

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )


class RowwiseParallel(TensorParallelLayer):
    """
    Row-wise parallel: weight is sharded on dim -1 (input features).
    Forward: input (optionally split) -> output partial -> all-reduce to replicate.

    Args:
        split_input: If True, splits replicated input before matmul. Use when input
                     comes from a non-parallelizable operation (chunk/slice).
                     Default False (expects pre-sharded input from colwise layer).
    """

    def __init__(self, split_input: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.split_input = split_input

    def _prepare_input_fn(self, mod, inputs, device_mesh):
        if hasattr(mod, "bias") and mod.bias is not None:
            mod._bias = mod.bias
            mod.bias = None

        input_tensor = inputs[0] if inputs else inputs

        if self.split_input:
            # Input is replicated, split it to match sharded weight
            return split(input_tensor, device_mesh)
        return input_tensor

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        outputs = all_reduce_forward(outputs, device_mesh)
        if hasattr(mod, "_bias") and mod._bias is not None:
            outputs = outputs + mod._bias
        return outputs

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        device_mesh = device_mesh or self.device_mesh
        empty_param = self.empty_param
        rank = rank if rank is not None else self.rank
        if param_type == "bias":
            parameter = param[...]
        else:
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -1, tensor_idx=tensor_idx)
        parameter = parameter.to(param_casting_dtype)
        return parameter, None

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        module._distribute_module_applied = True
        distribute_module(
            module,
            device_mesh,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )


class PackedColwiseParallel(ColwiseParallel):
    """Packed column-wise parallel for fused weights like gate_up_proj."""

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        device_mesh = device_mesh or self.device_mesh
        empty_param = self.empty_param
        rank = rank if rank is not None else self.rank

        if param_type == "bias":
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -1, tensor_idx)
        else:
            parameter = get_packed_weights(param, empty_param, device_mesh, rank, -2)

        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        return parameter, None


class PackedRowwiseParallel(RowwiseParallel):
    """Packed row-wise parallel for fused weights like gate_up_proj."""

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        device_mesh = device_mesh or self.device_mesh
        empty_param = self.empty_param
        rank = rank if rank is not None else self.rank

        if param_type == "bias":
            parameter = param[...]
        else:
            parameter = get_packed_weights(param, empty_param, device_mesh, rank, -1)

        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        return parameter, None


class EmbeddingParallel(TensorParallelLayer):
    """EmbeddingParallel: shards embedding table, handles masked lookups for vocab parallelism."""

    def __init__(self, *, embedding_dim_sharding: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim_sharding = embedding_dim_sharding

    def _prepare_input_fn(self, mod, inputs, device_mesh):
        input_tensor = inputs[0] if inputs else inputs

        # For vocab-parallel (dim 0), we need to handle masking and offsetting
        if self.embedding_dim_sharding == 0:
            rank = device_mesh.get_local_rank()

            # Get vocab range for this rank
            # Use weight.shape[0] to get the actual local (sharded) size, not num_embeddings
            # which may not be updated after sharding
            per_partition_size = mod.weight.shape[0]
            vocab_start_index = rank * per_partition_size
            vocab_end_index = vocab_start_index + per_partition_size

            # Build mask for out-of-vocabulary tokens
            input_mask = (input_tensor < vocab_start_index) | (input_tensor >= vocab_end_index)
            mod._input_mask = input_mask

            # Offset input to local indices and mask invalid ones
            masked_input = input_tensor.clone() - vocab_start_index
            masked_input[input_mask] = 0  # Set to valid local index

            return masked_input

        return input_tensor

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        # For vocab-parallel (dim 0), zero out embeddings for out-of-range tokens before all-reduce
        if self.embedding_dim_sharding == 0 and hasattr(mod, "_input_mask"):
            input_mask = mod._input_mask
            # Use multiplication instead of in-place assignment to preserve gradients
            mask_expanded = input_mask.unsqueeze(-1).expand_as(outputs)
            outputs = outputs * (~mask_expanded).float()
            del mod._input_mask

        return all_reduce_forward(outputs, device_mesh)

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        device_mesh = device_mesh or self.device_mesh
        empty_param = self.empty_param
        rank = rank if rank is not None else self.rank
        if param_type == "bias":
            parameter = get_tensor_shard(param, empty_param, device_mesh, rank, -1, tensor_idx=tensor_idx)
        else:
            parameter = get_tensor_shard(
                param, empty_param, device_mesh, rank, self.embedding_dim_sharding, tensor_idx=tensor_idx
            )
        parameter = parameter.to(param_casting_dtype)
        return parameter, None

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        module._distribute_module_applied = True
        distribute_module(
            module,
            device_mesh,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )


class SequenceParallel(TensorParallelLayer):
    """
    Sequence Parallel: input/output sharded on sequence dimension.
    Weights are replicated.
    """

    def __init__(self, sequence_dim: int = 1, use_local_output: bool = False, use_dtensor=False, **kwargs):
        super().__init__(**kwargs)
        self.sequence_dim = sequence_dim

    def _prepare_input_fn(self, mod, inputs, device_mesh):
        input_tensor = inputs[0] if inputs else inputs
        # For sequence parallel, input is sharded on sequence dim
        # All-gather for the layer, then reduce-scatter after
        return all_gather(input_tensor, device_mesh)

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        return reduce_scatter(outputs, device_mesh)

    def shard_tensor(
        self,
        param: torch.Tensor,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        parameter = param[...].to(param_casting_dtype)
        return parameter, None

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )


class GroupedGemmParallel(TensorParallelLayer):
    """
    Applies Expert Parallelism to MoE experts by loading the correct experts on each device.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def shard_tensor(
        self,
        param: torch.Tensor,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx: int | None = None,
    ) -> torch.Tensor:
        device_mesh = device_mesh or self.device_mesh
        rank = rank if rank is not None else self.rank
        global_num_experts = self.empty_param.shape[0]
        if global_num_experts % device_mesh.size() != 0:
            raise ValueError(
                f"Global number of experts must be divisible by number of devices: {global_num_experts} % {device_mesh.size()} != 0"
            )
        local_num_experts = global_num_experts // device_mesh.size()
        parameter = param[rank * local_num_experts : (rank + 1) * local_num_experts].to(param_casting_dtype)
        return parameter, None


class RouterParallel(TensorParallelLayer):
    """
    Allows to reshape the router scores to support running expert parallel.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
        return inputs[0] if inputs else inputs

    @staticmethod
    def _prepare_output_fn(mod, outputs, device_mesh):
        """
        Imagine if you had 4 tokens, top_k = 4, and 128experts.
        With EP = 8. The num_local_expert should be 128/8 = 16
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

        Thus for say rank 0, you fill with 16 (num_local_expert) the index tensor

        [ 16, 16, 16, 16],
        [ 16, 16, 16, 16],
        [ 16, 16, 4, 16],
        [ 16, 16, 16, 11],

        This works well. For another rank you need to make sure you round to num_local_expert
        because the next operation will one hot encode the router index vector.

        This allows us to know directly which local expert is hit.
        Similarly the scores are indexed with something created form
        router_indices.

        The kinda naive training loop that we use for device_map "auto" uses a similar logic.
        Here we are just making each rank believe that he is alone, and he computes his part of the hiddenstates.
        Mask invalid indices with num_local_expert for one-hot encoding, so the computes will skip the masking index.
        """
        ep_rank, ep_size = device_mesh.get_local_rank(), device_mesh.size()
        if mod.num_experts % ep_size != 0:
            raise ValueError(
                f"The number of experts must be divisible by number of ep_size: {mod.num_experts} % {ep_size} != 0"
            )
        num_local_experts = mod.num_experts // ep_size
        router_logits, router_scores, router_indices = outputs
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_scores)
        router_scores = router_scores[:, ep_rank * num_local_experts : (ep_rank + 1) * num_local_experts]
        router_indices = router_indices.masked_fill((router_indices // num_local_experts) != ep_rank, -1)
        # As -1 % 1 is 0, we can only use mask fill when num_local_experts is 1
        if num_local_experts > 1:
            router_indices = torch.fmod(router_indices, num_local_experts)
        else:
            router_indices = router_indices.masked_fill(router_indices > 0, 0).masked_fill(router_indices < 0, -1)
        router_indices = router_indices.masked_fill(router_indices == -1, num_local_experts)
        return router_logits, router_scores, router_indices

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        parameter = param[...].to(param_casting_dtype)
        return parameter, None

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )


class AllReduceOutput(TensorParallelLayer):
    """
    Module-level parallel that applies all-reduce on output (forward) and input gradient (backward).

    Use this when a module's internal computation produces partial sums that need
    to be reduced across TP ranks, but the module uses custom forward logic
    (e.g., nn.functional.linear) rather than nn.Linear modules where RowwiseParallel
    output hooks would be triggered.

    This implements the correct gradient flow for row-parallel style computations:
    - Forward: all_reduce(partial_output) -> full_output
    - Backward: all_reduce(partial_input_grad) -> full_input_grad
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
        return inputs

    @staticmethod
    def _prepare_output_fn(mod, outputs, device_mesh):
        return all_reduce_forward(outputs, device_mesh)

    def shard_tensor(
        self,
        param,
        param_type=None,
        param_casting_dtype=None,
        to_contiguous=None,
        rank=None,
        device_mesh=None,
        tensor_idx=None,
    ):
        # This class doesn't shard tensors - sharding is handled by packed_colwise/rowwise
        # on the individual weight tensors (gate_up_proj/down_proj)
        parameter = param[...].to(param_casting_dtype)
        return parameter, None

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )
        # Store device_mesh on module so MoE forward functions can use it for gradient sync
        module._device_mesh = device_mesh
        print(f"[DEBUG] AllReduceOutput.prepare_module_tp: Set _device_mesh on {module.__class__.__name__}")

class ParallelInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given entry)
    _global_mapping = (
        {
            "embedding_rowwise": EmbeddingParallel(embedding_dim_sharding=0),
            "colwise_gather_output": ColwiseParallel(gather_output=True),
            "colwise": ColwiseParallel(),
            "rowwise": RowwiseParallel(),
            "rowwise_split_input": RowwiseParallel(split_input=True),
            "packed_colwise": PackedColwiseParallel(),
            "packed_rowwise": PackedRowwiseParallel(),
            "sequence_parallel": SequenceParallel(),
            "grouped_gemm": GroupedGemmParallel(),
            "ep_router": RouterParallel(),
            "all_reduce_output": AllReduceOutput(),
        }
        if is_torch_available() and _torch_distributed_available
        else {}
    )


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


# =============================================================================
# High-Level API Functions
# =============================================================================


def gather_full_tensor(local_tensor: torch.Tensor, shard_dim: int, device_mesh) -> torch.Tensor:
    """
    All-gather a sharded tensor along the specified dimension to reconstruct the full tensor.

    Args:
        local_tensor: The local shard of the tensor on this rank
        shard_dim: The dimension along which the tensor was sharded
        device_mesh: The device mesh for distributed communication

    Returns:
        The full reconstructed tensor (same on all ranks)
    """
    world_size = device_mesh.size()

    # Normalize negative dimension
    if shard_dim < 0:
        shard_dim = local_tensor.ndim + shard_dim

    # Gather all shards
    gathered_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor.contiguous())

    # Concatenate along the shard dimension
    return torch.cat(gathered_tensors, dim=shard_dim)


def gather_state_dict_for_save(
    state_dict: dict[str, torch.Tensor],
    tp_plan: dict[str, str],
    device_mesh,
    tp_size: int,
) -> dict[str, torch.Tensor]:
    """
    Gather sharded tensors to reconstruct full tensors for saving.

    This function all-gathers each sharded tensor along its shard dimension
    to reconstruct the full unsharded tensor for checkpoint saving.

    Args:
        state_dict: The model state dict with local sharded tensors
        tp_plan: The tensor parallel plan mapping layer patterns to shard styles
        device_mesh: The device mesh for distributed communication
        tp_size: The tensor parallel world size

    Returns:
        State dict with full (gathered) tensors
    """
    # Map plan names to sharding dimensions
    # For weights: colwise shards dim -2, rowwise shards dim -1
    # For embedding: rowwise shards dim 0 (vocab), colwise shards dim -2 (hidden)
    plan_to_weight_dim = {
        "colwise": -2,
        "colwise_gather_output": -2,
        "packed_colwise": -2,
        "rowwise": -1,
        "rowwise_split_input": -1,
        "packed_rowwise": -1,
        "embedding_rowwise": 0,
        "sequence_parallel": None,
    }

    # Bias sharding: colwise shards bias, rowwise doesn't (bias is replicated and all-reduced)
    plan_to_bias_dim = {
        "colwise": -1,
        "colwise_gather_output": -1,
        "packed_colwise": -1,
        "rowwise": None,
        "rowwise_split_input": None,
        "packed_rowwise": None,
        "embedding_rowwise": None,
        "sequence_parallel": None,
    }

    result = {}
    for key, tensor in state_dict.items():
        # Find the matching TP plan for this parameter
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        param_type = key.rsplit(".", 1)[1] if "." in key else None
        generic_param_name = re.sub(r"\d+", "*", param_name)

        # Check if this parameter has a TP plan
        current_plan = None
        if generic_param_name in tp_plan:
            current_plan = tp_plan[generic_param_name]
        elif "." in generic_param_name:
            parent_param_name = generic_param_name.rsplit(".", 1)[0]
            if parent_param_name in tp_plan:
                current_plan = tp_plan[parent_param_name]

        if current_plan is None or current_plan not in plan_to_weight_dim:
            # Not sharded, keep as-is
            result[key] = tensor
            continue

        # Determine sharding dimension based on param type
        if param_type == "bias":
            shard_dim = plan_to_bias_dim.get(current_plan)
        else:
            shard_dim = plan_to_weight_dim.get(current_plan)

        if shard_dim is None:
            # Replicated, keep as-is
            result[key] = tensor
            continue

        # Gather full tensor and handle packed weights repacking
        full_tensor = gather_full_tensor(tensor, shard_dim, device_mesh)
        if current_plan in ("packed_colwise", "packed_rowwise"):
            full_tensor = repack_weights(full_tensor, shard_dim, tp_size, 2)
        result[key] = full_tensor.contiguous()

    return result


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
):
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
    tp_plan = model.tp_plan or {}
    module_to_tp = model.get_submodule(param_name)
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
            tp_layer.empty_param = empty_param
            tp_layer.device_mesh = device_mesh
            tp_layer.rank = rank
            param, _ = tp_layer.shard_tensor(
                param, param_type=param_type, param_casting_dtype=param_casting_dtype, tensor_idx=None
            )
            if is_contiguous:
                param = param.contiguous()
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
    return param


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.
    """

    if tp_plan is None:
        return

    generic_keys = {replace_layer_number_by_wildcard(key) for key in expected_keys}
    unsharded_layers = set(generic_keys)
    unused_rules = tp_plan.copy()

    for key in generic_keys:
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = re.sub(r"\d+", "*", param_name)

        if generic_param_name in tp_plan:
            unused_rules.pop(generic_param_name, None)
            unsharded_layers.discard(key)
        elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
            unused_rules.pop(parent_param_name, None)
            unsharded_layers.discard(key)

    if len(unused_rules) > 0:
        logger.warning(f"The following TP rules were not applied on any of the layers: {unused_rules}")
    if len(unsharded_layers) > 0:
        logger.warning(f"The following layers were not sharded: {', '.join(unsharded_layers)}")


def distribute_model(model, tp_plan, distributed_config, device_mesh, tp_size):
    """Distribute a model according to the TP plan."""
    model._tp_size = tp_size
    model._device_mesh = device_mesh
    if distributed_config is not None:
        if isinstance(distributed_config, dict):
            distributed_config = DistributedConfig.from_dict(distributed_config)
        model.config.distributed_config = distributed_config
    # Set the new requested tp_plan on the model
    if isinstance(tp_plan, dict):
        model.tp_plan = tp_plan
    model_plan = model.tp_plan
    if model_plan is not None and _torch_distributed_available:
        for v in model_plan.values():
            if v not in ALL_PARALLEL_STYLES:
                raise ValueError(f"Unsupported tensor parallel style {v}. Supported styles are {ALL_PARALLEL_STYLES}")
        for name, module in model.named_modules():
            if not getattr(module, "_is_hooked", False):
                plan = _get_parameter_tp_plan(parameter_name=name, tp_plan=model_plan, is_weight=False)
                add_tensor_parallel_hooks_to_module(
                    model=model,
                    module=module,
                    tp_plan=model_plan,
                    layer_name="",
                    current_module_plan=plan,
                    device_mesh=device_mesh,
                )
            module._is_hooked = True
    return model
