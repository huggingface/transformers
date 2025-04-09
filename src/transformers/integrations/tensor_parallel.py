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

import re
from functools import lru_cache, partial
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ..utils import is_torch_greater_or_equal, logging


ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

logger = logging.get_logger(__name__)

# Cache this result has it's a C FFI call which can be pretty time-consuming
_torch_distributed_available = torch.distributed.is_available()


if is_torch_greater_or_equal("2.5") and _torch_distributed_available:
    from torch.distributed.tensor import DTensor, Placement, Replicate, Shard


def _blocks_to_block_sizes(total_size: int, blocks: Union[int, List[int]]) -> List[int]:
    """
    Convert block count or proportions to block sizes.

    This function accepts

    - The number of blocks (int), in which case the block size is
      total_size//blocks; or
    - A list of block sizes (List[int]).

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
    if slice_dtype == "F8_E4M3":
        slice_ = slice_[...].to(torch.float16)

    if dim == 0:
        tensor = slice_[tensors_slices, ...]
    elif dim == 1 or dim == -2:
        tensor = slice_[:, tensors_slices, ...]
    elif dim == 2 or dim == -1:
        tensor = slice_[..., tensors_slices]
    else:
        raise ValueError(f"Unsupported dim {dim}, only dim 0, 1 or 2 are supported")
    return tensor.to(str_to_torch_dtype[slice_dtype])


def get_tensor_shard(param, empty_param, device_mesh, rank, dim):
    if dim == 0:
        size_ = empty_param.shape[0]
        param = param[rank * (size_ // device_mesh.size()) : (rank + 1) * (size_ // device_mesh.size()), ...]
    elif dim == 1 or dim == -2:
        size_ = empty_param.shape[-2]
        param = param[..., rank * (size_ // device_mesh.size()) : (rank + 1) * (size_ // device_mesh.size()), :]
    elif dim == 2 or dim == -1:
        size_ = empty_param.shape[-1]
        param = param[..., rank * (size_ // device_mesh.size()) : (rank + 1) * (size_ // device_mesh.size())]
    else:
        raise ValueError(f"Unsupported dim {dim}, only dim 0, 1 or 2 are supported")
    return param


def distribute_module(
    module: nn.Module,
    device_mesh=None,
    input_fn=None,
    output_fn=None,
) -> nn.Module:
    """
    Copy pasted from torch's function but we remove the communications (partitionning)
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
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = output_layouts
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        if inputs and isinstance(inputs[0], DTensor):
            inputs = inputs[0].to_local()
        return inputs

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # this op cannot be asynch, otherwise it completely breaks the outputs of models
        torch.distributed.all_reduce(outputs[0], op=torch.distributed.ReduceOp.SUM, async_op=False)
        return outputs


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

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        distribute_module(
            module,
            device_mesh,
            partial(self._prepare_input_fn, None, None),
            partial(self._prepare_output_fn, None, None),
        )


class ColwiseParallel(TensorParallelLayer):
    """
    General tensor parallel layer for transformers.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
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
            parameter = DTensor.from_local(parameter, device_mesh, shard, run_check=False)
        return nn.Parameter(parameter)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=False)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs


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
        return nn.Parameter(parameter)


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
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
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
            parameter = DTensor.from_local(parameter, device_mesh, shard, run_check=False)
        return nn.Parameter(parameter)

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
        return outputs.to_local() if use_local_output else outputs

    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
        module._distribute_module_applied = True
        if self.use_dtensor:
            if isinstance(module, nn.Linear):
                # rowwise linear runtime sharding requires input tensor shard on last dim
                self.desired_input_layouts: Tuple[Placement, ...] = (Shard(-1),)
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
        return nn.Parameter(parameter)


class SequenceParallel(TensorParallelLayer):
    """
    SequenceParallel replicates a compatible ``nn.Module`` parameters and runs the sharded computation with
    input sharded on the sequence dimension. This currently supports ``nn.LayerNorm``, ``nn.Dropout``, and the
    `RMSNorm python implementation <https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34>`__

    This style implements the operation that is described in the paper
    `Reducing Activation Recomputation in Large Transformer Models <https://arxiv.org/abs/2205.05198>`__

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
        parameter = param[:]
        parameter = parameter.to(param_casting_dtype)
        if to_contiguous:
            parameter = parameter.contiguous()
        if self.use_dtensor:
            parameter = DTensor.from_local(parameter, device_mesh, [Replicate()], run_check=False)
        return nn.Parameter(parameter)


SUPPORTED_TP_STYLES = {
    "colwise",
    "rowwise",
    "colwise_rep",
    "rowwise_rep",
    "local_colwise",
    "local_rowwise",
    "local",
    "gather",
    "local_packed_rowwise",
    "sequence_parallel",
}


@lru_cache
def translate_to_torch_parallel_style(style: str):
    """
    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we translate them into torch.distributed tensor-parallel
    types.
    """
    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "local_colwise":
        return ColwiseParallel(use_dtensor=False)
    elif style == "local_rowwise":
        return RowwiseParallel(use_dtensor=False)
    elif style == "local":
        return IsolatedParallel()
    elif style == "gather":
        return GatherParallel()
    elif style == "local_packed_rowwise":
        return PackedRowwiseParallel(use_dtensor=False)
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unsupported parallel style value: {style}")


def add_tensor_parallel_hooks_to_module(model, module, tp_plan, layer_name, current_module_plan, device_mesh):
    """
    Add hooks to the module holding the layer. Meaning:
    ```
    class MyModel(nn.Module):
        def __init__(self):
            self.layer = nn.Linear(10, 10)
    ```
    has state_dict like:
    ```
    {
        "layer.weight": torch.Tensor,
        "layer.bias": torch.Tensor
    }
    ```
    we add hooks to `MyModel` as well as `layer` to make sure that the tensors are correctly sharded and gathered.
    """

    # 1. We add hooks to the layer being loaded:
    if current_module_plan is not None:
        tp_layer = translate_to_torch_parallel_style(current_module_plan)
        try:
            tp_layer.prepare_module_tp(module, device_mesh)
        except NotImplementedError as e:
            print(
                f"Trying to prepare {layer_name}, but it's not supported. Corresponding module: {module} Fix it's TP plan: {e}"
            )

    # 2. We add hooks to the parrent module if needed
    if "." in layer_name:
        parrent_layer_name = layer_name.rsplit(".", 1)[0]
        generic_name = re.sub(r"\d+", "*", parrent_layer_name)
        # The module itself needs hooks
        if module_plan := tp_plan.get(generic_name, False):
            tp_layer = translate_to_torch_parallel_style(module_plan)
            module_to_tp_ = model.get_submodule(parrent_layer_name)
            tp_layer.prepare_module_tp(module_to_tp_, device_mesh)


def shard_and_distribute_module(
    model, param, empty_param, parameter_name, param_casting_dtype, is_contiguous, rank, device_mesh
):
    r"""
    Main uses cases:
    - column / rowise parallelism, you just shard all the weights of the layer (weight and bias)
    - packed layers: you slice the weights, then shard like above
    - custom operation:
        - you want to add an all-gather at the end of a local layer.
        - you want to have a layer that is isolated from the rest of the world (because torch.DTensor does not work well with `.view` for instance)

    """
    param_name, param_type = parameter_name.rsplit(".", 1) if "." in parameter_name else parameter_name
    tp_plan = model._tp_plan
    module_to_tp = model.get_submodule(param_name)
    current_module_plan = None
    rank = int(rank)
    generic_param_name = re.sub(r"\d+", "*", parameter_name)
    if generic_param_name in tp_plan:
        current_module_plan = tp_plan[generic_param_name]
    elif "." in generic_param_name and generic_param_name.rsplit(".", 1)[0] in tp_plan:
        current_module_plan = tp_plan[generic_param_name.rsplit(".", 1)[0]]

    # Add hooks to the module if not done yet
    # add_tensor_parallel_hooks_to_module(model, module_to_tp, tp_plan, param_name, current_module_plan, device_mesh)
    if not getattr(module_to_tp, "_is_hooked", False):
        add_tensor_parallel_hooks_to_module(model, module_to_tp, tp_plan, param_name, current_module_plan, device_mesh)
        module_to_tp._is_hooked = True

    if current_module_plan is not None:
        try:
            tp_layer = translate_to_torch_parallel_style(current_module_plan)
            param = tp_layer.partition_tensor(
                param, empty_param, param_type, param_casting_dtype, is_contiguous, rank, device_mesh
            )
        except NotImplementedError as e:
            print(
                f"Trying to prepare {parameter_name}, but it's not supported. Corresponding module: {module_to_tp} Fix it's TP plan, current layer: {tp_layer} : {e}"
            )
    else:
        # TODO log no plan modules in set
        # print("No plan for", parameter_name,end ="\n")
        param = param[...].to(param_casting_dtype)
        if is_contiguous:
            param = param.contiguous()

    # SUPER IMPORTANT we have to use setattr
    # otherwise loading is crazy slow
    if not isinstance(param, torch.nn.Parameter):
        param = torch.nn.Parameter(param)
    setattr(module_to_tp, param_type, param)
    # module_to_tp.load_state_dict({param_type: param}, strict=False, assign=True)
    return param
