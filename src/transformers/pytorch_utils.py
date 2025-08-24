# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
from functools import lru_cache, wraps
from typing import Callable

import torch
from safetensors.torch import storage_ptr, storage_size
from torch import nn

from .utils import is_torch_greater_or_equal, is_torch_xla_available, is_torchdynamo_compiling, logging


ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

logger = logging.get_logger(__name__)

is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
is_torch_greater_or_equal_than_2_4 = is_torch_greater_or_equal("2.4", accept_dev=True)
is_torch_greater_or_equal_than_2_3 = is_torch_greater_or_equal("2.3", accept_dev=True)
is_torch_greater_or_equal_than_2_2 = is_torch_greater_or_equal("2.2", accept_dev=True)
is_torch_greater_or_equal_than_2_1 = is_torch_greater_or_equal("2.1", accept_dev=True)

# For backwards compatibility (e.g. some remote codes on Hub using those variables).
is_torch_greater_or_equal_than_2_0 = is_torch_greater_or_equal("2.0", accept_dev=True)
is_torch_greater_or_equal_than_1_13 = is_torch_greater_or_equal("1.13", accept_dev=True)
is_torch_greater_or_equal_than_1_12 = is_torch_greater_or_equal("1.12", accept_dev=True)

# Cache this result has it's a C FFI call which can be pretty time-consuming
_torch_distributed_available = torch.distributed.is_available()

if is_torch_greater_or_equal("2.5") and _torch_distributed_available:
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


def softmax_backward_data(parent, grad_output, output, dim, self):
    """
    A function that calls the internal `_softmax_backward_data` PyTorch method and that adjusts the arguments according
    to the torch version detected.
    """

    from torch import _softmax_backward_data

    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer ([`~pytorch_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer: nn.Linear | Conv1D, index: torch.LongTensor, dim: int | None = None) -> nn.Linear | Conv1D:
    """
    Prune a Conv1D or linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`Union[torch.nn.Linear, Conv1D]`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear` or [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError(f"Can't prune layer of class {layer.__class__}")


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors,
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def meshgrid(*tensors: torch.Tensor | list[torch.Tensor], indexing: str | None = None) -> tuple[torch.Tensor, ...]:
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    return torch.meshgrid(*tensors, indexing=indexing)


def id_tensor_storage(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    """
    if tensor.device.type == "xla" and is_torch_xla_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)


def isin_mps_friendly(elements: torch.Tensor, test_elements: torch.Tensor | int) -> torch.Tensor:
    """
    Same as `torch.isin` without flags, but MPS-friendly. We can remove this function when we stop supporting
    torch <= 2.3. See https://github.com/pytorch/pytorch/issues/77764#issuecomment-2067838075

    Args:
        elements (`torch.Tensor`): Input elements
        test_elements (`torch.Tensor` or `int`): The elements to check against.

    Returns:
        `torch.Tensor`: A boolean tensor of the same shape as `elements` that is True for `elements` in `test_elements`
        and False otherwise
    """

    if elements.device.type == "mps" and not is_torch_greater_or_equal_than_2_4:
        test_elements = torch.tensor(test_elements)
        if test_elements.ndim == 0:
            test_elements = test_elements.unsqueeze(0)
        return elements.tile(test_elements.shape[0], 1).eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze()
    else:
        # Note: don't use named arguments in `torch.isin`, see https://github.com/pytorch/pytorch/issues/126045
        return torch.isin(elements, test_elements)


# TODO need to add the __repr__ that shows that it is a colwise parallel
# See https://github.com/pytorch/pytorch/issues/145726
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
    else:
        raise ValueError(f"Unsupported parallel style value: {style}")


def compile_compatible_method_lru_cache(*lru_args, **lru_kwargs):
    """
    LRU cache decorator from standard functools library, but with a workaround to disable
    caching when torchdynamo is compiling. Expected to work with class methods.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not is_torchdynamo_compiling():
                # Cache the function only if the model is not being compiled
                # check if the function is already cached, otherwise create it
                if not hasattr(self, f"_cached_{func.__name__}"):
                    self.__setattr__(
                        f"_cached_{func.__name__}", lru_cache(*lru_args, **lru_kwargs)(func.__get__(self))
                    )
                return self.__getattribute__(f"_cached_{func.__name__}")(*args, **kwargs)
            else:
                # Otherwise, just call the original function
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def distribute_module(
    module: nn.Module,
    device_mesh=None,
    partition_fn=None,
    input_fn=None,
    output_fn=None,
) -> nn.Module:
    """
    This function expose three functions to control the parameters/inputs/outputs of the module:

    1. To perform sharding on the module before runtime execution by specifying the
    ``partition_fn`` (i.e. allow user to convert Module parameters to :class:`DTensor`
    parameters according to the `partition_fn` specified).
    2. To control the inputs or outputs of the module during runtime execution by
    specifying the ``input_fn`` and ``output_fn``. (i.e. convert the input to
    :class:`DTensor`, convert the output back to ``torch.Tensor``)

    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the ``device_mesh``). If ``partition_fn`` is not specified,
            by default we replicate all module parameters of ``module`` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. ``input_fn`` will be installed as a module
            ``forward_pre_hook`` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. ``output_fn`` will be
            installed as a module ``forward_hook`` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all ``DTensor`` s.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_module``
        return nn.Module with PyTorch/XLA SPMD annotated parameters. See
        `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.

    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_module")

    device_mesh = device_mesh

    # register input_fn as module forward pre hook
    if input_fn is not None:
        # check the input_fn signature
        num_args = len(inspect.signature(input_fn).parameters)
        if num_args == 2:
            # input_fn only takes in inputs and device mesh
            logger.warning(
                "Deprecating input_fn that takes two arguments (inputs, device_mesh), "
                "please use input_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs, device_mesh))  # type: ignore[call-arg]
        elif num_args == 3:
            # input_fn takes in module, inputs, device mesh
            module.register_forward_pre_hook(lambda mod, inputs: input_fn(mod, inputs, device_mesh))
        else:
            raise ValueError(f"input_fn should take in 3 arguments, but got {num_args} arguments!")
    # register output_fn as module forward hook
    if output_fn is not None:
        num_args = len(inspect.signature(output_fn).parameters)
        if num_args == 2:
            # output_fn only takes in outputs and device mesh
            logger.warning(
                "Deprecating output_fn that takes two arguments (inputs, device_mesh), "
                "please use output_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(outputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            module.register_forward_hook(lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh))
        else:
            raise ValueError(f"output_fn should take in 3 arguments, but got {num_args} arguments!")

    return module
