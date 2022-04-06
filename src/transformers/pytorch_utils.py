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
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from packaging import version
from torch import Tensor, _softmax_backward_data, device, nn

from .generation_utils import GenerationMixin
from .modeling_utils import logger
from .utils import logging


logger = logging.get_logger(__name__)

is_torch_less_than_1_8 = version.parse(torch.__version__) < version.parse("1.8.0")
is_torch_less_than_1_11 = version.parse(torch.__version__) < version.parse("1.11")


def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    if is_torch_less_than_1_8:
        return tensor1 // tensor2
    else:
        return torch.div(tensor1, tensor2, rounding_mode="floor")


def softmax_backward_data(parent, grad_output, output, dim, self):
    """
    A function that calls the internal `_softmax_backward_data` PyTorch method and that adjusts the arguments according
    to the torch version detected.
    """

    if is_torch_less_than_1_11:
        return _softmax_backward_data(grad_output, output, parent.dim, self)
    else:
        return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
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


def get_parameter_device(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


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
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

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
        layer ([`~modeling_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~modeling_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
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


def prune_layer(
    layer: Union[nn.Linear, Conv1D], index: torch.LongTensor, dim: Optional[int] = None
) -> Union[nn.Linear, Conv1D]:
    """
    Prune a Conv1D or linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`Union[torch.nn.Linear, Conv1D]`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear` or [`~modeling_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError(f"Can't prune layer of class {layer.__class__}")


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
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


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except (ImportError):
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except (ImportError):
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None

    def add_memory_hooks(self):
        """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero
        with `model.reset_memory_hooks_state()`.
        """
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype in [torch.bfloat16, torch.float32]:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask

    def create_extended_attention_mask_for_decoder(self, input_shape, attention_mask, device):
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        else:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            return 0

    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)
