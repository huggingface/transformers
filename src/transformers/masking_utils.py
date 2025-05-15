# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
import itertools
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .cache_utils import Cache
from .configuration_utils import PretrainedConfig
from .modeling_utils import GeneralInterface
from .utils.import_utils import is_torchdynamo_compiling, is_torch_greater_or_equal

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return and_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask


def chunked_overlay(chunk_size: int) -> Callable:
    """
    This is an overlay depicting a chuned attention pattern. Add it on top of a causal mask for a proper chunked
    attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx // chunk_size == q_idx // chunk_size

    return inner_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def chunked_causal_mask_function(chunk_size: int) -> Callable:
    """
    This return the mask_function function to create a chunked attention mask.
    """
    return and_masks(chunked_overlay(chunk_size), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)

    return inner_mask


def _vmap_for_q_idx_kv_idx(mask_function: Callable) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).

    Args:
        mask_function (`Callable`):
            The mask_function to vmap.

    Returns:
        Callable: The vmapped function.
    """
    # We vmap the function 3 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
    ]

    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


class LayerPattern(object):
    """
    This is a structure to describe the attention pattern used in a DecoderLayer.

    Args:
        pattern (`str`):
            The attention pattern for the given layer. Must be one of `("full", "sliding", "chunked")`.
        local_size (`int`, *optional*):
            The size of the local attention, if `pattern` is one of `("sliding", "chunked")`. That is, the size
            of the sliding window or the chunk size. If `pattern="full"`, this argument is ignored.
    """

    pattern: str
    local_size: Optional[int] = None

    def __init__(self, pattern: str, local_size: Optional[int] = None):
        allowed_patterns = ("full", "sliding", "chunked")
        if pattern not in allowed_patterns:
            raise ValueError(
                f"Could not recognize attention pattern `{pattern}` It should be one of {allowed_patterns}"
            )
        self.pattern = pattern
        self.local_size = local_size
        # Ignore the potential `local_size` passed if the layer uses full attention pattern
        if self.pattern == "full":
            self.local_size = None

    def get_mask_factory_function(self) -> Callable:
        """
        Return the mask function describing this LayerPattern.
        """
        # Standard causal attention
        if self.pattern == "full":
            mask_function = causal_mask_function
        # Sliding window attention
        elif self.pattern == "sliding":
            mask_function = sliding_window_causal_mask_function(self.local_size)
        # Chunked attention
        elif self.pattern == "chunked":
            mask_function = chunked_causal_mask_function(self.local_size)
        return mask_function


def prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True
) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necesary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        # For flex, we should not slice them, only use an offset
        if _slice:
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def sdpa_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    layer_pattern: LayerPattern = LayerPattern("full"),
    attention_mask: Optional[torch.Tensor] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    For older torch versions, if `allow_torch_fix=True` (the default), rows corresponding to query tokens that do not attend
    to any other tokens (due to padding) will be fully attended to instead, in order to avoid `nan` propagation (this does
    not change the final result).

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        layer_pattern (`LayerPattern`):
            The structure describing the attention pattern used by a DecoderLayer.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.


    ## Creating a simple causal mask:

    To create the following causal mask:

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ■ ■ ■ ■ ⬚
        4 ■ ■ ■ ■ ■

    You can do

    ```python
    >>> create_4d_causal_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5)
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [ True,  True,  True,  True, False],
                  [ True,  True,  True,  True,  True]]]])
    ```

    ## Creating a sliding window mask:

    To create the following sliding window mask (`sliding_window=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ■ ■ ■ ⬚
        4 ⬚ ⬚ ■ ■ ■

    You can do

    ```python
    >>> create_4d_causal_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, layer_pattern=LayerPattern("sliding", 3))
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [False,  True,  True,  True, False],
                  [False, False,  True,  True,  True]]]])
    ```

    ## Creating a chunked attention mask

    To create the following chunked attention mask (`chunk_size=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ⬚ ⬚ ■ ⬚
        4 ⬚ ⬚ ⬚ ■ ■

    You can do

    ```python
    >>> create_4d_causal_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, layer_pattern=LayerPattern("chunked", 3))
    >>> tensor([[[[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False, False, False,  True, False],
                [False, False, False,  True,  True]]]])
    ```

    """
    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, layer_pattern.local_size):
        return None

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    # Return the mask function based on the layer type
    mask_function = layer_pattern.get_mask_factory_function()

    # This creates the 4D mask easily. Note that we do not include vmap over the batch_idx dimension as well,
    # as vmap cannot handle slicing a tensor from scalar tensor (it internally calls `.item()` which vmap does not allow
    # However, in more recent version of Pytorch, a trick was introduced to handle it, so the code below could be
    # replaced by a simpler call to `torch.nn.attention.flex_attention.create_mask` in the future (it would just mean
    # adding the padding_mask_function, and adding the correct offsets before calling the function)
    causal_mask = _vmap_for_q_idx_kv_idx(mask_function)(None, None, cache_position, kv_arange)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]

    # Due to a bug in some older torch version, we need to update the mask in case a query is not attending to any
    # tokens (due to padding). See details in https://github.com/pytorch/pytorch/issues/110213
    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
    return causal_mask


def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    layer_pattern: LayerPattern = LayerPattern("full"),
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
    the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
    it should not.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        layer_pattern (`LayerPattern`):
            The structure describing the attention pattern used by a DecoderLayer.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        dtype (`torch.dtype`, optional):
            The dtype to use for the mask. By default, `torch.float32`.
    """
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        layer_pattern=layer_pattern,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_torch_fix=False,
        **kwargs,
    )
    min_dtype = torch.finfo(dtype).min
    # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
    return mask


def flash_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    layer_pattern: LayerPattern = LayerPattern("full"),
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Create the attention mask necesary to use FA2. Since FA2 is un-padded by definition, here we simply return
    `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.
    We just slice it in case of sliding window.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        layer_pattern (`LayerPattern`):
            The structure describing the attention pattern used by a DecoderLayer.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
    # Raise if using chunked attention on context too large
    if layer_pattern.pattern == "chunked" and kv_length + kv_offset > layer_pattern.local_size:
        raise ValueError(
            "Flash attention 2 cannot handle attention chunked attention, and the key-value length is larger than the chunk size "
            "so the chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    if attention_mask is not None:
        # Here we need to slice from the right if using sliding or chunked (for full attention, this is equivalent to doing nothing)
        attention_mask = attention_mask[:, -kv_length:]
        # We only return an actual mask if there is at least 1 padding token, otherwise we return `None` and use `is_causal` in FA2
        # (note that the attention_mask is a boolean dtype here)
        if attention_mask.all():
            attention_mask = None

    return attention_mask


def flex_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    layer_pattern: LayerPattern = LayerPattern("full"),
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> "BlockMask":
    """
    Create a 4D block mask which is a compressed representation of the full 4D block causal mask. BlockMask is essential
    for performant computation of flex attention. See: https://pytorch.org/blog/flexattention/

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        layer_pattern (`LayerPattern`):
            The structure describing the attention pattern used by a DecoderLayer.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
    q_length, q_offset = cache_position.shape[0], cache_position[0]

    # Return the mask function based on the layer type
    mask_function = layer_pattern.get_mask_factory_function()

    # Potentially add the padding 2D mask
    if attention_mask is not None:
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Add the offsets on top (because flex interface only allows length, not start and end indices)
    mask_function = add_offsets_to_mask_function(mask_function, q_offset, kv_offset)

    # Finally create the block mask
    block_mask = create_block_mask(
        mask_mod=mask_function,
        B=batch_size,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=cache_position.device,
        _compile=True,
    )
    return block_mask


class AttentionMaskInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask,
        "flex_attention": flex_attention_mask,
    }


# Global AttentionMaskInterface shared by all models which do not need to overwrite any of the existing ones
ALL_MASK_CREATION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def _create_mask(
    layer_pattern: LayerPattern,
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    layer_idx: int,
    output_attentions: bool = False,
) -> Optional[Union[torch.Tensor, "BlockMask"]]:
    """
    General utility function that is able to generate all types of masks, for all attention functions. This is
    intended to be reused by each public mask creation function (e.g. `create_causal_mask`, `create_sliding_window_causal_mask`,
    etc)

    Args:
        layer_pattern (`LayerPattern`):
            The structure describing the attention pattern used by a DecoderLayer.
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        output_attentions (`bool`, optional):
            Whether we return the attention scores or not. By default `False`.
    """
    # For BC -> if the mask is passed in 4D directly, simply replicate it on all layers
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim == 4
        and config._attn_implementation != "flash_attention_2"
    ):
        return attention_mask

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    if config._attn_implementation not in ALL_MASK_CREATION_FUNCTIONS:
        return None

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    # If using a cache, it can give all informations about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    mask_interface = ALL_MASK_CREATION_FUNCTIONS[config._attn_implementation]
    # Sdpa fallbacks to eager in the Attention modules if `output_attentions=True`
    if config._attn_implementation == "sdpa" and output_attentions:
        mask_interface = ALL_MASK_CREATION_FUNCTIONS["eager"]

    # We now create the mask
    return mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        layer_pattern=layer_pattern,
        attention_mask=attention_mask,
        # Additional kwargs for eager
        dtype=dtype,
        # Pass the config as well, in case someone wants to easily have their own mask_interface
        config=config,
    )


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    output_attentions: bool = False,
) -> Optional[Union[torch.Tensor, "BlockMask"]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an HybridCache structure, this function will return the mask corresponding to one of the "full" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        output_attentions (`bool`, optional):
            Whether we return the attention scores or not. By default `False`.
    """
    layer_pattern = LayerPattern("full")
    # If we have an HybridCache structure, here we want to create the mask for the full layers
    try:
        layer_idx = past_key_values.is_sliding.index(False)
    except (ValueError, AttributeError):
        layer_idx = 0
    return _create_mask(
        layer_pattern=layer_pattern,
        config=config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        layer_idx=layer_idx,
        output_attentions=output_attentions,
    )


def create_sliding_window_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    output_attentions: bool = False,
) -> Optional[Union[torch.Tensor, "BlockMask"]]:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an HybridCache structure, this
    function will return the mask corresponding to one of the "sliding" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        output_attentions (`bool`, optional):
            Whether we return the attention scores or not. By default `False`.
    """
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config")

    layer_pattern = LayerPattern("sliding", sliding_window)
    # If we have an HybridCache structure, here we want to create the mask for the sliding layers
    try:
        layer_idx = past_key_values.is_sliding.index(True)
    except (ValueError, AttributeError):
        layer_idx = 0
    return _create_mask(
        layer_pattern=layer_pattern,
        config=config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        layer_idx=layer_idx,
        output_attentions=output_attentions,
    )


def create_chunked_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    output_attentions: bool = False,
) -> Optional[Union[torch.Tensor, "BlockMask"]]:
    """
    Create a chunked attention causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Llama4. If `past_key_values` has an HybridCache structure, this
    function will return the mask corresponding to one of the "sliding" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        output_attentions (`bool`, optional):
            Whether we return the attention scores or not. By default `False`.
    """
    chunk_size = getattr(config, "attention_chunk_size", None)
    if chunk_size is None:
        raise ValueError("Could not find an `attention_chunk_size` argument in the config")

    layer_pattern = LayerPattern("chunked", chunk_size)
    # If we have an HybridCache structure, here we want to create the mask for the sliding layers
    try:
        layer_idx = past_key_values.is_sliding.index(True)
    except (ValueError, AttributeError):
        layer_idx = 0
    return _create_mask(
        layer_pattern=layer_pattern,
        config=config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        layer_idx=layer_idx,
        output_attentions=output_attentions,
    )


def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    # local_attention_size = sliding_window or chunk_size

    if padding_mask is None:
        # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
        # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
        # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
        # `ignore_causal_mask = True` if we are not tracing
        if (
            not is_tracing
            and (query_length == 1 or kv_length == query_length)
            and (local_attention_size is None or kv_length < local_attention_size)
        ):
            return True
    elif local_attention_size is None or kv_length < local_attention_size:
        if len(padding_mask.shape) == 4:
            return False
        elif not is_tracing and torch.all(padding_mask == 1):
            if query_length == 1 or kv_length == query_length:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                return True

            # Unfortunately, for query_length > 1 and kv_length != query_length, we cannot generally ignore
            # the attention mask, as SDPA causal mask is the upper triangular part instead of lower part. We will set
            # `is_causal=False` in SDPA and rely on Transformers causal_mask instead, hence not setting it to None here.
            # Reference: https://github.com/pytorch/pytorch/issues/108108

    return False


LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full": create_causal_mask,
    "sliding": create_sliding_window_causal_mask,
    "chunked": create_chunked_causal_mask,
}


def create_masks_for_generate(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    output_attentions: bool = False,
):
    """
    This function mimics how we create the masks in the `modeling_xxx.py` files, and is used in `generate` in order
    to easily create the masks in advance, when we compile the forwards with Static caches.

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        output_attentions (`bool`, optional):
            Whether we return the attention scores or not. By default `False`.
    """
    # Prepare the mask args
    mask_kwargs = {
        "config": config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "output_attentions": output_attentions,
    }

    # If the attribute exist, we need several masks
    if hasattr(config, "layer_attention_patterns"):
        causal_masks = {}
        for layer_pattern in set(config.layer_attention_patterns):
            causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
        return causal_masks
    # In this case, all layers are sliding
    elif getattr(config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    # In this case, all layxers are chunked
    elif getattr(config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    # All layers use standard causal attention
    return create_causal_mask(**mask_kwargs)


# Below are utilities to pretty-print the different masks
# Print the matrix with words as row labels
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"
GREY_SQUARE = "∙"
LOW_TRIANGLE = "⬕"
UPPER_TRIANGLE = "⬔"


def get_style(style):
    if style == "majong":
        BLACK_SQUARE = "🀞"  # Full block (represents "on" or active)
        BLACK_SQUARE = "🀙"  # Full block (represents "on" or active)
        WHITE_SQUARE = "🀆"  # "▒"  # Light shade (represents "off" or inactive)
        LOW_TRIANGLE = "🀛"  # Lower left triangle (stylized indication)
        UPPER_TRIANGLE = "🀛"  # Upper left triangle (stylized indication)
    else:
        BLACK_SQUARE = "█"  # Full block (represents "on" or active)
        WHITE_SQUARE = "░"  # "▒"  # Light shade (represents "off" or inactive)
        LOW_TRIANGLE = "▙"  # Lower left triangle (stylized indication))
        UPPER_TRIANGLE = "▜"  # Upper left triangle (stylized indication)

    return BLACK_SQUARE, WHITE_SQUARE, LOW_TRIANGLE, UPPER_TRIANGLE


# LOW_TRIANGLE = UPPER_TRIANGLE = "⟍"   # Upper right triangle (stylized indication)

YELLOW_SQUARE = f"{YELLOW}{BLACK_SQUARE}{RESET}"
GREEN_SQUARE = f"{GREEN}{BLACK_SQUARE}{RESET}"


def tensor_to_mask_visual(original_tensor: torch.Tensor, grid_size=(20, 40), style="majong") -> str:
    BLACK_SQUARE, WHITE_SQUARE, LOW_TRIANGLE, UPPER_TRIANGLE = get_style(style)
    h, w = original_tensor.shape
    max_h, max_w = grid_size
    if not (h < max_h and w < max_w):
        # Preserve aspect ratio within max grid size
        aspect_ratio = 2 * w / h
        if aspect_ratio > 1:
            w = max_w
            h = min(max_h, max(1, round(max_w / aspect_ratio)))
        else:
            h = max_h
            w = max(1, round(max_h * aspect_ratio))

        # Step 1: Rescale tensor by average pooling
        tensor = original_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        tensor = F.adaptive_avg_pool2d(tensor, output_size=(h, w))[0, 0]  # Remove extra dims
    else:
        tensor = original_tensor

    # Step 3: Build the string representation
    result = []
    for i in range(h):
        row = ""
        for j in range(w):
            if tensor[i, j] == 1:
                row += BLACK_SQUARE
            elif tensor[i, j] == 0:
                row += WHITE_SQUARE
            else:
                if j > 0:
                    if tensor[i, j - 1] == 1:
                        row += LOW_TRIANGLE
                    elif tensor[i, j - 1] == 0:
                        row += UPPER_TRIANGLE
                    else:
                        row += BLACK_SQUARE if tensor[i, j] == 1 else WHITE_SQUARE
                else:
                    row += (
                        BLACK_SQUARE
                        if tensor[i, j] == 1
                        else (
                            WHITE_SQUARE
                            if tensor[i, j] == 0
                            else (UPPER_TRIANGLE if tensor[i, j + 1] == 1 else LOW_TRIANGLE)
                        )
                    )
        result.append(row)

    return "\n".join(result)


class AttentionMask(torch.Tensor):
    def __new__(cls, data, style=None):
        # Create a new instance of AttentionMask as a Tensor
        cls.style = style
        return torch.Tensor._make_subclass(cls, data, require_grad=False)

    def __init__(self, data):
        # You can initialize any additional metadata here if needed
        pass

    def to_string(self, grid_size=(20, 40), limit=4):
        """Returns a string representation of the block mask."""
        dense_mask = self
        *batch_dims, num_rows, num_cols = dense_mask.shape
        total_vis = []

        for idx, batch_idx in enumerate(itertools.product(*[range(i) for i in batch_dims])):
            if idx == limit:
                total_vis.append("...")
                total_vis.append("To print out more, set AttentionMask.to_string(limit=N)")
                total_vis.append("You can also index (AttentionMask[batch, head]) to choose a specific batch or head")
                break
            block_vis = tensor_to_mask_visual(dense_mask[batch_idx], grid_size=grid_size, style=self.style)
            total_vis.append(block_vis)

        total_vis.append(f"torch.Tensor(shape={tuple(self.shape)}, dtype={self.dtype})")
        return "\n".join(total_vis)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, style: Optional[str] = None) -> "AttentionMask":
        res = cls(tensor)
        res.style = style
        return res
