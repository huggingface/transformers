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
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .cache_utils import Cache
from .modeling_utils import GeneralInterface, LayerPattern
from .utils.import_utils import is_torchdynamo_compiling


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


def get_mask_factory_function(layer_pattern: LayerPattern) -> Callable:
    """
    Return the mask function describing the given `layer_pattern`.

    Args:
        layer_pattern (`LayerPattern`):
            The structure describing the attention pattern used by a DecoderLayer.
    """
    # Standard causal attention
    if layer_pattern.pattern == "full":
        mask_function = causal_mask_function
    # Sliding window attention
    elif layer_pattern.pattern == "sliding":
        mask_function = sliding_window_causal_mask_function(layer_pattern.local_size)
    # Chunked attention
    elif layer_pattern.pattern == "chunked":
        mask_function = chunked_causal_mask_function(layer_pattern.local_size)
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
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.

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
    mask_function = get_mask_factory_function(layer_pattern)

    # This creates the 4D mask easily. Note that we do not include vmap over the batch_idx dimension as well,
    # as vmap cannot handle slicing a tensor from scalar tensor (it internally calls `.item()` which vmap does not allow
    # However, in more recent version of Pytorch, a trick was introduced to handle it, so the code below could be
    # replaced by a simpler call to `torch.nn.attention.flex_attention.create_mask` in the future (it would just mean
    # adding the padding_mask_function, and adding the correct offsets before calling the function)
    causal_mask = _vmap_for_q_idx_kv_idx(mask_function)(None, None, cache_position, kv_arange)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]
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
        **kwargs,
    )
    min_dtype = torch.finfo(dtype).min
    # we need 0s where the tokens should be taken into account, and -inf otherwise
    mask = torch.where(mask == 1, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
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

    # Here we need to slice from the right (padding is always left)
    if layer_pattern.pattern != "full":
        if attention_mask is not None:
            attention_mask = attention_mask[:, -kv_length:]

    # We only return an actual mask if there is at least 1 padding token, otherwise we return `None` and use `is_causal` in
    # FA2
    if attention_mask is not None and (attention_mask == 0.0).any():
        return attention_mask
    return None


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
    mask_function = get_mask_factory_function(layer_pattern)

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


@dataclass(unsafe_hash=True)
class MaskParameterization(object):
    """
    This is a structure to describe both the attention pattern used in a DecoderLayer, as well as the KV length
    and offsets. This is used to be able to compare the masks needed at every Layer.
    This structure may mutate the `layer_pattern` passed to it based on the `kv_length` and `kv_offset` in order
    to simplify mask creation and always create the minimum number of causal masks.

    Args:
        layer_pattern (`LayerPattern`):
            The attention pattern used for a given Layer.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """

    layer_pattern: LayerPattern
    kv_length: int
    kv_offset: int

    def __init__(self, layer_pattern: LayerPattern, kv_length: int, kv_offset: int):
        # If the layer is sliding or chunked, but the cumulative sequence length is lower than the window/chunk,
        # the layer behaves as a full layer from an attention perspective (we are not yet beyond the window/chunk size)
        if layer_pattern.pattern in ("sliding", "chunked") and kv_offset + kv_length <= layer_pattern.local_size:
            self.layer_pattern = LayerPattern("full")
        else:
            self.layer_pattern = layer_pattern

        self.kv_length = kv_length
        self.kv_offset = kv_offset

    def __iter__(self):
        """To be able to easily get fields as a tuple/list."""
        yield self.layer_pattern
        yield self.kv_length
        yield self.kv_offset


def get_causal_masks(
    layer_patterns: list[LayerPattern],
    attn_implementation: str,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, list[torch.Tensor]]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    output_attentions: bool = False,
) -> list[Optional[Union[torch.Tensor, "BlockMask"]]]:
    """
    Create all required masks, one per layer. This is the only function to be called inside the modeling files.

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor` or `list[torch.Tensor]`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is replicated for each layer. Can also be
            a list of masks, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        output_attentions (`bool`, optional):
            Whether we return the attention scores or not. By default `False`.
    """
    num_layers = len(layer_patterns)
    # It means the masks were already prepared outside the `forward`, e.g. by `generate` when compiling - return immediately
    if isinstance(attention_mask, list):
        return attention_mask

    # For BC -> if the mask is passed in 4D directly, simply replicate it on all layers
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim == 4
        and attn_implementation in ("sdpa", "eager")
    ):
        return [attention_mask] * num_layers

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    if attn_implementation not in ALL_MASK_CREATION_FUNCTIONS:
        return [None] * num_layers

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype

    # If using a cache, it can give all informations about mask sizes based on seen tokens
    if past_key_values is not None:
        mask_sizes = past_key_values.get_mask_sizes(cache_position, num_layers)
    # We are either training, or running inference without cache -> extract patterns from config
    else:
        kv_length = input_embeds.shape[1]
        mask_sizes = [(kv_length, 0)] * num_layers

    # Casting to `MaskParameterization` allows to very easily find unique needed patterns
    mask_configs = [
        MaskParameterization(layer_pattern, *sizes) for layer_pattern, sizes in zip(layer_patterns, mask_sizes)
    ]
    unique_mask_configs = list(set(mask_configs))
    # Compute the layer mapping from unique configs
    layer_to_mask_mapping = [unique_mask_configs.index(config) for config in mask_configs]

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    mask_interface = ALL_MASK_CREATION_FUNCTIONS[attn_implementation]
    # Sdpa fallbacks to eager in the Attention modules if `output_attentions=True`
    if attn_implementation == "sdpa" and output_attentions:
        mask_interface = ALL_MASK_CREATION_FUNCTIONS["eager"]
    # We now create all the masks
    masks = []
    # for kv_length, kv_offset, window, chunk in sizes_and_patterns:
    for effective_layer_pattern, kv_length, kv_offset in unique_mask_configs:
        causal_mask = mask_interface(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            layer_pattern=effective_layer_pattern,
            attention_mask=attention_mask,
            # Additional kwargs for eager
            dtype=dtype,
            # Pass the config as well, in case someone wants to easily have their own mask_interface
            # config=config,
        )
        masks.append(causal_mask)

    # Map back to each layer (note that this does not incur any copy of the masks)
    mask_per_layers = [masks[layer_to_mask_mapping[i]] for i in range(num_layers)]
    return mask_per_layers


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
