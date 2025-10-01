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

from .cache_utils import Cache
from .configuration_utils import PretrainedConfig
from .utils import is_torch_xpu_available, logging
from .utils.generic import GeneralInterface
from .utils.import_utils import is_torch_flex_attn_available, is_torch_greater_or_equal, is_torchdynamo_compiling


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks
    BlockMask = torch.Tensor

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()

if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex


logger = logging.get_logger(__name__)


def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask


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


def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a chunked attention pattern. Add it on top of a causal mask for a proper chunked
    attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (kv_idx - left_padding[batch_idx]) // chunk_size == (q_idx - left_padding[batch_idx]) // chunk_size

    return inner_mask


def _legacy_chunked_overlay(chunk_size: int) -> Callable:
    """
    Same as the above function, but do not correctly account for left padding tokens.
    Only kept for compatibility with older torch versions (< 2.6).
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx // chunk_size == q_idx // chunk_size

    return inner_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def chunked_causal_mask_function(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This return the mask_function function to create a chunked attention mask.
    """
    if not _is_torch_greater_or_equal_than_2_6:
        return and_masks(_legacy_chunked_overlay(chunk_size), causal_mask_function)
    return and_masks(chunked_overlay(chunk_size, left_padding), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask


def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)

    return inner_mask


def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs. Optionally, vmap over
    the batch and head indices as well if `bh_indices=True`.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).

    Args:
        mask_function (`Callable`):
            The mask_function to vmap.
        bh_indices (`bool`, optional):
            Whether to vmap over the batch and head indices as well, or only q and kv indices.

    Returns:
        Callable: The vmapped function.
    """
    # We vmap the function 2 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices:
        # We extend broadcasting over the [batch_idx, head_idx] dimensions
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
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
        # Pad it if necessary
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


def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
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
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    if (
        not is_tracing
        # only cases when lower and upper diags are the same, see https://github.com/pytorch/pytorch/issues/108108
        and (query_length == 1 or (kv_length == query_length or _is_torch_xpu_available))
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
        # In this case, we need to add padding to the mask, so cannot be skipped otherwise
        and (
            padding_mask is None
            or (
                padding_mask.all()
                if not _is_torch_xpu_available or query_length == 1
                else padding_mask[:, :query_length].all()
            )
        )
    ):
        return True

    return False


def sdpa_mask_recent_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    This function can only be used with torch>=2.5, as the context manager is otherwise not available.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
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
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5)
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
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=sliding_window_causal_mask_function(3))
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
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=chunked_causal_mask_function(3, torch.zeros(1, dtype=int)))
    >>> tensor([[[[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False, False, False,  True, False],
                [False, False, False,  True,  True]]]])
    ```

    """
    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=cache_position.device)
    head_arange = torch.arange(1, device=cache_position.device)
    # This creates the 4D mask easily. Note that we need this context manager as vmap cannot handle slicing a tensor from
    # scalar tensor (it internally calls `.item()` which vmap does not allow, but this context works around it
    # We don't need to add an offset to the mask_function either, as we vmap directly the correct indices for k and kv indices
    with TransformGetItemToIndex():
        causal_mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, cache_position, kv_arange)

    return causal_mask


def sdpa_mask_older_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    NOTE: This function is only used when torch version is torch<2.5 - see `sdpa_mask_recent_torch` otherwise.

    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    If `allow_torch_fix=True` (the default), rows corresponding to query tokens that do not attend
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
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.
    """
    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    # This creates the 4D mask easily. Note that we do not include vmap over the batch_idx dimension as well,
    # as vmap cannot handle slicing a tensor from scalar tensor (it internally calls `.item()` which vmap does not allow
    # However, in more recent version of Pytorch, a trick was introduced to handle it - which is the reason we have
    # `sdpa_mask_recent_torch`, as it allows more general `mask_function`
    causal_mask = _vmap_for_bhqkv(mask_function, bh_indices=False)(None, None, cache_position, kv_arange)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]

    # Due to a bug in versions of torch<2.5, we need to update the mask in case a query is not attending to any
    # tokens (due to padding). See details in https://github.com/pytorch/pytorch/issues/110213
    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
    return causal_mask


# We use the version with newer torch whenever possible, as it is more general and can handle arbitrary mask functions
# (especially mask_function indexing a tensor, such as the padding mask function)
sdpa_mask = sdpa_mask_recent_torch if _is_torch_greater_or_equal_than_2_6 else sdpa_mask_older_torch


def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
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
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        dtype (`torch.dtype`, optional):
            The dtype to use for the mask. By default, `torch.float32`.
    """
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    _ = kwargs.pop("allow_is_causal_skip", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
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
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Create the attention mask necessary to use FA2. Since FA2 is un-padded by definition, here we simply return
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
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
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
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> BlockMask:
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
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
    q_length, q_offset = cache_position.shape[0], cache_position[0]

    # Potentially add the padding 2D mask
    if attention_mask is not None:
        # Older torch (2.5.x) cannot handle sequences not in multiples of 128 (default block size)
        # Hence we pad to multiples of this as a minimum to ensure this
        pad_len = ((attention_mask.shape[1] // flex_default_block_size) + 1) * flex_default_block_size
        pad_len = pad_len - attention_mask.shape[1]
        if not _is_torch_greater_or_equal_than_2_6 and pad_len > 0:
            attention_mask = torch.nn.functional.pad(attention_mask, value=0, pad=(0, pad_len))

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
        _compile=_is_torch_greater_or_equal_than_2_6,
    )
    return block_mask


class AttentionMaskInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask,
        "flash_attention_3": flash_attention_mask,
        "flex_attention": flex_attention_mask,
    }


# Global AttentionMaskInterface shared by all models which do not need to overwrite any of the existing ones
ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].
    """
    # What separate different sequences is when 2 consecutive positions_ids are separated by more than 1. So
    # taking the diff (by prepending the first value - 1 to keep correct indexing) and applying cumsum to the result
    # gives exactly the sequence indices
    # Note that we assume that a single sequence cannot span several batch dimensions, i.e. 1 single sequence
    # cannot be part of the end of the first batch dim and the start of the 2nd one for example
    first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)

    # Here it would be nice to return None if we did not detect packed sequence format, i.e. if `packed_sequence_mask[:, -1] == 0`
    # but it causes issues with export
    return packed_sequence_mask


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, BlockMask]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[torch.Tensor, BlockMask]], int, int]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

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
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`torch.Tensor` or `BlockMask` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`torch.Tensor`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    # Note: it's not ideal to check the `_global_mapping` attribute instead of the object itself, however otherwise
    # full graph dynamo tracing (i.e. torch.export or compile with `fullgraph=True`) will fail on Python<3.11
    # with `torch._dynamo.exc.Unsupported: 'inline in skipfiles:Mapping.__contains__ | __contains__, skipped
    # according trace_rules.lookup SKIP_DIRS'` -- can be removed when we require Python>=3.11
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    # We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
    # and we don't have past_key_values, i.e. generally a training setup)
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        # The position ids are sometimes just unsqueezed, without being expanded
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an hybrid cache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
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
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # If we have an hybrid cache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    if _is_torch_xpu_available:
        allow_is_causal_skip = True
    else:
        allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # Allow slight deviations from causal mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # If we detected packing format
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def create_sliding_window_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "sliding_attention" layers (to align to what is needed in the
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
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
    """
    # If we have an hybrid cache structure, here we want to create the mask for the sliding layers
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # Allow slight deviations from causal mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # If we detected packing format
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=sliding_window,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def create_chunked_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a chunked attention causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Llama4. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "chunked_attention" layers (to align to what is needed in the
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
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
    """
    # If we have an hybrid cache structure, here we want to create the mask for the sliding layers
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    chunk_size = getattr(config, "attention_chunk_size", None)
    if chunk_size is None:
        raise ValueError("Could not find an `attention_chunk_size` argument in the config, or it is not set")

    # Raise if using chunked attention on context too large with FA2
    if config._attn_implementation == "flash_attention_2" and kv_length + kv_offset > chunk_size:
        raise ValueError(
            "Flash attention 2 cannot handle chunked attention, and the key-value length is larger than the chunk size so the "
            "chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    # For chunked attention and batched inputs, we need to take the number of left padding tokens into account
    # to start the chunk from the actual start of the sequence for the padded sequence
    if attention_mask is not None:
        # Only count the left padding tokens, not all of them
        left_padding_tokens = (attention_mask.cumsum(dim=-1) == torch.zeros_like(attention_mask)).sum(dim=-1)
    else:
        left_padding_tokens = torch.zeros(batch_size, device=cache_position.device, dtype=int)
    # Raise a warning for older versions if the problematic left-padding situation arises
    if (
        not _is_torch_greater_or_equal_than_2_6
        and kv_length + kv_offset > chunk_size
        and (left_padding_tokens > 0).any()
    ):
        logger.warning_once(
            "Due to limitations of your current torch version, we cannot correctly account for the left-padding "
            "when computing the chunked attention pattern. This will lead to a wrong attention mask for the padded "
            "sequences. Behavior will be undefined. Please upgrade to `torch>=2.6` to solve this issue."
        )
    mask_factory_function = chunked_causal_mask_function(chunk_size, left_padding_tokens)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # Allow slight deviations from causal mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # If we detected packing format
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=chunk_size,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
}


def create_masks_for_generate(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
    **kwargs,
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
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # The attribute reside in the text config for composite models
    effective_config = config.get_text_config()
    # Prepare the mask args
    mask_kwargs = {
        "config": effective_config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
    }

    # If the attribute exist, we need several masks
    if hasattr(effective_config, "layer_types"):
        causal_masks = {}
        for layer_pattern in set(effective_config.layer_types):
            causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
        return causal_masks
    # In this case, all layers are sliding
    elif getattr(effective_config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    # In this case, all layers are chunked
    elif getattr(effective_config, "attention_chunk_size", None) is not None:
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
