from typing import Optional, Union

import torch

def create_4d_causal_mask(
    batch_size: int,
    kv_length: int,
    cache_position: torch.Tensor,
    kv_offset: int = 0,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        sliding_window (`int`, optional):
            An optional sliding window length, if we are using sliding window attention. Mutually exclusive with `chunk_size`.
        chunk_size (`int`, optional):
            An optional chunk size, if we are using chunked attention. Mutually exclusive with `sliding_window`.


    ## Creating a simple causal mask:

    To create the following causal mask:

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ■ ■ ■ ■ ⬚
        4 ■ ■ ■ ■ ■

    You can do

    ```python
    >>> create_4d_causal_mask(1, 5, torch.arange(5), kv_offset=0)
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
    >>> create_4d_causal_mask(1, 5, torch.arange(5), kv_offset=0, sliding_window=3)
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
    >>> create_4d_causal_mask(1, 5, torch.arange(5), kv_offset=0, chunk_size=3)
    >>> tensor([[[[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False, False, False,  True, False],
                [False, False, False,  True,  True]]]])
    ```

    """
    if sliding_window is not None and chunk_size is not None:
        raise ValueError("`sliding_window` and `chunk_size` are mutually exclusive for mask creation")

    kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)
    reshaped_cache_position = cache_position.view(-1, 1)

    # Simplest and most efficient way to obtain a causal mask
    causal_mask = kv_arange <= reshaped_cache_position
    # If using sliding window, add the sliding mask
    if sliding_window is not None:
        sliding_mask_overlay = kv_arange > reshaped_cache_position - sliding_window
        causal_mask *= sliding_mask_overlay
    # If using chunk attention, add the chunked mask
    elif chunk_size is not None:
        chunked_mask_overlay = kv_arange // chunk_size == reshaped_cache_position // chunk_size
        causal_mask *= chunked_mask_overlay

    # Make it 4D
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    return causal_mask


def merge_2d_padding_mask_into_4d_mask(padding_mask: Optional[torch.Tensor], causal_mask: torch.Tensor) -> torch.Tensor:
    """
    Merge the 2d attention mask (corresponding to padded tokens) into the general 4d mask, corresponding to the
    masking pattern (causal attention, sliding window attention, or chunked attention).

    Args:
        padding_mask: (`torch.Tensor`):
            The 2d mask of shape `(batch_size, all_processed_tokens)`, returned by a Tokenizer or `generate`.
        causal_mask: (`torch.Tensor`):
            General 4d mask of shape `(batch_size, 1, query_length, kv_length)`, usually returned by `create_4d_causal_mask`.
    """
    if padding_mask is not None:
        padding_mask = padding_mask.to(device=causal_mask.device, dtype=torch.bool)
        causal_mask[:, :, :, :padding_mask.shape[-1]] *= padding_mask[:, None, None, :]
    return causal_mask


def flash_attention_mask(attention_mask: Optional[torch.Tensor]):
    """
    Create the attention mask necesary to use FA2. Since FA2 is un-padded by definition, here we simply return
    `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.

    Args:
        attention_mask (`torch.Tensor`, , *optional*):
            A 2D attention mask of shape `(batch_size, key_value_length)`.
    """
    if attention_mask is not None and (attention_mask == 0.0).any():
        return attention_mask
    return None


def _update_causal_mask(
    self,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool = False,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None
    if self.config._attn_implementation == "flex_attention":
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = make_flex_block_causal_mask(attention_mask)
        return attention_mask

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None

    dtype = input_tensor.dtype
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type in ["cuda", "xpu", "npu"]
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        min_dtype = torch.finfo(dtype).min
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask

