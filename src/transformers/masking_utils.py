import itertools
from typing import Optional

import torch
import torch.nn.functional as F

from .cache_utils import Cache, HybridCache, HybridChunkedCache, SlidingWindowCache, StaticCache


# Print the matrix with words as row labels
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"
GREY_SQUARE = "∙"
LOW_TRIANGLE = "⬕"
UPPER_TRIANGLE = "⬔"


BLACK_SQUARE = "█"  # Full block (represents "on" or active)
WHITE_SQUARE = "░"  # Light shade (represents "off" or inactive)
LOW_TRIANGLE = "▙"  # Lower left triangle (stylized indication)
UPPER_TRIANGLE = "▜"  # Upper left triangle (stylized indication)

# LOW_TRIANGLE = UPPER_TRIANGLE = "⟍"   # Upper right triangle (stylized indication)

YELLOW_SQUARE = f"{YELLOW}{BLACK_SQUARE}{RESET}"
GREEN_SQUARE = f"{GREEN}{BLACK_SQUARE}{RESET}"


def tensor_to_mask_visual(original_tensor: torch.Tensor, grid_size=(20, 40)) -> str:
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
    def __new__(cls, data):
        # Create a new instance of AttentionMask as a Tensor
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
            block_vis = tensor_to_mask_visual(dense_mask[batch_idx], grid_size=grid_size)
            total_vis.append(block_vis)

        total_vis.append(f"torch.Tensor(shape={tuple(self.shape)}, dtype={self.dtype})")
        return "\n".join(total_vis)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(tensor)


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
    return AttentionMask.from_tensor(causal_mask)


def flash_attention_mask(attention_mask: Optional[torch.Tensor]):
    """
    Create the attention mask necesary to use FA2. Since FA2 is un-padded by definition, here we simply return
    `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.

    Args:
        attention_mask (`torch.Tensor`, , *optional*):
            A 2D attention mask of shape `(batch_size, key_value_length)`.
    """
    if attention_mask is not None and (attention_mask == 0.0).any():
        return AttentionMask.from_tensor(attention_mask)
    return None


def sdpa_mask(
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Cache,
    batch_size: int,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> Optional[torch.Tensor]:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` to avoid materializing
    # the mask
    # if not isinstance(past_key_values, (StaticCache, SlidingWindowCache, HybridCache, HybridChunkedCache)):
    #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
    #         attention_mask,
    #         inputs_embeds=input_tensor,
    #         past_key_values_length=past_seen_tokens,
    #         is_training=self.training,
    #     ):
    #         return None

    query_length = cache_position.shape[0]
    # The kv_length and offset are the same for sliding attention or chunked attention -> only the mask pattern changes
    local_attention_size = sliding_window or chunk_size
    first_cache_position = cache_position[0]

    # THIS IS VERY UGLY AS-IS -> THIS SHOULD PROBABLY BE A CACHE METHOD FOR GENERALITY
    sizes_and_patterns = []
    if isinstance(past_key_values, SlidingWindowCache):
        # torch.clamp() is equivalent to max() but should be compile-friendly/exportable as first_cache_position is a Tensor
        kv_offset = torch.clamp(first_cache_position - local_attention_size + 1, min=0)
        # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
        kv_length = max(query_length, past_key_values.get_max_cache_shape())

        sizes_and_patterns = [(kv_offset, kv_length, sliding_window, chunk_size)]
    elif isinstance(past_key_values, StaticCache):
        kv_offset = 0
        kv_length = past_key_values.get_max_cache_shape()

        sizes_and_patterns = [(kv_offset, kv_length, sliding_window, chunk_size)]
    elif isinstance(past_key_values, HybridCache):
        local_mask_kv_offset = torch.clamp(first_cache_position - local_attention_size + 1, min=0)
        # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
        local_mask_kv_length = max(query_length, past_key_values.sliding_window)

        full_mask_kv_offset = 0
        full_mask_kv_length = past_key_values.get_max_cache_shape()

        # In this case, we only need to use a single mask everywhere
        if local_mask_kv_length == full_mask_kv_length:
            sizes_and_patterns = [(full_mask_kv_offset, full_mask_kv_length, sliding_window, chunk_size)]
        else:
            sizes_and_patterns = [
                (local_mask_kv_offset, local_mask_kv_length, sliding_window, chunk_size),
                (full_mask_kv_offset, full_mask_kv_length, sliding_window, chunk_size),
            ]
    elif isinstance(past_key_values, HybridChunkedCache):
        local_mask_kv_offset = torch.clamp(first_cache_position - local_attention_size + 1, min=0)
        # This is the true general case for any Cache using local attention (sliding or chunked)
        if first_cache_position >= local_attention_size:
            # Here the Cache is already full
            local_mask_kv_length = local_attention_size + query_length - 1
        elif (
            first_cache_position < local_attention_size and first_cache_position + query_length > local_attention_size
        ):
            # Here the Cache becomes full with the new input
            local_mask_kv_length = first_cache_position + query_length
        else:
            # Here the Cache is still smaller than the local size, but we return the local size as it's static
            local_mask_kv_length = local_attention_size

        full_mask_kv_offset = 0
        full_mask_kv_length = past_key_values.get_max_cache_shape()

        # Here Llama4 does not use chunk attention on the full mask (which is the reason why we need to add the patterns everywhere)
        if local_mask_kv_length == full_mask_kv_length:
            sizes_and_patterns = [(full_mask_kv_offset, full_mask_kv_length, None, None)]
        else:
            sizes_and_patterns = [
                (local_mask_kv_offset, local_mask_kv_length, sliding_window, chunk_size),
                (full_mask_kv_offset, full_mask_kv_length, None, None),
            ]
    else:
        kv_offset = 0
        kv_length = attention_mask.shape[-1] if attention_mask is not None else past_seen_tokens + query_length

        sizes_and_patterns = [(kv_offset, kv_length, sliding_window, chunk_size)]

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    # We now create all the masks
    masks = []
    for kv_offset, kv_length, window, chunk in sizes_and_patterns:
        causal_mask = create_4d_causal_mask(batch_size, kv_length, cache_position, kv_offset, window, chunk)
        # Merge the padding mask into the causal mask if needed
        if attention_mask is not None:
            # Offset the padding mask as well
            local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]
            # It may be smaller, in which case we pad with 0s to indicate the entries should not take part in the attention
            if (padding_length := kv_length - local_padding_mask.shape[-1]) > 0:
                local_padding_mask = torch.nn.functional.pad(local_padding_mask, (0, padding_length))
            # merge both
            causal_mask = causal_mask * local_padding_mask[:, None, None, :]

        masks.append(causal_mask)

    # CHECK VERSIONS BUT PROBABLY DROP THIS OR AT LEAST KEEP IT ONLY FOR FAULTY VERSION AS WE WANT TO KEEP MASK IN BOOL
    # ALSO THE VERY BEST WOULD BE TO HAVE BOOLS REPRESENTED AS 1 BIT INSTEAD OF DEFAULT 1 BYTE IN TORCH
    # if (
    #     attention_mask is not None
    #     and attention_mask.device.type in ["cuda", "xpu", "npu"]
    # ):
    #     # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    #     # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    #     # Details: https://github.com/pytorch/pytorch/issues/110213
    #     min_dtype = torch.finfo(dtype).min
    #     causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return masks[0] if len(masks) == 1 else masks
