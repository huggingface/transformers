import itertools
from typing import Optional

import torch
import torch.nn.functional as F

from .cache_utils import Cache
from .configuration_utils import PretrainedConfig
from .utils.import_utils import is_torchdynamo_compiling


# Print the matrix with words as row labels
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"
GREY_SQUARE = "∙"
LOW_TRIANGLE = "⬕"
UPPER_TRIANGLE = "⬔"


r"""
🀞🀞🀞🀞🀞🀞🀛🀆🀆🀆🀆🀆🀆🀆🀆🀆
🀞🀞🀞🀞🀞🀞🀞🀛🀆🀆🀆🀆🀆🀆🀆🀆
🀞🀞🀞🀞🀞🀞🀞🀞🀛🀆🀆🀆🀆🀆🀆🀆
🀞🀞🀞🀞🀞🀞🀞🀞🀞🀛🀆🀆🀆🀆🀆🀆
🀞🀞🀞🀞🀞🀞🀞🀞🀞🀞🀛🀆🀆🀆🀆🀆
🀞🀞🀞🀞🀞🀞🀞🀞🀞🀞🀞🀛🀆🀆🀆🀆
⛰⛰⛰⛰⛰⛰⛰⛰⛰⚋⚋⚋⚋⚋⚋⚋◹◥
✅✅✅✅✅🟥🟥🟥🟥🟥
▚▚▚▚▚▚▚▚▚▚▚▚▚⍂░⋱
⎕⎕⎕⎕⍂⎕⎕⎕⎕⎕⎕⎕
⏺⏺⏺⏺⏺⏲⏲⏲⏲⏲
⏺⏺⏺⏺⏺⏲⏲⏲⏲⏲

"""


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

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset
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


def flash_attention_mask(attention_mask: Optional[torch.Tensor], **kwargs):
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
    past_key_values: Optional[Cache],
    batch_size: int,
    num_layers: int,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    query_length = cache_position.shape[0] if cache_position is not None else attention_mask.shape[-1]
    if past_key_values is not None:
        sizes_and_patterns = past_key_values.get_mask_size_and_pattern(cache_position, num_layers)
    else:
        kv_offset = 0
        kv_length = attention_mask.shape[-1]
        # HERE MODELS WITH HYBRID CACHE STRUCTURE SHOULD STILL ALTERNATE BETWEEN USING LOCAL ATTENTION OR NOT EVEN WITHOUT CACHE!!!!
        # PROBABLY THE EASIEST IS TO HAVE A FUNCTION RETURNING IT BASED ON THE CONFIG???
        sizes_and_patterns = [(kv_offset, kv_length, sliding_window, chunk_size)] * num_layers

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    unique_mask_sizes_and_patterns = list(set(sizes_and_patterns))
    layer_to_mask_mapping = {
        idx: unique_mask_sizes_and_patterns.index(size_and_pattern)
        for idx, size_and_pattern in enumerate(sizes_and_patterns)
    }

    # We now create all the masks
    masks = []
    for kv_offset, kv_length, window, chunk in unique_mask_sizes_and_patterns:
        # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
        if allow_is_causal_skip and _ignore_causal_mask_sdpa(attention_mask, query_length, kv_length, window, chunk):
            masks.append(None)
            continue
        causal_mask = create_4d_causal_mask(batch_size, kv_length, cache_position, kv_offset, window, chunk)
        # Merge the padding mask into the causal mask if needed
        if attention_mask is not None:
            # Offset the padding mask as well
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indexes = torch.arange(min(kv_length, attention_mask.shape[-1]), device=cache_position.device)
            mask_indexes += kv_offset
            local_padding_mask = attention_mask[:, mask_indexes]
            # It may be smaller, in which case we pad with 0s to indicate the entries should not take part in the attention
            if (padding_length := kv_length - local_padding_mask.shape[-1]) > 0:
                local_padding_mask = torch.nn.functional.pad(local_padding_mask, (0, padding_length))
            # merge both
            causal_mask = causal_mask * local_padding_mask[:, None, None, :]

        masks.append(causal_mask)

    return masks, layer_to_mask_mapping


def eager_mask(
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    batch_size: int,
    num_layers: int,
    dtype: torch.dtype,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
    **kwargs,
):
    masks, layer_to_mask_mapping = sdpa_mask(
        attention_mask,
        cache_position,
        past_key_values,
        batch_size,
        num_layers,
        sliding_window,
        chunk_size,
        allow_is_causal_skip=False,
    )
    min_dtype = torch.finfo(dtype).min
    # Loop over them instead of comprehension to avoid keeping the 2 full lists in memory at once
    for i, mask in enumerate(masks):
        # we need 0s where the tokens should be taken into account, and -inf otherwise
        masks[i] = torch.where(mask == 1, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)

    return masks, layer_to_mask_mapping


ALL_MASK_CREATION_FUNCTIONS = {
    "sdpa": sdpa_mask,
    "eager": eager_mask,
    "flash_attention_2": flash_attention_mask,
}


def get_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
):
    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    num_layers = config.num_hidden_layers

    # HERE WE SHOULD EXTRACT ALL WAYS TO GET SLIDING_WINDOW/CHUNK SIZE IN THE CONFIG IF WE DON'T USE CACHE (OTHERWISE IT'S EMBEDDED
    # IN THE CACHE ALREADY)
    sliding_window = getattr(config, "sliding_window", None)

    mask_interface = ALL_MASK_CREATION_FUNCTIONS[config._attn_implementation]
    masks, layer_to_mask_mapping = mask_interface(
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        batch_size=batch_size,
        num_layers=num_layers,
        dtype=dtype,
        # USE THE ONE EXTRACTED FROM CONFIG ABOVE
        sliding_window=sliding_window,
        chunk_size=None,
    )

    mask_per_layers = [masks[layer_to_mask_mapping[i]] for i in range(num_layers)]

    return mask_per_layers


def _ignore_causal_mask_sdpa(
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the optional user-specified attention_mask & the automatically created causal mask can be
    ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = torch.jit.is_tracing() or isinstance(attention_mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    local_attention_size = sliding_window or chunk_size

    if attention_mask is None:
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
        if len(attention_mask.shape) == 4:
            return False
        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1 or kv_length == query_length:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                return True

            # Unfortunately, for query_length > 1 and kv_length != query_length, we cannot generally ignore
            # the attention mask, as SDPA causal mask is the upper triangular part instead of lower part. We will set
            # `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
            # Reference: https://github.com/pytorch/pytorch/issues/108108

    return False
