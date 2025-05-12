import itertools
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

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


def and_masks(*mask_mods: list[Callable]) -> Callable:
    """Returns a mask fucntion that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_mods:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return and_mask


def causal_mask_mod(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
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


def sliding_window_causal_mask_mod(sliding_window: int) -> Callable:
    """
    This return the mask_mod function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_mod)


def chunked_causal_mask_mod(chunk_size: int) -> Callable:
    """
    This return the mask_mod function to create a chunked attention mask.
    """
    return and_masks(chunked_overlay(chunk_size), causal_mask_mod)


def padding_mask_mod(padding_mask: torch.Tensor) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_mod as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def add_offsets_to_mask_mod(mask_mod: Callable, q_offset: int, kv_offset: int) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_mod(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)

    return inner_mask


def _vmap_for_q_idx_kv_idx(mask_mod: Callable) -> Callable:
    """
    Used to vmap our mask_mods over the q_idx and kv_idx dimensions of the inputs.

    Args:
        mask_mod (`Callable`):
            The mask_mod to vmap.

    Returns:
        Callable: The vmapped function.
    """
    # We vmap the function 3 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
    ]

    for dims in dimensions:
        mask_mod = torch.vmap(mask_mod, in_dims=dims, out_dims=0)
    return mask_mod


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
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
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
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        sliding_window (`int`, optional):
            An optional sliding window length, if we are using sliding window attention. Mutually exclusive with `chunk_size`.
        chunk_size (`int`, optional):
            An optional chunk size, if we are using chunked attention. Mutually exclusive with `sliding_window`.
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
    q_length = cache_position.shape[0]
    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(
        attention_mask, q_length, kv_length, sliding_window, chunk_size
    ):
        return None

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    # Sliding window mask
    if sliding_window is not None:
        mask_mod = sliding_window_causal_mask_mod(sliding_window)
    # CHunked attention mask
    elif chunk_size is not None:
        mask_mod = chunked_causal_mask_mod(chunk_size)
    # Simple causal mask
    else:
        mask_mod = causal_mask_mod

    # This creates the 4D mask easily. Note that we do not include vmap over the batch_idx dimension as well,
    # as vmap cannot handle slicing a tensor from scalar tensor (it internally calls `.item()` which vmap does not allow
    # However, in more recent version of Pytorch, a trick was introduced to handle it, so the code below could be
    # replaced by a simpler call to `torch.nn.attention.flex_attention.create_mask` in the future (it would just mean
    # adding the padding_mask_mod, and adding the correct offsets before calling the function)
    causal_mask = _vmap_for_q_idx_kv_idx(mask_mod)(None, None, cache_position, kv_arange)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if attention_mask is not None:
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)
        causal_mask = causal_mask * padding_mask[:, None, None, :]
    return causal_mask


def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
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
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        sliding_window (`int`, optional):
            An optional sliding window length, if we are using sliding window attention. Mutually exclusive with `chunk_size`.
        chunk_size (`int`, optional):
            An optional chunk size, if we are using chunked attention. Mutually exclusive with `sliding_window`.
        dtype (`torch.dtype`, optional):
            The dtype to use for the mask. By default, `torch.float32`.
    """
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        attention_mask=attention_mask,
        sliding_window=sliding_window,
        chunk_size=chunk_size,
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
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
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
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        sliding_window (`int`, optional):
            An optional sliding window length, if we are using sliding window attention. Mutually exclusive with `chunk_size`.
        chunk_size (`int`, optional):
            An optional chunk size, if we are using chunked attention. Mutually exclusive with `sliding_window`.
    """
    # Raise if using chunked attention on context too large
    if chunk_size is not None and kv_offset + kv_length > chunk_size:
        raise ValueError(
            "Flash attention 2 cannot handle attention chunked attention, and the key-value length is larger than the chunk size "
            "so the chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    # Here we need to slice from the right (padding is always left)
    if sliding_window is not None or chunk_size is not None:
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
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    chunk_size: Optional[int] = None,
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
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        sliding_window (`int`, optional):
            An optional sliding window length, if we are using sliding window attention. Mutually exclusive with `chunk_size`.
        chunk_size (`int`, optional):
            An optional chunk size, if we are using chunked attention. Mutually exclusive with `sliding_window`.
    """
    q_length, q_offset = cache_position.shape[0], cache_position[0]

    # Sliding window mask
    if sliding_window is not None:
        mask_mod = sliding_window_causal_mask_mod(sliding_window)
    # CHunked attention mask
    elif chunk_size is not None:
        mask_mod = chunked_causal_mask_mod(chunk_size)
    # Simple causal mask
    else:
        mask_mod = causal_mask_mod

    # Potentially add the padding 2D mask
    if attention_mask is not None:
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        mask_mod = and_masks(mask_mod, padding_mask_mod(padding_mask))

    # Add the offsets on top (because flex interface only allows length, not start and end indices)
    mask_mod = add_offsets_to_mask_mod(mask_mod, q_offset, kv_offset)

    # Finally create the block mask
    block_mask = create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=cache_position.device,
        _compile=True,
    )
    return block_mask


ALL_MASK_CREATION_FUNCTIONS = {
    "sdpa": sdpa_mask,
    "eager": eager_mask,
    "flash_attention_2": flash_attention_mask,
    "flex_attention": flex_attention_mask,
}


def get_causal_masks(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, list[torch.Tensor]]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
) -> list[Optional[Union[torch.Tensor, "BlockMask"]]]:
    num_layers = config.num_hidden_layers

    # It means the masks were already prepared outside the `forward`, e.g. by `generate` when compiling - return immediately
    if isinstance(attention_mask, list):
        return attention_mask

    # For BC -> if the mask is passed in 4D directly, simply replicate it on all layers
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim == 4
        and config._attn_implementation in ("sdpa", "eager")
    ):
        return [attention_mask] * num_layers

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    if config._attn_implementation not in ALL_MASK_CREATION_FUNCTIONS:
        return [None] * num_layers

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype

    # If using a cache, it can give all informations about mask size and patterns based on seen tokens
    if past_key_values is not None:
        sizes_and_patterns, layer_to_mask_mapping = past_key_values.get_mask_sizes_and_patterns(
            cache_position, num_layers
        )
    # We are either training, or running inference without cache -> extract patterns from config
    else:
        kv_length = input_embeds.shape[1]
        sizes_and_patterns, layer_to_mask_mapping = infer_mask_sizes_patterns_from_config(config, kv_length)

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    # We now create all the masks
    mask_interface = ALL_MASK_CREATION_FUNCTIONS[config._attn_implementation]
    masks = []
    for kv_length, kv_offset, window, chunk in sizes_and_patterns:
        # Raise if both are provided somehow
        if window is not None and chunk is not None:
            raise ValueError("`sliding_window` and `chunk_size` are mutually exclusive for mask creation")

        causal_mask = mask_interface(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            attention_mask=attention_mask,
            sliding_window=window,
            chunk_size=chunk,
            # Additional kwargs for eager
            dtype=dtype,
            # Pass the config as well, in case someone wants to easily have their own mask_interface????
            config=config,
        )
        masks.append(causal_mask)

    # Map back to each layer (note that this does not incur any copy of the masks)
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


def infer_mask_sizes_patterns_from_config(config: PretrainedConfig, kv_length: int) -> tuple[list[tuple], list[int]]:
    """
    Return a list of tuples (kv_length, kv_offset, sliding_window, chunk_size), corresponding to all unique mask pattern we may need,
    as well as a mapping of indices from the pattern to each layers in the model.
    The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (sliding_window, chunk_size), and
    mapped back to the corresponding layers to be easily indexed in the modleing code (in case of different patterns par layers).
    """

    # Extract sliding window and chunk size from config
    sliding_window = getattr(config, "sliding_window", None)
    chunk_size = getattr(config, "attention_chunk_size", None)

    # Default sizes and patterns, and mapping when all layers are similar
    sizes_and_patterns = [(kv_length, 0, sliding_window, chunk_size)]
    layer_to_mask_mapping = [0] * config.num_hidden_layers

    # sliding_window_pattern is the attribute used by models with alternate full/sliding layers
    sliding_window_pattern = getattr(config, "sliding_window_pattern", None)
    # no_rope_layers is the attribute used by models with alternate full/chunked layers
    is_chunked = getattr(config, "no_rope_layers", None)

    if sliding_window_pattern is not None and is_chunked is not None:
        raise ValueError("We detected both an alternating sliding and chunked pattern, which is incompatible")
    # This is for models with HybridCache
    elif sliding_window_pattern is not None:
        if sliding_window is None:
            raise ValueError("We detected a sliding window pattern, but no sliding window")
        # If this is not the case, we can still use only a single mask as the speciall pattern has no importance
        if kv_length > sliding_window:
            is_sliding = [bool((i + 1) % sliding_window_pattern) for i in range(config.num_hidden_layers)]
            layer_to_mask_mapping = [1 if sliding else 0 for sliding in is_sliding]
            sizes_and_patterns = [(kv_length, 0, None, None), (kv_length, 0, sliding_window, None)]
    # For models with HybridChunkedCache
    elif is_chunked is not None:
        if chunk_size is None:
            raise ValueError("We detected a chunked attention pattern, but no chunk size")
        # If this is not the case, we can still use only a single mask as the speciall pattern has no importance
        if kv_length > chunk_size:
            layer_to_mask_mapping = [1 if chunked else 0 for chunked in is_chunked]
            sizes_and_patterns = [(kv_length, 0, None, None), (kv_length, 0, None, chunk_size)]

    return sizes_and_patterns, layer_to_mask_mapping
