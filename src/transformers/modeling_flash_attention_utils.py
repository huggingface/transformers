# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
import warnings
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F

from .utils import (
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_flash_attn_greater_or_equal,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_npu_available,
    logging,
)


logger = logging.get_logger(__name__)
flash_attn_func = None


def _index_first_axis(tensor, indices):
    """
    A local implementation of the PyTorch indexing operation `tensor[indices]` on the first axis,
    after flattening the first two dimensions of the tensor. This is functionally equivalent to
    FA2's `index_first_axis` and replaces the need to import it.
    """
    # The input tensor is expected to be of shape (batch, seq_len, ...). We flatten the first
    # two dimensions to get (total_tokens, ...) before indexing.
    reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
    return reshaped_tensor[indices]


def _fa3_unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    FA3-compatible unpad_input function.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        _index_first_axis(hidden_states, indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def _fa3_pad_input(hidden_states, indices, batch, seqlen):
    """
    FA3-compatible pad_input function.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


FA_VERSION = None
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func as flash_attn_2_func
    from flash_attn import flash_attn_varlen_func as flash_attn_2_varlen_func
    from flash_attn.bert_padding import pad_input as pad_input_fa2
    from flash_attn.bert_padding import unpad_input as unpad_input_fa2
    from flash_attn.layers.rotary import apply_rotary_emb

    HAS_FA2 = True
    FA_VERSION = 2
elif is_torch_npu_available():
    # patch functions in package `flash-attn` when using flash-attention on Ascend NPU.
    from .integrations.npu_flash_attention import npu_apply_rotary_emb as apply_rotary_emb  # noqa: F401
    from .integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_2_func
    from .integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_2_varlen_func
    from .integrations.npu_flash_attention import pad_input as pad_input_fa2
    from .integrations.npu_flash_attention import unpad_input as unpad_input_fa2

    HAS_FA2 = True
    FA_VERSION = 2
else:
    flash_attn_2_func = None
    flash_attn_2_varlen_func = None
    pad_input_fa2 = None
    unpad_input_fa2 = None
    apply_rotary_emb = None
    HAS_FA2 = False

if is_flash_attn_3_available():
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func

    pad_input_fa3 = _fa3_pad_input
    unpad_input_fa3 = _fa3_unpad_input
    HAS_FA3 = True
    FA_VERSION = 3
else:
    flash_attn_3_func = None
    flash_attn_3_varlen_func = None
    pad_input_fa3 = None
    unpad_input_fa3 = None
    HAS_FA3 = False


# Current Flash Attention implementations
if FA_VERSION:
    flash_attn_func = globals()[f"flash_attn_{FA_VERSION}_func"]
    flash_attn_varlen_func = globals()[f"flash_attn_{FA_VERSION}_varlen_func"]
    unpad_input = globals()[f"unpad_input_fa{FA_VERSION}"]
    pad_input = globals()[f"pad_input_fa{FA_VERSION}"]


_flash_supports_window_size = False


if flash_attn_func:
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


def is_flash_attn_available():
    """Determine whether flash-attention can be used or not."""

    if is_flash_attn_3_available():
        return True

    # if package `flash-attn` is available, flash-attention can be used natively.
    if is_flash_attn_2_available():
        return True

    # flash-attention can be used on Ascend NPU without package `flash-attn`
    if is_torch_npu_available():
        return True

    return False


def flash_attn_supports_top_left_mask():
    """Determine whether flash-attention uses top-left or down-right mask"""

    if is_flash_attn_3_available():
        return False

    if is_flash_attn_2_available():
        # top-left mask is used in package `flash-attn` with version lower than 2.1.0
        return not is_flash_attn_greater_or_equal_2_10()

    if is_torch_npu_available():
        # down-right mask is used on Ascend NPU by default, set env `NPU_FA2_SPARSE_MODE=2` to activate top-left mask.
        from .integrations.npu_flash_attention import is_npu_fa2_top_left_aligned_causal_mask

        return is_npu_fa2_top_left_aligned_causal_mask()

    return False


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # NOTE: Similar to the `.item()` in prepare_fa2_from_position_ids, with torch compile,
    # this might cause a graph break
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    unpad_input_func,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.
        unpad_input_func:
            The function to use for unpadding the input tensors.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

    # With static caches, the k/v states may be larger than the mask -> we need to slice them to avoid generating garbage
    # It's a bit of an anti-pattern, but otherwise we silently compute wrong attentions scores
    if key_layer.shape[1] > (seq_len := attention_mask.shape[-1]):
        key_layer, value_layer = key_layer[:, :seq_len, :, :], value_layer[:, :seq_len, :, :]

    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = _index_first_axis(key_layer, indices_k)
    value_layer = _index_first_axis(value_layer, indices_k)
    if query_length == kv_seq_len:
        query_layer = _index_first_axis(query_layer, indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input_func(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def _prepare_flash_attention_from_position_ids(query, key, value, position_ids):
    """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cumulative lengths of each examples in the batch will be extracted from position_ids.

    NOTE: ideally cumulative lengths should be prepared at the data collator stage

    Arguments:
        query (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        query (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )

    # NOTE: With torch compile, this will cause a graph break if you don't set
    # `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` in the environment or call
    # `torch._dynamo.config.capture_scalar_outputs = True` before doing the forward pass.
    # This is a limitation of flash attention API, as the function `flash_attn_varlen_func`
    # requires `max_length_q`, `max_length_k` to be passed as `int` and not `torch.Tensor`.
    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    max_length = position_ids.max().item() + 1

    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))


def prepare_fa2_from_position_ids(*args, **kwargs):
    warnings.warn(
        "The function `prepare_fa2_from_position_ids` in `transformers.modeling_flash_attention_utils` is deprecated and will be removed in a future version. Please use `_prepare_flash_attention_from_position_ids` instead.",
        FutureWarning,
    )
    return _prepare_flash_attention_from_position_ids(*args, **kwargs)


def fa_peft_integration_check(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!

    Args:
        query (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        target_dtype (`torch.dtype`, *optional*):
            The dtype to convert the attention tensors to. Conversion can be ignored by
            not providing the target dtype.
    """
    if target_dtype is None:
        return query, key, value

    input_dtype = query.dtype
    if input_dtype == torch.float32:
        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    return query, key, value


flash_241 = is_flash_attn_greater_or_equal("2.4.1")
deterministic_g = None


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        attn_implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
    """
    if attn_implementation is None:
        _flash_attn_varlen_func = flash_attn_varlen_func
        _flash_attn_func = flash_attn_func
        _pad_input = pad_input
        _unpad_input = unpad_input
        _is_fa3 = HAS_FA3
    elif attn_implementation == "flash_attention_3":
        _flash_attn_varlen_func = flash_attn_3_varlen_func
        _flash_attn_func = flash_attn_3_func
        _pad_input = pad_input_fa3
        _unpad_input = unpad_input_fa3
        _is_fa3 = True
    elif attn_implementation == "flash_attention_2":
        _flash_attn_varlen_func = flash_attn_2_varlen_func
        _flash_attn_func = flash_attn_2_func
        _pad_input = pad_input_fa2
        _unpad_input = unpad_input_fa2
        _is_fa3 = False

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if _is_fa3:
        if dropout > 0.0:
            logger.warning_once("Flash Attention 3 does not support dropout. Setting dropout to 0.0.")
    else:
        flash_kwargs["dropout_p"] = dropout

    if flash_241:
        if deterministic is None:
            global deterministic_g
            if deterministic_g is None:
                deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # We will use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    # under two cases:
    # Case 1. If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Case 2. Some models pass directly pre-computed `cu_seqlens` so we don't need to infer it from position ids. It is safe to
    # use `flash_attn_varlen_func` knowing we already have all necessary the kwargs. NOTE: it is user's responsibility
    # to take care of flattenning `position_ids` if that's needed by the model. See #39121 for more information
    is_fa2_with_position_ids = (
        position_ids is not None
        and query_states.shape[0] == 1
        and (max_length_q is not None or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all()))
    )
    is_fa2_with_varlen_kwargs = all(
        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, _unpad_input
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = _flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = _pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    elif is_fa2_with_varlen_kwargs or is_fa2_with_position_ids:
        batch_size = query_states.size(0)

        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                _prepare_flash_attention_from_position_ids(query_states, key_states, value_states, position_ids)
            )

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens

        else:
            query_states = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            key_states = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            value_states = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        attn_output = _flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = _flash_attn_func(
            query_states, key_states, value_states, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    if isinstance(attn_output, tuple):
        return attn_output[0]
    return attn_output


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cumulative_seqlens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cumulative_seqlens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cumulative_seqlens_q: Optional[torch.LongTensor]
    cumulative_seqlens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]
