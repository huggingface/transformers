# Copyright 2025 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
from functools import partial
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F

from .utils import (
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_npu_available,
    logging,
)


logger = logging.get_logger(__name__)


# TODO Deprecate when all models have the attention interface
def flash_attn_supports_top_left_mask():
    if is_flash_attn_3_available():
        return False
    if is_flash_attn_2_available():
        return not is_flash_attn_greater_or_equal_2_10()

    from .integrations.npu_flash_attention import is_npu_fa2_top_left_aligned_causal_mask

    return is_npu_fa2_top_left_aligned_causal_mask()


# TODO Deprecate when all models have the attention interface
def is_flash_attn_available():
    return is_flash_attn_3_available() or is_flash_attn_2_available() or is_torch_npu_available()


# `globals()` is not compatible with dynamo, hence we have do define them in global scope ourselves
_flash_fn = None
_flash_varlen_fn = None
_pad_fn = None
_unpad_fn = None

# function that processes kwargs, generalized to handle any supported kwarg within the function
_process_flash_kwargs_fn = None
# exceptions where hf API doesn't match the original flash attention API
_hf_api_to_flash_mapping = {
    "dropout": "dropout_p",
    "sliding_window": "window_size",
}


def _lazy_imports(implementation: Optional[str]):
    """
    Lazy loads the respective flash attention implementations.

    Return:
        flash_attn_func: The base flash attention function.
        flash_attn_varlen_func: The flash attention function supporting variable sequence lengths,
                                e.g. for padding-free training.
        pad_input: The function to pad inputs into one sequence and returning the respective kwargs.
        unpad_input: The function to unpad outputs based on the kwargs (from pad_input).
    """
    is_fa2 = is_flash_attn_2_available()
    is_fa3 = is_flash_attn_3_available()

    pad_input, unpad_input = _pad_input, _unpad_input

    if (implementation == "flash_attention_2" and is_fa2) or (implementation is None and is_fa2 and not is_fa3):
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
    elif is_torch_npu_available():
        # Package `flash-attn` is unavailable on Ascend NPU, which will cause ImportError
        # Flash-Attention2 related apis for Ascend NPU must be imported from `.integrations.npu_flash_attention` module
        from .integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
        from .integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
    else:
        if implementation == "flash_attention_3" or (implementation is None and is_fa3):
            from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        # Kernels fallback
        else:
            flash_attn_func = getattr(implementation, "flash_attn_func", None)
            flash_attn_varlen_func = getattr(implementation, "flash_attn_varlen_func", None)
            if flash_attn_varlen_func is None or flash_attn_func is None:
                raise ValueError(
                    f"Could not find the currently requested flash attention implementation at `{implementation}`."
                    f"Make sure that you request a valid kernel from the hub, e.g. `kernels-community/flash-attn`."
                )

    return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input


def _lazy_define_process_function(flash_function):
    """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """

    flash_parameters = inspect.signature(flash_function).parameters
    process_parameters = inspect.signature(_process_flash_attention_kwargs).parameters

    supports_mapping = {}
    for param in process_parameters:
        fa_param = _hf_api_to_flash_mapping.get(param, param)
        supports_mapping[fa_param] = fa_param in flash_parameters

    return partial(_process_flash_attention_kwargs, supports_mapping=supports_mapping)


def lazy_import_flash_attention(implementation: Optional[str], force_import: Optional[bool] = False):
    """
    Lazily import flash attention and return the respective functions + flags.

    NOTE: For fullgraph, this needs to be called before compile, while no fullgraph can
    work without preloading. See `load_and_register_kernel` in `integrations.hub_kernels`.
    """
    global _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn
    if force_import or any(k is None for k in [_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn]):
        _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn = _lazy_imports(implementation)

    global _process_flash_kwargs_fn
    if force_import or _process_flash_kwargs_fn is None:
        _process_flash_kwargs_fn = _lazy_define_process_function(_flash_varlen_fn)

    return (_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn), _process_flash_kwargs_fn


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


def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    unpad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

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


def _pad_input(hidden_states, indices, batch, seqlen):
    """
    pad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

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


def prepare_fa_kwargs_from_position_ids(position_ids):
    """
    This function returns all the necessary kwargs to call `flash_attn_varlen_func` extracted from position_ids.

    Arguments:
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into
            ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length_q = cu_seq_lens_q.diff().max()
    # NOTE: With torch compile, this will cause a graph break if you don't set
    # `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` in the environment or call
    # `torch._dynamo.config.capture_scalar_outputs = True` before doing the forward pass.
    # This is a limitation of flash attention API, as the function `flash_attn_varlen_func`
    # requires `max_length_q`, `max_length_k` to be passed as `int` and not `torch.Tensor`.
    max_length_q = max_length_q.item()
    max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)


def _prepare_from_posids(query, key, value, position_ids):
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
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))

    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    return (query, key, value, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k))


def _is_packed_sequence(position_ids, batch_size):
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()


def fa_peft_integration_check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!
    """
    if target_dtype and q.dtype == torch.float32:
        logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-attn compatibility.")
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
    return q, k, v


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[torch.Tensor] = None,
    supports_mapping: Optional[dict[str, bool]] = None,
    **kwargs,
):
    """
    Returns a set of kwargs that are passed down to the according flash attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        dropout (`float`):
            Attention dropout.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        sliding_window (`int`, *optional*):
            The size of the sliding window, i.e. we look at a max of `sliding_window` tokens back.
        use_top_left_mask (`bool`):
            Deprecated behavior of older versions of flash attention requiring different masking.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        s_aux (`torch.Tensor`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
    flash_kwargs = {
        "causal": is_causal and not (use_top_left_mask and query_length == 1),
        "softmax_scale": softmax_scale,
    }

    if supports_mapping["dropout_p"]:
        flash_kwargs["dropout_p"] = dropout

    if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
        # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
        # and the current token for a total size of 5. However, we usually define our window sizes by
        # their total window size (when causal). Encoder models as of now seldom use SWA and when they
        # do, they have a custom workaround (e.g. ModernBERT) which would align with this symmetric logic, i.e.
        # for a total of `2*sliding_window + 1`.
        flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)

    if supports_mapping["deterministic"]:
        flash_kwargs["deterministic"] = (
            deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        )

    if supports_mapping["softcap"] and softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Only within kernel implementation atm
    if supports_mapping["s_aux"] and s_aux is not None:
        flash_kwargs["s_aux"] = s_aux

    return flash_kwargs


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
    implementation: Optional[str] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_attention_kwargs` and `FlashAttentionKwargs`.

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
        implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
    """
    (flash_fn, flash_varlen_fn, pad_fn, unpad_fn), process_flash_kwargs_fn = lazy_import_flash_attention(
        implementation
    )

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # Extract the flash attention kwargs that have been requested (and are supported by the implementation)
    flash_kwargs = process_flash_kwargs_fn(
        query_length=query_length,
        key_length=key_states.size(1),
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        use_top_left_mask=use_top_left_mask,
        softcap=softcap,
        deterministic=deterministic,
        **kwargs,
    )

    # We will use `flash_varlen_fn` to prevent cross-example attention and also allow padding free approach under two cases:
    # Case 1. If position ids is provided and the position ids indicate packed sequences, see `_is_packed_sequence`.
    # Case 2. Some models pass directly pre-computed `cu_seqlens` so we don't need to infer it from position ids. It is safe to
    # use `flash_varlen_fn` knowing we already have all necessary the kwargs.
    #
    # NOTE: it is user's responsibility to take care of flattening `position_ids` if that's needed by the model.
    # See #39121 for more information.
    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    is_fa_with_varlen_kwargs = all(
        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_fn
        )

        # TODO for now this is required to work with
        # https://huggingface.co/kernels-community/metal-flash-sdpa/blob/main/torch-ext/metal_flash_sdpa/__init__.py
        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out_unpad = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **flash_kwargs,
        )
        if isinstance(out_unpad, tuple):
            out_unpad = out_unpad[0]

        out = pad_fn(out_unpad, indices_q, query_states.size(0), query_length)

    # Padding free, i.e. sequences flattened into one total sequence
    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _prepare_from_posids(
                query_states, key_states, value_states, position_ids
            )
        else:
            q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        # TODO for now this is required to work with
        # https://huggingface.co/kernels-community/metal-flash-sdpa/blob/main/torch-ext/metal_flash_sdpa/__init__.py
        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **flash_kwargs,
        )
        if isinstance(out, tuple):
            out = out[0]

        out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))

    # No padding
    else:
        out = flash_fn(query_states, key_states, value_states, **flash_kwargs)
        if isinstance(out, tuple):
            out = out[0]

    return out
