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

from transformers.utils.import_utils import is_kernels_available

from .utils import (
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_flash_attn_greater_or_equal,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_npu_available,
    logging,
)


logger = logging.get_logger(__name__)


def _index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    reshaped = tensor.contiguous().reshape(-1, *tensor.shape[2:])
    return reshaped[indices]


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
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
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
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length = cu_seq_lens.diff().max().item()
    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))


def _prepare_flash_attention_from_position_ids(query, key, value, position_ids):
    warnings.warn(
        "prepare_fa2_from_position_ids is deprecated, use _prepare_from_posids",
        FutureWarning,
    )
    return _prepare_from_posids(query, key, value, position_ids)


def fa_peft_integration_check(q, k, v, target_dtype: Optional[torch.dtype] = None):
    if target_dtype and q.dtype == torch.float32:
        logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-attn compatibility.")
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
    return q, k, v


def _lazy_imports(impl: Optional[str]):
    # returns funcs and pad/unpad based on impl
    is_fa2 = is_flash_attn_2_available() or is_torch_npu_available()
    is_fa3 = is_flash_attn_3_available()
    if impl == "flash_attention_2" or (impl is None and is_fa2 and not is_fa3):
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            from flash_attn.bert_padding import pad_input, unpad_input

            return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input, False

        except ImportError as e:
            if not globals().get("use_remote_fa2", None):
                use_remote_fa2 = (
                    input(
                        "Unable to import the official flash attention, do you want to try to use `kernels-community/flash-attn` (trust remote code) Yes or No? "
                    )
                    .strip()
                    .lower()
                )
                globals()["use_remote_fa2"] = use_remote_fa2 in {"yes", "y", "1"}
            if globals()["use_remote_fa2"]:
                if not is_kernels_available():
                    raise ImportError("You need to install kernels: `pip install kernels`")
                from kernels import get_kernel

                impl = get_kernel("kernels-community/flash-attn")
                pad_input, unpad_input = _fa3_pad_input, _fa3_unpad_input
                return (
                    getattr(impl, "flash_attn_func", None),
                    getattr(impl, "flash_attn_varlen_func"),
                    pad_input,
                    unpad_input,
                    True,
                )

            else:
                raise ImportError(
                    "Failed to import flash attention 2, please install it or use another implementation."
                ) from e
    if impl == "flash_attention_3" or (impl is None and is_fa3):
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

        pad_input, unpad_input = _fa3_pad_input, _fa3_unpad_input
        return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input, True
    else:
        pad_input, unpad_input = _fa3_pad_input, _fa3_unpad_input
        return (
            getattr(impl, "flash_attn_func", None),
            getattr(impl, "flash_attn_varlen_func"),
            pad_input,
            unpad_input,
            True,
        )


_flash_supports_window = None


def is_flash_attn_available():
    return is_flash_attn_3_available() or is_flash_attn_2_available() or is_torch_npu_available()


def flash_attn_supports_top_left_mask():
    if is_flash_attn_3_available():
        return False
    if is_flash_attn_2_available():
        return not is_flash_attn_greater_or_equal_2_10()

    from .integrations.npu_flash_attention import is_npu_fa2_top_left_aligned_causal_mask

    return is_npu_fa2_top_left_aligned_causal_mask()


class FlashAttentionKwargs(TypedDict, total=False):
    cumulative_seqlens_q: Optional[torch.LongTensor]
    cumulative_seqlens_k: Optional[torch.LongTensor]


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
    if not all(k in globals() for k in ("_flash_fn", "_flash_varlen_fn", "_pad_fn", "_unpad_fn", "_is_fa3")):
        flash_fn, flash_varlen_fn, pad_fn, unpad_fn, is_fa3 = _lazy_imports(implementation)
        globals()["_flash_fn"] = flash_fn
        globals()["_flash_varlen_fn"] = flash_varlen_fn
        globals()["_pad_fn"] = pad_fn
        globals()["_unpad_fn"] = unpad_fn
        globals()["_is_fa3"] = is_fa3
        flash_supports_window = "window_size" in inspect.signature(flash_varlen_fn).parameters
        globals()["_flash_supports_window"] = flash_supports_window
    else:
        flash_fn = globals()["_flash_fn"]
        flash_varlen_fn = globals()["_flash_varlen_fn"]
        pad_fn = globals()["_pad_fn"]
        unpad_fn = globals()["_unpad_fn"]
        is_fa3 = globals()["_is_fa3"]
        flash_supports_window = globals()["_flash_supports_window"]

    causal = is_causal and not (use_top_left_mask and query_length == 1)
    use_sw = (
        (_flash_supports_window or flash_supports_window) and sliding_window and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sw else {}
    if not is_fa3:
        flash_kwargs["dropout_p"] = dropout
    if is_flash_attn_greater_or_equal("2.4.1"):
        det = deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = det
    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )
    use_mask = position_ids is not None or all(
        k is not None for k in [cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k]
    )
    if attention_mask is not None:
        q, k, v, idx, (cu_q, cu_k), (mq, mk) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_fn
        )
        # TODO for now this is required to work with https://huggingface.co/kernels-community/metal-flash-sdpa/blob/main/torch-ext/metal_flash_sdpa/__init__.p
        if "mps" in str(q.device):
            cu_k = cu_k.clone()
        out_unpad = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_q.to(torch.int32),
            cu_seqlens_k=cu_k.to(torch.int32),
            max_seqlen_q=mq,
            max_seqlen_k=mk,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        if isinstance(out_unpad, tuple):
            out_unpad = out_unpad[0]
        out = pad_fn(out_unpad, idx, query_states.shape[0], query_length)
    elif use_mask:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            if position_ids is None:
                raise ValueError(
                    "Position ids should be passed if the attention mask is not passed and the cu_seq-lens are not passed."
                )
            q, k, v, idx, (cu_q, cu_k), (mq, mk) = _prepare_from_posids(
                query_states, key_states, value_states, position_ids
            )
        else:
            q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))
            mq, mk = max_length_q, max_length_k
            cu_q, cu_k = cu_seq_lens_q, cu_seq_lens_k
        if "mps" in str(q.device):
            cu_k = cu_k.clone()
        out = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_q.to(torch.int32),
            cu_seqlens_k=cu_k.to(torch.int32),
            max_seqlen_q=mq,
            max_seqlen_k=mk,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(query_states.shape[0], -1, out.size(-2), out.size(-1))
    else:
        out = flash_fn(
            query_states, key_states, value_states, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return out[0] if isinstance(out, tuple) else out
