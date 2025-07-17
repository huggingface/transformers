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


def _index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    reshaped = tensor.reshape(-1, *tensor.shape[2:])
    return reshaped[indices]


def _fa3_unpad_input(hidden_states, attention_mask, unused_mask=None):
    masks = attention_mask + unused_mask if unused_mask is not None else attention_mask
    lengths = masks.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(masks.flatten(), as_tuple=False).flatten()
    max_len = lengths.max().item()
    cu = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    return (
        _index_first_axis(hidden_states, indices),
        indices,
        cu,
        max_len,
        attention_mask.sum(dim=-1, dtype=torch.int32),
    )


def _fa3_pad_input(hidden_states, indices, batch: int, seqlen: int):
    out = torch.zeros(
        (batch * seqlen), *hidden_states.shape[1:], device=hidden_states.device, dtype=hidden_states.dtype
    )
    out[indices] = hidden_states
    return out.view(batch, seqlen, *hidden_states.shape[1:])


def _get_unpad_data(attn_mask: torch.Tensor):
    seqlens = attn_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attn_mask.flatten(), as_tuple=False).flatten()
    max_len = seqlens.max().item()
    cu = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu, max_len


def _upad_input(q, k, v, attn_mask: torch.Tensor, q_len: int, unpad_fn):
    indices_k, cu_k, max_k = _get_unpad_data(attn_mask)
    # trim KV if too long
    if k.shape[1] > attn_mask.shape[-1]:
        k, v = k[:, : attn_mask.shape[-1]], v[:, : attn_mask.shape[-1]]
    q = (
        _index_first_axis(q, indices_k)
        if q_len == attn_mask.shape[-1]
        else _index_first_axis(q.squeeze(1), torch.arange(q.shape[0], device=q.device))
    )
    k = _index_first_axis(k, indices_k)
    v = _index_first_axis(v, indices_k)
    cu_q = cu_k if q_len == attn_mask.shape[-1] else torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device)
    max_q = max_k if q_len == attn_mask.shape[-1] else 1
    return q, k, v, indices_k, (cu_q, cu_k), (max_q, max_k)


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
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length = cu_seq_lens.diff().max().item()

    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))


def _prepare_flash_attention_from_position_ids(query, key, value, position_ids):
    warnings.warn(
        "prepare_fa2_from_position_ids is deprecated, use _prepare_flash_attention_from_position_ids",
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
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.layers.rotary import apply_rotary_emb

        return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input, False
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
            False,
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
    flash_fn, flash_varlen_fn, pad_fn, unpad_fn, is_fa3 = _lazy_imports(implementation)
    causal = is_causal and not (use_top_left_mask and query_length == 1)
    use_sw = (
        (_flash_supports_window or "window_size" in inspect.signature(flash_varlen_fn).parameters)
        and sliding_window
        and key_states.shape[1] > sliding_window
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
    use_varlen = position_ids is not None or all([cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k])
    if attention_mask is not None:
        q, k, v, idx, (cu_q, cu_k), (mq, mk) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_fn
        )
        out_unpad = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=mq,
            max_seqlen_k=mk,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        out = pad_fn(out_unpad, idx, query_states.shape[0], query_length)
    elif use_varlen:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            if position_ids is None:
                raise ValueError(
                    "Position ids should be passed if the attention mask is not passed and the cu_seq-lens are not passed."
                )
            q, k, v, idx, (cu_q, cu_k), (mq, mk) = _prepare_from_posids(
                query_states, key_states, value_states, position_ids
            )
        else:
            q = query_states.view(-1, query_states.size(-2), query_states.size(-1))
            k = key_states.view(-1, key_states.size(-2), key_states.size(-1))
            v = value_states.view(-1, value_states.size(-2), value_states.size(-1))
            mq, mk = max_length_q, max_length_k
            cu_q, cu_k = cu_seq_lens_q, cu_seq_lens_k
        out = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=mq,
            max_seqlen_k=mk,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        out = out.view(query_states.shape[0], -1, out.size(-2), out.size(-1))
    else:
        out = flash_fn(
            query_states, key_states, value_states, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return out[0] if isinstance(out, tuple) else out
