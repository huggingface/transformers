import os
from collections.abc import Callable
from typing import cast

import torch

from ..utils import is_torch_mps_available, is_torch_npu_available, is_torch_xpu_available, logging
from ..utils.import_utils import is_torch_greater_or_equal, is_torchdynamo_compiling


logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_greater_or_equal_than_2_11 = is_torch_greater_or_equal("2.11", accept_dev=True)
_is_torch_greater_or_equal_than_2_12 = is_torch_greater_or_equal("2.12", accept_dev=True)
_is_torch_greater_or_equal_than_2_13 = is_torch_greater_or_equal("2.13", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()
_is_torch_mps_available = is_torch_mps_available()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def pack_gqa(hidden_states: torch.Tensor | None, n_rep: int) -> torch.Tensor | None:
    """
    For a single query token: (batch, num_attention_heads, 1, dim) -> (batch, num_key_value_heads, n_rep, dim), so
    that attention runs on the key/value heads without `repeat_kv`. Masks shared by all heads are returned as is.
    """
    if hidden_states is None or hidden_states.shape[1] == 1:
        return hidden_states
    batch, num_attention_heads, _, dim = hidden_states.shape
    return hidden_states.reshape(batch, num_attention_heads // n_rep, n_rep, dim)


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor, value: torch.Tensor) -> bool:
    if os.environ.get("TRANSFORMERS_SDPA_FORCE_GQA", "0") == "1":
        return True
    # XPU and MPS support masked GQA under more recent torch versions
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    elif _is_torch_mps_available:
        return _is_torch_greater_or_equal_than_2_13
    # CUDA and Ascend NPU require no mask and supported head dims to avoid falling back to the math kernel.
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


def create_position_bias_mask(
    position_bias: torch.Tensor,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """
    Create a floating-point dtype mask to use with sdpa. The mask contains the values of `position_bias` to positions where we should
    attend to tokens, and -inf where we should not. It will be added to the QK^T result in the attention, before the softmax. Note
    that using such a mask will usually prevent sdpa from dispatching to the most efficient kernel implementations.

    Note that we cannot create this in advance when we create the mask in the model, as the position_bias is usually learned
    differently in every layer.
    """
    min_dtype = torch.finfo(key.dtype).min
    # If we don't have a mask already, we need to check causality to be sure to respect it
    if attention_mask is None:
        # If we were gonna rely on `is_causal`, we need to create a mask to respect causality on top of the position_bias mask
        if is_causal:
            device = key.device
            q_length, kv_length = query.shape[2], key.shape[2]
            causal_mask = (
                torch.arange(q_length, device=device)[:, None] >= torch.arange(kv_length, device=device)[None, :]
            )
            causal_mask = causal_mask.view(1, 1, q_length, kv_length)
            position_bias_mask = torch.where(causal_mask, position_bias, min_dtype)
        # If it's not causal, we can simply use the position_bias as the additive mask in sdpa
        else:
            position_bias_mask = position_bias
    elif attention_mask.dtype == torch.bool:
        # If we have a boolean mask, superpose both into a single float additive mask to use in sdpa
        position_bias_mask = torch.where(attention_mask, position_bias, min_dtype)
    else:
        # A custom float mask is already additive, so we can simply add it to the position_bias
        position_bias_mask = position_bias + attention_mask

    return position_bias_mask.contiguous()


def sdpa_varlen_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    sliding_window: int | None = None,
    sequence_lengths: list[int] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Attention over sequences packed along the length, `(batch, heads, length, head_dim)`, delimited by `cu_seq_lens_q`
    (and `cu_seq_lens_k`) over the flattened batch.

    Runs torch's varlen kernel when it supports the inputs: CUDA, half precision, no dropout, and a head size that is a
    multiple of 8 up to 256, the same for values. Otherwise, reads the lengths on the host and attends over each
    sequence separately.
    """
    batch_size, num_heads, _, head_dim = query.shape
    cu_seq_lens_k = cu_seq_lens_q if cu_seq_lens_k is None else cu_seq_lens_k
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    # One packed row of all the sequences: a view for a batch of one
    query, key, value = (states.transpose(0, 1).flatten(1, 2)[None] for states in (query, key, value))
    use_varlen = (
        _is_torch_greater_or_equal_than_2_11
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and dropout == 0.0
        # Head sizes the flash kernel supports, the same for values (unlike MLA's)
        and head_dim % 8 == 0
        and head_dim <= 256
        and value.shape[-1] == head_dim
    )
    if use_varlen:
        from torch.nn.attention.varlen import varlen_attn

        # The kernel only uses the maximum lengths to size its launch: without precomputed ones, the total lengths are
        # valid bounds that need no host sync
        max_length_q = query.shape[2] if max_length_q is None else max_length_q
        max_length_k = key.shape[2] if max_length_k is None else max_length_k
        # Keys a query sees on its left and right, -1 for all of them
        window_size = (-1 if sliding_window is None else sliding_window - 1, 0 if is_causal else -1)
        varlen_kwargs = {}
        if query.shape[1] != key.shape[1]:
            if _is_torch_greater_or_equal_than_2_12:
                varlen_kwargs["enable_gqa"] = True
            else:
                key = repeat_kv(key, query.shape[1] // key.shape[1])
                value = repeat_kv(value, query.shape[1] // value.shape[1])
        # Without `return_aux`, a single tensor
        attn_output = cast(
            torch.Tensor,
            varlen_attn(
                query[0].transpose(0, 1),
                key[0].transpose(0, 1),
                value[0].transpose(0, 1),
                cu_seq_lens_q.to(torch.int32),
                cu_seq_lens_k.to(torch.int32),
                max_length_q,
                max_length_k,
                scale=scaling,
                window_size=window_size,
                **varlen_kwargs,
            ),
        )
        return attn_output.view(batch_size, -1, num_heads, head_dim), None

    attn_output, _ = per_sequence_attention_forward(
        sdpa_attention_forward,
        module,
        query,
        key,
        value,
        cu_seq_lens_q,
        cu_seq_lens_k,
        sequence_lengths=sequence_lengths,
        is_causal=is_causal,
        sliding_window=sliding_window,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )
    return attn_output.view(batch_size, -1, num_heads, head_dim), None


def per_sequence_attention_forward(
    attention_fn: Callable,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    sequence_lengths: list[int] | None = None,
    is_causal: bool = True,
    sliding_window: int | None = None,
    causal_needs_mask: bool = False,
    return_attention_weights: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """
    Calls `attention_fn` on each of the sequences packed along the length of a batch of one, with a mask only where
    the attention needs it: a (causal) window narrower than the sequence, or causal attention itself when
    `causal_needs_mask` (eager has no causal flag). Returns the outputs concatenated along the length, `(1, length,
    heads, head_dim)`, and each sequence's weights if `return_attention_weights` (otherwise freed as it goes).
    """
    # Splitting needs the lengths on the host: read once per forward when the model passes them (shared by queries
    # and keys), here otherwise
    if sequence_lengths is not None:
        lengths_q = lengths_k = sequence_lengths
    else:
        lengths_q = (cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]).tolist()
        lengths_k = (cu_seq_lens_k[1:] - cu_seq_lens_k[:-1]).tolist()
    attn_outputs, attn_weights = [], []
    for q, k, v in zip(query.split(lengths_q, dim=2), key.split(lengths_k, dim=2), value.split(lengths_k, dim=2)):
        mask = None
        if (is_causal and causal_needs_mask) or (sliding_window is not None and k.shape[2] > sliding_window):
            visible = torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=q.device)
            visible = visible.tril() if is_causal else visible
            visible = visible.triu(1 - sliding_window) if sliding_window is not None else visible
            mask = torch.zeros(visible.shape, dtype=q.dtype, device=q.device)
            mask = mask.masked_fill(~visible, torch.finfo(q.dtype).min)[None, None]
        attn_output, weights = attention_fn(module, q, k, v, mask, is_causal=is_causal, **kwargs)
        attn_outputs.append(attn_output)
        if return_attention_weights:
            attn_weights.append(weights)
        del weights
    return torch.cat(attn_outputs, dim=1), attn_weights if return_attention_weights else None


def eager_packed_sequences_attention_forward(
    eager_attention_forward: Callable,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sliding_window: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    A model's `eager_attention_forward`, attending over each sequence of packed rows separately when the mask is their
    cumulative lengths over the flattened batch (from `create_causal_mask`): small per-sequence masks instead of a
    dense one over the whole rows. The weights are block-diagonal, kept only when attentions are recorded.
    """
    if attention_mask is None or attention_mask.ndim != 1:
        return eager_attention_forward(
            module, query, key, value, attention_mask, sliding_window=sliding_window, **kwargs
        )
    batch_size, num_heads, length, head_dim = query.shape
    record_attentions = kwargs.get("output_attentions", getattr(module.config, "output_attentions", False))
    # One packed row of all the sequences: a view for a batch of one
    query, key, value = (states.transpose(0, 1).flatten(1, 2)[None] for states in (query, key, value))
    attn_output, sequence_weights = per_sequence_attention_forward(
        eager_attention_forward,
        module,
        query,
        key,
        value,
        attention_mask,
        attention_mask,
        is_causal=getattr(module, "is_causal", True),
        sliding_window=sliding_window,
        causal_needs_mask=True,
        return_attention_weights=record_attentions,
        **kwargs,
    )
    attn_weights = None
    if sequence_weights is not None:
        attn_weights = query.new_zeros(batch_size, num_heads, length, length)
        start = 0
        for weights in sequence_weights:
            row, offset = divmod(start, length)
            end = offset + weights.shape[-1]
            attn_weights[row, :, offset:end, offset:end] = weights[0]
            start += weights.shape[-1]
    return attn_output.view(batch_size, length, num_heads, head_dim), attn_weights


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("softcap") is not None:
        raise ValueError(
            "`sdpa` attention does not support `softcap`. Try using `eager`, `flex_attention`, or the few FA kernels supporting it"
        )
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # Sequences packed along the length, delimited by their cumulative lengths rather than by a mask: passed as the mask
    # itself, 1D, for packed text rows (`create_causal_mask`), or as `cu_seq_lens_q` (packed encoders)
    if attention_mask is not None and attention_mask.ndim == 1:
        kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = attention_mask
        attention_mask = None
    if kwargs.get("cu_seq_lens_q") is not None and attention_mask is None:
        return sdpa_varlen_attention_forward(
            module, query, key, value, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
        )

    sdpa_kwargs = {}
    q_length = query.shape[2]
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        if use_gqa_in_sdpa(attention_mask, key, value):
            sdpa_kwargs["enable_gqa"] = True
        elif q_length == 1 and not is_torchdynamo_compiling():
            query = pack_gqa(query, module.num_key_value_groups)
            attention_mask = pack_gqa(attention_mask, module.num_key_value_groups)
            position_bias = pack_gqa(position_bias, module.num_key_value_groups)
        else:
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)

    kv_length = key.shape[2]

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    # TODO: under `torch.export` with dynamic shapes, `q_length > 1` is a `SymBool` and `and` returns the first falsy
    #   operand, so `is_causal` can end up being that `SymBool` (e.g. the seamless_m4t / seamless_m4t_v2 speech
    #   encoders). Reordering the conditions fixes it but breaks the compile requirement above, so it should rather be
    #   handled on the exporter side. See https://github.com/huggingface/transformers/pull/46196#discussion_r3717333141
    is_causal = q_length > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # This scenario can only happen during prefill with an empty StaticCache. Technically, since sdpa's `is_causal` mask alignment
    # is upper-left, `is_causal=True` is enough to correctly compute the attention. However, sdpa will only dispatch to
    # flash kernel if and only if q_length == kv_length, therefore it is more efficient to slice here and remove the masked tokens
    # rather than to use the other available kernels for such a case.
    # Note that we never compile prefill, and even if the user is doing it on its own, prefill and decode are 2 separate graphs
    # anyway, so altering the shapes is fine here
    if is_causal and attention_mask is None and q_length > 1 and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]
        # If we have a position_bias, we need to crop it as well (on last dim, which is the kv seq_len dim)
        if position_bias is not None:
            position_bias = position_bias[:, :, :, :q_length]

    # If we have a position_bias, create the correct floating-point mask by combining it with the existing mask, or a causal mask
    # if `is_causal=True`
    if position_bias is not None:
        attention_mask = create_position_bias_mask(position_bias, attention_mask, is_causal, query, key)
        is_causal = False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    # Undoes `pack_gqa`, a no-op otherwise. The batch comes from the output, as SDPA broadcasts it (e.g. learned
    # queries with batch 1)
    attn_output = attn_output.reshape(attn_output.shape[0], -1, q_length, value.shape[-1])
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
