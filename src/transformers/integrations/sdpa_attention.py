import torch

from ..utils import is_torch_npu_available, is_torch_xpu_available, logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()


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


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor, value: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key head_dim == value head_dim <= 256 (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


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
    else:
        # If we have a mask already, it's always of boolean dtype here. We only have to use the superpose both mask to float
        # dtype to use as additive mask in sdpa
        position_bias_mask = torch.where(attention_mask, position_bias, min_dtype)

    return position_bias_mask


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
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        if not use_gqa_in_sdpa(attention_mask, key, value):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    q_length = query.shape[2]
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
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
