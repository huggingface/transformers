import torch

from ..modeling_flash_attention_utils import (
    _flash_attention_forward,
    cast_to_flash_compatible_dtype,
    flash_attn_supports_top_left_mask,
)
from ..utils import logging


logger = logging.get_logger(__name__)

_use_top_left_mask = flash_attn_supports_top_left_mask()


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    s_aux: torch.Tensor | None = None,  # alias: learnable attention sink
    **kwargs,
) -> tuple[torch.Tensor, None]:
    batch_size, num_heads, seq_len, q_head_dim = query.shape
    v_head_dim = value.shape[-1]

    # Check for incompatible kwargs
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "Flash Attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # FA2 uses non-transposed inputs
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))

    # FlashAttention requires the query and value have the same head dim; pad `value` up to the query head dim and crop
    # the output below. This happens for example in MLA, where `v_head_dim < qk_head_dim`.
    if v_head_dim != q_head_dim:
        value = torch.nn.functional.pad(value, [0, q_head_dim - v_head_dim])

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query, key, value = cast_to_flash_compatible_dtype(module, query, key, value)

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        attn_implementation=module.config._attn_implementation,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        s_aux=(
            s_aux.to(query.dtype)  # FA only accepts half precision
            if s_aux is not None
            else None
        ),
        **kwargs,
    )

    if v_head_dim != head_dim:
        attn_output = attn_output[..., :v_head_dim]

    return attn_output, None
