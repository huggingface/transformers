import torch
from ..modeling_flash_attention_utils import FlashAttentionKwargs, _flash_attention_forward

def flash_attention_forward(
    config, query, key, value, mask, target_dtype=torch.float16, training=False, layer_idx=0, **kwargs
):
    if mask is not None:
        seq_len = mask.shape[1]
        query = query[:, :, :seq_len]
        value = value[:, :, :seq_len]
    else:
        seq_len = query.shape[1]

    dropout_rate = config.attention_dropout if training else 0.0

    input_dtype = query.dtype
    if input_dtype == torch.float32:
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        mask,
        seq_len,
        dropout=dropout_rate,
        softmax_scale=getattr(config, "scaling", 1.0),
        is_causal=getattr(config, "is_causal", False),
        sliding_window=getattr(config, "sliding_window", None),
        use_top_left_mask=getattr(config, "_flash_attn_uses_top_left_mask", False),
        layer_idx=layer_idx,
        **kwargs,
    )

    return attn_output, None