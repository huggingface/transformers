from typing import Optional

import torch

from ..modeling_flash_attention_utils import _flash_attention_forward
from ..utils import is_flash_attn_greater_or_equal_2_10


_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    target_dtype: torch.dtype = torch.float16,
    **kwargs,
):
    if attention_mask is not None:
        seq_len = attention_mask.shape[1]
        query = query[:, :, :seq_len]
        value = value[:, :, :seq_len]
    else:
        seq_len = query.shape[1]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    input_dtype = query.dtype
    if input_dtype == torch.float32:
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        seq_len,
        module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        **kwargs,
    )

    return attn_output, None
