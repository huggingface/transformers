import math
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    repeat_kv,
)


class MockModule(torch.nn.Module):
    def __init__(self, num_key_value_groups: int):
        self.num_key_value_groups = num_key_value_groups
        self.training = False


# Old code from src/transformers/models/llama/modelling_llama.py
def old_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Old code from src/transformers/integrations/sdpa_attention.py
def old_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


@pytest.mark.parametrize(
    "batch_size, num_heads, num_query_groups, q_len, kv_len, head_size",
    [
        (3, 16, 2, 128, 128, 8),
        (1, 8, 1, 5, 128, 16),
        (3, 16, 16, 20, 20, 8),
        (5, 14, 2, 21, 27, 8),
    ],
)
def test_eager_attention_forward(batch_size, num_heads, num_query_groups, q_len, kv_len, head_size):
    dtype = torch.float32
    query = torch.randn((batch_size, num_heads, q_len, head_size), dtype=dtype)
    key = torch.randn((batch_size, num_query_groups, kv_len, head_size), dtype=dtype)
    value = torch.randn((batch_size, num_query_groups, kv_len, head_size), dtype=dtype)
    scale = 1.0 / math.sqrt(head_size)
    attn_mask = torch.rand((batch_size, num_heads, q_len, kv_len)) <= 0.75
    mock_module = MockModule(num_key_value_groups=num_heads // num_query_groups)

    output1, weights1 = eager_attention_forward(
        module=mock_module,
        query=query,
        key=key,
        value=value,
        attention_mask=attn_mask,
        scaling=scale,
    )
    output2, weights2 = old_eager_attention_forward(
        module=mock_module,
        query=query,
        key=key,
        value=value,
        attention_mask=attn_mask,
        scaling=scale,
    )
    torch.testing.assert_close(output1, output2)
    torch.testing.assert_close(weights1, weights2)


@pytest.mark.parametrize(
    "batch_size, num_heads, num_query_groups, q_len, kv_len, head_size",
    [
        (3, 16, 2, 128, 128, 8),
        (1, 8, 1, 5, 128, 16),
        (3, 16, 16, 20, 20, 8),
        (5, 14, 2, 21, 27, 8),
    ],
)
def test_sdpa_attention_forward(batch_size, num_heads, num_query_groups, q_len, kv_len, head_size):
    dtype = torch.float32
    query = torch.randn((batch_size, num_heads, q_len, head_size), dtype=dtype)
    key = torch.randn((batch_size, num_query_groups, kv_len, head_size), dtype=dtype)
    value = torch.randn((batch_size, num_query_groups, kv_len, head_size), dtype=dtype)
    scale = 1.0 / math.sqrt(head_size)
    attn_mask = torch.rand((batch_size, num_heads, q_len, kv_len)) <= 0.75
    mock_module = MockModule(num_key_value_groups=num_heads // num_query_groups)

    output1, _ = sdpa_attention_forward(
        module=mock_module,
        query=query,
        key=key,
        value=value,
        attention_mask=attn_mask,
        scaling=scale,
    )
    output2, _ = old_sdpa_attention_forward(
        module=mock_module,
        query=query,
        key=key,
        value=value,
        attention_mask=attn_mask,
        scaling=scale,
    )
    torch.testing.assert_close(output1, output2)
