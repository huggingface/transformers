from typing import Optional

import torch
from torch import nn


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


def eager_paged_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],  # shape [seqlen_q, seqlen_k]
    scaling: float,
    **kwargs,
):
    # Add KV cache to the key and value tensors
    cache = kwargs.pop("cache", None)
    if cache is not None:
        # This changes the shape of k and v from [1, num_kv_heads, seqlen_kv, head_dim] to [-1, num_kv_heads, head_dim]
        key, value = cache.update(key, value, module.layer_idx, **kwargs)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

    # Repeat the key and value tensors for each group of key-value heads
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # Get the right causal mask for the current layer
    if isinstance(attention_mask, dict):
        sliding_window = getattr(module, "sliding_window", 1)
        layer_type = "full_attention" if sliding_window == 1 or sliding_window is None else "sliding_attention"
        causal_mask = attention_mask[layer_type]
    else:
        causal_mask = attention_mask

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if causal_mask is not None:
        attn_weights = attn_weights + causal_mask

    # Handle attention sinks if the model has them
    if hasattr(module, "sinks"):
        # Retrieve the sink and add it to the attention weights
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        attn_weights = torch.cat([attn_weights, sinks], dim=-1)
        # Normalize the attention weights for better numerical stability
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
        # Apply softmax and drop the sink. Not exactly the same code as eager w/ sink, but the same code does not produce the same results.
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = attn_weights[..., :-1]
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
