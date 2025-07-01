from typing import Optional

import torch

from kernels import get_kernel

paged_attention_kernel = get_kernel("kernels-community/paged-attention")

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


def sdpa_attention_paged_forward__(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    cache = kwargs.pop("cache", None)
    if cache is not None:
        key, value = cache.update(key, value, module.layer_idx, **kwargs)
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def sdpa_attention_paged_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    cache = kwargs.pop("cache", None)
    if cache is not None:
        key, value = cache.update(key, value, module.layer_idx, **kwargs)
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # Get parameters
    batch_size, seq_len, num_heads, head_size = query.shape
    num_kv_heads = key.shape[2]
    block_size = kwargs.get("block_size", 32)
    max_seq_len = kwargs.get("max_seqlen_k", seq_len)
    x = 16  # Key cache formatting parameter
    
    # For paged attention, we need to handle each sequence separately
    # Reshape query to [batch_size, num_heads, head_size] - assuming seq_len=1 for generation
    if seq_len == 1:
        # Generation case - single token per batch
        query_reshaped = query.squeeze(1)  # [batch_size, num_heads, head_size]
    else:
        # Prefill case - need to handle multiple tokens
        query_reshaped = query.reshape(batch_size * seq_len, num_heads, head_size)
        batch_size = batch_size * seq_len
    
    # Calculate number of blocks needed
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * max_num_blocks_per_seq
    
    key_cache = torch.zeros(num_blocks, num_kv_heads, head_size // x, block_size, x, device=query.device, dtype=key.dtype)
    value_cache = torch.zeros(num_blocks, num_kv_heads, head_size, block_size, device=query.device, dtype=value.dtype)
    
    key_input = key.reshape(-1, num_kv_heads, head_size).contiguous()
    value_input = value.reshape(-1, num_kv_heads, head_size).contiguous()
    
    slot_mapping = torch.arange(key_input.shape[0], device=query.device)
    
    paged_attention_kernel.reshape_and_cache(
        key_input,
        value_input,
        key_cache,
        value_cache,
        slot_mapping,
        kwargs.get("kv_cache_dtype", "auto"),
        kwargs.get("k_scale", torch.tensor(1.0, device=query.device)),
        kwargs.get("v_scale", torch.tensor(1.0, device=query.device)),
    )
    
    # Create proper sequence lengths and block tables
    seq_lens = kwargs.get("cumulative_seqlens_k", None)
    if seq_lens is None:
        # Default: assume each sequence has seq_len tokens
        seq_lens = torch.full((batch_size,), seq_len, device=query.device, dtype=torch.int32)
    
    block_tables = kwargs.get("block_tables", None)
    # if block_tables is None:
    #     # Create default block tables
    #     block_tables_lst = []
    #     for i in range(batch_size):
    #         seq_length = seq_lens[i].item() if seq_lens is not None else seq_len
    #         num_blocks_needed = (seq_length + block_size - 1) // block_size
    #         block_table = []
            
    #         for j in range(max_num_blocks_per_seq):
    #             if j < num_blocks_needed:
    #                 block_table.append(i * max_num_blocks_per_seq + j)
    #             else:
    #                 block_table.append(0)  # Padding
            
    #         block_tables_lst.append(block_table)
        
    #     block_tables = torch.tensor(block_tables_lst, dtype=torch.int32, device=query.device)
    
    # Prepare query and output tensors
    query_reshaped = query_reshaped.contiguous()
    attn_output = torch.empty_like(query_reshaped, device=query.device)
    
    # Ensure proper scaling
    scale = scaling
    if scale is None:
        scale = torch.tensor(1.0 / (head_size ** 0.5), device=query.device)
    elif not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=query.device)
    
    torch.mps.synchronize()
    paged_attention_kernel.paged_attention_v1(
        attn_output,
        query_reshaped,
        key_cache,      # Now using proper cache format
        value_cache,    # Now using proper cache format
        num_kv_heads=num_kv_heads,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
        max_seq_len=max_seq_len,
        kv_cache_dtype=kwargs.get("kv_cache_dtype", "auto"),
        scale=scale,
        k_scale=kwargs.get("k_scale", torch.tensor(1.0, device=query.device)),
        v_scale=kwargs.get("v_scale", torch.tensor(1.0, device=query.device)),
        alibi_slopes=kwargs.get("alibi_slopes", None),
    )    
    
    # Reshape output back to original format
    if seq_len == 1:
        attn_output = attn_output.unsqueeze(1)  # Add seq_len dimension back
    else:
        attn_output = attn_output.reshape(query.shape[0], seq_len, num_heads, head_size)
    
    attn_output = attn_output.contiguous()
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
