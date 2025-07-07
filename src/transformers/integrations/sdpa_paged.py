from typing import Optional

import torch

from kernels import get_kernel

paged_attention_kernel = get_kernel("kernels-community/paged-attention")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_kv_heads * n_rep, slen, head_dim)


def sdpa_attention_paged_forward__(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    reshaping_function=None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    cache = kwargs.pop("cache", None)
    if cache is not None:
        key, value = cache.update(key, value, module.layer_idx, reshaping_function=reshaping_function, **kwargs)

    # because of the kernel, the shape of the cache is different
    # it return [num_tokens, num_kv_heads, head_dim]
    # print(f"key.shape: {key.shape}")
    # if key.ndim == 3:
    key = key.permute(1, 0, 2)
    value = value.permute(1, 0, 2)
    print(f"key.shape: {key.shape}")
    print(f"value.shape: {value.shape}")
    # else:
    #     key = key.view(-1, key.shape[-2], key.shape[-1]).permute(1, 0, 2)
    #     value = value.view(-1, value.shape[-2], value.shape[-1]).permute(1, 0, 2)
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    causal_mask = attention_mask
    # print(f"causal_mask.shape: {causal_mask.shape}")
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        scale=scaling,
        dropout_p=dropout,
        is_causal=is_causal,
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
    reshaping_function = paged_attention_kernel.reshape_and_cache_flash
    max_seqlen_q = kwargs.get("max_seqlen_q")
    is_decoding = max_seqlen_q is not None and (max_seqlen_q == 1)
    is_decoding = False
    if not is_decoding:
        return sdpa_attention_paged_forward__(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            reshaping_function=reshaping_function,
            is_causal=False,
            **kwargs,
        )
    else:
        print("#########################")
        num_kv_heads = key.shape[1]
        cache = kwargs.pop("cache", None)
        key, value = cache.update(key, value, module.layer_idx, reshaping_function=reshaping_function, **kwargs)
        print(f"query shape: {query.shape}")
        print(f"key shape: {key.shape}")
        print(f"value shape: {value.shape}")
        batch_size, num_heads, seq_len, head_size = query.shape
        query = query.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_size).contiguous()

        # if not hasattr(module, "_attn_output"):
        _attn_output = torch.empty_like(query, device=query.device)

        x = 16 // key.element_size()
        key = key.view(cache.num_blocks, cache.block_size, num_kv_heads, head_size // x, x).permute(0, 2, 3, 1, 4).contiguous()
        value = value.permute(0, 2, 3, 1).contiguous()
        seq_lens = kwargs.get("cumulative_seqlens_k")[1:]  # .flatten()
        block_tables = kwargs.get("block_tables")
        # print(f"block_tables: {block_tables.shape}")
        # print(f"seq_lens: {seq_lens}")
        # Write block_tables to file and print all elements
        with open("block_tables_log.txt", "a") as f:
            if block_tables is not None:
                f.write("######################## All elements:\n")
                for i, table in enumerate(block_tables):
                    f.write(f"  row {i}: {table.tolist()}\n")

        block_size = kwargs.get("block_size", 32)
        
        # Pre-create scale tensors to avoid CUDA graph capture issues
        if not hasattr(module, "_k_scale_tensor"):
            module._k_scale_tensor = torch.tensor(1.0, device=key.device, dtype=key.dtype)
        if not hasattr(module, "_v_scale_tensor"):
            module._v_scale_tensor = torch.tensor(1.0, device=value.device, dtype=value.dtype)

        torch.cuda.synchronize()
        paged_attention_kernel.paged_attention_v1(
            _attn_output,
            query,
            key,  # → [num_blocks, num_kv_heads, head_dim // x, block_size, x], x depends on the dtype
            value,  # # → [num_blocks, num_kv_heads, head_dim, block_size]
            num_kv_heads=num_kv_heads,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
            max_seq_len=20,
            kv_cache_dtype=kwargs.get("kv_cache_dtype", "auto"),
            scale=scaling,
            k_scale=module._k_scale_tensor,
            v_scale=module._v_scale_tensor,
            alibi_slopes=None,
        )
        
        torch.cuda.synchronize()
    
        attn_output = _attn_output.view(batch_size, seq_len, num_heads, head_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
