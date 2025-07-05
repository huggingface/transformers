from typing import Optional

import torch

from kernels import get_kernel

paged_attention_kernel = get_kernel("kernels-community/paged-attention")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)


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
    key = key.permute(1, 0, 2)
    value = value.permute(1, 0, 2)
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
    reshaping_function = paged_attention_kernel.reshape_and_cache_flash
    is_decoding = kwargs.get("max_seqlen_q", -1) == 1
    if not is_decoding:
        return sdpa_attention_paged_forward__(
            module,
            query,
            key,
            value,
            attention_mask,
            reshaping_function=reshaping_function,
            **kwargs,
        )
    else:
        num_kv_heads = key.shape[1]
        cache = kwargs.pop("cache", None)
        if cache is not None:
            key, value = cache.update(
                key, value, module.layer_idx, reshaping_function=reshaping_function, kernel=True, **kwargs
            )
        batch_size, num_heads, seq_len, head_size = query.shape
        query = query.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_size)

        if not hasattr(module, "_attn_out"):
            module._attn_output = torch.empty_like(query, device=query.device)

        seq_lens = kwargs.get("cumulative_seqlens_k")
        block_tables = kwargs.get("block_tables")
        block_size = kwargs.get("block_size", 32)
        torch.mps.synchronize()
        paged_attention_kernel.paged_attention_v1(
            module._attn_output,
            query,
            key,  # key should be in the correct paged format after cache.update
            value,  # value should be in the correct paged format after cache.update
            num_kv_heads=num_kv_heads,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
            max_seq_len=kwargs.get("max_seqlen_k"),
            kv_cache_dtype=kwargs.get("kv_cache_dtype", "auto"),
            scale=module.scaling,
            k_scale=None,
            v_scale=None,
            alibi_slopes=None,
        )

        attn_output = module._attn_output.reshape(batch_size, seq_len, num_heads, head_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
