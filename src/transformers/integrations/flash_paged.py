import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..modeling_flash_attention_utils import _flash_attention_forward


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,  # Unused in flash
    cache: PagedAttentionCache,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor | dict[str, torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int | dict[str, int],
    block_table: torch.Tensor | None,
    sliding_window: int | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Performs the forward pass of attention with paged key-value cache. This function handles the cache updates and
    performs the attention computation. For decode-only batches (when block_table is provided), uses
    `flash_attn_with_kvcache` for fused attention + cache update. Otherwise uses `flash_attn_varlen_func`.

    Args:
        q: (1, nheads, total_q, headdim), where total_q = total number of query tokens in the batch.
        k: (1, nheads_k, total_k, headdim), where total_k = total number of key tokens in the batch.
        v: (1, nheads_k, total_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seq_lens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seq_lens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        block_table: (num_groups, batch_size, max_blocks_per_seq), dtype int32. Block table for paged KV cache.
            If provided, uses flash_attn_with_kvcache for fused attention + cache update. For each request, the block
            table is a vector of size (max_blocks_per_seq,) with indices indicating the physical location of the cache
            to read from and write to. The kernel, using the cache_seqlens for that request, knows how much cache to
            read and dispatches the read using the block table. Same for the write. If a request has less  blocksthan
            max_blocks_per_seq blocks, the block table is padded with -1s to indicate that this cache is not allocated.
    """
    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else module.is_causal

    # Retrieve the cumulative sequence lengths for the current layer
    layer_type = "full_attention" if sliding_window is None else "sliding_attention"
    if isinstance(cu_seq_lens_k, dict):
        cu_seq_lens_k = cu_seq_lens_k[layer_type]
        max_seqlen_k = max_seqlen_k[layer_type]

    if block_table is None:
        # If no block table is provided, use flash_attn_varlen_func with read/write indices

        # .update changes the shape of k and v from [1, num_kv_heads, seqlen_kv, head_dim] to [-1, num_kv_heads, head_dim]
        k, v = cache.update(
            key_states=k,
            value_states=v,
            layer_idx=module.layer_idx,
            read_index=kwargs["read_index"],
            write_index=kwargs["write_index"],
        )
        # [1, num_q_heads, seqlen_q, head_dim] to [1, seqlen_q, num_q_heads, head_dim] (we reshape during the base fa forward)
        q = q.transpose(1, 2).contiguous()

        # Prepare relevant kwargs
        flash_kwargs = {
            "cu_seq_lens_q": cu_seq_lens_q.to(torch.int32),
            "cu_seq_lens_k": cu_seq_lens_k.to(torch.int32),
            "max_length_q": max_seqlen_q,
            "max_length_k": max_seqlen_k,
        }
    else:
        # Otherwise, use flash_attn_with_kvcache which updates the cache in-place and computes attention

        # Get layer group index for this layer
        group_idx, layer_idx_in_group = cache.layer_index_to_group_indices[module.layer_idx]
        # KV cache shape: [num_pages, num_kv_heads, head_dim] -> [num_blocks, block_size, num_kv_heads, head_dim]
        k_cache = cache.key_cache[layer_idx_in_group].view(
            -1, cache.block_size, cache.num_key_value_heads, cache.head_dim
        )
        v_cache = cache.value_cache[layer_idx_in_group].view(
            -1, cache.block_size, cache.num_key_value_heads, cache.head_dim
        )
        # Reshape Q, K, V from [1, num_*_heads, batch_size, head_dim] to [batch_size, 1, num_*_heads, head_dim]
        q = q.permute(2, 0, 1, 3).contiguous()
        k = k.permute(2, 0, 1, 3).contiguous()
        v = v.permute(2, 0, 1, 3).contiguous()
        # Compute cache_seqlens from cu_seq_lens_k (current cache length BEFORE adding new tokens)
        # cu_seq_lens_k is cumulative, so seqlens[i] = cu_seq_lens_k[i+1] - cu_seq_lens_k[i] - 1 (subtract 1 for the new token)
        batch_size = k.size(0)
        cache_seq_lens = cu_seq_lens_k[1 : batch_size + 1] - cu_seq_lens_k[:batch_size] - 1

        # Prepare relevant kwargs
        flash_kwargs = {
            "block_table": block_table[group_idx],
            "cache_seq_lens": cache_seq_lens.to(torch.int32),
            "cached_key_states": k_cache,
            "cached_value_states": v_cache,
        }

    # Filter base kwargs just in case
    kwargs = {k: v for k, v in kwargs.items() if k not in flash_kwargs}

    attn_output = _flash_attention_forward(
        q,
        k,
        v,
        attention_mask=None,  # unused as we force varlen
        query_length=None,  # unused as we force varlen
        is_causal=is_causal,
        **flash_kwargs,
        **kwargs,
    )

    if block_table is not None:
        # Reshape output from [batch_size, 1, num_heads, head_dim] to [batch_size, num_heads, head_dim]
        attn_output = attn_output.squeeze(1)

    return attn_output, None
