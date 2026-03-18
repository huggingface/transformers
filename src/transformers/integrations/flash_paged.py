import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..modeling_flash_attention_utils import lazy_import_paged_flash_attention


"""
This file contains the paged attention forward function used in continuous batching. It is essentially a wrapper
around two versions of the flash attention kernel:

1. `flash_attn_varlen_func` for variable length batches, also called the "varlen" path.

1.a Cache behavior:
    This version of flash attention has no mechanism to interact with the paged cache, so we manually read and write
    from `PagedAttentionCache` using the `update` method. This can become a significant bottlenect when sequence length
    grows large, so this path is recommended for batches with a large number of request in prefill.

1.b Indexing mechanism:
    This version of flash attention uses max sequence length (max_seqlen_q or _kv) and cumulative sequence lengths
    (cu_seq_lens_q or _k) to compute the attention for each sequence.

1.c Example
    For a batch of 3 sequences, with query length [10, 3, 1] and key length [0, 1, 7], you get:
        cu_seq_lens_q = [0, 10, 13, 14]
        cu_seq_lens_k = [0, 0, 1, 8]
        max_seqlen_q = 10
        max_seqlen_k = 7

    And inputs shapes:
        Q:      [1, 10+3+1, num_heads, head_dim] = [1, 14, num_heads, head_dim]
        K or V: [1, 0+1+7, num_kv_heads, head_dim] = [1, 8, num_kv_heads, head_dim]

    The kernel is capable of assigning each query and key / value token to a sequence using the cumulative sequence
    lengths:

        Q request index: [r0, r0, r0, r0, r0, r0, r0, r0, r0, r0, r1, r1, r1, r2]
        cu_seq_lens_q:  0_____________________________________10__________13__14

        K request index: [r1, r2, r2, r2, r2, r2, r2, r2]
        cu_seq_lens_k:  0__1___________________________8

2. `flash_attn_with_kvcache` for decode-only batches, also called the "decode" path.

2.a Cache behavior:
    This version of flash attention has a mechanism to interact with the paged cache, so we can use the `block_table`
    to index into the cache and directly update it in-place. This is more efficient than calling `update` as in the
    varlen path, but assumes that each sequence in the batch only has one token. Hence this is not viable for batches
    where some requests are still prefilling.
    The block table is a tensor of shape (batch_size, max_blocks_per_seq), where batch_size is the number of sequences
    in the batch, and max_blocks_per_seq is the maximum number of blocks per sequence. For each request, the block table
    is a vector containing the physical location of the request's cache in the KV cache tensor.

2.b Indexing mechanism:
    This version of flash attention uses `cache_seqlens` to retrieve the length of the cache for each sequence. It has
    no mechanism to retrieve which query token belongs to which sequence, because it assumes each query token belongs to
    a different sequence.

2.c Example:
    Consider a batch of 3 sequences, with query lengths [1, 1, 1] and key lengths [30, 32, 70]. As stated in 2.a, each
    sequence has only one query token, which is needed to use this kernel. We also set the cache block size to 32: this
    is the number of tokens in a KV cache block. Also, we set the maximum number of blocks allocated per sequence to 4.

    The cache seqlens is easy to compute: for each sequence, it is the number of key tokens in the sequence. So:
        cache_seqlens = [30, 32, 70]

    Since there are 3 sequences and 4 block max per sequence, the shape of the block table is (3, 4). Using random
    adresses for this example, the block table looks like this:
        block table: [[2, -1, -1, -1],
                      [0,  1, -1, -1],
                      [3,  5,  6, -1]]
    -1 means that there is no block allocated for this index.

    Sequence 0, with 30 cached tokens, has its cache located in KV_cache[2]. There is enough space to write the
    new KV cache token: 30 + 1 = 31 < 32 (block size).
    Sequence 1, with 32 cached tokens, has its cache located in KV_cache[0] and KV_cache[1]. Since the
    new KV cache token produced will not fit in the first block (KV_cache[0]), there needs to be a second block
    (KV_cache[1]) already allocated before the kernel is called.
    Sequence 2, with 70 cached tokens, has its cache located in KV_cache[3], KV_cache[5], and KV_cache[6]. Note that the
    blocks are not necessarily contiguous, which is no problem for the kernel, and the real advantage of paged cache.
    Since the new KV cache token fits in the third block (KV_cache[6]), there is no need to allocated fourth block.
"""


@torch.compiler.disable
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
    # Retrieve the flash attention functions
    flash_attn_varlen_func, flash_attn_with_kvcache = lazy_import_paged_flash_attention(
        module.config._attn_implementation
    )

    # Retrieve the cumulative sequence lengths for the current layer
    sliding_window = (-1, -1) if not getattr(module, "sliding_window", False) else (module.sliding_window - 1, 0)
    layer_type = "full_attention" if sliding_window == (-1, -1) else "sliding_attention"
    if isinstance(cu_seq_lens_k, dict):
        cu_seq_lens_k = cu_seq_lens_k[layer_type]
        max_seqlen_k = max_seqlen_k[layer_type]

    # If no block table is provided, use flash_attn_varlen_func with read/write indices
    if block_table is None:
        # .update changes the shape of k and v from [1, num_kv_heads, seqlen_kv, head_dim] to [-1, num_kv_heads, head_dim]
        k, v = cache.update(
            key_states=k,
            value_states=v,
            layer_idx=module.layer_idx,
            read_index=kwargs["read_index"],
            write_index=kwargs["write_index"],
        )
        custom_kwargs = {"s_aux": kwargs.get("s_aux")} if "s_aux" in kwargs else {}
        attn_output = flash_attn_varlen_func(
            q.transpose(1, 2).squeeze(0).contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seq_lens_q.to(torch.int32),
            cu_seq_lens_k.to(torch.int32).clone(),
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=module.scaling,
            causal=True,  # kind of a must, it automatically aligns the mask for q < k
            window_size=sliding_window,  # -1 means infinite context window
            **custom_kwargs,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

    # Otherwise, use flash_attn_with_kvcache which updates the cache in-place and computes attention
    else:
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
        cache_seqlens = (cu_seq_lens_k[1 : batch_size + 1] - cu_seq_lens_k[:batch_size] - 1).to(torch.int32)
        # The arg name for the block table is not the same in VLLM's kernel and Tri Dao's kernel, so we need to parse it
        flash_kwargs = {cache.get_block_table_key(flash_attn_with_kvcache): block_table[group_idx]}
        if "s_aux" in kwargs:
            flash_kwargs["s_aux"] = kwargs["s_aux"]  # this is only available in VLLM's FA3
        # Call flash_attn_with_kvcache - this updates cache in-place and computes attention
        attn_output = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            softmax_scale=module.scaling,
            causal=True,
            window_size=sliding_window,
            **flash_kwargs,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        # Reshape output from [batch_size, 1, num_heads, head_dim] to [batch_size, num_heads, head_dim]
        attn_output = attn_output.squeeze(1)
    return attn_output, None
