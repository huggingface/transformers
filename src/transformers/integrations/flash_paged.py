import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..utils import is_flash_attn_2_available

from kernels import get_kernel

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func


def paged_attention_forward_(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor = None,
    cache: PagedAttentionCache = None,
    cumulative_seqlens_q=None,
    cumulative_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    block_tables=None,
    **kwargs,
) -> torch.Tensor:
    r"""Perform the forward pass of attention with paged key-value cache.

    This function handles the cache updates and performs the attention computation
    using the flash_attn_varlen_func for efficient processing.

    Args:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full k
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full v
        cumulative_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cumulative_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
    """
    k, v = cache.update(k, v, module.layer_idx, cumulative_seqlens_k=cumulative_seqlens_k, **kwargs)

    attn_output = flash_attn_varlen_func(
        q.transpose(1, 2).squeeze(0),
        k.transpose(1, 2).squeeze(0),
        v.transpose(1, 2).squeeze(0),
        cumulative_seqlens_q.to(torch.int32),
        cumulative_seqlens_k.to(torch.int32),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=module.scaling,
        causal=True,  # kind of a must, it automatically aligns the mask for q < k
        window_size=(-1, -1),  # -1 means infinite context window
        # block_table=block_tables, -> torch.Tensor
        # **kwargs,
    )

    return attn_output, None


paged_attention_kernel = get_kernel("kernels-community/paged-attention")


def paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor = None,
    cache: PagedAttentionCache = None,
    cumulative_seqlens_q=None,
    cumulative_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    block_tables=None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Wrapper for paged attention forward that uses flash attention."""
    reshaping_function = paged_attention_kernel.reshape_and_cache_flash
    is_decoding = kwargs.get("max_seqlen_q", -1) == 1
    if not is_decoding:
        return paged_attention_forward_(
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
        key, value = cache.update(key, value, module.layer_idx, reshaping_function=reshaping_function, **kwargs)

        batch_size, num_heads, seq_len, head_size = query.shape
        query = query.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_size)

        if not hasattr(module, "_attn_out"):
            module._attn_output = torch.empty_like(query, device=query.device)

        x = 16 // key.element_size()
        key = key.view(cache.num_blocks, cache.block_size, num_kv_heads, head_size // x, x).permute(0, 2, 3, 1, 4)
        value = value.permute(0, 2, 3, 1).contiguous()
        seq_lens = kwargs.get("cumulative_seqlens_k")  # .flatten()
        block_tables = kwargs.get("block_tables")
        block_size = kwargs.get("block_size", 32)
        torch.mps.synchronize()
        paged_attention_kernel.paged_attention_v1(
            module._attn_output,
            query,
            key,  # → [num_blocks, num_kv_heads, head_dim // x, block_size, x], x depends on the dtype
            value,  # # → [num_blocks, num_kv_heads, head_dim, block_size]
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
