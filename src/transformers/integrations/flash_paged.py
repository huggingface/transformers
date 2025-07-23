import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..utils import is_flash_attn_2_available


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func  # noqa: F401


def paged_attention_forward(
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
    implementation=None,
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

    if implementation is not None:
        flash_attn_varlen_func = implementation.flash_attn_varlen_func
    attn_output = flash_attn_varlen_func(
        q.transpose(1, 2).squeeze(0).contiguous(),
        k.transpose(1, 2).squeeze(0).contiguous(),
        v.transpose(1, 2).squeeze(0).contiguous(),
        cumulative_seqlens_q.to(torch.int32),
        cumulative_seqlens_k.to(torch.int32).clone(),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=module.scaling,
        causal=True,  # kind of a must, it automatically aligns the mask for q < k
        window_size=(-1, -1),  # -1 means infinite context window
        # block_table=block_tables, -> torch.Tensor
        # **kwargs,
    )

    return attn_output, None
