import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..modeling_flash_attention_utils import lazy_import_paged_flash_attention


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cache: PagedAttentionCache = None,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    **kwargs,
) -> torch.Tensor:
    r"""Perform the forward pass of attention with paged key-value cache.

    This function handles the cache updates and performs the attention computation
    using the flash_attn_varlen_func for efficient processing.

    Args:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full k
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full v
        cu_seq_lens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seq_lens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
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
    flash_attn_varlen_func = lazy_import_paged_flash_attention(module.config._attn_implementation)

    sliding_window = (-1, -1) if not getattr(module, "sliding_window", False) else (module.sliding_window - 1, 0)
    layer_type = "full_attention" if sliding_window == (-1, -1) else "sliding_attention"

    # .update changes the shape of k and v from [1, num_kv_heads, seqlen_kv, head_dim] to [-1, num_kv_heads, head_dim]
    if cache is not None:
        k, v = cache.update(
            key_states=k,
            value_states=v,
            layer_idx=module.layer_idx,
            read_index=kwargs["read_index"],
            write_index=kwargs["write_index"],
        )

    # Retrieve the cumulative sequence lengths for the current layer
    if isinstance(cu_seq_lens_k, dict):
        cu_seq_lens_k = cu_seq_lens_k[layer_type]
        max_seqlen_k = max_seqlen_k[layer_type]

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
    return attn_output, None
