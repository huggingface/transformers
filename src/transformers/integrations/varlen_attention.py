from typing import cast

import torch

from ..utils.import_utils import is_torch_greater_or_equal
from .sdpa_attention import repeat_kv, sdpa_attention_forward


# Torch's varlen attention gained `scale` and `window_size` in 2.11 (the minimum for `varlen_attention`), and GQA in 2.12
_is_torch_greater_or_equal_than_2_12 = is_torch_greater_or_equal("2.12", accept_dev=True)


def varlen_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    sliding_window: int | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Torch's varlen attention (`torch.nn.attention.varlen`) over sequences packed along the length, delimited by their
    cumulative lengths: the 1D mask of packed text rows (`create_causal_mask`), or the `cu_seq_lens_q` (and
    `cu_seq_lens_k`) of packed encoders. Everything else, and what the kernel doesn't support (CPU, fp32, dropout, head
    sizes not a multiple of 8 up to 256, value heads of another size), is SDPA.
    """
    if attention_mask is not None and attention_mask.ndim == 1:
        cu_seq_lens_q = cu_seq_lens_k = attention_mask
        attention_mask = None
    head_dim = query.shape[-1]
    use_varlen = (
        cu_seq_lens_q is not None
        and attention_mask is None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and dropout == 0.0
        and head_dim % 8 == 0
        and head_dim <= 256
        and value.shape[-1] == head_dim
    )
    if not use_varlen:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            sliding_window=sliding_window,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            **kwargs,
        )

    from torch.nn.attention.varlen import varlen_attn

    batch_size, num_heads = query.shape[:2]
    cu_seq_lens_q = cast(torch.Tensor, cu_seq_lens_q)
    cu_seq_lens_k = cu_seq_lens_q if cu_seq_lens_k is None else cu_seq_lens_k
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    if key.shape[1] != num_heads and not _is_torch_greater_or_equal_than_2_12:
        key = repeat_kv(key, num_heads // key.shape[1])
        value = repeat_kv(value, num_heads // value.shape[1])
    # One packed row of all the sequences, `(total_length, heads, head_dim)`
    query, key, value = (states.transpose(1, 2).flatten(0, 1) for states in (query, key, value))
    # The kernel only uses the maximum lengths to size its launch: without precomputed ones, the total lengths are
    # valid bounds that need no host sync
    max_length_q = query.shape[0] if max_length_q is None else max_length_q
    max_length_k = key.shape[0] if max_length_k is None else max_length_k
    # Keys a query sees on its left and right, -1 for all of them
    window_size = (-1 if sliding_window is None else sliding_window - 1, 0 if is_causal else -1)
    # Without `return_aux`, a single tensor
    attn_output = cast(
        torch.Tensor,
        varlen_attn(
            query,
            key,
            value,
            cu_seq_lens_q.to(torch.int32),
            cu_seq_lens_k.to(torch.int32),
            max_length_q,
            max_length_k,
            scale=scaling,
            window_size=window_size,
            **({"enable_gqa": True} if key.shape[1] != num_heads else {}),
        ),
    )
    return attn_output.view(batch_size, -1, num_heads, head_dim), None
