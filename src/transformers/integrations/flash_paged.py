import warnings

import torch

from ..generation.continuous_batching import PagedAttentionCache
from .flash_attention import flash_attention_forward


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache: PagedAttentionCache,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor | dict[str, torch.Tensor],
    max_length_q: int,
    max_length_k: int | dict[str, int],
    block_table: torch.Tensor | None,
    sliding_window: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Deprecated function that used to be the way to compute attention with flash attention + paged cache. Please
    use flash_attention_forward instead."""
    warnings.warn(
        "paged_attention_forward is deprecated and will be removed in v5.23. Use flash_attention_forward instead.",
        FutureWarning,
        stacklevel=2,
    )
    return flash_attention_forward(
        module=module,
        query=q,
        key=k,
        value=v,
        attention_mask=attention_mask,
        sliding_window=sliding_window,
        cache=cache,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_length_q=max_length_q,
        max_length_k=max_length_k,
        block_table=block_table,
        **kwargs,
    )
