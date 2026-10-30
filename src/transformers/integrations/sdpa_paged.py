import warnings

import torch

from .sdpa_attention import repeat_kv as repeat_kv_
from .sdpa_attention import sdpa_attention_forward


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Deprecated helper for the sdpa_attention_paged_forward."""
    warnings.warn(
        "repeat_kv is deprecated and will be removed in v5.23. Use repeat_kv from the sdpa_attention module instead.",
        FutureWarning,
        stacklevel=2,
    )
    return repeat_kv_(hidden_states, n_rep)


def sdpa_attention_paged_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Deprecated function that used to be the way to compute attention with SDPA + paged cache. Please use
    sdpa_attention_forward instead."""
    warnings.warn(
        "sdpa_attention_paged_forward is deprecated and will be removed in v5.23. Use sdpa_attention_forward instead.",
        FutureWarning,
        stacklevel=2,
    )
    return sdpa_attention_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )
