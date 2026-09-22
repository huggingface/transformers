import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..modeling_flash_attention_utils import (
    _flash_attention_forward,
    cast_to_flash_compatible_dtype,
    flash_attn_supports_top_left_mask,
)
from ..utils import logging


logger = logging.get_logger(__name__)

_use_top_left_mask = flash_attn_supports_top_left_mask()


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    s_aux: torch.Tensor | None = None,  # alias: learnable attention sink
    cache: PagedAttentionCache | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    q_head_dim = query.shape[-1]
    v_head_dim = value.shape[-1]

    # Check for incompatible kwargs
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "Flash Attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # _flash_attention_forward needs non-transposed inputs, with shape [batch_size, seq_len, num_heads, head_dim]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))

    # If there is a paged cache, now is the time to prepare the kwargs for the flash attention forward pass
    is_paged_cache = isinstance(cache, PagedAttentionCache)
    if is_paged_cache:
        query, key, value = (x.contiguous() for x in (query, key, value))
        update_cache = cache.specialize_kwargs(module.layer_idx, key.size(1), kwargs)  # type: ignore
        if update_cache:
            key, value = cache.update(  # type: ignore
                key_states=key,
                value_states=value,
                layer_idx=module.layer_idx,
                read_index=kwargs["read_index"],
                write_index=kwargs["write_index"],
            )

    # FlashAttention requires the query and value have the same head dim; pad `value` up to the query head dim and crop
    # the output below. This happens for example in MLA, where `v_head_dim < qk_head_dim`.
    if v_head_dim != q_head_dim:
        value = torch.nn.functional.pad(value, [0, q_head_dim - v_head_dim])

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query, key, value = cast_to_flash_compatible_dtype(module, query, key, value)
    s_aux = s_aux.to(query.dtype) if s_aux is not None else None

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else module.is_causal

    # Call the flash attention, either compiled or uncompiled
    all_kwargs = dict(
        query_states=query,
        key_states=key,
        value_states=value,
        attention_mask=attention_mask,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        attn_implementation=module.config._attn_implementation,  # type: ignore <- the config is cached on the module
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        s_aux=s_aux,
        **kwargs,
    )
    if is_paged_cache:
        attn_output = _uncompiled_flash_attention_forward(**all_kwargs)
    else:
        attn_output = _flash_attention_forward(**all_kwargs)

    if v_head_dim != q_head_dim:
        attn_output = attn_output[..., :v_head_dim]

    return attn_output, None


@torch.compiler.disable
def _uncompiled_flash_attention_forward(**kwargs) -> torch.Tensor:
    return _flash_attention_forward(**kwargs)
