import torch

from ..modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask
from ..utils import logging
from .mla import mla


logger = logging.get_logger(__name__)

_use_top_left_mask = flash_attn_supports_top_left_mask()


def get_target_dtype(query: torch.Tensor, module: torch.nn.Module) -> torch.dtype:
    """If the query is in float32, return a target dtype compatible with flash attention. Return None otherwise."""
    if query.dtype == torch.float32:
        device_type = query.device.type
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_is_quantized"):
            return module.config.dtype
        else:
            return next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
    return None


@mla
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
    s_aux: torch.Tensor | None = None,  # alias: learnable attention sin
    qv_latents: torch.Tensor | None = None,  # MLA/DSA latents
    indices: torch.IntTensor | None = None,  # DSA topk indices
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "Flash Attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = get_target_dtype(query, module)

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # FlashAttention requires the query and value to share a head dim; pad `value` up to the
    # query head dim and crop the output below.
    # NOTE: The only exception is MLA (`qv_latents`) where different head dims are expected
    head_dim, v_head_dim = query.shape[-1], value.shape[-1]
    if qv_latents is None and v_head_dim != head_dim:
        value = torch.nn.functional.pad(value, [0, head_dim - v_head_dim])

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        attn_implementation=module.config._attn_implementation,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        s_aux=(
            s_aux.to(query.dtype)  # FA only accepts half precision
            if s_aux is not None
            else None
        ),
        qv_latents=qv_latents,
        indices=indices,
        **kwargs,
    )

    if qv_latents is None and v_head_dim != head_dim:
        attn_output = attn_output[..., :v_head_dim]

    return attn_output, None
