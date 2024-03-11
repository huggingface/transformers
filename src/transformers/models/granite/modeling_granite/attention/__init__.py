from ...configuration_granite import GraniteConfig
from ..enums import AttentionHeadType, AttentionImplementation, PositionEmbeddingType
from .base import Attention
from .flash import FlashAttention
from .math import MathAttention
from .padding_free import PaddingFreeAttention
from .sdpa import SDPA


_ATTENTION_MODULES = {
    AttentionImplementation.eager.value: MathAttention,
    AttentionImplementation.math.value: MathAttention,
    AttentionImplementation.sdpa.value: SDPA,
    AttentionImplementation.flash.value: FlashAttention,
    AttentionImplementation.padding_free.value: PaddingFreeAttention,
}


def get_attention_module(
    config: GraniteConfig,
    causal: bool,
    layer_idx: int,
    attention_implementation: AttentionImplementation = AttentionImplementation.sdpa,
) -> Attention:
    attention_implementaton = AttentionImplementation(attention_implementation)

    if attention_implementaton.value in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementaton.value](
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_head_type=AttentionHeadType(config.attention_head_type),
            position_embedding_type=PositionEmbeddingType(config.position_embedding_type),
            causal=causal,
            add_bias=config.add_bias,
            scale_attention_weights=config.scale_attn_weights,
            attention_softmax_in_fp32=config.attention_softmax_in_fp32,
            scale_attention_softmax_in_fp32=config.scale_attention_softmax_in_fp32,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            layer_idx=layer_idx,
        )

    raise ValueError(f"unexpected `attention_implementation` {attention_implementaton}")
