import torch
import torch.utils.checkpoint

from ...cache_utils import Cache, SlidingWindowCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ..llama.modeling_llama import (
    LlamaForMultipleChoice,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,apply_rotary_pos_emb, eager_attention_forward
)
from .configuration_olmo import OlmoConfig
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
class OlmoLayerNorm(nn.Module):
    """LayerNorm but with no learnable weight or bias."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        return F.layer_norm(hidden_states.to(dtype=torch.float32), self.normalized_shape, None, None, eps=1e-5).to(
            orig_dtype
        )


class OlmoRMSNorm(LlamaRMSNorm):
    pass


class OlmoMLP(LlamaMLP):
    pass

class OlmoAttention(LlamaAttention):
       
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OlmoDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OlmoConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = OlmoAttention(config=config, layer_idx=layer_idx)
        self.mlp = OlmoMLP(config)
        self.input_layernorm = OlmoLayerNorm(config.hidden_size)
        self.post_attention_layernorm = OlmoLayerNorm(config.hidden_size)


class OlmoForTokenClassification(LlamaForTokenClassification):
    pass


class OlmoForSequenceClassification(LlamaForSequenceClassification):
    pass


class OlmoForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class OlmoForMultipleChoice(LlamaForMultipleChoice):
    pass
