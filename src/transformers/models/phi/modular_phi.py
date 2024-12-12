from .configuration_phi import PhiConfig
from ..llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb, LlamaMLP, LlamaForCausalLM, LlamaForSequenceClassification, LlamaForTokenClassification, LlamaForQuestionAnswering
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
import torch.nn as nn

import math
import torch
from typing import Optional, Tuple, Callable
from transformers.cache_utils import Cache


def eager_attention_forward(attention_class: nn.Module, query, key, value, attention_mask=None, layer_head_mask=None, **_kwargs):
    key_states = repeat_kv(key_states, attention_class.num_key_value_groups)
    value_states = repeat_kv(value_states, attention_class.num_key_value_groups)

    # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
    attn_weights = torch.matmul(
            query.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
    ) / math.sqrt(attention_class.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights += causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=attention_class.attention_dropout, training=attention_class.training)

    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, attn_weights


class PhiAttention(LlamaAttention):

    def __init__(self, config: PhiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.rotary_ndims = int(self.head_dim * config.partial_rotary_factor)
        self.dense = self.o_proj
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
    
    def forward(self, 
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        cos, sin = position_embeddings
        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_ndims],
            query_states[..., self.rotary_ndims :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_ndims],
            key_states[..., self.rotary_ndims :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

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

class PhiMLP(LlamaMLP):
    pass
class PhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig, layer_idx: int):
        super().__init__()
        self.self_attn = PhiAttention(config, layer_idx=layer_idx)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class PhiForCausalLM(LlamaForCausalLM):
    pass

class PhiForSequenceClassification(LlamaForSequenceClassification):
    pass

class PhiForTokenClassification(LlamaForTokenClassification):
    pass

class PhiForQuestionAnswering(LlamaForQuestionAnswering):
    pass


