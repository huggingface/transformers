# coding=utf-8
# Copyright 2026 Cohere Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..cohere2.modeling_cohere2 import (
    Cohere2Attention,
    Cohere2DecoderLayer,
    Cohere2ForCausalLM,
    Cohere2LayerNorm,
    Cohere2MLP,
    Cohere2Model,
    Cohere2PreTrainedModel,
    Cohere2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llama.modeling_llama import LlamaRMSNorm
from .configuration_cohere2_moe import Cohere2MoeConfig


class Cohere2MoeMLP(Cohere2MLP):
    def __init__(self, config: Cohere2MoeConfig, intermediate_size=None):
        super().__init__(config)
        if intermediate_size is not None:
            self.intermediate_size = intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class Cohere2MoeRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.expert_selection_fn = config.expert_selection_fn
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight).float()
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)

        if self.expert_selection_fn == "softmax":
            routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float)
        elif self.expert_selection_fn == "sigmoid":
            routing_weights = torch.sigmoid(routing_weights)
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError("Expert selection function can only be either softmax or sigmoid.")

        return router_logits, routing_weights, selected_experts


class Cohere2MoeExperts(nn.ModuleList):
    """Collection of experts stored as a ModuleList to preserve per-expert weight names
    (experts.{i}.gate_proj / up_proj / down_proj) matching the checkpoint layout."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=len(self)).permute(2, 1, 0)

        for expert_idx, expert in enumerate(self):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_hidden_states = expert(hidden_states[token_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Cohere2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_shared_experts = config.num_shared_experts
        self.shared_expert_combination_strategy = config.shared_expert_combination_strategy

        self.gate = Cohere2MoeRouter(config)
        self.experts = Cohere2MoeExperts([Cohere2MoeMLP(config) for _ in range(config.num_experts)])

        if self.num_shared_experts > 0:
            self.shared_experts = Cohere2MoeMLP(
                config,
                intermediate_size=config.intermediate_size * config.num_shared_experts,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        _, routing_weights, selected_experts = self.gate(hidden_states_flat)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.experts(hidden_states_flat, selected_experts, routing_weights)

        if self.num_shared_experts > 0:
            shared_expert_output = self.shared_experts(hidden_states_flat)
            if self.shared_expert_combination_strategy == "sum":
                final_hidden_states = final_hidden_states + shared_expert_output
            elif self.shared_expert_combination_strategy == "average":
                final_hidden_states = (final_hidden_states + shared_expert_output) / 2

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Cohere2MoeRMSNorm(LlamaRMSNorm):
    pass


class Cohere2MoeLayerNorm(Cohere2LayerNorm):
    pass


class Cohere2MoeAttention(Cohere2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Cohere2MoeConfig, layer_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        prefix_dense_sliding_window_pattern = getattr(config, "prefix_dense_sliding_window_pattern", 1)
        self.force_rope = (
            first_k_dense_replace
            and prefix_dense_sliding_window_pattern == 1
            and self.layer_idx < first_k_dense_replace
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.sliding_window is not None or self.force_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Cohere2MoeDecoderLayer(Cohere2DecoderLayer):
    def __init__(self, config: Cohere2MoeConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)

        self.hidden_size = config.hidden_size
        self.self_attn = Cohere2MoeAttention(config=config, layer_idx=layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        rms_norm_eps = getattr(config, "rms_norm_eps", None)
        self.input_layernorm = (
            Cohere2MoeRMSNorm(hidden_size=config.hidden_size, eps=rms_norm_eps)
            if rms_norm_eps is not None
            else Cohere2MoeLayerNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        )

        if layer_idx < config.first_k_dense_replace:
            self.mlp = Cohere2MoeMLP(config, config.prefix_dense_intermediate_size)
        else:
            self.mlp = Cohere2MoeSparseMoeBlock(config)


@auto_docstring
class Cohere2MoePreTrainedModel(Cohere2PreTrainedModel):
    config_class = Cohere2MoeConfig
    _no_split_modules = ["Cohere2MoeDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": Cohere2MoeDecoderLayer,
        "attentions": Cohere2MoeAttention,
    }


class Cohere2MoeRotaryEmbedding(Cohere2RotaryEmbedding):
    pass


@auto_docstring
class Cohere2MoeModel(Cohere2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Cohere2MoeDecoderLayer`]
    Args:
        config: Cohere2MoeConfig
    """

    def __init__(self, config: Cohere2MoeConfig):
        Cohere2MoePreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Cohere2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        rms_norm_eps = getattr(config, "rms_norm_eps", None)
        self.norm = (
            Cohere2MoeRMSNorm(hidden_size=config.hidden_size, eps=rms_norm_eps)
            if rms_norm_eps is not None
            else Cohere2MoeLayerNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        )
        self.rotary_emb = Cohere2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Cohere2MoeForCausalLM(Cohere2ForCausalLM):
    _tp_plan = {"lm_head": "colwise_rep"}


__all__ = ["Cohere2MoeForCausalLM", "Cohere2MoeModel", "Cohere2MoePreTrainedModel"]
