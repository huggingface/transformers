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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring
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
from ..mixtral.modeling_mixtral import MixtralExperts
from .configuration_cohere2_moe import Cohere2MoeConfig


class Cohere2MoeMLP(Cohere2MLP):
    def __init__(self, config: Cohere2MoeConfig, intermediate_size=None):
        super().__init__(config)
        if intermediate_size is not None:
            self.intermediate_size = intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class Cohere2MoeExperts(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.intermediate_dim = config.intermediate_size


class Cohere2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.expert_selection_fn = config.expert_selection_fn
        self.norm_topk_prob = config.norm_topk_prob
        self.num_shared_experts = config.num_shared_experts
        self.shared_expert_combination_strategy = config.shared_expert_combination_strategy

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Cohere2MoeExperts(config)

        if self.num_shared_experts > 0:
            self.shared_experts = Cohere2MoeMLP(
                config,
                intermediate_size=config.intermediate_size * config.num_shared_experts,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.expert_selection_fn == "softmax":
            routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float)
        elif self.expert_selection_fn == "sigmoid":
            routing_weights = F.sigmoid(routing_weights)
            if self.norm_topk_prob:
                routing_weights = routing_weights / torch.sum(routing_weights, dim=-1, keepdims=True)
        else:
            raise NotImplementedError("Expert selection function can only be either Softmax or Sigmoid.")

        # we cast back to the input dtype
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

    def __init__(self, config: Cohere2MoeConfig, layer_idx: int | None = None):
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
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
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
        super().__init__()
        self.input_layernorm = (
            Cohere2MoeRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            if config.rms_norm_eps is not None
            else Cohere2MoeLayerNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        )
        self.mlp = (
            Cohere2MoeMLP(config, config.prefix_dense_intermediate_size)
            if layer_idx < config.first_k_dense_replace
            else Cohere2MoeSparseMoeBlock(config)
        )


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
        super().__init__()
        self.norm = (
            Cohere2MoeRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            if config.rms_norm_eps is not None
            else Cohere2MoeLayerNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        )


@auto_docstring
class Cohere2MoeForCausalLM(Cohere2ForCausalLM):
    _tp_plan = {"lm_head": "colwise_rep"}


__all__ = ["Cohere2MoeForCausalLM", "Cohere2MoeModel", "Cohere2MoePreTrainedModel"]
