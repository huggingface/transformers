# coding=utf-8
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Optional, Tuple
from ...cache_utils import Cache
import torch
from torch import nn
from ...activations import ACT2FN
from ...integrations.flex_attention import flex_attention_forward
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel
from .configuration_openai_moe import OpenaiConfig


logger = logging.get_logger(__name__)


class OpenaiRMSNorm(LlamaRMSNorm):
    pass


def swiglu(x, alpha: float = 1.702):
    # Note we add an extra bias of 1 to the linear layer
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)

class OpenaiExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.expert_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[:, None, :]
        swiglu_ = swiglu(gate_up)
        next_states = torch.bmm(swiglu_, self.down_proj) + self.down_proj_bias[:,None,:]
        return next_states


class OpenaiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_local_experts = config.num_local_experts
        self.experts = OpenaiExperts(config)
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)

    def forward(self, hidden_states):
        # we don't slice weight as its not compile compatible
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1, sorted=True)
        router_scores = (
            torch.full_like(router_logits, float(0)).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        )

        routed_in = hidden_states.repeat(self.num_local_experts, 1)
        routed_out = self.experts(routed_in)
        routed_out = routed_out * router_scores.reshape(self.num_local_experts, -1, 1)
        hidden_states = routed_out.sum(dim=0)[None, ...]
        return hidden_states, router_scores


class OpenaiRotaryEmbedding(LlamaRotaryEmbedding):
    pass


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.cat([attn_weights, module.sinks], dim=-1)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def openai_flex_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    sinks = module.sinks

    def attention_sink(score, b, h, q_idx, kv_idx):
        score = torch.cat([score, sinks], dim=-1)
        return score

    return flex_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        dropout=dropout,
        attention_sink=attention_sink,
        score_mod=attention_sink,
        **kwargs,
    )


ALL_ATTENTION_FUNCTIONS.register("openai_flex_attention", openai_flex_attention_forward)


class OpenaiAttention(LlamaAttention):
    def __init__(self, config: OpenaiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.register_buffer("sinks", torch.empty(config.num_attention_heads), persistent=True)

class OpenaiDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OpenaiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = OpenaiAttention(config=config, layer_idx=layer_idx)
        self.mlp = OpenaiMLP(config)
        self.input_layernorm = OpenaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OpenaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if kwargs.get("output_router_logits", False):
            outputs += (router_logits,)
        return outputs
    
class OpenaiPreTrainedModel(LlamaPreTrainedModel):
    config_class = OpenaiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OpenaiDecoderLayer"]


class OpenaiModel(MixtralModel, OpenaiPreTrainedModel):
    _no_split_modules = ["OpenaiDecoderLayer"]

    def __init__(self, config: OpenaiConfig):
        super().__init__(config)
        self.rope = OpenaiRotaryEmbedding(config)
        self.layers = nn.ModuleList([OpenaiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = OpenaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class OpenaiForCausalLM(MixtralForCausalLM, OpenaiPreTrainedModel):
    def __init__(self, config: OpenaiConfig):
        super().__init__(config)
        self.model = OpenaiModel(config)


__all__ = [
    "OpenaiForCausalLM",
    "OpenaiModel",
    "OpenaiPreTrainedModel",
]
