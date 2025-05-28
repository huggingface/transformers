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
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn

from ...integrations.flex_attention import flex_attention_forward
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from ..llama4.modeling_llama4 import Llama4TextExperts
from .configuration_openai import OpenaiConfig


logger = logging.get_logger(__name__)


class OpenaiRMSNorm(LlamaRMSNorm):
    pass


class OpenaiExperts(Llama4TextExperts):
    pass


class OpenaiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_local_experts = config.num_local_experts
        self.experts = OpenaiExperts(config)
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = (
            torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)
        routed_in = hidden_states.repeat(self.num_local_experts, 1)
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        out.add_(routed_out.reshape(self.num_local_experts, -1, self.hidden_dim).sum(dim=0))
        return out, router_scores


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

    attn_weights = torch.cat([attn_weights, module.sink], dim=-1)
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
    sink = module.sink

    def attention_sink(score, b, h, q_idx, kv_idx):
        score = torch.cat([score, sink], dim=-1)
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
        self.sinks = torch.empty(config.num_attention_heads)

class OpenaiDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OpenaiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = OpenaiAttention(config=config, layer_idx=layer_idx)
        self.mlp = OpenaiMLP(config)
        self.input_layernorm = OpenaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OpenaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class OpenaiPreTrainedModel(LlamaPreTrainedModel):
    config_class = OpenaiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OpenaiDecoderLayer"]


class OpenaiModel(LlamaModel):
    def __init__(self, config: OpenaiConfig):
        super().__init__(config)
        self.rope = OpenaiRotaryEmbedding(config)
        self.layers = nn.ModuleList([OpenaiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = OpenaiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class OpenaiForCausalLM(LlamaForCausalLM):
    def __init__(self, config: OpenaiConfig):
        super().__init__(config)
        self.model = OpenaiModel(config)


__all__ = [
    "OpenaiForCausalLM",
    "OpenaiModel",
    "OpenaiPreTrainedModel",
]
