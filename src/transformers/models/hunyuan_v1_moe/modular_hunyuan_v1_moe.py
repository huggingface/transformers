# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HunYuanMoEV1 model."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1RotaryEmbedding
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mixtral.modeling_mixtral import MixtralExperts
from .configuration_hunyuan_v1_moe import HunYuanMoEV1Config


logger = logging.get_logger(__name__)


class HunYuanMoEV1RMSNorm(LlamaRMSNorm):
    pass


class HunYuanMoEV1MLP(LlamaMLP):
    def __init__(self, config: HunYuanMoEV1Config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class HunYuanMoEV1Attention(LlamaAttention):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.query_layernorm = HunYuanMoEV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanMoEV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HunYuanMoEV1Gate(nn.Module):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        self.wg = nn.Linear(config.hidden_size, num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        return logits


class HunYuanMoEV1Experts(MixtralExperts):
    pass


class HunYuanMoEV1Moe(nn.Module):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        self.top_k = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[layer_idx]
        self.gate = HunYuanMoEV1Gate(config, layer_idx=layer_idx)
        self.experts = HunYuanMoEV1Experts(config)
        self.shared_mlp = HunYuanMoEV1MLP(config)

    def route_tokens_to_experts(self, hidden_states):
        routing_weights = F.softmax(hidden_states, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return selected_experts, routing_weights.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_mlp = self.shared_mlp(hidden_states)
        router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_dim)
        selected_experts, routing_weights = self.route_tokens_to_experts(router_logits)
        final_hidden_states = self.experts(hidden_states, selected_experts, routing_weights).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states + hidden_states_mlp


class HunYuanMoEV1DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanMoEV1Attention(config=config, layer_idx=layer_idx)
        self.mlp = HunYuanMoEV1Moe(config, layer_idx=layer_idx)
        self.input_layernorm = HunYuanMoEV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanMoEV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx


class HunYuanMoEV1PreTrainedModel(LlamaPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, HunYuanMoEV1Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        # DynamicNTKAlphaRotary - unique to this model
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            if module.rope_type == "dynamic" and module.config.rope_parameters.get("alpha"):
                dim = module.config.head_dim
                rope_theta = module.config.rope_parameters["rope_theta"]
                alpha = module.config.rope_parameters["alpha"]

                base = rope_theta * alpha ** (dim / (dim - 2))
                buffer_value = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            else:
                rope_fn = (
                    ROPE_INIT_FUNCTIONS[module.rope_type]
                    if module.rope_type != "default"
                    else module.compute_default_rope_parameters
                )
                buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)


class HunYuanMoEV1RotaryEmbedding(HunYuanDenseV1RotaryEmbedding):
    pass


class HunYuanMoEV1Model(LlamaModel):
    pass


class HunYuanMoEV1ForCausalLM(LlamaForCausalLM):
    pass


class HunYuanMoEV1ForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "HunYuanMoEV1ForCausalLM",
    "HunYuanMoEV1Model",
    "HunYuanMoEV1PreTrainedModel",
    "HunYuanMoEV1ForSequenceClassification",
]
