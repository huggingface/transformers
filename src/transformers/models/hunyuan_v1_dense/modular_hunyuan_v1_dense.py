# coding=utf-8
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
"""PyTorch HunYuanDenseV1 model."""

from typing import Callable, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.utils import (
    logging,
)

from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_hunyuan_v1_dense import HunYuanDenseV1Config


logger = logging.get_logger(__name__)


class HunYuanDenseV1RMSNorm(LlamaRMSNorm):
    pass


class HunYuanDenseV1MLP(LlamaMLP):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx=None, is_shared_mlp=False):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class HunYuanDenseV1Attention(LlamaAttention):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.query_layernorm = HunYuanDenseV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanDenseV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
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

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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


class HunYuanDenseV1DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx


class HunYuanDenseV1PreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HunYuanDenseV1RotaryEmbedding(LlamaRotaryEmbedding):
    def get_rope_frequencies(self, device, layer_type=None):
        # Some layer types have no RoPE, e.g. conv or mamba layers. Skip them
        rope_params = self.config.rope_scaling[layer_type] if layer_type is not None else self.config.rope_scaling
        if rope_params is None:
            return None, None, None

        rope_type = rope_params["rope_type"]
        # Diff from Llama - DynamicNTKAlphaRotary
        if rope_type == "dynamic" and rope_params["alpha"]:
            base = rope_params["rope_theta"] * rope_params["alpha"] ** (
                self.config.head_dim / (self.config.head_dim - 2)
            )
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.config.head_dim))
            attention_scaling = 1.0
        else:
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
        return rope_type, inv_freq, attention_scaling


class HunYuanDenseV1Model(LlamaModel):
    pass


class HunYuanDenseV1ForCausalLM(LlamaForCausalLM):
    pass


class HunYuanDenseV1ForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "HunYuanDenseV1ForCausalLM",
    "HunYuanDenseV1Model",
    "HunYuanDenseV1PreTrainedModel",
    "HunYuanDenseV1ForSequenceClassification",
]
