# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import can_return_tuple, logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

logger = logging.get_logger(__name__)


class SmolLM3RMSNorm(LlamaRMSNorm):
    pass


class SmolLM3MLP(LlamaMLP):
    pass


class SmolLM3Attention(LlamaAttention):
    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)

        self.use_rope = config.no_rope_layers[layer_idx]

        self.sliding_window = config.sliding_window if (
                config.use_sliding_window and
                config.sliding_window is not None and
                not self.use_rope
        ) else None

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if self.use_rope:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            else:
                cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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


class SmolLM3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = SmolLM3Attention(config=config, layer_idx=layer_idx)


class SmolLM3RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class SmolLM3PreTrainedModel(LlamaPreTrainedModel):
    pass


class SmolLM3Model(LlamaModel):
    def __init__(self, config: SmolLM3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [SmolLM3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class SmolLM3ForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SmolLM3Model(config)
        self.post_init()


class SmolLM3ForSequenceClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = SmolLM3Model(config)
        self.post_init()


class SmolLM3ForTokenClassification(LlamaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = SmolLM3Model(config)
        self.post_init()


class SmolLM3ForQuestionAnswering(LlamaForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = SmolLM3Model(config)
        self.post_init()


__all__ = [
    "SmolLM3Config",
    "SmolLM3PreTrainedModel",
    "SmolLM3Model",
    "SmolLM3ForCausalLM",
    "SmolLM3ForSequenceClassification",
    "SmolLM3ForTokenClassification",
    "SmolLM3ForQuestionAnswering",
]