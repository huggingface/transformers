# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..glm4.configuration_glm4 import Glm4Config
from ..glm4.modeling_glm4 import Glm4DecoderLayer, Glm4ForCausalLM, Glm4MLP, Glm4Model, Glm4PreTrainedModel


def build_mode_mix_input(vision_input, txt_input, type_ids, dim=2):
    if dim == 1:
        mix_length = type_ids.shape[0]
        mix_input = torch.zeros(mix_length, dtype=vision_input.dtype)
        mix_input[type_ids == 1] = vision_input
        mix_input[type_ids == 0] = txt_input
        return mix_input

    elif dim == 2:
        v_t, c = vision_input.shape[0], vision_input.shape[1]
        mix_length = type_ids.shape[0]
        if txt_input is not None:
            t_t = txt_input.shape[0]
            assert v_t + t_t == mix_length, "v_token + t_token = type_ids.shape[0]"

        mix_input = torch.zeros(mix_length, c, dtype=vision_input.dtype)
        index = torch.nonzero(type_ids == 1, as_tuple=True)
        mix_input[index] = vision_input
        if txt_input is not None:
            index2 = torch.nonzero(type_ids == 0, as_tuple=True)
            mix_input[index2] = txt_input
        return mix_input


class GlmImageConfig(Glm4Config):
    def __init__(
        self,
        vocab_size: Optional[int] = 151552,
        vision_vocab_size: Optional[int] = 16512,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 13696,
        num_hidden_layers: Optional[int] = 40,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 2,
        head_dim: Optional[int] = 128,
        hidden_act: Optional[str] = "silu",
        attention_dropout: Optional[float] = 0.0,
        max_position_embeddings: Optional[int] = 131072,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 0.00000015625,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        pad_token_id: Optional[int] = 151329,
        eos_token_id: Optional[list[int]] = [151329, 151336, 151338],
        bos_token_id: Optional[int] = None,
        attention_bias: Optional[bool] = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.vision_vocab_size = vision_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class GlmImageMLP(Glm4MLP):
    pass


class GlmImageDecoderLayer(Glm4DecoderLayer):
    pass


class GlmImagePreTrainedModel(Glm4PreTrainedModel):
    pass


class GlmImageModel(Glm4Model):
    def __init__(self, config: GlmImageConfig):
        super().__init__(config)
        self.vision_embed_tokens = nn.Embedding(config.vision_vocab_size, config.hidden_size, self.padding_idx)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        vision_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if vision_inputs_embeds is None:
            vision_inputs_embeds: torch.Tensor = self.vision_embed_tokens(vision_input_ids)

        inputs_embeds = build_mode_mix_input(
            vision_inputs_embeds,
            inputs_embeds,
            type_ids=None,
            dim=vision_inputs_embeds.dim(),
        )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
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


class GlmImageForCausalLM(Glm4ForCausalLM):
    def __init__(self, config: GlmImageConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vision_vocab_size, bias=False)


__all__ = ["GlmImageConfig", "GlmImagePreTrainedModel", "GlmImageModel", "GlmImageForCausalLM"]
