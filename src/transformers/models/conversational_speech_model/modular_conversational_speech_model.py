# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from ...modeling_utils import PreTrainedModel

from ..moshi.modeling_moshi import MoshiForCausalLM, MoshiForConditionalGeneration

from ..llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaForCausalLM, CausalLMOutputWithPast
from ..llama.configuration_llama import LlamaConfig
from ..auto.modeling_auto import AutoModel

from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import rope_config_validation
from ...processing_utils import Unpack
from ...cache_utils import Cache
from ...utils import (
    LossKwargs,
)


class ConversationalSpeechModelDepthDecoderConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=128256,
        audio_vocab_size=2048,
        audio_num_codebooks=32,
        hidden_size=1024,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000,
        rope_scaling=None, #TODO: to change
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ConversationalSpeechModelConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000,
        rope_scaling=None, #TODO: to change
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    

class ConversationalSpeechModelEmbeddingsWithProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding((config.audio_vocab_size + 3) * config.audio_num_codebooks, config.hidden_size)
        self.projector = nn.Linear(config.audio_vocab_size, config.hidden_size)

    def forward(self, input_ids):
        return self.projector(self.embed_tokens(input_ids))


def _codebook_head_selector(module, args, kwargs):
    # select the correct lm_head depending on the position

    input_ids = kwargs.get("input_ids", None)
    cache_position = kwargs.get("cache_position", None)
    position_ids = kwargs.get("position_ids", None)
    past_key_values = kwargs.get("past_key_values", None)
    inputs_embeds = kwargs.get("inputs_embeds", None)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if inputs_embeds is not None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        else:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    codebook_idx = position_ids[:, -1]
    module.lm_head = module.codebooks_heads[codebook_idx - 1]
    return args, kwargs


class ConversationalSpeechModelDepthDecoder(LlamaForCausalLM):
    config_class = ConversationalSpeechModelDepthDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechModelEmbeddingsWithProjector(config)
        self.codebooks_heads = [
            nn.Linear(config.hidden_size, config.audio_vocab_size + 3, bias=False) for _ in range(config.audio_num_codebooks - 1)
        ]
        self.lm_head = None 
        self.register_forward_pre_hook(_codebook_head_selector, with_kwargs=True)

class ConversationalSpeechModelForCausalLM(LlamaForCausalLM):
    pass 


class ConversationalSpeechModelForConditionalGeneration(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.audio_encoder = AutoModel.from_config(config.audio_encoder_config)
        self.depth_decoder = ConversationalSpeechModelDepthDecoder(config.depth_decoder_config)
        self.audio_inputs_embeds = nn.Linear(config.audio_encoder_config.hidden_size, config.depth_decoder_config.hidden_size)



__all__ = [
    "ConversationalSpeechModelDepthDecoderConfig",
    "ConversationalSpeechModelConfig",
    "ConversationalSpeechModelDepthDecoder",
    "ConversationalSpeechModelEmbeddings",
    "ConversationalSpeechModelModel",
    "ConversationalSpeechModelForCausalLM",
    "ConversationalSpeechModelForConditionalGeneration",
]