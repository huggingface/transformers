# coding=utf-8
# Copyright 2024 Cohere team. All rights reserved.
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

# This file is based on the LLama model definition file in transformers

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, LlamaMLP, LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM
import torch
from typing import *
import torch.nn as nn
from transformers import CohereConfig


class CohereLayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-5, bias=False):
        """The hidden size can be a tuple or an int. The tuple is used for QKNorm to normalize across head_dim"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)

class CohereRotaryEmbedding(LlamaRotaryEmbedding):

    def rotate_half(self, x):
        # Split and rotate
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rot_x = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return rot_x

    def forward(self, q, k, position_ids=None, unsqueeze_dim=1):
        dtype = q.dtype
        q,k  = q.float(), k.float()
        cos, sin = self.comput_cos_sin(q, position_ids)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed.to(dtype=dtype), k_embed.to(dtype=dtype)

CohereMLP = LlamaMLP 
CohereAttention = LlamaAttention 
CohereSdpaAttention = LlamaAttention 
CohereFlashAttention2 = LlamaAttention 

COHERE_ATTENTION_CLASSES = {"eager": CohereAttention, "flash_attention_2": CohereFlashAttention2, "sdpa": CohereSdpaAttention}

class CohereDecoderLayer(nn.Module):
    def __init__(self, config: CohereConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = COHERE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = CohereMLP(config)
        self.input_layernorm = CohereLayerNorm(hidden_size=(config.hidden_size), eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states_attention, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        # Add everything together (main diff with llama )
        hidden_states = residual + hidden_states_attention + hidden_states_mlp

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

CoherePreTrainedModel = LlamaPreTrainedModel
CohereModel = LlamaModel

from typing import TypedDict, Unpack


class CohereForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    # Ignore copy
    def __init__(self, config):
        super().__init__(config)
        self.logit_scale = config.logit_scale
        self.tie_word_embeddings = config.tie_word_embeddings

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        logits = self.lm_head(output[1])
        logits = logits * self.logit_scale