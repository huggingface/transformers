# coding=utf-8
# Copyright 2025 Apple Inc. and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import torch
from .configuration_aimv2 import AIMv2Config
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ..llama.modeling_llama import LlamaRMSNorm
from ..siglip.modeling_siglip import SiglipAttention, SiglipEncoder


class AIMv2PreTrainedModel(PreTrainedModel):
    pass

class AIMv2RMSNorm(LlamaRMSNorm):
    pass 

class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        in_features = config.hidden_size
        out_features = config.intermediate_size
        self.act_fn = config.hidden_act

        self.fc1 = nn.Linear(in_features, out_features, bias=config.use_bias)
        self.fc2 = nn.Linear(out_features, in_features, bias=config.use_bias)
        self.fc3 = nn.Linear(in_features, out_features, bias=config.use_bias)

    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        fc3_out = self.fc3(hidden_states)
        fc1_out = self.fc1(hidden_states)
        hidden_states = ACT2FN[self.act_fn](fc1_out) * fc3_out
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class AIMv2Embeddings(nn.Module):
    def __init__(self, config:AIMv2Config ):
        self.patch_embed = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size= config.patch_size, stride=config.patch_size)
        self.rms_norm = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)

    @staticmethod
    def build_2d_sincos_position_embedding(height, width, embed_dim):
        pass

    def forward(self, pixel_values:torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states)

        _, num_patches, _ = hidden_states.size()

        # added logic for native in build s2d sincos pos embed
        hidden_states = hidden_states +self.position_embeddings

        return hidden_states

class AIMv2Attention(SiglipAttention):
    pass

class AIMv2EncoderLayer(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.attention = AIMv2Attention(config)
        self.ffn = AIMv2SwiGLUFFN(config)
        self.rms_norm1 = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rms_norm2 = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask, output_attention:Optional[bool]=False) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states = self.rms_norm1(hidden_states)
        attn_output, attn_wights = self.attention(hidden_states=norm_hidden_states,
                                          attention_mask=attention_mask,
                                          output_attention=output_attention)

        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.rms_norm2(hidden_states)
        mlp_output = self.ffn(norm_hidden_states)

        hidden_states = hidden_states + mlp_output
        return (hidden_states, attn_wights) if output_attention else (hidden_states,)

class AIMv2Encoder(SiglipEncoder):
    pass

class AIMv2Model(nn.Module):
    def __init__(self,config: AIMv2Config):
        super().__init__()
        self.config = config
        self.embeddings = AIMv2Embeddings(config)
        self.encoder = AIMv2Encoder(config)
        self.rms_norm = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.rms_norm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

__all__ = ["AIMv2Model"]