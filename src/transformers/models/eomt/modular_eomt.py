# coding=utf-8
# Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file ehidden_statescept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ehidden_statespress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch EoMT model."""

import torch
from torch import nn

from ...activations import ACT2FN
from ..dinov2.modeling_dinov2 import (
    Dinov2Attention,
    Dinov2DropPath,
    Dinov2Embeddings,
    Dinov2Encoder,
    Dinov2Layer,
    Dinov2LayerScale,
    Dinov2MLP,
    Dinov2PreTrainedModel,
)
from .configuration_eomt import EoMTConfig


class EoMTEmbeddings(Dinov2Embeddings):
    pass


class EoMTAttention(Dinov2Attention):
    pass


class EoMTMLP(Dinov2MLP):
    pass


class EoMTLayerScale(Dinov2LayerScale):
    pass


class EoMTDropPath(Dinov2DropPath):
    pass


class EoMTLayer(Dinov2Layer, nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: EoMTConfig) -> None:
        nn.Module().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = EoMTAttention(config)
        self.layer_scale1 = EoMTLayerScale(config)
        self.drop_path = EoMTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = EoMTMLP(config)
        self.layer_scale2 = EoMTLayerScale(config)


class EoMTScaleBlock(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.deconv1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.activation = ACT2FN[config.hidden_act]
        self.deconv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
            bias=False,
        )
        self.layernorm2d = nn.GroupNorm(num_groups=1, num_channels=hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.tensor) -> torch.Tensor:
        hidden_states = self.deconv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.deconv2(hidden_states)
        hidden_states = self.layernorm2d(hidden_states)
        return hidden_states


class EoMTEncoder(Dinov2Encoder):
    pass


class EoMTPreTrainedModel(Dinov2PreTrainedModel):
    pass


class EoMTModel(EoMTPreTrainedModel):
    pass
