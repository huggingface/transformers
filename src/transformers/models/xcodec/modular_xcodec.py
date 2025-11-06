# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ..auto.modeling_auto import AutoModel
from ..dac.modeling_dac import DacDecoder, DacDecoderBlock, DacEncoder, DacModel
from ..encodec.modeling_encodec import EncodecResidualVectorQuantizer
from .configuration_xcodec import XcodecConfig


class XcodecAcousticEncoder(DacEncoder): ...


class XcodecSemanticEncoderResidualLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn = nn.ELU()
        self.conv1 = nn.Conv1d(config.semantic_hidden_size, config.semantic_hidden_size, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(config.semantic_hidden_size, config.semantic_hidden_size, kernel_size=1, bias=False)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.conv1(self.act_fn(hidden_state))
        hidden_state = self.conv2(self.act_fn(hidden_state))
        hidden_state = self.conv3(residual + hidden_state)
        return hidden_state


class XcodecSemanticEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.residual_layers = nn.ModuleList(
            [XcodecSemanticEncoderResidualLayer(config) for _ in range(config.num_residual_layers)]
        )
        self.conv = nn.Conv1d(config.semantic_hidden_size, config.semantic_hidden_size, kernel_size=3, padding=1)
    
    def forward(self, hidden_state):
        for layer in self.residual_layers:
            hidden_state = layer(hidden_state)
        hidden_state = self.conv(hidden_state)
        return hidden_state


class XcodecSemanticEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_conv = nn.Conv1d(config.semantic_hidden_size, config.semantic_hidden_size, kernel_size=3, padding=1, bias=False)
        self.layers = nn.ModuleList(
            [XcodecSemanticEncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, hidden_state):
        hidden_state = self.input_conv(hidden_state)
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class XcodecEncoder(nn.Module):
    def __init__(self, config: XcodecConfig):
        super().__init__()
        self.acoustic_encoder = XcodecAcousticEncoder(config)
        self.semantic_encoder = XcodecSemanticEncoder(config)
        self.semantic_model = AutoModel.from_config(config.semantic_config)
        self.linear = nn.Linear(
            config.semantic_hidden_size + config.acoustic_hidden_size,
            config.semantic_hidden_size + config.acoustic_hidden_size
        )

        self.sample_rate = config.sample_rate
        self.semantic_sample_rate = config.semantic_sample_rate

    def _extract_semantic_features(self, input_values):
        with torch.no_grad():
            input_values = torchaudio.functional.resample(
                input_values, self.config.sampling_rate, self.config.semantic_sample_rate
            )
            input_values = input_values[:, 0, :]
            input_values = F.pad(input_values, (self.config.pad, self.config.pad))
            outputs = self.semantic_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            stacked = torch.stack(hidden_states, dim=1)
            semantic_features = stacked.mean(dim=1)
            semantic_features = semantic_features[:, :: self.config.semantic_downsample_factor, :]
            return semantic_features

    def forward(self, input_values: torch.Tensor):
        acoustic_embeds = self.acoustic_encoder(input_values)

        semantic_features = self._extract_semantic_features(input_values)
        semantic_embeds = self.semantic_encoder(semantic_features.transpose(1, 2))

        input_embeds = torch.cat([acoustic_embeds, semantic_embeds], dim=1)
        return input_embeds


class XcodecAcousticDecoderBlock(DacDecoderBlock):
    def __init__(self, config, stride: int = 1, stride_index: int = 1):
        super().__init__(config)
        input_dim = config.decoder_hidden_size // 2**stride_index
        output_dim = config.decoder_hidden_size // 2 ** (stride_index + 1)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=(stride % 2,),
        )


class XcodecAcousticDecoder(DacDecoder):
    def __init__(self, config: XcodecConfig):
        super().__init__(config)
        input_channel = config.acoustic_hidden_size
        del self.tanh

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        for layer in self.block:
            hidden_states = layer(hidden_states)

        hidden_states = self.snake1(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return hidden_states


class XcodecSemanticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_conv = nn.Conv1d(config.semantic_hidden_size, config.semantic_hidden_size, kernel_size=3, padding=1, bias=False)
        self.layers = nn.ModuleList(
            [XcodecSemanticEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.output_conv = nn.Conv1d(config.semantic_hidden_size, config.semantic_hidden_size, kernel_size=3, padding=1, bias=False)


class XcodecResidualVectorQuantizer(EncodecResidualVectorQuantizer): ...


# TODO: @eustlb, consider adding bandwidth handling in dac in order to standardize codec usage
class XcodecModel(DacModel):
    def __init__(self, config: XcodecConfig):
        super().__init__(config)
        self.decoder = XcodecAcousticDecoder(config)
        self.semantic_decoder = XcodecSemanticDecoder(config)
        self.semantic_proj = nn.Linear(
            config.semantic_hidden_size + config.acoustic_hidden_size,
            config.semantic_hidden_size
        )
        self.acoustic_proj = nn.Linear(
            config.semantic_hidden_size + config.acoustic_hidden_size,
            config.acoustic_hidden_size
        )


__all__ = ["XcodecModel"]
