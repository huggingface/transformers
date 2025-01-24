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
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .configuration_style_text_to_speech_2 import StyleTextToSpeech2Config
from ...modeling_utils import PreTrainedModel
from ...utils import logging


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "hexgrad/Kokoro-82M"
_CONFIG_FOR_DOC = "StyleTextToSpeech2Config"


class AcousticTextEncoderLayer(nn.Module):
    def __init__(self, config: StyleTextToSpeech2Config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.kernel_size, padding=config.kernel_size // 2)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.leaky_relu_slope = config.leaky_relu_slope
        self.dropout = nn.Dropout(config.acoustic_text_encoder_dropout)
        
        # apply weight norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states.transpose(1, -1))
        hidden_states = self.norm(hidden_states.transpose(1, -1))
        hidden_states = F.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class StyleTextToSpeech2AcousticTextEncoderPretrainedModel(PreTrainedModel):
    config_class = StyleTextToSpeech2Config
    base_model_prefix = "acoustic_text_encoder"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)


class StyleTextToSpeech2AcousticTextEncoder(StyleTextToSpeech2AcousticTextEncoderPretrainedModel):
    def __init__(self, config: StyleTextToSpeech2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([AcousticTextEncoderLayer(config) for _ in range(config.acoustic_text_encoder_num_hidden_layers)])
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, input_ids, mask=None, input_lengths=None):
        if mask is None:
            mask = torch.full((*input_ids.shape, 1), 1, dtype=torch.int, device=input_ids.device)
        else:
            mask = mask.unsqueeze(-1)
        
        if input_lengths is None:
            input_lengths = mask.sum(dim=1).view(-1).tolist()
    
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            hidden_states = hidden_states.masked_fill(~mask.bool(), 0)
        
        hidden_states = nn.utils.rnn.pack_padded_sequence(hidden_states, input_lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True)
    
        return hidden_states


__all__ = ["StyleTextToSpeech2AcousticTextEncoder"]