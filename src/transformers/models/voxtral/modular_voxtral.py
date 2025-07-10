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

from ...activations import ACT2FN
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder, Qwen2AudioForConditionalGeneration
from ..whisper.modeling_whisper import WhisperEncoder
from .configuration_voxtral import VoxtralConfig

from torch import nn


class VoxtralEncoder(Qwen2AudioEncoder):
    pass


class VoxtralMultiModalProjector(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VoxtralForConditionalGeneration(Qwen2AudioForConditionalGeneration):
    pass


__all__ = [
    "VoxtralEncoder",
    "VoxtralForConditionalGeneration"
]
