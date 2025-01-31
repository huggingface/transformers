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

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class StyleTextToSpeech2Config(PretrainedConfig):
    model_type = "style_text_to_speech_2"
    def __init__(
        self,
        vocab_size=178,
        acoustic_text_encoder_hidden_size=512,
        acoustic_text_encoder_num_hidden_layers=3,
        acoustic_text_encoder_dropout=0.2,
        acoustic_text_encoder_kernel_size=5,
        acoustic_text_encoder_leaky_relu_slope=0.2,
        prosodic_encoder_hidden_size=768,
        prosodic_encoder_num_hidden_layers=12,
        prosodic_encoder_num_attention_heads=12,  
        prosodic_encoder_intermediate_size=2048,
        prosodic_encoder_max_position_embeddings=512,
        prosodic_encoder_dropout=0.1,
        style_hidden_size=128,
        max_duration=50,
        duration_encoder_hidden_size=512,
        duration_encoder_num_layers=3,
        duration_encoder_dropout=0.2,
        **kwargs,
    ):
        self.vocab_size = vocab_size

        self.acoustic_text_encoder_hidden_size = acoustic_text_encoder_hidden_size
        self.acoustic_text_encoder_num_hidden_layers = acoustic_text_encoder_num_hidden_layers
        self.acoustic_text_encoder_dropout = acoustic_text_encoder_dropout
        self.acoustic_text_encoder_kernel_size = acoustic_text_encoder_kernel_size
        self.acoustic_text_encoder_leaky_relu_slope = acoustic_text_encoder_leaky_relu_slope

        self.prosodic_encoder_hidden_size = prosodic_encoder_hidden_size
        self.prosodic_encoder_num_hidden_layers = prosodic_encoder_num_hidden_layers
        self.prosodic_encoder_num_attention_heads = prosodic_encoder_num_attention_heads
        self.prosodic_encoder_intermediate_size = prosodic_encoder_intermediate_size
        self.prosodic_encoder_max_position_embeddings = prosodic_encoder_max_position_embeddings
        self.prosodic_encoder_dropout = prosodic_encoder_dropout

        self.style_hidden_size = style_hidden_size
        self.max_duration = max_duration
        self.duration_encoder_hidden_size = duration_encoder_hidden_size
        self.duration_encoder_num_layers = duration_encoder_num_layers
        self.duration_encoder_dropout = duration_encoder_dropout

        super().__init__(**kwargs)

__all__ = ["StyleTextToSpeech2Config"]