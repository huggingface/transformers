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
        hidden_size=512,
        acoustic_text_encoder_num_hidden_layers=3,
        acoustic_text_encoder_dropout=0.2,
        kernel_size=5,
        leaky_relu_slope=0.2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.acoustic_text_encoder_num_hidden_layers = acoustic_text_encoder_num_hidden_layers
        self.acoustic_text_encoder_dropout = acoustic_text_encoder_dropout
        self.kernel_size = kernel_size
        self.leaky_relu_slope = leaky_relu_slope

        super().__init__(**kwargs)
    

__all__ = ["StyleTextToSpeech2Config"]