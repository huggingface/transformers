# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""CLAP model configuration"""

import os

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MSClapConfig(PretrainedConfig): 

    model_type = "msclap"

    def __init__(self, 
        text_model = 'gpt2', 
        text_len = 77, 
        transformer_embed_dim = 768, 
        freeze_text_encoder_weights = True, 
        audioenc_name = 'HTSAT', 
        out_emb = 768,
        sample_rate = 44100, 
        duration = 7, 
        fmin = 50, 
        fmax = 8000, 
        n_fft = 1024, 
        hop_size = 320, 
        mel_bins = 64, 
        window_size = 1024, 
        d_proj = 1024, 
        temperature = 0.003, 
        num_classes = 527, 
        batch_size = 1024, 
        demo = False, 
        ): 
        
        # TEXT ENCODER CONFIG
        self.text_model = text_model
        self.text_len = text_len
        self.transformer_embed_dim = transformer_embed_dim
        self.freeze_text_encoder_weights = freeze_text_encoder_weights

        # AUDIO ENCODER CONFIG
        self.audioenc_name = audioenc_name
        self.out_emb = out_emb
        self.sample_rate = sample_rate
        self.duration = duration
        self.fmin = fmin
        self.fmax = fmax #14000 
        self.n_fft = n_fft # 1028 
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.window_size = window_size

        # PROJECTION SPACE CONFIG 
        self.d_proj = d_proj
        self.temperature = temperature

        # TRAINING AND EVALUATION CONFIG
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.demo = demo


