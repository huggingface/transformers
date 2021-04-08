# coding=utf-8
# Copyright 2021 The Ontocord team, the G2P, Melgan, and Fastspeech2 Authors, and the HuggingFace Inc. team. All rights reserved.
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

# This software is based on other open source code. A huge thanks to
# the Huggingface team, and the authors of the Fastspeech2 and Melgan
# papers and the following authors who originally implemented the
# various modules, from which this code is based:

# Chung-Ming Chien's Fastspeech2 implementation is under the MIT license: https://github.com/ming024/FastSpeech2
# Seung-won Park 박승원's Meglan implementation is under BSD-3 license: https://github.com/seungwonpark/melgan
# Kyubyong Park's G2P implementation is under the Apache 2 license: https://github.com/Kyubyong/g2p, and also here for pytorch specifics https://github.com/Kyubyong/nlp_made_easy/blob/master/PyTorch%20seq2seq%20template%20based%20on%20the%20g2p%20task.ipynb
""" FastSpeech2 model configuration """

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


import os

class FastSpeech2Config(PretrainedConfig):
    model_type = "fastspeech2"

    def __init__(self,
        # Audio and mel
        ### for LJSpeech ###
        sampling_rate = 22050,
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,

        ### for Blizzard2013 ###
        #sampling_rate = 16000
        #filter_length = 800
        #hop_length = 200
        #win_length = 800

        max_wav_value = 32768.0,
        n_mel_channels = 80,
        mel_fmin = 0.0,
        mel_fmax = 8000.0,
        # FastSpeech 2
        encoder_layer = 4,
        encoder_head = 2,
        encoder_hidden = 256,
        decoder_layer = 4,
        decoder_head = 2,
        decoder_hidden = 256,
        fft_conv1d_filter_size = 1024,
        fft_conv1d_kernel_size = (9, 1),
        encoder_dropout = 0.2,
        decoder_dropout = 0.2,

        variance_predictor_filter_size = 256,
        variance_predictor_kernel_size = 3,
        variance_predictor_dropout = 0.5,

        max_seq_len = 1000,

        # Quantization for F0 and energy
        ### for LJSpeech ###
        f0_min = 71.0,
        f0_max = 795.8,
        energy_min = 0.0,
        energy_max = 315.0,
        ### for Blizzard2013 ###
        #f0_min = 71.0
        #f0_max = 786.7
        #energy_min = 21.23
        #energy_max = 101.02
        dropout=0.1,
        n_bins = 256,
        vocab_size = 154,
        use_postnet=True,
        initializer_range=0.02,
        # Log-scaled duration
        log_offset = 1.,
        use_g2p = False,
        use_lm_linear_in = False,
        lm_output_dim = 512,
        lm_output_dim_upsample_factor = 6,
        pad_token_id=0,
        bos_token_id =152,
        **kwargs

        ):
        super().__init__(**kwargs, pad_token_id=pad_token_id)
        self.initializer_range = initializer_range
        self.dropout = dropout
        self.use_postnet = use_postnet
        # Audio and mel
        ### for LJSpeech ###
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_wav_value = max_wav_value
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        # FastSpeech 2
        self.encoder_layer = encoder_layer
        self.encoder_head = encoder_head
        self.encoder_hidden = encoder_hidden
        self.decoder_layer = decoder_layer
        self.decoder_head = decoder_head
        self.decoder_hidden = decoder_hidden
        self.fft_conv1d_filter_size = fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = fft_conv1d_kernel_size
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

        self.variance_predictor_filter_size = variance_predictor_filter_size
        self.variance_predictor_kernel_size = variance_predictor_kernel_size
        self.variance_predictor_dropout = variance_predictor_dropout

        self.max_seq_len = max_seq_len

        # Quantization for F0 and energy
        ### for LJSpeech ###
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.n_bins = n_bins
        self.vocab_size = vocab_size

        # Log-scaled duration
        self.log_offset = log_offset
        self.use_lm_linear_in = use_lm_linear_in
        self.lm_output_dim = lm_output_dim
        self.lm_output_dim_upsample_factor = lm_output_dim_upsample_factor
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id


    
