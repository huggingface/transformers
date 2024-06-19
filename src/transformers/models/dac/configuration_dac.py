# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
"""Dac model configuration"""

import numpy as np

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

class DacConfig(PretrainedConfig):
    r"""
    ```"""

    model_type = "dac"

    def __init__(
        self,
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=False,
        sample_rate=44100,
        **kwargs,
    ):
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = int(np.prod(encoder_rates))

        super().__init__(**kwargs)
