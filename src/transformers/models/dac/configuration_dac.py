# coding=utf-8
# Copyright 2024 Descript and The HuggingFace Inc. team. All rights reserved.
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
    This is the configuration class to store the configuration of an [`DacModel`]. It is used to instantiate a
    Dac model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [descript/dac_16khz](https://huggingface.co/descript/dac_16khz) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_hidden_size (`int`, *optional*, defaults to 64):
            Intermediate representation dimension for the encoder.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 4, 8, 8]`):
            Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 1536):
            Intermediate representation dimension for the decoder.
        n_codebooks (`int`, *optional*, defaults to 9):
            Number of codebooks in the VQVAE.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of discrete codes in each codebook.
        codebook_dim (`int`, *optional*, defaults to 8):
            Dimension of the codebook vectors. If not defined, uses `encoder_hidden_size`.
        quantizer_dropout (`bool`, *optional*, defaults to `False`):
            Whether to apply dropout to the quantizer.

    Example:

    ```python
    >>> from transformers import DacModel, DacConfig

    >>> # Initializing a "descript/dac_16khz" style configuration
    >>> configuration = DacConfig()

    >>> # Initializing a model (with random weights) from the "descript/dac_16khz" style configuration
    >>> model = DacModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    """
    model_type = "dac"

    def __init__(
        self,
        encoder_hidden_size=64,
        downsampling_ratios=[2, 4, 8, 8],
        decoder_hidden_size=1536,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0,
        **kwargs,
    ):
        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.decoder_hidden_size = decoder_hidden_size
        self.upsampling_ratios = downsampling_ratios[::-1]
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout

        self.hidden_size = encoder_hidden_size * (2 ** len(downsampling_ratios))

        self.hop_length = int(np.prod(downsampling_ratios))

        super().__init__(**kwargs)
