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
"""Xcodec model configuration"""

import math
from typing import Optional, Union

import numpy as np

from transformers import DacConfig, HubertConfig

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class XcodecConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`XcodecModel`]. It is used to instantiate a
    Xcodec model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [Manel/X-Codec](https://huggingface.co/Manel/X-Codec) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        target_bandwidths (`List[float]`, *optional*, defaults to `[0.5, 1, 1.5, 2, 4]`):
            The range of different bandwidths (in kbps) the model can encode audio with.
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        sample_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized, in hertz (Hz).
        input_channels (`int`, *optional*, defaults to 768):
            Number of channels of the input to the first convolution in the semantic encoder.
        encoder_channels (`int`, *optional*, defaults to 768):
            Number of hidden channels in each semantic encoder block.
        kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the initial semantic convolution.
        channel_ratios (`List[float]`, *optional*, defaults to `[1, 1]`):
            Expansion factors for the number of output channels in each semantic block.
        strides (`List[int]`, *optional*, defaults to `[1, 1]`):
            Strides for each semantic encoder block.
        block_dilations (`List[int]`, *optional*, defaults to `[1, 1]`):
            Dilation factors for the residual units in semantic blocks.
        unit_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size inside each ResidualUnit in semantic blocks.
        decoder_channels (`int`, *optional*, defaults to 768):
            Number of hidden channels in each semantic decoder block.
        output_channels (`int`, *optional*, defaults to 768):
            Number of output channels in the semantic decoder.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of entries in each residual quantizer’s codebook.
        num_quantizers (`int`, *optional*, defaults to 8):
            Number of sequential quantizers (codebooks) in the RVQ stack.
        codebook_dim (`int`, *optional*, defaults to 1024):
            Dimensionality of each codebook vector.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer for all weight matrices.
        hidden_dim (`int`, *optional*, defaults to 1024):
            Dimensionality of the joint acoustic+semantic FC layer.
        intermediate_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the next FC layer in the decoder path.
        output_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the final FC layer before feeding into the acoustic decoder.
        acoustic_model_config (`Union[Dict, DacConfig]`, *optional*):
            An instance of the configuration for the acoustic (DAC) model.
        semantic_model_config (`Union[Dict, HubertConfig]`, *optional*):
            An instance of the configuration object for the semantic (HuBERT) model.

    Example:

    ```python
    >>> from transformers import XcodecModel, XcodecConfig

    >>> # Initializing a " " style configuration
    >>> configuration = XcodecConfig()

    >>> # Initializing a model (with random weights) from the " " style configuration
    >>> model = XcodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xcodec"

    sub_configs = {
        "acoustic_model_config": DacConfig,
        "semantic_model_config": HubertConfig,
    }

    def __init__(
        self,
        target_bandwidths: Optional[list[float]] = None,
        audio_channels: int = 1,
        sample_rate: int = 16000,
        input_channels: int = 768,
        encoder_channels: int = 768,
        kernel_size: int = 3,
        channel_ratios: list[float] = [1, 1],
        strides: list[int] = [1, 1],
        block_dilations: list[int] = [1, 1],
        unit_kernel_size: int = 3,
        decoder_channels: int = 768,
        output_channels: int = 768,
        codebook_size: int = 1024,
        num_quantizers: int = 8,
        codebook_dim: int = 1024,
        initializer_range: float = 0.02,
        hidden_dim: int = 1024,
        intermediate_dim: int = 768,
        output_dim: int = 256,
        acoustic_model_config: Union[dict, DacConfig] = None,
        semantic_model_config: Union[dict, HubertConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if acoustic_model_config is None:
            self.acoustic_model_config = DacConfig(
                encoder_hidden_size=64,
                downsampling_ratios=[8, 5, 4, 2],
                decoder_hidden_size=1024,
                upsampling_ratios=[8, 5, 4, 2],
                hidden_size=256,
            )
        elif isinstance(acoustic_model_config, dict):
            self.acoustic_model_config = DacConfig(**acoustic_model_config)
        elif isinstance(acoustic_model_config, DacConfig):
            self.acoustic_model_config = acoustic_model_config

        if semantic_model_config is None:
            self.semantic_model_config = HubertConfig()
        elif isinstance(semantic_model_config, dict):
            self.semantic_model_config = HubertConfig(**semantic_model_config)
        elif isinstance(semantic_model_config, HubertConfig):
            self.semantic_model_config = semantic_model_config

        if target_bandwidths is None:
            target_bandwidths = [0.5, 1, 1.5, 2, 4]

        self.target_bandwidths = target_bandwidths
        self.audio_channels = audio_channels
        self.sample_rate = sample_rate
        self.input_channels = input_channels
        self.encoder_channels = encoder_channels
        self.kernel_size = kernel_size
        self.channel_ratios = channel_ratios
        self.strides = strides
        self.block_dilations = block_dilations
        self.unit_kernel_size = unit_kernel_size
        self.decoder_channels = decoder_channels
        self.output_channels = output_channels
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.codebook_dim = codebook_dim
        self.initializer_range = initializer_range
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sample_rate / np.prod(self.acoustic_model_config.upsampling_ratios))

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.acoustic_model_config.downsampling_ratios))


__all__ = ["XcodecConfig"]
