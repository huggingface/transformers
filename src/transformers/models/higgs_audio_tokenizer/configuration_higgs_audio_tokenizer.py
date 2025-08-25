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
"""HiggsAudioTokenizerConfig."""

import math
from typing import Optional, Union

import numpy as np

from ...configuration_utils import PretrainedConfig
from ..dac import DacConfig
from ..hubert import HubertConfig


class HiggsAudioTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`HiggsAudioTokenizer`]. It is used to instantiate a
    HiggsAudioTokenizer according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [bosonai/higgs-audio-v2-tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) architecture.

    Args:
        target_bandwidths (`List[float]`, *optional*, defaults to `[0.5, 1, 1.5, 2, 4]`):
            The range of different bandwidths (in kbps) the model can encode audio with.
        sample_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized, in hertz (Hz).
        semantic_sample_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the semantic model was trained with, in hertz (Hz).
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
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of entries in each residual quantizer’s codebook.
        num_quantizers (`int`, *optional*, defaults to 8):
            Number of sequential quantizers (codebooks) in the RVQ stack.
        codebook_dim (`int`, *optional*, defaults to 64):
            Dimensionality of each codebook vector.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer for all weight matrices.
        acoustic_model_config (`Union[Dict, DacConfig]`, *optional*):
            An instance of the configuration for the acoustic (DAC) model.
        semantic_model_config (`Union[Dict, HubertConfig]`, *optional*):
            An instance of the configuration object for the semantic (HuBERT) model.
        downsample_mode (`str`, *optional*, defaults to `"step_down"`):
            The downsample mode for the semantic features.
        pad (`int`, *optional*, defaults to 160):
            Padding size for the input to the semantic model.
        downsample_factor (`int`, *optional*, defaults to 320):
            The downsample factor used to compute actual downsample factor for the semantic features.

    Example:

    ```python
    >>> from transformers import HiggsAudioTokenizer, HiggsAudioTokenizerConfig

    >>> # Initializing a " " style configuration
    >>> configuration = HiggsAudioTokenizerConfig()

    >>> # Initializing a model (with random weights) from the " " style configuration
    >>> model = HiggsAudioTokenizer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "higgs_audio_tokenizer"

    sub_configs = {
        "acoustic_model_config": DacConfig,
        "semantic_model_config": HubertConfig,
    }

    def __init__(
        self,
        target_bandwidths: Optional[list[float]] = None,
        sample_rate: int = 24000,
        semantic_sample_rate: int = 16000,
        kernel_size: int = 3,
        channel_ratios: list[float] = [1, 1],
        strides: list[int] = [1, 1],
        block_dilations: list[int] = [1, 1],
        unit_kernel_size: int = 3,
        codebook_size: int = 1024,
        num_quantizers: int = 8,
        codebook_dim: int = 64,
        initializer_range: float = 0.02,
        acoustic_model_config: Union[dict, DacConfig] = None,
        semantic_model_config: Union[dict, HubertConfig] = None,
        downsample_mode: str = "step_down",
        pad: int = 160,
        downsample_factor: int = 320,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if acoustic_model_config is None:
            self.acoustic_model_config = DacConfig(
                encoder_hidden_size=64,
                downsampling_ratios=[8, 5, 4, 2, 3],
                decoder_hidden_size=1024,
                upsampling_ratios=[8, 5, 4, 2, 3],
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
        self.sample_rate = sample_rate
        self.semantic_sample_rate = semantic_sample_rate
        self.kernel_size = kernel_size
        self.channel_ratios = channel_ratios
        self.strides = strides
        self.block_dilations = block_dilations
        self.unit_kernel_size = unit_kernel_size
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.codebook_dim = codebook_dim
        self.initializer_range = initializer_range
        self.downsample_mode = downsample_mode
        self.pad = pad
        self.downsample_factor = downsample_factor

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sample_rate / np.prod(self.acoustic_model_config.upsampling_ratios))

    @property
    def sampling_rate(self):
        return self.sample_rate

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.acoustic_model_config.downsampling_ratios))

    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_model_config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self.acoustic_model_config.hidden_size + self.semantic_model_config.hidden_size


__all__ = ["HiggsAudioTokenizerConfig"]
