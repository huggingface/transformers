# coding=utf-8
# Copyright The HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import numpy as np

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


class XcodecConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`XcodecModel`]. It is used to instantiate a
    XcodecModel according to the specified arguments, defining the model architecture.

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
            Number of entries in each residual quantizer's codebook.
        num_quantizers (`int`, *optional*, defaults to 8):
            Number of sequential quantizers (codebooks) in the RVQ stack.
        codebook_dim (`int`, *optional*, defaults to 64):
            Dimensionality of each codebook vector.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer for all weight matrices.
        acoustic_model_config (`Union[Dict, DacConfig]`, *optional*):
            An instance of the configuration for the acoustic (DAC) model.
        semantic_config (`Union[Dict, HubertConfig]`, *optional*):
            An instance of the configuration object for the semantic (HuBERT) model.
        downsample_mode (`str`, *optional*, defaults to `"step_down"`):
            The downsample mode for the semantic features.
        pad (`int`, *optional*, defaults to 160):
            Padding size for the input to the semantic model.
        downsample_factor (`int`, *optional*, defaults to 320):
            The downsample factor used to compute actual downsample factor for the semantic features.

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

    model_type = "higgs_audio_tokenizer"
    sub_configs = {"semantic_config": AutoConfig}
    attribute_map = {
        "sampling_rate": "sample_rate",
    }
    def __init__(
        self, 
        encoder_hidden_size=64,
        acoustic_hidden_size=256,
        downsampling_ratios=[8, 5, 4, 2, 3],
        decoder_hidden_size=1024,
        upsampling_ratios=[8, 5, 4, 2, 3],
        n_codebooks=9,
        strides=[1, 1],
        channel_ratios=[1, 1],
        kernel_size=3,
        block_dilations=[1, 1],
        semantic_config=None,
        unit_kernel_size=3,
        target_bandwidths=[0.5, 1, 1.5, 2, 4],
        sample_rate=24000,
        semantic_sample_rate=16000,
        codebook_size=1024,
        num_quantizers=8,
        codebook_dim=64,
        initializer_range=0.02,
        pad=160,
        **kwargs,
    ):
        # Handle semantic model config
        if isinstance(semantic_config, dict):
            semantic_config["model_type"] = semantic_config.get("model_type", "hubert")
            semantic_config = CONFIG_MAPPING[semantic_config["model_type"]](**semantic_config)
        elif semantic_config is None:
            semantic_config = CONFIG_MAPPING["hubert"]()

        self.semantic_config = semantic_config

        self.acoustic_hidden_size = acoustic_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.n_codebooks = n_codebooks

        self.strides = strides
        self.channel_ratios = channel_ratios
        self.kernel_size = kernel_size
        self.block_dilations = block_dilations
        self.unit_kernel_size = unit_kernel_size

        self.decoder_hidden_size = decoder_hidden_size
        self.target_bandwidths = target_bandwidths
        self.sample_rate = sample_rate
        self.semantic_sample_rate = semantic_sample_rate
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.codebook_dim = codebook_dim
        self.initializer_range = initializer_range
        self.pad = pad
        self.upsampling_ratios = upsampling_ratios

        super().__init__(**kwargs)

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sample_rate / np.prod(self.upsampling_ratios))

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.downsampling_ratios))

    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self.acoustic_hidden_size + self.semantic_hidden_size

    @property
    def semantic_downsample_factor(self) -> int:
        return int(
            self.hop_length / (self.sample_rate / self.semantic_sample_rate) / self.downsample_factor
        )


__all__ = ["XcodecConfig"]
