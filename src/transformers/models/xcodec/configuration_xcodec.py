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
"""Xcodec model configuration"""

import math
from dataclasses import dataclass

import numpy as np
from huggingface_hub.dataclasses import strict

from transformers import AutoConfig, DacConfig, HubertConfig

from ...configuration_utils import PreTrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class XcodecConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`XcodecModel`]. It is used to instantiate a
    Xcodec model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [Manel/X-Codec](https://huggingface.co/Manel/X-Codec) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        target_bandwidths (`List[float]`, *optional*, defaults to `[0.5, 1, 1.5, 2, 4]`):
            The range of different bandwidths (in kbps) the model can encode audio with.
        sample_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized, in hertz (Hz).
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
        codebook_dim (`int`, *optional*):
            Dimensionality of each codebook vector. Defaults to sum of hidden size of acoustic and semantic models.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer for all weight matrices.
        acoustic_model_config (`Union[Dict, DacConfig]`, *optional*):
            An instance of the configuration for the acoustic (DAC) model.
        semantic_model_config (`Union[Dict, HubertConfig, WavLMConfig]`, *optional*):
            An instance of the configuration object for the semantic (HuBERT) model.

    Example:

    ```python
    >>> from transformers import XcodecModel, XcodecConfig

    >>> # Initializing configuration
    >>> configuration = XcodecConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XcodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xcodec"

    sub_configs = {
        "acoustic_model_config": AutoConfig,
        "semantic_model_config": AutoConfig,
    }

    target_bandwidths: list[float | int] | None = None
    sample_rate: int = 16000
    kernel_size: int = 3
    channel_ratios: list[int] | tuple[int, ...] = (1, 1)
    strides: list[int] | tuple[int, ...] = (1, 1)
    block_dilations: list[int] | tuple[int, ...] = (1, 1)
    unit_kernel_size: int = 3
    codebook_size: int = 1024
    codebook_dim: int | None = None
    initializer_range: float = 0.02
    acoustic_model_config: dict | DacConfig | None = None
    semantic_model_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.acoustic_model_config is None:
            self.acoustic_model_config = CONFIG_MAPPING["dac"](
                encoder_hidden_size=64,
                # NOTE: original DAC uses [2, 4, 8, 8] `downsampling ratios`, namely reverse of `upsampling_ratios`
                # (not sure if intentional by Xcodec but we keep it)
                downsampling_ratios=[8, 5, 4, 2],
                decoder_hidden_size=1024,
                upsampling_ratios=[8, 5, 4, 2],
                hidden_size=256,
            )
        elif isinstance(self.acoustic_model_config, dict):
            self.acoustic_model_config["model_type"] = self.acoustic_model_config.get("model_type", "dac")
            self.acoustic_model_config = CONFIG_MAPPING[self.acoustic_model_config["model_type"]](
                **{**self._default_acoustic_model_config_kwargs, **self.acoustic_model_config}
            )

        if self.semantic_model_config is None:
            self.semantic_model_config = CONFIG_MAPPING["hubert"]()
        elif isinstance(self.semantic_model_config, dict):
            self.semantic_model_config["model_type"] = self.semantic_model_config.get("model_type", "hubert")
            self.semantic_model_config = CONFIG_MAPPING[self.semantic_model_config["model_type"]](
                **{**self._default_semantic_model_config_kwargs, **self.semantic_model_config}
            )

        self.target_bandwidths = self.target_bandwidths or [0.5, 1, 1.5, 2, 4]
        if self.codebook_dim is None:
            self.codebook_dim = self.acoustic_model_config.hidden_size + self.semantic_model_config.hidden_size

        super().__post_init__(**kwargs)

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sample_rate / self.hop_length)

    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_model_config.hidden_size

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.acoustic_model_config.downsampling_ratios))

    @property
    def codebook_nbits(self) -> int:
        return math.ceil(math.log2(self.codebook_size))

    @property
    def hidden_size(self) -> int:
        return self.acoustic_model_config.hidden_size + self.semantic_model_config.hidden_size

    @property
    def num_quantizers(self) -> int:
        return int(1000 * self.target_bandwidths[-1] // (self.frame_rate * self.codebook_nbits))


__all__ = ["XcodecConfig"]
