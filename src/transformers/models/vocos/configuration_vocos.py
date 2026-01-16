# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Vocos model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class VocosConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VocosModel`]. It is used to
    instantiate a Vocos vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        n_mels (`int`, *optional*, defaults to 100):
            Number of mel filterbanks.
        hidden_size (`int`, *optional*, defaults to 512):
            Hidden dimension for the ConvNeXt backbone.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimension of the feed-forward layers inside each ConvNeXt block.
        num_layers (`int`, *optional*, defaults to 8):
            Number of ConvNeXt blocks to stack.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for depthwise convolutions.
        padding (`int`, *optional*, defaults to 3):
            Padding applied to depthwise convolutions.
        layer_scale_init_value (`float`, *optional*, defaults to 0.125):
            Initial value for layer-scale (if >0, enables per-block scaling).
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for all LayerNorm operations.
        n_fft (`int`, *optional*, defaults to 1024):
            FFT size for ISTFT in the decoder.
        hop_length (`int`, *optional*, defaults to 256):
            Hop length between STFT frames used in `VocosISTFTHead`.
        istft_padding (`str`, *optional*, defaults to `"center"`):
            Padding mode for spectrogram inversion (`"center"` or `"same"`).
        sample_rate (`int`, *optional*, defaults to 24000):
            The sample rate at which the audio waveform should be digitalized expressed in Hertz (Hz).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function used inside the model.

    Example:

    ```python
    >>> from transformers import VocosModel, VocosConfig
    >>> config = VocosConfig()
    >>> model = VocosModel(config)
    ```
    """

    model_type = "vocos"

    def __init__(
        self,
        n_mels=100,
        hidden_size=512,
        intermediate_size=1536,
        num_layers=8,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=0.125,
        layer_norm_eps=1e-6,
        n_fft=1024,
        hop_length=256,
        istft_padding="center",
        sample_rate=24000,
        hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.layer_scale_init_value = layer_scale_init_value
        self.layer_norm_eps = layer_norm_eps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.istft_padding = istft_padding
        self.sample_rate = sample_rate
        self.hidden_act = hidden_act


__all__ = ["VocosConfig"]
