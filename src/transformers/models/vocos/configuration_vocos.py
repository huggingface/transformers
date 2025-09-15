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
"""Vocos model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class VocosConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VocosModel`]. It is used to
    instantiate a Vocos vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Manel/vocos-mel-24khz](https://huggingface.co/Manel/vocos-mel-24khz) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        input_channels (`int`, *optional*, defaults to 100):
            Number of mel-spectrogram input channels (i.e. number of mel filter bins).
        hidden_dim (`int`, *optional*, defaults to 512):
            Hidden dimension for the ConvNeXt backbone.
        intermediate_dim (`int`, *optional*, defaults to 1536):
            Dimension of the feed-forward layers inside each ConvNeXt block.
        num_layers (`int`, *optional*, defaults to 8):
            Number of ConvNeXt blocks to stack.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for depthwise convolutions.
        padding (`int`, *optional*, defaults to 3):
            Padding applied to those convolutions.
        layer_scale_init_value (`float`, *optional*, defaults to `1/8`):
            Initial value for layer-scale (if >0, enables per-block scaling).
        use_adaptive_norm (`bool`, *optional*, defaults to `False`):
            Whether to use adaptive layer normalization.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for all LayerNorm operations.
        n_fft (`int`, *optional*, defaults to 1024):
            FFT size for STFT/ISTFT used in `VocosISTFTHead`.
        hop_length (`int`, *optional*, defaults to 256):
            Hop length between STFT frames used in `VocosISTFTHead`.
        spec_padding (`str`, *optional*, defaults to `"center"`):
            Padding mode for spectrogram inversion (`"center"` or `"same"`).
        bandwidths (`List[float]`, *optional*, defaults to `[1.5, 3.0, 6.0, 12.0]`):
            Supported target bandwidths in kbps, This determines the number of quantizers/codebooks used in RVQ part of
            EnCodec, namely [2, 4, 6, 8].
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in Hertz (Hz).

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
        input_channels=100,
        hidden_dim=512,
        intermediate_dim=1536,
        num_layers=8,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1 / 8,
        use_adaptive_norm=False,
        layer_norm_eps=1e-6,
        n_fft=1024,
        hop_length=256,
        spec_padding="center",
        bandwidths=[1.5, 3.0, 6.0, 12.0],
        sampling_rate=24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.layer_scale_init_value = layer_scale_init_value
        self.use_adaptive_norm = use_adaptive_norm
        self.layer_norm_eps = layer_norm_eps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec_padding = spec_padding
        self.bandwidths = list(bandwidths)
        self.sampling_rate = sampling_rate


__all__ = ["VocosConfig"]
