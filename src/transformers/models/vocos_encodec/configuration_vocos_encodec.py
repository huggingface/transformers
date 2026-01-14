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
"""VocosEncodec model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class VocosEncodecConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VocosEncodecConfig`]. It is used to
    instantiate a Vocos vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    [charactr/vocos-encodec-24khz](https://huggingface.co/charactr/vocos-encodec-24khz).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        codebook_dim (`int`, *optional*, defaults to 128):
            Encodec codebook dimension. Codebook weights have shape `(num_quantizers, codebook_dim)`.
        num_quantizers (`int`, *optional*, defaults to 16384):
            Number of quantizers from the Encodec codebook. Codebook weights have shape `(num_quantizers, codebook_dim)`.
        hidden_size (`int`, *optional*, defaults to 384):
            Hidden dimension for the ConvNeXt backbone.
        intermediate_size (`int`, *optional*, defaults to 1152):
            Dimension of the feed-forward layers inside each ConvNeXt block.
        num_layers (`int`, *optional*, defaults to 8):
            Number of ConvNeXt blocks to stack.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for depthwise convolutions.
        padding (`int`, *optional*, defaults to 3):
            Padding applied to depthwise convolutions.
        layer_scale_init_value (`float`, *optional*, defaults to `1/8`):
            Initial value for layer-scale (if >0, enables per-block scaling).
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for all LayerNorm operations.
        n_fft (`int`, *optional*, defaults to 1280):
            FFT size for ISTFT in the decoder.
        hop_length (`int`, *optional*, defaults to 320):
            Hop length between STFT frames used in `VocosISTFTHead`.
        istft_padding (`str`, *optional*, defaults to `"same"`):
            Padding mode for spectrogram inversion (`"center"` or `"same"`).
        sample_rate (`int`, *optional*, defaults to 24000):
            The sample rate at which the audio waveform should be digitalized expressed in Hertz (Hz).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function used inside the model.
        bandwidths (`List[float]`, *optional*, defaults to `[1.5, 3.0, 6.0, 12.0]`):
            Supported target bandwidths in kbps, This determines the number of quantizers/codebooks used in RVQ part of
            EnCodec, namely [1.5, 3, 6, 12].

    Example:

    ```python
    >>> from transformers import VocosEncodecModel, VocosEncodecConfig
    >>> config = VocosEncodecConfig()
    >>> model = VocosEncodecModel(config)
    ```
    """

    model_type = "vocos_encodec"

    def __init__(
        self,
        codebook_dim=128,
        num_quantizers=16384,
        hidden_size=384,
        intermediate_size=1152,
        num_layers=8,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=0.125,
        layer_norm_eps=1e-6,
        n_fft=1280,
        hop_length=320,
        istft_padding="same",
        sample_rate=24000,
        hidden_act="gelu",
        bandwidths=[1.5, 3.0, 6.0, 12.0],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers
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
        self.bandwidths = bandwidths


__all__ = ["VocosEncodecConfig"]
