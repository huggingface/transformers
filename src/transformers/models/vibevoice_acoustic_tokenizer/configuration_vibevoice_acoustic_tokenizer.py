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

import numpy as np

from ...configuration_utils import PretrainedConfig


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceAcousticTokenizerModel`]. It is used to
    instantiate a VibeVoice acoustic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    VibeVoice architecture.

    e.g. [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        channels (`int`, *optional*, defaults to 1):
            Number of input channels.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of latent representations.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for convolutional layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon value for RMSNorm layers.
        bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in convolution and feed-forward layers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            Initial value for layer scaling.
        initializer_range (`float`, *optional*, defaults to 0.01):
            Standard deviation for weight initialization.
        n_filters (`int`, *optional*, defaults to 32):
            Number of filters in initial convolutional layer, and doubles after each downsampling.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
            Downsampling ratios for each layer.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
            Number of ConvNeXt blocks at each stage.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to 4):
            Expansion factor for feed-forward networks.
        vae_std (`float`, *optional*, defaults to 0.625):
            Standard deviation used during VAE sampling.
    Example:

    ```python
    >>> from transformers import VibeVoiceAcousticTokenizerModel, VibeVoiceAcousticTokenizerConfig

    >>> # Initializing a VibeVoice Acoustic Tokenizer configuration
    >>> configuration = VibeVoiceAcousticTokenizerConfig()

    >>> # Initializing a model (with random weights)
    >>> model = VibeVoiceAcousticTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels=1,
        hidden_size=64,
        kernel_size=7,
        rms_norm_eps=1e-5,
        bias=True,
        layer_scale_init_value=1e-6,
        initializer_range=1e-2,
        n_filters=32,
        downsampling_ratios=[2, 2, 4, 5, 5, 8],
        depths=[3, 3, 3, 3, 3, 3, 8],
        hidden_act="gelu",
        ffn_expansion=4,
        vae_std=0.625,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.kernel_size = kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.bias = bias
        self.layer_scale_init_value = layer_scale_init_value
        self.ffn_expansion = ffn_expansion
        self.initializer_range = initializer_range
        self.n_filters = n_filters
        self.downsampling_ratios = downsampling_ratios
        self.depths = depths
        self.vae_std = vae_std

    @property
    def hop_length(self) -> int:
        return np.prod(self.downsampling_ratios)

    @property
    def upsampling_ratios(self):
        return self.downsampling_ratios[::-1]

    @property
    def decoder_depths(self):
        return self.depths[::-1]


__all__ = ["VibeVoiceAcousticTokenizerConfig"]
