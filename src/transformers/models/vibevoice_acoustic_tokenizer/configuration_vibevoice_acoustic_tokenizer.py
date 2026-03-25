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


import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PretrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/VibeVoice-1.5B")
@strict
class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    r"""
    channels (`int`, *optional*, defaults to 1):
        Number of input channels.
    hidden_size (`int`, *optional*, defaults to 64):
        Dimensionality of latent representations.
    kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for convolutional layers.
    num_filters (`int`, *optional*, defaults to 32):
        Number of filters in initial convolutional layer, and doubles after each downsampling.
    downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
        Downsampling ratios for each layer.
    depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
        Number of ConvNeXt blocks at each stage.
    ffn_expansion (`int`, *optional*, defaults to 4):
        Expansion factor for feed-forward networks.
    vae_std (`float`, *optional*, defaults to 0.625):
        Standard deviation used for VAE sampling after encoder.

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

    channels: int = 1
    hidden_size: int = 64
    kernel_size: int = 7
    rms_norm_eps: float = 1e-5
    layer_scale_init_value: float = 1e-6
    initializer_range: float = 1e-2
    num_filters: int = 32
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 2, 4, 5, 5, 8)
    depths: list[int] | tuple[int, ...] = (3, 3, 3, 3, 3, 3, 8)
    hidden_act: str = "gelu"
    ffn_expansion: int = 4
    vae_std: float = 0.625

    @property
    def hop_length(self):
        return int(math.prod(self.downsampling_ratios))

    @property
    def encoder_config(self):
        return VibeVoiceAcousticTokenizerEncoderConfig(**self.to_dict())

    @property
    def decoder_config(self):
        config_dict = self.to_dict()
        config_dict["depths"] = list(reversed(config_dict["depths"]))
        return VibeVoiceAcousticTokenizerDecoderConfig(**config_dict)


@auto_docstring(checkpoint="microsoft/VibeVoice-1.5B")
@strict
class VibeVoiceAcousticTokenizerEncoderConfig(VibeVoiceAcousticTokenizerConfig):
    r"""
    channels (`int`, *optional*, defaults to 1):
        Number of input channels.
    hidden_size (`int`, *optional*, defaults to 64):
        Dimensionality of latent representations.
    kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for convolutional layers.
    num_filters (`int`, *optional*, defaults to 32):
        Number of filters in initial convolutional layer, and doubles after each downsampling.
    downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
        Downsampling ratios for each layer.
    depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
        Number of ConvNeXt blocks at each stage.
    ffn_expansion (`int`, *optional*, defaults to 4):
        Expansion factor for feed-forward networks.
    vae_std (`float`, *optional*, defaults to 0.625):
        Standard deviation used for VAE sampling after encoder.
    """

    model_type = "vibevoice_acoustic_tokenizer_encoder"
    base_config_key = "encoder_config"

    @property
    def encoder_config(self):
        return None


@auto_docstring(checkpoint="microsoft/VibeVoice-1.5B")
@strict
class VibeVoiceAcousticTokenizerDecoderConfig(VibeVoiceAcousticTokenizerConfig):
    r"""
    channels (`int`, *optional*, defaults to 1):
        Number of input channels.
    hidden_size (`int`, *optional*, defaults to 64):
        Dimensionality of latent representations.
    kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for convolutional layers.
    num_filters (`int`, *optional*, defaults to 32):
        Number of filters in initial convolutional layer, and doubles after each downsampling.
    downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
        Downsampling ratios for each layer.
    depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
        Number of ConvNeXt blocks at each stage.
    ffn_expansion (`int`, *optional*, defaults to 4):
        Expansion factor for feed-forward networks.
    vae_std (`float`, *optional*, defaults to 0.625):
        Standard deviation used for VAE sampling after encoder.
    """

    model_type = "vibevoice_acoustic_tokenizer_decoder"
    base_config_key = "decoder_config"

    @property
    def encoder_config(self):
        return None

    @property
    def decoder_config(self):
        return None

    @property
    def upsampling_ratios(self):
        return list(reversed(self.downsampling_ratios))


__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceAcousticTokenizerEncoderConfig",
    "VibeVoiceAcousticTokenizerDecoderConfig",
]
