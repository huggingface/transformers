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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ...utils.type_validators import positive_int_field


@auto_docstring(checkpoint="bezzam/yue2-acoustic-tokenizer")
@strict
class YuE2AcousticTokenizerConfig(PreTrainedConfig):
    r"""
    audio_channels (`int`, *optional*, defaults to 2):
        Number of channels of the audio the encoder takes and the decoder returns; the released checkpoint is stereo.
    hidden_size (`int`, *optional*, defaults to 64):
        Number of channels of the outermost convolutions, at the audio rate. Each encoder block multiplies it by the
        next entry of `channel_multipliers`, and each decoder block divides it back.
    channel_multipliers (`list[int]`, *optional*, defaults to `[1, 2, 4, 8, 16, 32]`):
        Channel multiplier of each encoder block's output (and decoder block's input), relative to `hidden_size`. The
        first encoder block keeps `hidden_size` channels at its input.
    downsampling_ratios (`list[int]`, *optional*, defaults to `[2, 2, 4, 4, 5, 6]`):
        Strides of the encoder blocks; the decoder blocks upsample by them in reverse order. Their product is the number
        of audio samples per latent frame (`hop_length`).
    latent_dim (`int`, *optional*, defaults to 64):
        Dimension of the continuous latents. The encoder outputs twice as many channels, the mean and the scale of the
        posterior.

    Example:

    ```python
    >>> from transformers import YuE2AcousticTokenizerConfig, YuE2AcousticTokenizerModel

    >>> # Initializing a configuration
    >>> configuration = YuE2AcousticTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = YuE2AcousticTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "yue2_acoustic_tokenizer"

    sampling_rate: int = positive_int_field(default=48000)
    audio_channels: int = positive_int_field(default=2)
    hidden_size: int = positive_int_field(default=64)
    channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 2, 4, 4, 5, 6)
    latent_dim: int = positive_int_field(default=64)

    def __post_init__(self, **kwargs):
        self.channel_multipliers = list(self.channel_multipliers)
        self.downsampling_ratios = list(self.downsampling_ratios)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if len(self.channel_multipliers) != len(self.downsampling_ratios):
            raise ValueError(
                f"`channel_multipliers` ({self.channel_multipliers}) and `downsampling_ratios` "
                f"({self.downsampling_ratios}) must have one entry per block."
            )

    @property
    def hop_length(self) -> int:
        """Number of audio samples per latent frame."""
        return math.prod(self.downsampling_ratios)

    @property
    def frame_rate(self) -> float:
        """Number of latent frames per second."""
        return self.sampling_rate / self.hop_length


__all__ = ["YuE2AcousticTokenizerConfig"]
