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

"""YuE2 acoustic tokenizer model configuration"""

import math
from dataclasses import field

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ...utils.type_validators import positive_int_field


@auto_docstring(checkpoint="m-a-p/YuE2-Vae")
@strict
class YuE2AcousticTokenizerEncoderConfig(PreTrainedConfig):
    r"""
    audio_channels (`int`, *optional*, defaults to 2):
        Number of channels of the input audio; the released checkpoint is stereo.
    hidden_sizes (`list[int]`, *optional*, defaults to `[64, 64, 128, 256, 512, 1024, 2048]`):
        Number of channels between the encoder's layers: the output of its input convolution, then the output of each
        block.
    downsampling_ratios (`list[int]`, *optional*, defaults to `[2, 2, 4, 4, 5, 6]`):
        Strides of the encoder blocks. Their product is the number of audio samples per latent frame (`hop_length`).
    latent_dim (`int`, *optional*, defaults to 64):
        Dimension of the continuous latents. The encoder outputs twice as many channels, the mean and the scale of the
        posterior.

    Example:

    ```python
    >>> from transformers import YuE2AcousticTokenizerEncoderConfig, YuE2AcousticTokenizerEncoderModel

    >>> # Initializing a configuration
    >>> configuration = YuE2AcousticTokenizerEncoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = YuE2AcousticTokenizerEncoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "yue2_acoustic_tokenizer_encoder"
    base_config_key = "encoder_config"

    audio_channels: int = positive_int_field(default=2)
    hidden_sizes: list[int] | tuple[int, ...] = (64, 64, 128, 256, 512, 1024, 2048)
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 2, 4, 4, 5, 6)
    latent_dim: int = positive_int_field(default=64)

    def __post_init__(self, **kwargs):
        # handling of the original checkpoint (m-a-p/YuE2-Vae)
        if "in_channels" in kwargs:
            self.audio_channels = kwargs.pop("in_channels")
            # the original counts the channels of both the mean and the scale of the posterior
            self.latent_dim //= 2
        if "c_mults" in kwargs:
            channels = kwargs.pop("channels")
            self.hidden_sizes = [channels * multiplier for multiplier in [1, *kwargs.pop("c_mults")]]
            self.downsampling_ratios = kwargs.pop("strides")
            if not kwargs.pop("use_snake", True):
                raise ValueError("Only the released variant, with SnakeBeta activations, is supported.")
        self.hidden_sizes = list(self.hidden_sizes)
        self.downsampling_ratios = list(self.downsampling_ratios)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if len(self.hidden_sizes) != len(self.downsampling_ratios) + 1:
            raise ValueError(
                f"`hidden_sizes` ({self.hidden_sizes}) must have one more entry than `downsampling_ratios` "
                f"({self.downsampling_ratios}): the input convolution's width, then each block's."
            )

    @property
    def hop_length(self) -> int:
        return math.prod(self.downsampling_ratios)


@auto_docstring(checkpoint="m-a-p/YuE2-Vae")
@strict
class YuE2AcousticTokenizerDecoderConfig(YuE2AcousticTokenizerEncoderConfig):
    r"""
    audio_channels (`int`, *optional*, defaults to 2):
        Number of channels of the output audio; the released checkpoint is stereo.
    hidden_sizes (`list[int]`, *optional*, defaults to `[64, 64, 128, 256, 512, 1024, 2048]`):
        Number of channels between the layers of the encoder, which the decoder goes through in reverse order: the
        input of its last convolution, then the input of each block from the last.
    downsampling_ratios (`list[int]`, *optional*, defaults to `[2, 2, 4, 4, 5, 6]`):
        Strides of the encoder blocks; the decoder blocks upsample by them in reverse order. Their product is the number
        of audio samples per latent frame (`hop_length`).
    latent_dim (`int`, *optional*, defaults to 64):
        Dimension of the continuous latents.

    Example:

    ```python
    >>> from transformers import YuE2AcousticTokenizerDecoderConfig, YuE2AcousticTokenizerDecoderModel

    >>> # Initializing a configuration
    >>> configuration = YuE2AcousticTokenizerDecoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = YuE2AcousticTokenizerDecoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "yue2_acoustic_tokenizer_decoder"
    base_config_key = "decoder_config"

    def __post_init__(self, **kwargs):
        # handling of the original checkpoint (m-a-p/YuE2-Vae)
        if "out_channels" in kwargs:
            self.audio_channels = kwargs.pop("out_channels")
            if kwargs.pop("final_tanh", False) or kwargs.pop("use_filter", False):
                raise ValueError("Only the released variant, without final `tanh` nor filter, is supported.")
            kwargs.pop("snake_type", None)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="m-a-p/YuE2-Vae")
@strict
class YuE2AcousticTokenizerConfig(PreTrainedConfig):
    r"""
    encoder_config (`Union[YuE2AcousticTokenizerEncoderConfig, dict]`, *optional*):
        Configuration of the encoder. Set it to `None` to drop the encoder, e.g. to only decode latents.
    decoder_config (`Union[YuE2AcousticTokenizerDecoderConfig, dict]`, *optional*):
        Configuration of the decoder. Set it to `None` to drop the decoder, e.g. to only encode audio.
    sampling_rate (`int`, *optional*, defaults to 48000):
        Sampling rate of the audio, in Hz.

    Example:

    ```python
    >>> from transformers import YuE2AcousticTokenizerConfig, YuE2AcousticTokenizerModel

    >>> # Initializing a configuration
    >>> configuration = YuE2AcousticTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = YuE2AcousticTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Its decoder alone
    >>> decoder_only = YuE2AcousticTokenizerModel(YuE2AcousticTokenizerConfig(encoder_config=None))
    ```
    """

    # the model type of the original checkpoint (m-a-p/YuE2-Vae), so that it loads directly
    model_type = "yue2_vae"
    sub_configs_defaults = {
        "encoder_config": SubConfigSpec(config_class=YuE2AcousticTokenizerEncoderConfig, optional=True),
        "decoder_config": SubConfigSpec(config_class=YuE2AcousticTokenizerDecoderConfig, optional=True),
    }

    # a half is dropped when set to `None`, so each defaults to an empty dict: the default configuration of the half
    encoder_config: YuE2AcousticTokenizerEncoderConfig | dict | None = field(default_factory=dict)
    decoder_config: YuE2AcousticTokenizerDecoderConfig | dict | None = field(default_factory=dict)
    sampling_rate: int = positive_int_field(default=48000)

    def __post_init__(self, **kwargs):
        # handling of the original checkpoint (m-a-p/YuE2-Vae)
        if "sample_rate" in kwargs:
            self.sampling_rate = kwargs.pop("sample_rate")
            for key in (
                "audio_channels",
                "latent_dim",
                "downsampling_ratio",
                "release_variant",
                "decode_core_frames",
                "decode_halo_frames",
            ):
                kwargs.pop(key, None)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.encoder_config is None and self.decoder_config is None:
            raise ValueError(
                "YuE2AcousticTokenizer needs an encoder, a decoder, or both: `encoder_config` and `decoder_config` are None."
            )

    @property
    def hop_length(self) -> int:
        return (self.encoder_config or self.decoder_config).hop_length

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.hop_length


__all__ = ["YuE2AcousticTokenizerConfig", "YuE2AcousticTokenizerEncoderConfig", "YuE2AcousticTokenizerDecoderConfig"]
