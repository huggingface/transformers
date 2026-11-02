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
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.type_validators import positive_int_field
from ..dac.modeling_dac import DacDecoderBlock, DacDecoderOutput, DacEncoderBlock, DacResidualUnit
from ..pocket_tts_audio_tokenizer.configuration_pocket_tts_audio_tokenizer import (
    PocketTTSAudioTokenizerDecoderConfig,
    PocketTTSAudioTokenizerEncoderConfig,
)
from ..pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer import PocketTTSAudioTokenizerModel
from ..qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniSnakeBeta


@auto_docstring(checkpoint="bezzam/yue2-acoustic-tokenizer")
@strict
class YuE2AcousticTokenizerConfig(PreTrainedConfig):
    r"""
    encoder_config (`None`, *optional*):
        Set to `None` to drop the encoder, e.g. to only decode latents. Otherwise it is derived from the other fields,
        which the encoder and the decoder share.
    decoder_config (`None`, *optional*):
        Set to `None` to drop the decoder, e.g. to only encode audio. Otherwise it is derived from the other fields.
    audio_channels (`int`, *optional*, defaults to 2):
        Number of channels of the audio the encoder takes and the decoder returns; the released checkpoint is stereo.
    hidden_sizes (`list[int]`, *optional*, defaults to `[64, 64, 128, 256, 512, 1024, 2048]`):
        Number of channels between the encoder's layers: the output of its input convolution, then the output of each
        block. The decoder goes through them in reverse order.
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

    >>> # Its decoder alone
    >>> decoder_only = YuE2AcousticTokenizerModel(YuE2AcousticTokenizerConfig(encoder_config=None))
    ```
    """

    model_type = "yue2_acoustic_tokenizer"

    sampling_rate: int = positive_int_field(default=48000)
    audio_channels: int = positive_int_field(default=2)
    hidden_sizes: list[int] | tuple[int, ...] = (64, 64, 128, 256, 512, 1024, 2048)
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 2, 4, 4, 5, 6)
    latent_dim: int = positive_int_field(default=64)

    def __post_init__(self, **kwargs):
        self.hidden_sizes = list(self.hidden_sizes)
        self.downsampling_ratios = list(self.downsampling_ratios)
        # set to `False` by `encoder_config=None` / `decoder_config=None`, see `encoder_config`
        self._has_encoder = self._has_decoder = True
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if len(self.hidden_sizes) != len(self.downsampling_ratios) + 1:
            raise ValueError(
                f"`hidden_sizes` ({self.hidden_sizes}) must have one more entry than `downsampling_ratios` "
                f"({self.downsampling_ratios}): the input convolution's width, then each block's."
            )

    @property
    def hop_length(self) -> int:
        """Number of audio samples per latent frame."""
        return math.prod(self.downsampling_ratios)

    @property
    def frame_rate(self) -> float:
        """Number of latent frames per second."""
        return self.sampling_rate / self.hop_length

    # derived, not `sub_configs`: the halves share every field, and YuE2AcousticTokenizer checkpoints are flat
    @property
    def encoder_config(self) -> "YuE2AcousticTokenizerEncoderConfig | None":
        if not self._has_encoder:
            return None
        return YuE2AcousticTokenizerEncoderConfig(**self.to_dict())

    @encoder_config.setter
    def encoder_config(self, value: None):
        if value is not None:
            raise ValueError(
                "`encoder_config` is derived from the fields of `YuE2AcousticTokenizerConfig`: it can only be set to `None`."
            )
        if not self._has_decoder:
            raise ValueError(
                "YuE2AcousticTokenizer needs an encoder, a decoder, or both: the decoder is already dropped."
            )
        self._has_encoder = False

    @property
    def decoder_config(self) -> "YuE2AcousticTokenizerDecoderConfig | None":
        if not self._has_decoder:
            return None
        return YuE2AcousticTokenizerDecoderConfig(**self.to_dict())

    @decoder_config.setter
    def decoder_config(self, value: None):
        if value is not None:
            raise ValueError(
                "`decoder_config` is derived from the fields of `YuE2AcousticTokenizerConfig`: it can only be set to `None`."
            )
        if not self._has_encoder:
            raise ValueError(
                "YuE2AcousticTokenizer needs an encoder, a decoder, or both: the encoder is already dropped."
            )
        self._has_decoder = False

    def to_dict(self):
        output = super().to_dict()
        # a dropped half is saved as `"encoder_config": null`, loaded back through its setter
        for half in ("encoder", "decoder"):
            if not output.pop(f"_has_{half}", True):
                output[f"{half}_config"] = None
        return output


# trf-ignore: TRF057
class YuE2AcousticTokenizerEncoderConfig(PocketTTSAudioTokenizerEncoderConfig):
    """
    Configuration of the encoder half of YuE2AcousticTokenizer, from audio to continuous latents. It shares every field
    with [`YuE2AcousticTokenizerConfig`] (see its documentation for the arguments), so it can be loaded directly from a
    full YuE2AcousticTokenizer checkpoint.
    """


# trf-ignore: TRF057
class YuE2AcousticTokenizerDecoderConfig(PocketTTSAudioTokenizerDecoderConfig):
    """
    Configuration of the decoder half of YuE2AcousticTokenizer, from continuous latents to audio. It shares every field
    with [`YuE2AcousticTokenizerConfig`] (see its documentation for the arguments), so it can be loaded directly from a
    full YuE2AcousticTokenizer checkpoint.
    """


@auto_docstring
@dataclass
class YuE2AcousticTokenizerEncoderOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
        Continuous latents of the audio, one frame per `config.hop_length` audio samples: the posterior mean, or a
        sample of the posterior with `sample=True`.
    """

    latents: torch.FloatTensor | None = None


class YuE2AcousticTokenizerDecoderOutput(DacDecoderOutput):
    pass


@auto_docstring
@dataclass
class YuE2AcousticTokenizerOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
        Continuous latents of the audio.
    audio_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
        Audio waveform reconstructed from the latents.
    """

    latents: torch.FloatTensor | None = None
    audio_values: torch.FloatTensor | None = None


class YuE2AcousticTokenizerSnakeBeta(Qwen2_5OmniSnakeBeta):
    pass


class YuE2AcousticTokenizerResidualUnit(DacResidualUnit):
    """A residual unit: two SnakeBeta-activated convolutions (the first dilated) and a skip connection."""

    def __init__(self, dimension: int, dilation: int = 1):
        super().__init__(dimension, dilation)
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(dimension)
        self.snake2 = YuE2AcousticTokenizerSnakeBeta(dimension)


class YuE2AcousticTokenizerEncoderBlock(DacEncoderBlock):
    """Three residual units of growing dilation, then a strided convolution that downsamples by `stride`."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        nn.Module.__init__(self)
        self.res_unit1 = YuE2AcousticTokenizerResidualUnit(in_channels, dilation=1)
        self.res_unit2 = YuE2AcousticTokenizerResidualUnit(in_channels, dilation=3)
        self.res_unit3 = YuE2AcousticTokenizerResidualUnit(in_channels, dilation=9)
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )


class YuE2AcousticTokenizerDecoderBlock(DacDecoderBlock):
    """A transposed convolution that upsamples by `stride`, then three residual units of growing dilation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        nn.Module.__init__(self)
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(in_channels)
        self.conv_t1 = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )
        self.res_unit1 = YuE2AcousticTokenizerResidualUnit(out_channels, dilation=1)
        self.res_unit2 = YuE2AcousticTokenizerResidualUnit(out_channels, dilation=3)
        self.res_unit3 = YuE2AcousticTokenizerResidualUnit(out_channels, dilation=9)


@auto_docstring
class YuE2AcousticTokenizerPreTrainedModel(PreTrainedModel):
    config: YuE2AcousticTokenizerConfig
    base_model_prefix = "yue2_acoustic_tokenizer"
    main_input_name = "input_values"
    input_modalities = "audio"
    _no_split_modules = ["YuE2AcousticTokenizerResidualUnit"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, YuE2AcousticTokenizerSnakeBeta):
            init.zeros_(module.alpha)
            init.zeros_(module.beta)


@auto_docstring(
    custom_intro="""
    The encoder half of YuE2AcousticTokenizer, the Oobleck encoder: from audio to the continuous latents.
    """
)
class YuE2AcousticTokenizerEncoderModel(YuE2AcousticTokenizerPreTrainedModel):
    config: YuE2AcousticTokenizerEncoderConfig
    base_model_prefix = "encoder"
    _keys_to_ignore_on_load_unexpected = [r"^decoder\."]

    def __init__(self, config: YuE2AcousticTokenizerEncoderConfig):
        super().__init__(config)
        hidden_sizes = config.hidden_sizes
        self.conv1 = nn.Conv1d(config.audio_channels, hidden_sizes[0], kernel_size=7, padding=3)
        self.block = nn.ModuleList(
            [
                YuE2AcousticTokenizerEncoderBlock(hidden_sizes[index], hidden_sizes[index + 1], stride)
                for index, stride in enumerate(config.downsampling_ratios)
            ]
        )
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(hidden_sizes[-1])
        self.conv2 = nn.Conv1d(hidden_sizes[-1], 2 * config.latent_dim, kernel_size=3, padding=1)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, input_values: torch.FloatTensor, sample: bool = False, **kwargs: Unpack[TransformersKwargs]
    ) -> YuE2AcousticTokenizerEncoderOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
            Audio waveform at `config.sampling_rate`.
        sample (`bool`, *optional*, defaults to `False`):
            Whether to sample the latents from the posterior instead of returning its mean.
        """
        hidden_states = self.conv1(input_values)
        for layer in self.block:
            hidden_states = layer(hidden_states)
        hidden_states = self.conv2(self.snake1(hidden_states))

        mean, scale = hidden_states.chunk(2, dim=1)
        latents = mean
        if sample:
            std = nn.functional.softplus(scale) + 1e-4
            latents = mean + std * torch.randn_like(mean)
        return YuE2AcousticTokenizerEncoderOutput(latents=latents.transpose(1, 2))


@auto_docstring(
    custom_intro="""
    The decoder half of YuE2AcousticTokenizer, the Oobleck decoder: from continuous latents to audio. Unlike DAC's
    decoder, the last convolution has no `tanh` after it.
    """
)
class YuE2AcousticTokenizerDecoderModel(YuE2AcousticTokenizerPreTrainedModel):
    config: YuE2AcousticTokenizerDecoderConfig
    main_input_name = "latents"
    base_model_prefix = "decoder"
    _keys_to_ignore_on_load_unexpected = [r"^encoder\."]

    def __init__(self, config: YuE2AcousticTokenizerDecoderConfig):
        super().__init__(config)
        hidden_sizes = config.hidden_sizes
        self.conv1 = nn.Conv1d(config.latent_dim, hidden_sizes[-1], kernel_size=7, padding=3)
        self.block = nn.ModuleList(
            [
                YuE2AcousticTokenizerDecoderBlock(hidden_sizes[index + 1], hidden_sizes[index], stride)
                for index, stride in reversed(list(enumerate(config.downsampling_ratios)))
            ]
        )
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(hidden_sizes[0])
        self.conv2 = nn.Conv1d(hidden_sizes[0], config.audio_channels, kernel_size=7, padding=3, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, latents: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> YuE2AcousticTokenizerDecoderOutput:
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
            Continuous latents to decode, one frame per `config.hop_length` audio samples.
        """
        hidden_states = self.conv1(latents.transpose(1, 2))
        for layer in self.block:
            hidden_states = layer(hidden_states)
        audio_values = self.conv2(self.snake1(hidden_states))
        return YuE2AcousticTokenizerDecoderOutput(audio_values=audio_values)


@auto_docstring(
    custom_intro="""
    The YuE2 acoustic tokenizer: the Oobleck variational autoencoder of stable-audio-tools, which maps 48 kHz stereo
    audio to continuous 64-dimensional latents at 25 Hz and back. YuE2 generates these latents with flow matching.
    """
)
class YuE2AcousticTokenizerModel(PocketTTSAudioTokenizerModel):
    def __init__(self, config: YuE2AcousticTokenizerConfig):
        super().__init__(config)
        del self.latent_proj

    @can_return_tuple
    def encode(self, input_values: torch.FloatTensor, sample: bool = False) -> YuE2AcousticTokenizerEncoderOutput:
        """
        Encodes audio into continuous latents, with [`YuE2AcousticTokenizerEncoderModel.forward`].

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
                Audio waveform at `config.sampling_rate`.
            sample (`bool`, *optional*, defaults to `False`):
                Whether to sample the latents from the posterior instead of returning its mean.
        """
        if self.encoder is None:
            raise ValueError(
                "Audio encoding was requested, but the model was initialized without an encoder (`encoder_config=None`)."
            )
        return self.encoder(input_values, sample=sample, return_dict=True)

    @can_return_tuple
    def decode(self, latents: torch.FloatTensor) -> YuE2AcousticTokenizerDecoderOutput:
        """
        Decodes continuous latents into audio, with [`YuE2AcousticTokenizerDecoderModel.forward`].

        Args:
            latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
                Continuous latents to decode, one frame per `config.hop_length` audio samples.
        """
        if self.decoder is None:
            raise ValueError(
                "Audio decoding was requested, but the model was initialized without a decoder (`decoder_config=None`)."
            )
        return self.decoder(latents, return_dict=True)

    @can_return_tuple
    @auto_docstring
    def forward(
        self, input_values: torch.FloatTensor, sample: bool = False, **kwargs: Unpack[TransformersKwargs]
    ) -> YuE2AcousticTokenizerOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
            Audio waveform at `config.sampling_rate`.
        sample (`bool`, *optional*, defaults to `False`):
            Whether to sample the latents from the posterior instead of using its mean.
        """
        latents = self.encode(input_values, sample=sample).latents
        audio_values = self.decode(latents).audio_values
        return YuE2AcousticTokenizerOutput(latents=latents, audio_values=audio_values)


__all__ = [
    "YuE2AcousticTokenizerConfig",
    "YuE2AcousticTokenizerEncoderConfig",
    "YuE2AcousticTokenizerDecoderConfig",
    "YuE2AcousticTokenizerPreTrainedModel",
    "YuE2AcousticTokenizerEncoderModel",
    "YuE2AcousticTokenizerDecoderModel",
    "YuE2AcousticTokenizerModel",
]
