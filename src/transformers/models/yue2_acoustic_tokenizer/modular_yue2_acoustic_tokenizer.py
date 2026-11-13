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
from torch import nn

from ... import initialization as init
from ...modeling_utils import PreTrainedAudioTokenizerBase, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoModel
from ..dac.modeling_dac import DacDecoderBlock, DacDecoderOutput, DacEncoderBlock, DacResidualUnit
from ..qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniSnakeBeta
from .configuration_yue2_acoustic_tokenizer import (
    YuE2AcousticTokenizerConfig,
    YuE2AcousticTokenizerDecoderConfig,
    YuE2AcousticTokenizerEncoderConfig,
)


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
            Audio waveform at the `sampling_rate` of [`YuE2AcousticTokenizerConfig`].
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
    The YuE2 acoustic tokenizer: the Oobleck variational autoencoder, which maps stereo audio to continuous latents
    and back.
    """
)
class YuE2AcousticTokenizerModel(YuE2AcousticTokenizerPreTrainedModel, PreTrainedAudioTokenizerBase):
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^decoder\."]

    def __init__(self, config: YuE2AcousticTokenizerConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config) if config.encoder_config is not None else None
        self.decoder = AutoModel.from_config(config.decoder_config) if config.decoder_config is not None else None
        self.post_init()

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
    "YuE2AcousticTokenizerPreTrainedModel",
    "YuE2AcousticTokenizerEncoderModel",
    "YuE2AcousticTokenizerDecoderModel",
    "YuE2AcousticTokenizerModel",
]
