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
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ..dac.modeling_dac import DacDecoderBlock, DacEncoder, DacEncoderBlock, DacResidualUnit
from .configuration_yue2_acoustic_tokenizer import YuE2AcousticTokenizerConfig


@auto_docstring
@dataclass
class YuE2AcousticTokenizerEncoderOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
        Continuous latents of the audio, one frame per `config.hop_length` audio samples: the posterior mean, or a
        sample of the posterior with `sample=True`.
    """

    latents: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class YuE2AcousticTokenizerDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
        Audio waveform decoded from the latents, unclipped.
    """

    audio_values: torch.FloatTensor | None = None


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


class YuE2AcousticTokenizerSnakeBeta(nn.Module):
    """
    The SnakeBeta activation of BigVGAN, `x + 1 / beta * sin(alpha * x) ** 2`, with a learnt `alpha` and `beta` per
    channel, both stored as logarithms.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        return hidden_states + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(hidden_states * alpha), 2)


class YuE2AcousticTokenizerResidualUnit(DacResidualUnit):
    """A residual unit: two SnakeBeta-activated convolutions (the first dilated) and a skip connection."""

    def __init__(self, dimension: int, dilation: int = 1):
        nn.Module.__init__(self)
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(dimension)
        self.conv1 = nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = YuE2AcousticTokenizerSnakeBeta(dimension)
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=1)


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


class YuE2AcousticTokenizerEncoder(DacEncoder):
    """The Oobleck encoder: audio to the mean and scale of the latent posterior."""

    def __init__(self, config: YuE2AcousticTokenizerConfig):
        nn.Module.__init__(self)
        channels = [config.hidden_size * multiplier for multiplier in [1] + config.channel_multipliers]
        self.conv1 = nn.Conv1d(config.audio_channels, channels[0], kernel_size=7, padding=3)
        self.block = nn.ModuleList(
            [
                YuE2AcousticTokenizerEncoderBlock(channels[index], channels[index + 1], stride)
                for index, stride in enumerate(config.downsampling_ratios)
            ]
        )
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(channels[-1])
        self.conv2 = nn.Conv1d(channels[-1], 2 * config.latent_dim, kernel_size=3, padding=1)


class YuE2AcousticTokenizerDecoder(nn.Module):
    """The Oobleck decoder: latents to audio, with no output nonlinearity."""

    def __init__(self, config: YuE2AcousticTokenizerConfig):
        super().__init__()
        channels = [config.hidden_size * multiplier for multiplier in [1] + config.channel_multipliers]
        self.conv1 = nn.Conv1d(config.latent_dim, channels[-1], kernel_size=7, padding=3)
        self.block = nn.ModuleList(
            [
                YuE2AcousticTokenizerDecoderBlock(channels[index + 1], channels[index], stride)
                for index, stride in reversed(list(enumerate(config.downsampling_ratios)))
            ]
        )
        self.snake1 = YuE2AcousticTokenizerSnakeBeta(channels[0])
        self.conv2 = nn.Conv1d(channels[0], config.audio_channels, kernel_size=7, padding=3, bias=False)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


@auto_docstring
class YuE2AcousticTokenizerPreTrainedModel(PreTrainedAudioTokenizerBase):
    config: YuE2AcousticTokenizerConfig
    base_model_prefix = "yue2_acoustic_tokenizer"
    main_input_name = "input_values"
    input_modalities = "audio"
    _no_split_modules = ["YuE2AcousticTokenizerResidualUnit"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, YuE2AcousticTokenizerSnakeBeta):
            # log-scale: alpha = beta = 1
            init.zeros_(module.alpha)
            init.zeros_(module.beta)


@auto_docstring(
    custom_intro="""
    The YuE2 acoustic tokenizer: the Oobleck variational autoencoder of stable-audio-tools, which maps 48 kHz stereo
    audio to continuous 64-dimensional latents at 25 Hz and back. YuE2 generates these latents with flow matching.
    """
)
class YuE2AcousticTokenizerModel(YuE2AcousticTokenizerPreTrainedModel):
    def __init__(self, config: YuE2AcousticTokenizerConfig):
        super().__init__(config)
        self.encoder = YuE2AcousticTokenizerEncoder(config)
        self.decoder = YuE2AcousticTokenizerDecoder(config)
        self.post_init()

    @can_return_tuple
    def encode(self, input_values: torch.FloatTensor, sample: bool = False) -> YuE2AcousticTokenizerEncoderOutput:
        """
        Encodes audio into continuous latents.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
                Audio waveform at `config.sampling_rate`.
            sample (`bool`, *optional*, defaults to `False`):
                Whether to sample the latents from the posterior instead of returning its mean.
        """
        mean, scale = self.encoder(input_values).chunk(2, dim=1)
        latents = mean
        if sample:
            std = nn.functional.softplus(scale) + 1e-4
            latents = mean + std * torch.randn_like(mean)
        return YuE2AcousticTokenizerEncoderOutput(latents=latents.transpose(1, 2))

    @can_return_tuple
    def decode(self, latents: torch.FloatTensor) -> YuE2AcousticTokenizerDecoderOutput:
        """
        Decodes continuous latents into audio.

        Args:
            latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
                Continuous latents to decode, one frame per `config.hop_length` audio samples.
        """
        audio_values = self.decoder(latents.transpose(1, 2))
        return YuE2AcousticTokenizerDecoderOutput(audio_values=audio_values)

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


__all__ = ["YuE2AcousticTokenizerModel", "YuE2AcousticTokenizerPreTrainedModel"]
