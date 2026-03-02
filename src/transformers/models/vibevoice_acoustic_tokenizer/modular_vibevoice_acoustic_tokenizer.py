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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ... import initialization as init
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ..auto.modeling_auto import AutoModel
from ..llama.modeling_llama import LlamaRMSNorm
from ..voxtral_realtime.modeling_voxtral_realtime import VoxtralRealtimeConv1dPaddingCache
from .configuration_vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerConfig


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerOutput(ModelOutput):
    r"""
    audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
        Decoded audio.
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed):
        A [`VibeVoiceAcousticTokenizerConv1dPaddingCache`] instance containing cached convolution states for each decoder
        layer that can be passed to subsequent forward calls.
    """

    audio: torch.FloatTensor | None = None
    latents: torch.FloatTensor | None = None
    padding_cache: Optional["VibeVoiceAcousticTokenizerConv1dPaddingCache"] = None


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerEncoderOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed):
        A [`VibeVoiceAcousticTokenizerConv1dPaddingCache`] instance containing cached convolution states for each encoder
        layer that can be passed to subsequent forward calls.
    """

    latents: torch.FloatTensor | None = None
    padding_cache: Optional["VibeVoiceAcousticTokenizerConv1dPaddingCache"] = None


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerDecoderOutput(ModelOutput):
    r"""
    audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
        Decoded audio.
    padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed):
        A [`VibeVoiceAcousticTokenizerConv1dPaddingCache`] instance containing cached convolution states for each decoder
        layer that can be passed to subsequent forward calls.
    """

    audio: torch.FloatTensor | None = None
    padding_cache: Optional["VibeVoiceAcousticTokenizerConv1dPaddingCache"] = None


class VibeVoiceAcousticTokenizerRMSNorm(LlamaRMSNorm):
    pass


class VibeVoiceAcousticTokenizerFeedForward(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, config.ffn_expansion * hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.ffn_expansion * hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.linear2(self.activation(self.linear1(hidden_states)))


class VibeVoiceAcousticTokenizerConv1dPaddingCache(VoxtralRealtimeConv1dPaddingCache):
    pass


# TODO: @eustlb, @ebezzam this should be latter factorized with other causalconv1d (e.g. VoxtralRealtimeCausalConv1d)
class VibeVoiceAcousticTokenizerCausalConv1d(nn.Module):
    """Conv1d with built-in causal padding and optional streaming support through a cache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups)
        self.causal_padding = (kernel_size - 1) * dilation - (stride - 1)
        if self.causal_padding < 0:
            raise ValueError(
                f"Invalid causal padding {self.causal_padding} for kernel_size={kernel_size}, "
                f"dilation={dilation}, stride={stride}."
            )
        self.cache_key = cache_key
        self.in_channels = in_channels
        self.left_pad = self.causal_padding

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: VibeVoiceAcousticTokenizerConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        if padding_cache is not None:
            hidden_states = padding_cache.update(hidden_states, self.cache_key, self)
        else:
            hidden_states = nn.functional.pad(hidden_states, (self.left_pad, 0))

        return self.conv(hidden_states)


class VibeVoiceAcousticTokenizerCausalConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in causal padding and optional streaming support through a cache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

        self.stride = stride
        self.cache_key = cache_key
        self.in_channels = in_channels
        self.padding_total = kernel_size - stride
        self.causal_padding = kernel_size - 1
        self.left_pad = self.causal_padding

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: Optional["VibeVoiceAcousticTokenizerConv1dPaddingCache"] = None,
    ) -> torch.Tensor:
        time_dim = hidden_states.shape[-1]

        if padding_cache is not None:
            hidden_states = padding_cache.update(hidden_states, self.cache_key, self)
        hidden_states = self.convtr(hidden_states)

        # Remove extra padding at the right side
        if self.padding_total > 0:
            hidden_states = hidden_states[..., : -self.padding_total]

        if padding_cache is not None:
            # For first chunk return full output, for subsequent chunks return only new output
            expected_new_output = time_dim * self.stride
            if hidden_states.shape[2] >= expected_new_output:
                hidden_states = hidden_states[:, :, -expected_new_output:]
        return hidden_states


class VibeVoiceAcousticTokenizerConvNext1dLayer(nn.Module):
    """ConvNeXt-like block adapted for 1D convolutions."""

    def __init__(self, config, hidden_size, dilation=1, stride=1, layer_idx=None):
        super().__init__()

        self.norm = VibeVoiceAcousticTokenizerRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = VibeVoiceAcousticTokenizerRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn = VibeVoiceAcousticTokenizerFeedForward(config, hidden_size)
        self.gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)
        self.ffn_gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)
        self.mixer = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=config.kernel_size,
            cache_key=f"convnext_layer_{layer_idx}",
            groups=hidden_size,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, hidden_states, padding_cache=None):
        # mixer
        residual = hidden_states
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.mixer(hidden_states, padding_cache=padding_cache)
        hidden_states = hidden_states * self.gamma.unsqueeze(-1)
        hidden_states = residual + hidden_states

        # ffn
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states.transpose(1, 2))
        hidden_states = self.ffn(hidden_states).transpose(1, 2)
        hidden_states = hidden_states * self.ffn_gamma.unsqueeze(-1)
        return residual + hidden_states


class VibeVoiceAcousticTokenizerEncoderStem(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=config.channels,
            out_channels=config.num_filters,
            kernel_size=config.kernel_size,
            cache_key="encoder_stem",
        )
        self.stage = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerConvNext1dLayer(
                    config,
                    hidden_size=config.num_filters,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(1, config.depths[0] + 1)
            ]
        )

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


class VibeVoiceAcousticTokenizerEncoderLayer(nn.Module):
    def __init__(self, config, stage_idx):
        super().__init__()

        depth_idx = stage_idx + 1  # first depth is for stem layer
        layer_idx = sum(depth + 1 for depth in config.depths[:depth_idx])
        intermediate_channels = int(config.num_filters * (2 ** (depth_idx)))

        self.conv = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=int(config.num_filters * (2**stage_idx)),
            out_channels=intermediate_channels,
            kernel_size=int(config.downsampling_ratios[stage_idx] * 2),
            cache_key=f"encoder_layer_{stage_idx}",
            stride=config.downsampling_ratios[stage_idx],
        )
        self.stage = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerConvNext1dLayer(
                    config, hidden_size=intermediate_channels, layer_idx=layer_idx + offset
                )
                for offset in range(1, config.depths[depth_idx] + 1)
            ]
        )

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


@auto_docstring
class VibeVoiceAcousticTokenizerPreTrainedModel(PreTrainedModel):
    config: VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"
    main_input_name = "input_values"
    _no_split_modules = ["VibeVoiceAcousticTokenizerEncoderModel", "VibeVoiceAcousticTokenizerDecoderModel"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, VibeVoiceAcousticTokenizerConvNext1dLayer):
            init.constant_(module.gamma, self.config.layer_scale_init_value)
            init.constant_(module.ffn_gamma, self.config.layer_scale_init_value)


class VibeVoiceAcousticTokenizerEncoderModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.stem = VibeVoiceAcousticTokenizerEncoderStem(config)
        self.conv_layers = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerEncoderLayer(config, stage_idx)
                for stage_idx in range(len(config.downsampling_ratios))
            ]
        )
        self.head = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=int(config.num_filters * (2 ** len(config.downsampling_ratios))),
            out_channels=config.hidden_size,
            kernel_size=config.kernel_size,
            cache_key="encoder_head",
        )
        self.post_init()

    def forward(self, hidden_states, padding_cache=None, use_cache=False, **kwargs):
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceAcousticTokenizerConv1dPaddingCache()

        hidden_states = self.stem(hidden_states, padding_cache=padding_cache)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        hidden_states = self.head(hidden_states, padding_cache=padding_cache)
        latents = hidden_states.permute(0, 2, 1)
        return VibeVoiceAcousticTokenizerEncoderOutput(latents=latents, padding_cache=padding_cache)


class VibeVoiceAcousticTokenizerDecoderStem(nn.Module):
    def __init__(self, config):
        super().__init__()

        intermediate_channels = int(config.num_filters * 2 ** (len(config.depths) - 1))
        self.conv = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=intermediate_channels,
            kernel_size=config.kernel_size,
            cache_key="decoder_stem",
        )
        self.stage = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerConvNext1dLayer(
                    config,
                    hidden_size=intermediate_channels,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(1, config.depths[0] + 1)
            ]
        )

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


class VibeVoiceAcousticTokenizerDecoderLayer(nn.Module):
    def __init__(self, config, stage_idx):
        super().__init__()

        depth_idx = stage_idx + 1  # first depth is for stem layer
        layer_idx = sum(depth + 1 for depth in config.depths[:depth_idx])
        intermediate_channels = int(config.num_filters * (2 ** (len(config.depths) - 2 - stage_idx)))

        self.convtr = VibeVoiceAcousticTokenizerCausalConvTranspose1d(
            in_channels=int(config.num_filters * (2 ** (len(config.depths) - 1 - stage_idx))),
            out_channels=intermediate_channels,
            kernel_size=int(config.upsampling_ratios[stage_idx] * 2),
            cache_key=f"decoder_layer_{stage_idx}",
            stride=config.upsampling_ratios[stage_idx],
        )
        self.stage = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerConvNext1dLayer(
                    config, hidden_size=intermediate_channels, layer_idx=layer_idx + offset
                )
                for offset in range(1, config.depths[depth_idx] + 1)
            ]
        )

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.convtr(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


class VibeVoiceAcousticTokenizerDecoderModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.stem = VibeVoiceAcousticTokenizerDecoderStem(config)
        self.conv_layers = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerDecoderLayer(config, stage_idx)
                for stage_idx in range(len(config.upsampling_ratios))
            ]
        )
        self.head = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=config.num_filters,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            cache_key="decoder_head",
        )
        self.post_init()

    def forward(self, hidden_states, padding_cache=None, use_cache=False, **kwargs):
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceAcousticTokenizerConv1dPaddingCache()

        hidden_states = self.stem(hidden_states, padding_cache=padding_cache)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        hidden_states = self.head(hidden_states, padding_cache=padding_cache)
        return VibeVoiceAcousticTokenizerDecoderOutput(audio=hidden_states, padding_cache=padding_cache)


@auto_docstring(
    custom_intro="""
    VibeVoice acoustic tokenizer with an encoder and decoder for continuous acoustic tokens.
    """
)
class VibeVoiceAcousticTokenizerModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.decoder = AutoModel.from_config(config.decoder_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def encode(self, input_values, padding_cache=None, use_cache=None, sample=True):
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representation.
        padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        sample (`bool`, *optional*):
            Whether to sample from the VAE. If False, no noise is added.
        """
        encoder_output = self.encoder(input_values, padding_cache=padding_cache, use_cache=use_cache)

        if sample:
            noise_std = self.config.vae_std * torch.randn(
                encoder_output.latents.shape[0],
                device=encoder_output.latents.device,
                dtype=encoder_output.latents.dtype,
            )
            encoder_output.latents = encoder_output.latents + noise_std[:, None, None] * torch.randn_like(
                encoder_output.latents
            )
        return encoder_output

    @can_return_tuple
    @auto_docstring
    def decode(self, latents, padding_cache=None, use_cache=False):
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input latent representation to be decoded back into audio.
        padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        latents = latents.permute(0, 2, 1)
        return self.decoder(latents, padding_cache=padding_cache, use_cache=use_cache)

    @can_return_tuple
    @auto_docstring
    def forward(self, input_values, padding_cache=None, use_cache=False, sample=True, **kwargs):
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representation.
        padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers. Note only used by decoder.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        sample (`bool`, *optional*):
            Whether to sample from the VAE latent distribution. If False, no noise is added to the latents.
        """
        encoder_output = self.encode(input_values, use_cache=use_cache, sample=sample)
        decoder_output = self.decode(encoder_output.latents, padding_cache=padding_cache, use_cache=use_cache)
        return VibeVoiceAcousticTokenizerOutput(
            audio=decoder_output.audio,
            latents=encoder_output.latents,
            padding_cache=decoder_output.padding_cache,
        )


__all__ = [
    "VibeVoiceAcousticTokenizerModel",
    "VibeVoiceAcousticTokenizerEncoderModel",
    "VibeVoiceAcousticTokenizerDecoderModel",
    "VibeVoiceAcousticTokenizerPreTrainedModel",
]
