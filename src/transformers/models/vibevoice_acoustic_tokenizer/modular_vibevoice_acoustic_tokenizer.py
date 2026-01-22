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

from ...activations import ACT2FN
from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ..llama.modeling_llama import LlamaRMSNorm
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from .configuration_vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerConfig


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerOutput(ModelOutput):
    r"""
    audio (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceAcousticTokenizerConv1dPaddingCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
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
    """

    latents: torch.FloatTensor | None = None


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerDecoderOutput(ModelOutput):
    r"""
    audio (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceAcousticTokenizerConv1dPaddingCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    audio: torch.FloatTensor | None = None
    padding_cache: Optional["VibeVoiceAcousticTokenizerConv1dPaddingCache"] = None


class VibeVoiceAcousticTokenizerRMSNorm(LlamaRMSNorm):
    pass


class VibeVoiceAcousticTokenizerFeedForward(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, config.ffn_expansion * hidden_size, bias=config.bias)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.ffn_expansion * hidden_size, hidden_size, bias=config.bias)

    def forward(self, hidden_states):
        return self.linear2(self.activation(self.linear1(hidden_states)))


class VibeVoiceAcousticTokenizerConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class VibeVoiceAcousticTokenizerCausalConv1d(nn.Module):
    """Conv1d with built-in causal padding and optional streaming support through a cache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )
        self.causal_padding = (kernel_size - 1) * dilation - (stride - 1)
        if self.causal_padding < 0:
            raise ValueError(
                f"Invalid causal padding {self.causal_padding} for kernel_size={kernel_size}, "
                f"dilation={dilation}, stride={stride}."
            )
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: VibeVoiceAcousticTokenizerConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        if padding_cache is not None:
            layer_padding = padding_cache.update(hidden_states, self.layer_idx)
        else:
            layer_padding = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.causal_padding,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        hidden_states = torch.cat([layer_padding, hidden_states], dim=-1)

        return self.conv(hidden_states)


class VibeVoiceAcousticTokenizerCausalConvTranspose1d(nn.Module):
    """Causal ConvTranspose1d with optional streaming support via VibeVoiceAcousticTokenizerConv1dPaddingCache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.stride = stride
        self.layer_idx = layer_idx
        self.padding_total = kernel_size - stride
        self.causal_padding = kernel_size - 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: Optional["VibeVoiceAcousticTokenizerConv1dPaddingCache"] = None,
    ) -> torch.Tensor:
        time_dim = hidden_states.shape[-1]

        if padding_cache is not None:
            layer_padding = padding_cache.update(hidden_states, self.layer_idx)
            hidden_states = torch.cat([layer_padding, hidden_states], dim=-1)
        hidden_states = self.convtr(hidden_states)

        # Remove extra padding at the right side
        if self.padding_total > 0:
            hidden_states = hidden_states[..., : -self.padding_total]

        if padding_cache is not None and layer_padding.shape[2] != 0:
            # For first chunk (layer_padding.shape[2] == 0) return full output
            # for subsequent chunks return only new output
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
            groups=hidden_size,
            bias=config.bias,
            dilation=dilation,
            stride=stride,
            layer_idx=layer_idx,
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
            out_channels=config.n_filters,
            kernel_size=config.kernel_size,
            bias=config.bias,
            layer_idx=0,
        )
        self.stage = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerConvNext1dLayer(
                    config,
                    hidden_size=config.n_filters,
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
        intermediate_channels = int(config.n_filters * (2 ** (depth_idx)))

        self.conv = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=int(config.n_filters * (2 ** stage_idx)),
            out_channels=intermediate_channels,
            kernel_size=int(config.downsampling_ratios[stage_idx] * 2),
            stride=config.downsampling_ratios[stage_idx],
            bias=config.bias,
            layer_idx=layer_idx,
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


class VibeVoiceAcousticTokenizerEncoder(nn.Module):
    """Encoder component for the VibeVoice tokenizer that converts audio to latent representations."""

    def __init__(self, config):
        super().__init__()

        self.stem = VibeVoiceAcousticTokenizerEncoderStem(config)
        self.conv_layers = nn.ModuleList(
            [VibeVoiceAcousticTokenizerEncoderLayer(config, stage_idx) for stage_idx in range(len(config.downsampling_ratios))]
        )
        self.head = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=int(config.n_filters * (2 ** len(config.downsampling_ratios))),
            out_channels=config.hidden_size,
            kernel_size=config.kernel_size,
            bias=config.bias,
            layer_idx=sum(depth + 1 for depth in config.depths),
        )

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.stem(hidden_states, padding_cache=padding_cache)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        hidden_states = self.head(hidden_states, padding_cache=padding_cache)
        return hidden_states.permute(0, 2, 1)


class VibeVoiceAcousticTokenizerDecoderStem(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        depth = config.decoder_depths[0]
        intermediate_channels = int(config.n_filters * 2 ** (len(config.decoder_depths) - 1))
        self.conv = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=intermediate_channels,
            kernel_size=config.kernel_size,
            bias=config.bias,
            layer_idx=layer_idx,
        )
        self.stage = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerConvNext1dLayer(
                    config,
                    hidden_size=intermediate_channels,
                    layer_idx=layer_idx + depth_idx + 1 if layer_idx is not None else None,
                )
                for depth_idx in range(depth)
            ]
        )
        self.num_layers = depth + 1

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


class VibeVoiceAcousticTokenizerDecoderLayer(nn.Module):
    def __init__(self, config, stage_idx):
        super().__init__()

        # `layer_idx` offset by stem layers and previous decoder layers
        layer_idx = config.decoder_depths[0] + 1
        for prev_stage_idx in range(stage_idx):
            layer_idx += config.decoder_depths[prev_stage_idx + 1] + 1

        depth = config.decoder_depths[stage_idx + 1]
        in_channels = int(config.n_filters * (2 ** (len(config.decoder_depths) - 1 - stage_idx)))
        intermediate_channels = int(config.n_filters * (2 ** (len(config.decoder_depths) - 2 - stage_idx)))
        self.convtr = VibeVoiceAcousticTokenizerCausalConvTranspose1d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=int(config.upsampling_ratios[stage_idx] * 2),
            stride=config.upsampling_ratios[stage_idx],
            bias=config.bias,
            layer_idx=layer_idx,
        )
        self.stage = nn.ModuleList(
            [
                VibeVoiceAcousticTokenizerConvNext1dLayer(
                    config, hidden_size=intermediate_channels, layer_idx=layer_idx + depth_idx + 1
                )
                for depth_idx in range(depth)
            ]
        )
        self.num_layers = depth + 1

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.convtr(hidden_states, padding_cache=padding_cache)
        for block in self.stage:
            hidden_states = block(hidden_states, padding_cache=padding_cache)
        return hidden_states


class VibeVoiceAcousticTokenizerDecoder(nn.Module):
    """Decoder component for the VibeVoice tokenizer that converts latent representations back to audio."""

    def __init__(self, config):
        super().__init__()

        layer_idx = 0
        self.stem = VibeVoiceAcousticTokenizerDecoderStem(config, layer_idx=layer_idx)
        layer_idx += self.stem.num_layers

        self.conv_layers = nn.ModuleList(
            VibeVoiceAcousticTokenizerDecoderLayer(config, stage_idx)
            for stage_idx in range(len(config.upsampling_ratios))
        )
        layer_idx += sum(layer.num_layers for layer in self.conv_layers)

        self.head = VibeVoiceAcousticTokenizerCausalConv1d(
            in_channels=config.n_filters,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            bias=config.bias,
            layer_idx=layer_idx,
        )

        # Parameters for cache creation
        self.num_conv_layers = layer_idx + 1
        self.per_conv_layer_padding_mode = ["constant"] * self.num_conv_layers
        self.per_conv_layer_padding = [self.stem.conv.causal_padding]
        self.per_conv_layer_in_channels = [self.stem.conv.conv.in_channels]
        for block in self.stem.stage:
            self.per_conv_layer_padding.append(block.mixer.causal_padding)
            self.per_conv_layer_in_channels.append(block.mixer.conv.in_channels)
        for layer in self.conv_layers:
            self.per_conv_layer_padding.append(layer.convtr.causal_padding)
            self.per_conv_layer_in_channels.append(layer.convtr.convtr.in_channels)
            for block in layer.stage:
                self.per_conv_layer_padding.append(block.mixer.causal_padding)
                self.per_conv_layer_in_channels.append(block.mixer.conv.in_channels)
        self.per_conv_layer_padding.append(self.head.causal_padding)
        self.per_conv_layer_in_channels.append(self.head.conv.in_channels)

    def forward(self, hidden_states, padding_cache=None):
        hidden_states = self.stem(hidden_states, padding_cache=padding_cache)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        hidden_states = self.head(hidden_states, padding_cache=padding_cache)
        return hidden_states


@auto_docstring
class VibeVoiceAcousticTokenizerPreTrainedModel(PreTrainedModel):
    config: VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"
    main_input_name = "audio"
    _no_split_modules = ["VibeVoiceAcousticTokenizerEncoder", "VibeVoiceAcousticTokenizerDecoder"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, VibeVoiceAcousticTokenizerConvNext1dLayer):
            init.constant_(module.gamma, self.config.layer_scale_init_value)
            init.constant_(module.ffn_gamma, self.config.layer_scale_init_value)


@auto_docstring
class VibeVoiceAcousticTokenizerModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    """VibeVoice speech tokenizer model combining encoder and decoder for acoustic tokens"""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = VibeVoiceAcousticTokenizerEncoder(config)
        self.decoder = VibeVoiceAcousticTokenizerDecoder(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def encode(self, audio):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        """
        latents = self.encoder(audio)
        return VibeVoiceAcousticTokenizerEncoderOutput(latents=latents)

    @can_return_tuple
    @auto_docstring
    def sample(self, latents):
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input latent representations to be sampled.
        """
        noise_std = self.config.vae_std * torch.randn(latents.shape[0], device=latents.device, dtype=latents.dtype)
        while noise_std.dim() < latents.dim():
            noise_std = noise_std.unsqueeze(-1)
        latents = latents + noise_std * torch.randn_like(latents)
        return VibeVoiceAcousticTokenizerEncoderOutput(latents=latents)

    @can_return_tuple
    @auto_docstring
    def decode(self, latents, padding_cache=None, use_cache=False):
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input latent representations to be decoded back into audio waveforms.
        padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceAcousticTokenizerConv1dPaddingCache(
                num_layers=self.decoder.num_conv_layers,
                per_layer_padding=self.decoder.per_conv_layer_padding,
                per_layer_padding_mode=self.decoder.per_conv_layer_padding_mode,
                per_layer_in_channels=self.decoder.per_conv_layer_in_channels,
            )

        latents = latents.permute(0, 2, 1)
        audio = self.decoder(latents, padding_cache=padding_cache)
        return VibeVoiceAcousticTokenizerDecoderOutput(audio=audio, padding_cache=padding_cache)

    @can_return_tuple
    @auto_docstring
    def forward(self, audio, padding_cache=None, use_cache=False, sample=True, **kwargs: Unpack[TransformersKwargs]):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceAcousticTokenizerConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers. Note only used by decoder.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        sample (`bool`, *optional*):
            Whether to sample from the output distribution of the encoder, or return the latent as is.
        """
        encoder_output = self.encode(audio)
        if sample:
            encoder_output = self.sample(encoder_output.latents)
        decoder_output = self.decode(encoder_output.latents, padding_cache=padding_cache, use_cache=use_cache)
        return VibeVoiceAcousticTokenizerOutput(
            audio=decoder_output.audio,
            latents=encoder_output.latents,
            padding_cache=decoder_output.padding_cache,
        )


__all__ = [
    "VibeVoiceAcousticTokenizerModel",
    "VibeVoiceAcousticTokenizerPreTrainedModel",
]
