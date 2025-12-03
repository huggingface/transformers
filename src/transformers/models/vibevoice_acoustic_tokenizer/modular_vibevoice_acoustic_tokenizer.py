# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ..llama.modeling_llama import LlamaRMSNorm
from ..vibevoice_semantic_tokenizer.configuration_vibevoice_semantic_tokenizer import VibeVoiceSemanticTokenizerConfig
from ..vibevoice_semantic_tokenizer.modeling_vibevoice_semantic_tokenizer import (
    VibeVoiceCausalConv1d,
    VibeVoiceConv1dCache,
    VibeVoiceConvNext1dLayer,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceSemanticTokenizerOutput,
    VibeVoiceSemanticTokenizerPreTrainedModel,
)


class VibeVoiceAcousticTokenizerConfig(VibeVoiceSemanticTokenizerConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceAcousticTokenizerModel`]. It is used to
    instantiate a VibeVoice acoustic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    acoustic tokenizer of [VibeVoice-1.5B](https://hf.co/microsoft/VibeVoice-1.5B).

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
        weight_init_value (`float`, *optional*, defaults to 0.01):
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
        vae_std (`float`, *optional*, defaults to 0.5):
            Standard deviation used during VAE sampling.
        vae_scaling_factor (`float`, *optional*, defaults to 0.8):
            Scaling factor applied to VAE standard deviation. (Hardcoded in original implementation.)
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
        weight_init_value=1e-2,
        n_filters=32,
        downsampling_ratios=[2, 2, 4, 5, 5, 8],
        depths=[3, 3, 3, 3, 3, 3, 8],
        hidden_act="gelu",
        ffn_expansion=4,
        vae_std=0.5,
        vae_scaling_factor=0.8,
        **kwargs,
    ):
        super().__init__(
            channels=channels,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            rms_norm_eps=rms_norm_eps,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
            weight_init_value=weight_init_value,
            n_filters=n_filters,
            downsampling_ratios=downsampling_ratios,
            depths=depths,
            hidden_act=hidden_act,
            ffn_expansion=ffn_expansion,
            **kwargs,
        )
        # NOTE (ebezzam) original hardcodes scaling within sampling: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L963
        # scaling moved here in case future implementations modify `vae_std` but keep internal scaling
        self.vae_std = vae_std
        self.vae_scaling_factor = vae_scaling_factor

    @property
    def upsampling_ratios(self):
        return self.downsampling_ratios[::-1]

    @property
    def decoder_depths(self):
        return self.depths[::-1]


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerOutput(ModelOutput):
    """
    audio (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceConv1dCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceConv1dCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    audio: Optional[torch.FloatTensor] = None
    latents: Optional[torch.FloatTensor] = None
    padding_cache: Optional["VibeVoiceConv1dCache"] = None


class VibeVoiceAcousticTokenizerEncoderOutput(VibeVoiceSemanticTokenizerOutput):
    pass


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerDecoderOutput(ModelOutput):
    """
    audio (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceConv1dCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceConv1dCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    audio: Optional[torch.FloatTensor] = None
    padding_cache: Optional["VibeVoiceConv1dCache"] = None


class VibeVoiceRMSNorm(LlamaRMSNorm):
    pass


class VibeVoiceCausalConvTranspose1d(nn.Module):
    """Causal ConvTranspose1d with optional streaming support via VibeVoiceConv1dCache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.stride = stride
        self.layer_idx = layer_idx
        # Different padding for transposed convolution: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L423
        self.padding_total = kernel_size - stride
        self.causal_padding = kernel_size - 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: Optional["VibeVoiceConv1dCache"] = None,
    ) -> torch.Tensor:
        time_dim = hidden_states.shape[-1]

        if padding_cache is not None:
            layer_padding = padding_cache.update(hidden_states, self.layer_idx, is_transpose=True)
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


class VibeVoiceAcousticTokenizerDecoder(nn.Module):
    """
    Decoder component for the VibeVoice tokenizer that converts latent representations back to audio.
    """

    def __init__(self, config):
        super().__init__()

        layer_idx = 0
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(
            VibeVoiceCausalConv1d(
                in_channels=config.hidden_size,
                out_channels=int(config.n_filters * 2 ** (len(config.decoder_depths) - 1)),
                kernel_size=config.kernel_size,
                bias=config.bias,
                layer_idx=layer_idx,
            )
        )
        layer_idx += 1
        for stage_idx in range(len(config.upsampling_ratios)):
            input_channels = int(config.n_filters * (2 ** (len(config.decoder_depths) - 1 - stage_idx)))
            output_channels = int(config.n_filters * (2 ** (len(config.decoder_depths) - 1 - stage_idx - 1)))
            upsample_layer = VibeVoiceCausalConvTranspose1d(
                input_channels,
                output_channels,
                kernel_size=int(config.upsampling_ratios[stage_idx] * 2),
                stride=config.upsampling_ratios[stage_idx],
                bias=config.bias,
                layer_idx=layer_idx,
            )
            self.upsample_layers.append(upsample_layer)
            layer_idx += 1

        self.stages = nn.ModuleList()
        for stage_idx in range(len(config.decoder_depths)):
            input_channels = config.n_filters * (2 ** (len(config.decoder_depths) - 1 - stage_idx))
            stage = nn.ModuleList(
                [
                    VibeVoiceConvNext1dLayer(config, hidden_size=input_channels, layer_idx=layer_idx + depth_idx)
                    for depth_idx in range(config.decoder_depths[stage_idx])
                ]
            )
            self.stages.append(stage)
            layer_idx += config.decoder_depths[stage_idx]

        self.head = VibeVoiceCausalConv1d(
            in_channels=input_channels,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            bias=config.bias,
            layer_idx=layer_idx,
        )

        # store parameters for cache creation
        self.num_layers = layer_idx + 1
        self.per_layer_padding = []
        self.per_layer_in_channels = []
        for upsample_layer in self.upsample_layers:
            self.per_layer_padding.append(upsample_layer.causal_padding)
            if hasattr(upsample_layer, "convtr"):
                self.per_layer_in_channels.append(upsample_layer.convtr.in_channels)
            else:
                self.per_layer_in_channels.append(upsample_layer.conv.in_channels)
        for stage in self.stages:
            for block in stage:
                self.per_layer_padding.append(block.mixer.causal_padding)
                self.per_layer_in_channels.append(block.mixer.conv.in_channels)
        self.per_layer_padding.append(self.head.causal_padding)
        self.per_layer_in_channels.append(self.head.conv.in_channels)

    def forward(self, hidden_states, padding_cache=None):
        for layer_idx, upsample_layer in enumerate(self.upsample_layers):
            hidden_states = upsample_layer(hidden_states, padding_cache=padding_cache)
            for block in self.stages[layer_idx]:
                hidden_states = block(hidden_states, padding_cache=padding_cache)
        hidden_states = self.head(hidden_states, padding_cache=padding_cache)
        return hidden_states


class VibeVoiceAcousticTokenizerPreTrainedModel(VibeVoiceSemanticTokenizerPreTrainedModel):
    _no_split_modules = ["VibeVoiceAcousticTokenizerEncoder", "VibeVoiceAcousticTokenizerDecoder"]


@auto_docstring
class VibeVoiceAcousticTokenizerModel(VibeVoiceSemanticTokenizerModel):
    """VibeVoice speech tokenizer model combining encoder and decoder for acoustic tokens"""

    def __init__(self, config):
        super().__init__(config)
        self.decoder = VibeVoiceAcousticTokenizerDecoder(config)
        self.vae_std = config.vae_std / config.vae_scaling_factor

    @can_return_tuple
    @auto_docstring
    def encode(self, audio, padding_cache=None, use_cache=False, sample=True):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceConv1dCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        sample (`bool`, *optional*):
            Whether to sample from the output distribution or return the latent as is.
        """

        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceConv1dCache(
                num_layers=self.encoder.num_layers,
                per_layer_padding=self.encoder.per_layer_padding,
                per_layer_in_channels=self.encoder.per_layer_in_channels,
            )

        latents = self.encoder(audio, padding_cache=padding_cache)

        if sample:
            batch_size = audio.shape[0]
            noise_std = self.vae_std * torch.randn(batch_size, device=latents.device, dtype=latents.dtype)
            while noise_std.dim() < latents.dim():
                noise_std = noise_std.unsqueeze(-1)
            latents = latents + noise_std * torch.randn_like(latents)

        return VibeVoiceAcousticTokenizerEncoderOutput(
            latents=latents,
            padding_cache=padding_cache if use_cache else None,
        )

    @can_return_tuple
    @auto_docstring
    def decode(self, latents, padding_cache=None, use_cache=False):
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input latent representations to be decoded back into audio waveforms.
        padding_cache (`VibeVoiceConv1dCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """

        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceConv1dCache(
                num_layers=self.decoder.num_layers,
                per_layer_padding=self.decoder.per_layer_padding,
                per_layer_in_channels=self.decoder.per_layer_in_channels,
            )

        latents = latents.permute(0, 2, 1)
        audio = self.decoder(latents, padding_cache=padding_cache)
        return VibeVoiceAcousticTokenizerDecoderOutput(audio=audio, padding_cache=padding_cache)

    @can_return_tuple
    @auto_docstring
    def forward(self, audio, padding_cache=None, use_cache=False, sample=True):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceConv1dCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers. Note only used by decoder.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        sample (`bool`, *optional*):
            Whether to sample from the output distribution of the encoder, or return the latent as is.
        """
        encoder_output = self.encode(audio, sample=sample)
        decoder_output = self.decode(encoder_output.latents, padding_cache=padding_cache, use_cache=use_cache)
        return VibeVoiceAcousticTokenizerOutput(
            audio=decoder_output.audio,
            latents=encoder_output.latents,
            padding_cache=decoder_output.padding_cache,
        )


__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceAcousticTokenizerModel",
    "VibeVoiceAcousticTokenizerPreTrainedModel",
]
