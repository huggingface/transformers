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
    VibeVoiceSemanticTokenizerOutput,
    VibeVoiceTokenizerCache,
    VibeVoiceSemanticTokenizerPreTrainedModel,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceStreamingConv1d,
    VibeVoiceConvNext1dLayer,
)


class VibeVoiceAcousticTokenizerConfig(VibeVoiceSemanticTokenizerConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceAcousticTokenizerModel`]. It is used to
    instantiate a VibeVoice acoustic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    acoustic tokenizer of [VibeVoice-1.5B](https://hf.co/microsoft/VibeVoice-1.5B).

    Args:
        channels (`int`, *optional*, defaults to `1`):
            Number of input channels.
        hidden_size (`int`, *optional*, defaults to `64`):
            Dimensionality of latent representations.
        kernel_size (`int`, *optional*, defaults to `7`):
            Kernel size for convolutional layers.
        rms_norm_eps (`float`, *optional*, defaults to `1e-5`):
            Epsilon value for RMSNorm layers.
        bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in convolution and feed-forward layers.
        layer_scale_init_value (`float`, *optional*, defaults to `1e-6`):
            Initial value for layer scaling.
        weight_init_value (`float`, *optional*, defaults to `1e-2`):
            Standard deviation for weight initialization.
        n_filters (`int`, *optional*, defaults to `32`):
            Number of filters in initial convolutional layer, and doubles after each downsampling.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
            Downsampling ratios for each layer.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
            Number of ConvNeXt blocks at each stage.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to `4`):
            Expansion factor for feed-forward networks.
        vae_std (`float`, *optional*, defaults to `0.5`):
            Standard deviation used during VAE sampling.
        vae_scaling_factor (`float`, *optional*, defaults to `0.8`):
            Scaling factor applied to VAE standard deviation. (Hardcoded in original implementation.)
    """

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
        **kwargs
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
            **kwargs
        )
        # TODO (ebezzam) original authors hardcode scaling within sampling: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L963
        # so moved scaling here in case they modify `vae_std` later but keep internal scaling
        self.vae_std = vae_std / vae_scaling_factor


    @property
    def hop_length(self) -> int:
        return np.prod(self.downsampling_ratios)
    
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
    past_conv_values (`VibeVoiceTokenizerCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceTokenizerCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    audio: Optional[torch.FloatTensor] = None
    latents: Optional[torch.FloatTensor] = None
    past_conv_values: Optional["VibeVoiceTokenizerCache"] = None


class VibeVoiceAcousticTokenizerEncoderOutput(VibeVoiceSemanticTokenizerOutput):
    pass


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerDecoderOutput(ModelOutput):
    """
    audio (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for acoustic tokens) at the output of the encoder.
    past_conv_values (`VibeVoiceTokenizerCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceTokenizerCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    audio: Optional[torch.FloatTensor] = None
    past_conv_values: Optional["VibeVoiceTokenizerCache"] = None


class VibeVoiceTokenizerCache(VibeVoiceTokenizerCache):
    pass


class VibeVoiceAcousticTokenizerRMSNorm(LlamaRMSNorm):
    pass


class VibeVoiceStreamingConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of streaming and causal padding."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int, 
        stride: int = 1, 
        bias: bool = True
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)

        # For streaming, we need to keep track of input history
        # Transposed conv needs to see multiple input samples to produce correct output
        self.context_size = kernel_size - stride
        self.stride = stride

    def forward(self, 
        hidden_states: torch.Tensor,
        past_conv_values: Optional[VibeVoiceTokenizerCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        """
        batch_size, channels, time_dim = hidden_states.shape

        # Validate cache parameters if provided
        if past_conv_values is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when past_conv_values is used.")
            if sample_indices is None:
                raise ValueError("sample_indices must be provided when past_conv_values is used.")
            if len(sample_indices) != batch_size:
                raise ValueError("sample_indices length must match batch size")

        # Get input for convolution (with context for streaming)
        if past_conv_values is None:
            conv_input = hidden_states
            is_first_chunk = True
        else:
            cached_input = past_conv_values.get(layer_idx, sample_indices)
            if cached_input is None:
                cached_input = torch.zeros(batch_size, channels, 0, device=hidden_states.device, dtype=hidden_states.dtype)
            conv_input = torch.cat([cached_input, hidden_states], dim=2)
            is_first_chunk = cached_input.shape[2] == 0

        # Apply transposed convolution and remove padding
        output = self.convtr(conv_input)
        if self.context_size > 0:
            output = output[..., :-self.context_size]

        # For streaming mode, extract only the new output (except first chunk)
        if past_conv_values is not None:
            if not is_first_chunk:
                expected_new_output = time_dim * self.stride
                if output.shape[2] >= expected_new_output:
                    output = output[:, :, -expected_new_output:]
            
            # Update cache with last context_size samples
            if self.context_size > 0:
                new_cache = conv_input[:, :, -self.context_size:]
                past_conv_values.update(layer_idx, sample_indices, new_cache)

        return output


class VibeVoiceAcousticTokenizerDecoder(nn.Module):
    """
    Decoder component for the VibeVoice tokenizer that converts latent representations back to audio.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()

        # stem and upsampling layers
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(
            VibeVoiceStreamingConv1d(config.hidden_size, config.n_filters * 2 ** (len(config.decoder_depths) - 1), config.kernel_size,
                    bias=config.bias)
        )
        for stage_idx in range(len(config.upsampling_ratios)):
            input_channels = config.n_filters * (2 ** (len(config.decoder_depths) - 1 - stage_idx))
            output_channels = config.n_filters * (2 ** (len(config.decoder_depths) - 1 - stage_idx - 1))
            upsample_layer = VibeVoiceStreamingConvTranspose1d(input_channels, output_channels,
                            kernel_size=config.upsampling_ratios[stage_idx] * 2, stride=config.upsampling_ratios[stage_idx],
                            bias=config.bias)
            self.upsample_layers.append(upsample_layer)

        # configure ConvNext1D blocks
        self.stages = nn.ModuleList()
        for stage_idx in range(len(config.decoder_depths)):
            input_channels = config.n_filters * (2 ** (len(config.decoder_depths) - 1 - stage_idx))
            stage = nn.ModuleList(
                [VibeVoiceConvNext1dLayer(config, hidden_size=input_channels) for _ in range(config.decoder_depths[stage_idx])]
            )
            self.stages.append(stage)

        self.head = VibeVoiceStreamingConv1d(input_channels, config.channels, kernel_size=config.kernel_size, bias=config.bias)

    def forward(self, hidden_states, past_conv_values=None, sample_indices=None):
        for layer_idx, upsample_layer in enumerate(self.upsample_layers):
            hidden_states = upsample_layer(
                hidden_states, past_conv_values=past_conv_values, sample_indices=sample_indices, layer_idx=layer_idx
            )
            for block in self.stages[layer_idx]:
                hidden_states = block(hidden_states)
        hidden_states = self.head(hidden_states, past_conv_values=past_conv_values, sample_indices=sample_indices, layer_idx=layer_idx + 1)
        return hidden_states


class VibeVoiceAcousticTokenizerPreTrainedModel(VibeVoiceSemanticTokenizerPreTrainedModel):
    _no_split_modules = ["VibeVoiceAcousticTokenizerEncoder", "VibeVoiceAcousticTokenizerDecoder"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, VibeVoiceAcousticTokenizerRMSNorm):
            nn.init.ones_(module.weight)


@auto_docstring
class VibeVoiceAcousticTokenizerModel(VibeVoiceSemanticTokenizerModel):
    """VibeVoice speech tokenizer model combining encoder and decoder for acoustic tokens"""

    def __init__(self, config):
        super().__init__(config)
        self.decoder = VibeVoiceAcousticTokenizerDecoder(config)
        self.vae_std = config.vae_std

    @can_return_tuple
    @auto_docstring
    def encode(self, audio, past_conv_values=None, sample_indices=None, use_cache=False, sample=True):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        past_conv_values (`VibeVoiceTokenizerCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        sample_indices (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices identifying each sample in the batch for cache management.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        sample (`bool`, *optional*):
            Whether to sample from the output distribution or return the latent as is.
        """
        # Input validation
        batch_size = audio.shape[0]
        if sample_indices is not None and len(sample_indices) != batch_size:
            raise ValueError(f"sample_indices length ({len(sample_indices)}) must match batch size ({batch_size})")

        if use_cache and past_conv_values is None:
            past_conv_values = VibeVoiceTokenizerCache()

        latents = self.encoder(audio, past_conv_values=past_conv_values, sample_indices=sample_indices)
        
        if sample:
            noise_std = self.vae_std * torch.ones(batch_size, device=latents.device, dtype=latents.dtype)
            while noise_std.dim() < latents.dim():
                noise_std = noise_std.unsqueeze(-1)
            latents = latents + noise_std * torch.randn_like(latents)
        
        return VibeVoiceAcousticTokenizerEncoderOutput(
            latents=latents,
            past_conv_values=past_conv_values if use_cache else None,
        )
    
    @can_return_tuple
    @auto_docstring
    def decode(self, latents, past_conv_values=None, sample_indices=None, use_cache=False):
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input latent representations to be decoded back into audio waveforms.
        past_conv_values (`VibeVoiceTokenizerCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        sample_indices (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices identifying each sample in the batch for cache management.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        # Input validation
        batch_size = latents.shape[0]
        if sample_indices is not None and len(sample_indices) != batch_size:
            raise ValueError(f"sample_indices length ({len(sample_indices)}) must match batch size ({batch_size})")

        if use_cache and past_conv_values is None:
            past_conv_values = VibeVoiceTokenizerCache()

        latents = latents.permute(0, 2, 1)
        audio = self.decoder(latents, past_conv_values=past_conv_values, sample_indices=sample_indices)
        return VibeVoiceAcousticTokenizerDecoderOutput(audio=audio, past_conv_values=past_conv_values)

    @can_return_tuple
    @auto_docstring
    def forward(self, audio, past_conv_values=None, sample_indices=None, use_cache=False, sample=True):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        past_conv_values (`VibeVoiceTokenizerCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers. Note only used by decoder.
        sample_indices (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices identifying each sample in the batch for cache management.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        sample (`bool`, *optional*):
            Whether to sample from the output distribution of the encoder, or return the latent as is.
        """
        encoder_output = self.encode(audio, sample=sample)
        decoder_output = self.decode(
            encoder_output.latents, 
            past_conv_values=past_conv_values, 
            sample_indices=sample_indices, 
            use_cache=use_cache
        )
        return VibeVoiceAcousticTokenizerOutput(
            audio=decoder_output.audio,
            latents=encoder_output.latents,
            past_conv_values=decoder_output.past_conv_values,
        )


__all__ = ["VibeVoiceAcousticTokenizerConfig", "VibeVoiceAcousticTokenizerModel"]