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
import torch.nn.functional as F

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from .configuration_vibevoice_semantic_tokenizer import VibeVoiceSemanticTokenizerConfig

from ..llama.modeling_llama import LlamaRMSNorm


@dataclass
@auto_docstring
class VibeVoiceSemanticTokenizerOutput(ModelOutput):
    """
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for semantic tokens) at the output of the encoder.
    past_conv_values (`VibeVoiceTokenizerCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceTokenizerCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    latents: Optional[torch.FloatTensor] = None
    past_conv_values: Optional["VibeVoiceTokenizerCache"] = None


class VibeVoiceRMSNorm(LlamaRMSNorm):
    pass


class VibeVoiceEncoderFeedForward(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, config.ffn_expansion * hidden_size, bias=config.bias)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.ffn_expansion * hidden_size, hidden_size, bias=config.bias)

    def forward(self, hidden_states):
        return self.linear2(self.activation(self.linear1(hidden_states)))


# TODO (ebezzam) move to `src/transformers/cache_utils.py` and make similar to `DynamicCache`?
class VibeVoiceTokenizerCache:
    """
    Cache for streaming convolution, where the past convolution outputs of each layer are stored.
    Similar to KV cache for attention-based models.
    """

    def __init__(self):
        self.cache = {}  # Dict mapping (layer_idx, sample_idx) to state tensor

    def get(self, layer_idx: int, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached states for given layer and sample indices"""
        if sample_indices.numel() == 0:
            return None

        # Collect states for all requested samples
        states = []
        sample_indices_list = sample_indices.tolist()
        for sample_idx in sample_indices_list:
            cache_key = (layer_idx, sample_idx)
            if cache_key not in self.cache:
                return None  # If any sample is missing, return None
            states.append(self.cache[cache_key])

        # Check if all states have the same shape for direct stacking
        first_shape = states[0].shape
        if all(state.shape == first_shape for state in states):
            return torch.stack(states, dim=0)

        # Pad to max length if shapes differ
        max_length = max(state.shape[-1] for state in states)
        padded_states = [
            F.pad(state, (max_length - state.shape[-1], 0)) if state.shape[-1] < max_length else state
            for state in states
        ]
        return torch.stack(padded_states, dim=0)

    def update(self, layer_idx: int, sample_indices: torch.Tensor, states: torch.Tensor):
        """Set cached states for given layer and sample indices"""
        sample_indices_list = sample_indices.tolist()
        for batch_idx, sample_idx in enumerate(sample_indices_list):
            self.cache[(layer_idx, sample_idx)] = states[batch_idx].detach()

    def set_to_zero(self, sample_indices: torch.Tensor):
        """Reset cached states for given sample indices"""
        if sample_indices.numel() == 0 or not self.cache:
            return

        sample_indices_set = set(sample_indices.tolist())
        # Remove keys (instead of zeroing them in original) for cleaner memory management
        keys_to_remove = [cache_key for cache_key in self.cache.keys() if cache_key[1] in sample_indices_set]
        for cache_key in keys_to_remove:
            del self.cache[cache_key]

    def clear(self):
        """Clear all cached states"""
        self.cache.clear()

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty"""
        return len(self.cache) == 0


class VibeVoiceStreamingConv1d(nn.Module):
    """Conv1d with built-in handling of streaming and causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )
        # Padding for causality: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L263C28-L263C72
        self.causal_padding = (kernel_size - 1) * dilation - (stride - 1)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_conv_values: Optional[VibeVoiceTokenizerCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.

        Original code: https://github.com/vibevoice-community/VibeVoice/blob/63a21e2b45e908be63765bf312a9ecfb3a588315/vibevoice/modular/modular_vibevoice_tokenizer.py#L327

        Args:
            hidden_states: Input tensor [batch_size, channels, time]
            past_conv_values: `VibeVoiceTokenizerCache` object for maintaining convolution states
            sample_indices: Indices identifying each sample for cache management

        Returns:
            Output tensor
        """
        # Early return for no causal padding case
        if self.causal_padding <= 0:
            return self.conv(hidden_states)

        batch_size, channels, _ = hidden_states.shape

        # Get cached context
        cached_states = None
        if past_conv_values is not None:
            if self.layer_idx is None:
                raise ValueError("layer_idx must be provided during initialization when past_conv_values is used.")
            if sample_indices is None:
                raise ValueError("sample_indices must be provided when past_conv_values is used.")
            if len(sample_indices) != batch_size:
                raise ValueError("sample_indices length must match batch size")
            cached_states = past_conv_values.get(self.layer_idx, sample_indices)

        # Initialize with zeros if no cache exists
        if cached_states is None:
            cached_states = torch.zeros(
                batch_size, channels, self.causal_padding, device=hidden_states.device, dtype=hidden_states.dtype
            )

        # Concatenate context with input
        input_with_context = torch.cat([cached_states, hidden_states], dim=2)

        # Update cache with the last causal_padding samples from the input
        if past_conv_values is not None:
            # Use the combined input for cache update to maintain continuity
            new_cache = input_with_context[:, :, -self.causal_padding :]
            past_conv_values.update(self.layer_idx, sample_indices, new_cache)

        return self.conv(input_with_context)


class VibeVoiceConvNext1dLayer(nn.Module):
    """
    ConvNeXt-like block adapted for 1D convolutions, used in VibeVoice tokenizer encoder.

    For reference, original 2D `ConvNextLayer`:
    https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L120
    """

    def __init__(self, config, hidden_size, drop_path=0.0, dilation=1, stride=1):
        super().__init__()

        self.norm = VibeVoiceRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = VibeVoiceRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn = VibeVoiceEncoderFeedForward(config, hidden_size)
        self.gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)
        self.ffn_gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)

        # TODO (ebezzam) original code has option for DropPath but is never actually used (and `nn.modules.DropPath` does not exist):
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L637
        # however, could be interesting feature for future versions of `ConvNext1dLayer` as the 2D version has it:
        # https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L146
        if drop_path > 0.0:
            # possible implementation (that may needed to be adapted for 1D):
            # https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L40
            raise NotImplementedError("DropPath is not implemented.")
        self.drop_path = nn.Identity()

        self.mixer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=config.kernel_size,
            groups=hidden_size,
            bias=config.bias,
            dilation=dilation,
            stride=stride,
        )
        # Padding for causality: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L266
        self.causal_padding = (config.kernel_size - 1) * dilation - (stride - 1)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        # Padding for causality: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L382
        hidden_states = F.pad(hidden_states, (self.causal_padding, 0))
        hidden_states = self.mixer(hidden_states)
        hidden_states = hidden_states * self.gamma.unsqueeze(-1)
        # (ebezzam) original code (https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L653)
        # as mentioned above, drop_path is not used and the VibeVoice authors don't use the `forward` method but a custom
        # call which does `residual + hidden_states` directly (see link below), which is same as using identity
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L768
        hidden_states = residual + self.drop_path(hidden_states)

        # ffn
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states.transpose(1, 2))  # [B, T, C]
        hidden_states = self.ffn(hidden_states)  # FFN expects [B, T, C]
        hidden_states = hidden_states.transpose(1, 2)  # Back to [B, C, T]
        hidden_states = hidden_states * self.ffn_gamma.unsqueeze(-1)
        # (ebezzam) see comment above
        hidden_states = residual + self.drop_path(hidden_states)
        return hidden_states


class VibeVoiceSemanticTokenizerEncoder(nn.Module):
    """
    Encoder component for the VibeVoice tokenizer that converts audio to latent representations.

    Paper (https://arxiv.org/pdf/2508.19205) says:
    "7 stages of modified Transformer blocks (using 1D depth-wise causal convolutions instead of self-attention module)
    for efficient streaming processing. Six downsampling layers achieve a cumulative 3200X downsampling rate from a
    24kHz input, yielding 7.5 tokens/frames per second."

    But each block is more like a ConvNeXt block (but 1D): https://arxiv.org/abs/2201.03545
    Hence the name `ConvNext1dLayer` in this code for the blocks.

    Args:
        config: Configuration object with model parameters
    """

    def __init__(self, config):
        super().__init__()

        # stem and intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            VibeVoiceStreamingConv1d(
                in_channels=config.channels,
                out_channels=config.n_filters,
                kernel_size=config.kernel_size,
                bias=config.bias,
                layer_idx=0,
            )
        )
        for stage_idx in range(len(config.downsampling_ratios)):
            downsample_layer = VibeVoiceStreamingConv1d(
                in_channels=config.n_filters * (2**stage_idx),
                out_channels=config.n_filters * (2 ** (stage_idx + 1)),
                kernel_size=config.downsampling_ratios[stage_idx] * 2,
                stride=config.downsampling_ratios[stage_idx],
                bias=config.bias,
                layer_idx=stage_idx + 1,
            )
            self.downsample_layers.append(downsample_layer)

        # configure ConvNext1D blocks
        self.stages = nn.ModuleList()
        for stage_idx in range(len(config.depths)):
            input_channels = config.n_filters * (2**stage_idx)
            stage = nn.ModuleList(
                [VibeVoiceConvNext1dLayer(config, hidden_size=input_channels) for _ in range(config.depths[stage_idx])]
            )
            self.stages.append(stage)

        self.head = VibeVoiceStreamingConv1d(
            in_channels=input_channels,
            out_channels=config.hidden_size,
            kernel_size=config.kernel_size,
            bias=config.bias,
            layer_idx=len(config.downsampling_ratios) + 1,
        )

    def forward(self, hidden_states, past_conv_values=None, sample_indices=None):
        for layer_idx, downsample_layer in enumerate(self.downsample_layers):
            hidden_states = downsample_layer(
                hidden_states, past_conv_values=past_conv_values, sample_indices=sample_indices
            )
            for block in self.stages[layer_idx]:
                hidden_states = block(hidden_states)
        hidden_states = self.head(
            hidden_states, past_conv_values=past_conv_values, sample_indices=sample_indices
        )
        return hidden_states.permute(0, 2, 1)


@auto_docstring
class VibeVoiceSemanticTokenizerPreTrainedModel(PreTrainedModel):
    config = VibeVoiceSemanticTokenizerConfig
    base_model_prefix = "vibevoice_semantic_tokenizer"
    main_input_name = "audio"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["VibeVoiceSemanticTokenizerEncoder"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, VibeVoiceRMSNorm):
            nn.init.ones_(module.weight)


@auto_docstring
class VibeVoiceSemanticTokenizerModel(VibeVoiceSemanticTokenizerPreTrainedModel):
    """Encoder-only VibeVoice tokenizer model for semantic tokens."""

    def __init__(self, config):
        super().__init__(config)

        self.encoder = VibeVoiceSemanticTokenizerEncoder(config)

        # Initialize weights
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def encode(self, audio, past_conv_values=None, sample_indices=None, use_cache=None):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        past_conv_values (`VibeVoiceTokenizerCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        sample_indices (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices identifying each sample in the batch for cache management.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        # Input validation
        batch_size = audio.shape[0]
        if sample_indices is not None and len(sample_indices) != batch_size:
            raise ValueError(f"sample_indices length ({len(sample_indices)}) must match batch size ({batch_size})")

        if use_cache and past_conv_values is None:
            past_conv_values = VibeVoiceTokenizerCache()

        latents = self.encoder(audio, past_conv_values=past_conv_values, sample_indices=sample_indices)

        return VibeVoiceSemanticTokenizerOutput(
            latents=latents,
            past_conv_values=past_conv_values if use_cache else None,
        )

    @can_return_tuple
    @auto_docstring
    def forward(self, audio, past_conv_values=None, sample_indices=None, use_cache=None):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        past_conv_values (`VibeVoiceTokenizerCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        sample_indices (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices identifying each sample in the batch for cache management.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        return self.encode(
            audio, past_conv_values=past_conv_values, sample_indices=sample_indices, use_cache=use_cache
        )


__all__ = ["VibeVoiceSemanticTokenizerModel"]
