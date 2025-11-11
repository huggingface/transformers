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

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ..llama.modeling_llama import LlamaRMSNorm
from .configuration_vibevoice_semantic_tokenizer import VibeVoiceSemanticTokenizerConfig


@dataclass
@auto_docstring
class VibeVoiceSemanticTokenizerOutput(ModelOutput):
    """
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for semantic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceConv1dCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        A [`VibeVoiceConv1dCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    latents: Optional[torch.FloatTensor] = None
    padding_cache: Optional["VibeVoiceConv1dCache"] = None


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


class VibeVoiceConv1dCache:
    """
    Similar to Mimi's Cache: https://github.com/huggingface/transformers/blob/cad7eeeb5e8a173f8d7d746ccdb6ef670ffe6be4/src/transformers/models/mimi/modeling_mimi.py#L76
    But with `batch_mask` support for selective cache updates.

    Original (uses unique key per layer and sample): https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L174
    """

    def __init__(self, num_layers: int):
        self.cache = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        context: int,
        hidden_states: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None,
    ):
        """
        Updates the padding cache with the new padding states for the layer `layer_idx` and returns the current cache.
        Similar to Mimi's update method.

        Parameters:
            layer_idx (`int`):
                The index of the layer to cache the states for.
            hidden_states (`torch.Tensor`):
                The hidden states to be partially cached.
            batch_mask (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Indices of samples to update cache for.
            context (`int`):
                The amount of context for this layer.

        Returns:
            `torch.Tensor`, the current padding cache for the specified samples.
        """
        batch_size, channels, _ = hidden_states.shape

        if batch_mask is None:
            batch_mask = torch.arange(batch_size, device=hidden_states.device)
        else:
            if len(batch_mask) != batch_size:
                raise ValueError("batch_mask length must match batch size")

        existing_cache = self.cache[layer_idx]
        if existing_cache is None:
            current_cache = torch.zeros(
                batch_size, channels, max(context, 0), device=hidden_states.device, dtype=hidden_states.dtype
            )
        else:
            current_cache = existing_cache[batch_mask]

        # Update the cache with padded input (otherwise not enough context):
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L345
        input_with_context = torch.cat([current_cache, hidden_states], dim=-1)
        if context > 0:
            new_cache = input_with_context[..., -context:]
            if existing_cache is None:
                self.cache[layer_idx] = new_cache
            else:
                self.cache[layer_idx][batch_mask] = new_cache
        return current_cache


class VibeVoiceCausalConv1d(nn.Module):
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
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )
        # Padding for causality: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L263C28-L263C72
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
        padding_cache: Optional[VibeVoiceConv1dCache] = None,
        batch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        Original code: https://github.com/vibevoice-community/VibeVoice/blob/63a21e2b45e908be63765bf312a9ecfb3a588315/vibevoice/modular/modular_vibevoice_tokenizer.py#L296
        """

        if padding_cache is not None:
            layer_padding = padding_cache.update(self.layer_idx, self.causal_padding, hidden_states, batch_mask)
        else:
            # non-streaming mode: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L365
            layer_padding = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.causal_padding,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        hidden_states = torch.cat([layer_padding, hidden_states], dim=-1)

        return self.conv(hidden_states)


class VibeVoiceConvNext1dLayer(nn.Module):
    """
    ConvNeXt-like block adapted for 1D convolutions, used in VibeVoice tokenizer encoder.

    For reference, original 2D `ConvNextLayer`:
    https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L120
    """

    def __init__(self, config, hidden_size, drop_path=0.0, dilation=1, stride=1, layer_idx=None):
        super().__init__()

        self.norm = VibeVoiceRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = VibeVoiceRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn = VibeVoiceEncoderFeedForward(config, hidden_size)
        self.gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)
        self.ffn_gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)

        # NOTE (ebezzam) original code has option for DropPath but is never actually used (and `nn.modules.DropPath` does not exist):
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L637
        # however, could be interesting feature for future versions of `ConvNext1dLayer` as the 2D version has it:
        # https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L146
        if drop_path > 0.0:
            # possible implementation (that may needed to be adapted for 1D):
            # https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L40
            raise NotImplementedError("DropPath is not implemented.")
        self.drop_path = nn.Identity()

        self.mixer = VibeVoiceCausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=config.kernel_size,
            groups=hidden_size,
            bias=config.bias,
            dilation=dilation,
            stride=stride,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states, padding_cache=None, batch_mask=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.mixer(hidden_states, padding_cache=padding_cache, batch_mask=batch_mask)
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
    """

    def __init__(self, config):
        super().__init__()

        layer_idx = 0
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            VibeVoiceCausalConv1d(
                in_channels=config.channels,
                out_channels=config.n_filters,
                kernel_size=config.kernel_size,
                bias=config.bias,
                layer_idx=layer_idx,
            )
        )
        layer_idx += 1
        for stage_idx in range(len(config.downsampling_ratios)):
            downsample_layer = VibeVoiceCausalConv1d(
                in_channels=config.n_filters * (2**stage_idx),
                out_channels=config.n_filters * (2 ** (stage_idx + 1)),
                kernel_size=config.downsampling_ratios[stage_idx] * 2,
                stride=config.downsampling_ratios[stage_idx],
                bias=config.bias,
                layer_idx=layer_idx,
            )
            self.downsample_layers.append(downsample_layer)
            layer_idx += 1

        self.stages = nn.ModuleList()
        for stage_idx in range(len(config.depths)):
            input_channels = config.n_filters * (2**stage_idx)
            stage = nn.ModuleList(
                [VibeVoiceConvNext1dLayer(config, hidden_size=input_channels, layer_idx=layer_idx+depth_idx) for depth_idx in range(config.depths[stage_idx])]
            )
            self.stages.append(stage)
            layer_idx += config.depths[stage_idx]

        self.head = VibeVoiceCausalConv1d(
            in_channels=input_channels,
            out_channels=config.hidden_size,
            kernel_size=config.kernel_size,
            bias=config.bias,
            layer_idx=layer_idx,
        )
        self.num_layers = layer_idx + 1

    def forward(self, hidden_states, padding_cache=None, batch_mask=None):
        for layer_idx, downsample_layer in enumerate(self.downsample_layers):
            hidden_states = downsample_layer(hidden_states, padding_cache=padding_cache, batch_mask=batch_mask)
            for block in self.stages[layer_idx]:
                hidden_states = block(hidden_states, padding_cache=padding_cache, batch_mask=batch_mask)
        hidden_states = self.head(hidden_states, padding_cache=padding_cache, batch_mask=batch_mask)
        return hidden_states.permute(0, 2, 1)


@auto_docstring
class VibeVoiceSemanticTokenizerPreTrainedModel(PreTrainedModel):
    config: VibeVoiceSemanticTokenizerConfig
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
    def encode(self, audio, padding_cache=None, batch_mask=None, use_cache=None):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceConv1dCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        batch_mask (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices identifying each sample in the batch for cache management.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceConv1dCache(num_layers=self.encoder.num_layers)

        latents = self.encoder(audio, padding_cache=padding_cache, batch_mask=batch_mask)

        return VibeVoiceSemanticTokenizerOutput(
            latents=latents,
            padding_cache=padding_cache if use_cache else None,
        )

    @can_return_tuple
    @auto_docstring
    def forward(self, audio, padding_cache=None, batch_mask=None, use_cache=None):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceConv1dCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        batch_mask (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices identifying each sample in the batch for cache management.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        return self.encode(
            audio, padding_cache=padding_cache, batch_mask=batch_mask, use_cache=use_cache
        )


__all__ = ["VibeVoiceSemanticTokenizerModel"]
