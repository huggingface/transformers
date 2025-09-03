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

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial, wraps
from math import ceil
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import int32, nn, sin, sinc
from torch.amp import autocast
from torch.nn import Module, Parameter

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoFeatureExtractor, AutoModel
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_xcodec2 import Xcodec2Config


# General docstring
_CONFIG_FOR_DOC = "Xcodec2Config"


@dataclass
class Xcodec2Output(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, 1, codes_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`.
        audio_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Xcodec2.
    """

    audio_codes: Optional[torch.LongTensor] = None
    audio_values: Optional[torch.FloatTensor] = None


@dataclass
class Xcodec2EncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, 1, codes_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`.
    """

    audio_codes: Optional[torch.LongTensor] = None


@dataclass
class Xcodec2DecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Xcodec2.
    """

    audio_values: Optional[torch.FloatTensor] = None


if is_torch_flex_attn_available():
    pass


logger = logging.get_logger(__name__)


# See here for their attention implementation:
# https://huggingface.co/HKUSTAudio/xcodec2/blob/main/vq/bs_roformer5.py
class Xcodec2Attention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Xcodec2Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=self.is_causal,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Xcodec2RMSNorm(LlamaRMSNorm):
    pass


class Xcodec2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Xcodec2MLP(nn.Module):
    def __init__(self, config: Xcodec2Config):
        super().__init__()
        dim = config.hidden_size

        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)  # No bias like original
        self.silu = nn.SiLU()  # TODO: Or ACT2FN[config.hidden_act] if using config activation
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)  # No bias like original

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x


class Xcodec2DecoderLayer(LlamaDecoderLayer):
    # Override forward to enforce non-causal attention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # This is the mask PASSED TO the layer
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,  # Non-causal typically doesn't use KV cache
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # Catches potential FlashAttention kwargs etc.
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if use_cache:
            logger.warning_once("KV Caching (`use_cache=True`) is typically not used with non-causal attention.")
            # Depending on use case, you might want to force use_cache = False here
            # or ensure the caching mechanism handles non-causal correctly (unlikely with standard KV cache).

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # Use potentially overridden norm

        # --- Self Attention (using parent's LlamaAttention instance) ---
        # We need to pass an attention_mask to self.self_attn that is *NOT* causal.
        # If the original attention_mask only contains padding info (e.g., from tokenizer),
        # it might be usable directly. If it's explicitly causal, we need to ignore/modify it.
        # Simplest for non-causal (assuming no padding or padding handled by mask): pass None
        # If padding needs to be handled, construct a non-causal padding mask here.

        # Example: Assuming `attention_mask` might be causal or handle padding.
        # We create a mask that only accounts for padding if present.
        non_causal_attn_mask = None
        if attention_mask is not None:
            # Check if the input mask handles padding (e.g., has 0s).
            # This is a basic check; robust padding handling might need more.
            if (attention_mask == 0).any():
                # Keep the original mask if it seems to handle padding.
                # WARNING: If this mask ALSO encodes causality, this won't work as intended.
                # A better approach might be to reconstruct the padding mask from input_ids
                # if available higher up, or assume the passed mask is ONLY for padding.
                non_causal_attn_mask = attention_mask
            # If the mask exists but doesn't seem to have padding (all 1s),
            # and we want non-causal, set it to None.
            # else: non_causal_attn_mask = None # Already initialized to None

        # Call the LlamaAttention forward method
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=non_causal_attn_mask,  # <<< Pass the non-causal mask
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # `LlamaAttention` internally uses `is_causal = True` by default IF the backend
        # (like SDPA) takes an `is_causal` flag and the mask allows it (e.g., mask is None).
        # By passing a non-causal mask (or None when appropriate), we prevent the causal path.

        hidden_states = residual + hidden_states

        # --- Fully Connected ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # Use potentially overridden norm
        hidden_states = self.mlp(hidden_states)  # Use potentially overridden MLP
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        # Note: past_key_value is returned unmodified if use_cache=False,
        # or potentially updated by self.self_attn if use_cache=True (check compatibility)

        return outputs


class Xcodec2SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
        alpha is initialized to 1 by default, higher values = higher-frequency.
        beta is initialized to 1 by default, higher values = higher-magnitude.
        alpha will be trained along with the rest of your model.
        """
        super(Xcodec2SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x):
        x = self.lowpass(x)

        return x


class Activation1d(nn.Module):
    def __init__(
        self, activation, up_ratio: int = 2, down_ratio: int = 2, up_kernel_size: int = 12, down_kernel_size: int = 12
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.activation1 = Activation1d(activation=Xcodec2SnakeBeta(dim, alpha_logscale=True))
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.activation2 = Activation1d(activation=Xcodec2SnakeBeta(dim, alpha_logscale=True))
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        return residual + x


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, dilations=(1, 3, 9)):
        super().__init__()
        self.residual_units = nn.ModuleList([ResidualUnit(dim // 2, dilation=d) for d in dilations])
        self.activation = Activation1d(activation=Xcodec2SnakeBeta(dim // 2, alpha_logscale=True))
        self.conv = nn.Conv1d(
            dim // 2,
            dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2 + stride % 2,
        )

    def forward(self, x):
        for residual_unit in self.residual_units:
            x = residual_unit(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class Xcodec2CodecEncoder(nn.Module):
    def __init__(
        self,
        d_model=48,
        downsampling_ratios=[2, 2, 4, 4, 5],
        dilations=(1, 3, 9),
        hidden_dim=1024,
    ):
        super().__init__()

        self.initial_conv = nn.Conv1d(1, d_model, kernel_size=7, padding=3)

        self.encoder_blocks = nn.ModuleList()
        for i, stride in enumerate(downsampling_ratios):
            d_model *= 2
            self.encoder_blocks.append(EncoderBlock(d_model, stride=stride, dilations=dilations))

        self.final_activation = Activation1d(activation=Xcodec2SnakeBeta(d_model, alpha_logscale=True))
        self.final_conv = nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Apply all encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        # Final processing
        x = self.final_activation(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x


def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_maybe_sync_seed(device, max_size=10_000):
    rand_int = torch.randint(0, max_size, (), device=device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()


class Backbone(nn.Module):
    """Base class for the decoder's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = h * torch.sigmoid(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(temb * torch.sigmoid(temb))[:, :, None, None]

        h = self.norm2(h)
        h = h * torch.sigmoid(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Xcodec2VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(self, config: Xcodec2Config):
        super().__init__()

        self.embed = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=7, padding=3)

        self.temb_ch = 0
        block_in = config.hidden_size
        dropout = 0.1

        self.prior_blocks = nn.ModuleList(
            [
                ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout),
                ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout),
            ]
        )

        # Initialize rotary embeddings
        self.rotary_emb = Xcodec2RotaryEmbedding(config=config)

        # Create transformer layers
        self.transformers = nn.ModuleList(
            [Xcodec2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.post_blocks = nn.ModuleList(
            [
                ResnetBlock(
                    in_channels=config.hidden_size,
                    out_channels=config.hidden_size,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                ),
                ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, hidden_dim]

        # Handle initial transformations
        x = x.transpose(1, 2)
        x = self.embed(x)

        # Process through prior_blocks
        for block in self.prior_blocks:
            x = block(x)

        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Generate position IDs and rotary embeddings
        batch_size, seq_length = x.shape[:2]
        position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        # Apply transformer layers with position embeddings
        for layer in self.transformers:
            x = layer(
                x,
                position_embeddings=position_embeddings,
            )[0]  # Only take hidden states, ignore attention weights

        # Handle final transformations
        x = x.transpose(1, 2)

        # Process through post_blocks
        for block in self.post_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)

        return x


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec.to(torch.complex64), self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio = self.istft(S).to(x_pred.dtype)
        return audio.unsqueeze(1), x_pred


def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def floor_ste(z):
    """Floor with straight through gradients."""
    zhat = z.floor()
    return z + (zhat - z).detach()


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if x is None:
            return x
        return fn(x, *args, **kwargs)

    return inner


class Xcodec2FSQ(Module):
    """
    Copied from https://github.com/lucidrains/vector-quantize-pytorch/blob/fe903ce2ae9c125ace849576aa6d09c5cec21fe4/vector_quantize_pytorch/finite_scalar_quantization.py#L61
    """

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32=True,
        preserve_symmetry: bool = False,
        noise_dropout=0.0,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = keep_num_codebooks_dim if keep_num_codebooks_dim is not None else num_codebooks > 1
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = dim if dim is not None else len(_levels) * num_codebooks

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias) if has_projections else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = math.prod(levels)
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def symmetry_preserving_bound(self, z):
        """
        # Section 3.2 in https://arxiv.org/abs/2411.19842
        QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1
        """
        levels_minus_1 = self._levels - 1
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.0) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z):
        _, _, noise_dropout, preserve_symmetry, half_width = (
            z.shape[0],
            z.device,
            self.noise_dropout,
            self.preserve_symmetry,
            (self._levels // 2),
        )
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        bounded_z = bound_fn(z)
        if self.training and noise_dropout > 0.0:
            offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
            offset = torch.rand_like(bounded_z) - 0.5
            bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return round_ste(bounded_z) / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings"""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert indices is not None

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = codes.reshape(*codes.shape[:-2], -1)

        codes = self.project_out(codes)

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        # z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)
        b, n, cd = z.shape  # (b, n, c·d)
        c = self.num_codebooks
        d = cd // c  # infer the per-codebook dimension

        z = z.view(b, n, c, d)  # now (b, n, c, d)
        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            # codes = rearrange(codes, 'b n c d -> b n (c d)')
            codes = codes.flatten(start_dim=2)
            codes = codes.to(orig_dtype)

        # project out

        out = self.project_out(codes)
        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(lambda t, *_, **__: t.squeeze(-1))(indices)

        # return quantized output and indices

        return out, indices


class Xcodec2ResidualFSQ(Module):
    """
    Copied from https://github.com/lucidrains/vector-quantize-pytorch/blob/fe903ce2ae9c125ace849576aa6d09c5cec21fe4/vector_quantize_pytorch/residual_fsq.py#L49
    """

    def __init__(
        self,
        *,
        levels: list[int],
        num_quantizers,
        dim: Optional[int] = None,
        is_channel_first=False,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        soft_clamp_input_value=None,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = len(levels)

        dim = codebook_dim if dim is None else dim

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers
        self.soft_clamp_input_value = soft_clamp_input_value
        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)

        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = Xcodec2FSQ(levels=levels, dim=codebook_dim, **kwargs)

            self.layers.append(fsq)

        self.codebook_size = self.layers[0].codebook_size

        self.register_buffer("scales", torch.stack(scales), persistent=False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        _, quantize_dim = indices.shape[0], indices.shape[-1]
        b, *spatial_dims, q = indices.shape
        indices_packed = indices.reshape(b, -1, q)
        ps = (tuple(spatial_dims),)

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0.0, (
                "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            )
            indices_packed = F.pad(indices_packed, (0, self.num_quantizers - quantize_dim), value=-1)

        mask = indices_packed == -1
        indices_proc = indices_packed.masked_fill(mask, 0)

        indices_permuted = indices_proc.permute(2, 0, 1)

        selected_codes = [self.codebooks[qi][indices_permuted[qi]] for qi in range(self.num_quantizers)]

        all_codes = torch.stack(selected_codes, dim=0)

        mask_permuted = mask.permute(2, 0, 1).unsqueeze(-1)
        all_codes = all_codes.masked_fill(mask_permuted, 0.0)

        scales_reshaped = self.scales.view(self.num_quantizers, 1, 1, -1)
        all_codes = all_codes * scales_reshaped

        spatial_shape = tuple(ps[0])
        q, b, _, d = all_codes.shape

        all_codes = all_codes.reshape(q, b, *spatial_shape, d)
        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = torch.sum(codes, dim=0)
        return self.project_out(codes_summed)

    def forward(self, x, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

        x = self.project_in(x)

        if self.soft_clamp_input_value is not None:
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        quantized_out = 0.0
        residual = self.layers[0].bound(x)

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout and torch.is_grad_enabled()

        if should_quantize_dropout:
            if rand_quantize_dropout_fixed_seed is None:
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = ceil((rand_quantize_dropout_index + 1) / quant_dropout_multiple_of) - 1

            null_indices = torch.full(x.shape[:2], -1.0, device=device, dtype=torch.long)

        with autocast("cuda", enabled=False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):
                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)

                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        quantized_out = self.project_out(quantized_out.to(x.dtype))

        all_indices = torch.stack(all_indices, dim=-1)

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        all_codes = self.get_codes_from_indices(all_indices)

        return (*ret, all_codes)


class Xcodec2CodecDecoderVocos(nn.Module):
    def __init__(
        self,
        config: Xcodec2Config,
    ):
        super().__init__()
        self.hop_length = config.hop_length

        self.quantizer = Xcodec2ResidualFSQ(
            dim=config.vq_dim, levels=config.vq_levels, num_quantizers=config.num_quantizers
        )

        self.backbone = Xcodec2VocosBackbone(config=config)

        self.head = ISTFTHead(
            dim=config.hidden_size, n_fft=self.hop_length * 4, hop_length=self.hop_length, padding="same"
        )

    def forward(self, x, vq=True):
        if vq is True:
            x = x.permute(0, 2, 1)
            x, q = self.quantizer(x)
            x = x.permute(0, 2, 1)
            q = q.permute(0, 2, 1)
            return x, q, None
        x = self.backbone(x)
        x, _ = self.head(x)

        return x, _

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None, :, :]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x):
        x = self.model(x)
        return x, None


class Xcodec2SemanticEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        encode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super(Xcodec2SemanticEncoder, self).__init__()

        # Initial convolution, maps input_channels to encode_channels
        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        # Residual block with two convolutional layers
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            encode_channels,
            encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            encode_channels,
            encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

        # Final convolution, maps encode_channels to code_dim
        self.final_conv = nn.Conv1d(
            in_channels=encode_channels,
            out_channels=code_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x):
        """
        Forward propagation method.

        Args:
            x (Tensor): Input tensor, shape (Batch, Input_channels, Length)

        Returns:
            Tensor: Encoded tensor, shape (Batch, Code_dim, Length)
        """
        x = self.initial_conv(x)  # (Batch, Encode_channels, Length)

        # Apply residual block operations
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = x + residual  # Residual connection

        x = self.final_conv(x)  # (Batch, Code_dim, Length)
        return x


class Xcodec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = Xcodec2Config
    base_model_prefix = "xcodec2"
    main_input_name = "input_values"
    supports_gradient_checkpointing = False
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Xcodec2SnakeBeta):
            module.alpha.data.fill_(1.0)
            module.beta.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


XCODEC2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`Xcodec2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XCODEC2_INPUTS_DOCSTRING = r"""
    args:
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`):
            The raw float values of the input audio waveform.
        audio_codes (`torch.LongTensor`  of shape `(batch_size, 1, codes_length)`:
            Discrete code indices computed using `model.encode`.
        return_dict (`bool`, *optional*):
            whether to return a `Xcodec2Output` or a plain tuple.
"""


# Taking inspiration form modeling code of original authors: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py
@add_start_docstrings(
    "The Xcodec2 neural audio codec model.",
    XCODEC2_INPUTS_DOCSTRING,
)
class Xcodec2Model(Xcodec2PreTrainedModel):
    config_class = Xcodec2Config

    def __init__(self, config: Xcodec2Config):
        super().__init__(config)

        self.semantic_model = AutoModel.from_config(config.semantic_model_config).eval()
        self.semantic_feature_extractor = AutoFeatureExtractor.from_pretrained(config.semantic_model_id)
        self.semantic_encoder = Xcodec2SemanticEncoder(
            config.semantic_hidden_size, config.semantic_hidden_size, config.semantic_hidden_size
        )

        self.acoustic_encoder = Xcodec2CodecEncoder(
            downsampling_ratios=config.downsampling_ratios, hidden_dim=config.encoder_hidden_size
        )
        self.decoder = Xcodec2CodecDecoderVocos(config=config)
        self.fc_prior = nn.Linear(config.intermediate_size, config.intermediate_size)
        self.fc_post_a = nn.Linear(config.intermediate_size, config.decoder_hidden_size)

        self.post_init()

    def apply_weight_norm(self, legacy=True):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            weight_norm = nn.utils.parametrizations.weight_norm

        # Weight norm was only applied in acoustic encoder of original model
        # -- to initial_conv
        weight_norm(self.acoustic_encoder.initial_conv)

        # -- to encoder blocks
        for encoder_block in self.acoustic_encoder.encoder_blocks:
            # -- to each residual unit in the block
            for residual_unit in encoder_block.residual_units:
                weight_norm(residual_unit.conv1)
                weight_norm(residual_unit.conv2)
            # -- to the final conv in the encoder block
            weight_norm(encoder_block.conv)

        # -- to final_conv
        weight_norm(self.acoustic_encoder.final_conv)

    def remove_weight_norm(self, legacy=True):
        """Remove weight normalization from layers that have it applied via apply_weight_norm."""
        remove_weight_norm = nn.utils.remove_weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            remove_weight_norm = torch.nn.utils.parametrize.remove_parametrizations

        # Remove weight norm from acoustic_encoder
        # -- from initial_conv
        try:
            remove_weight_norm(self.acoustic_encoder.initial_conv)
        except (ValueError, RuntimeError):
            raise ValueError("Not able to remove weight norm. Have you run `apply_weight_norm?`")

        # -- from encoder blocks
        for encoder_block in self.acoustic_encoder.encoder_blocks:
            # -- from each residual unit in the block
            for residual_unit in encoder_block.residual_units:
                remove_weight_norm(residual_unit.conv1)
                remove_weight_norm(residual_unit.conv2)
            # -- from the final conv in the encoder block
            remove_weight_norm(encoder_block.conv)

        # -- from final_conv
        remove_weight_norm(self.acoustic_encoder.final_conv)

    def encode(
        self,
        input_values,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Xcodec2EncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `codebook` of shape `[batch_size, num_codebooks, frames]`, the discrete encoded codes for the input audio waveform.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        channels = input_values.shape[1]
        if channels != 1:
            raise ValueError(f"Audio must be mono, but got {channels}")

        # 1) Get semantic embedding
        # -- apply feature extractor: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L111
        input_features = (
            self.semantic_feature_extractor(
                input_values.cpu(),
                sampling_rate=self.semantic_feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            .input_features.to(self.dtype)
            .to(self.device)
        )
        # -- extract 16th layer of semantic model: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L64
        semantic_output = self.semantic_model(input_features, output_hidden_states=True)
        semantic_hidden_16 = semantic_output.hidden_states[16]
        semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)
        semantic_encoded = self.semantic_encoder(semantic_hidden_16)

        # 2) Get acoustic embedding
        vq_emb = self.acoustic_encoder(input_values)
        vq_emb = vq_emb.transpose(1, 2)

        # 3) Concat embeddings and apply final layers
        if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
            vq_emb = vq_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]
        concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # 4) Get codes for decoder
        _, vq_code, _ = self.decoder(concat_emb, vq=True)

        if not return_dict:
            return vq_code

        return Xcodec2EncoderOutput(
            audio_codes=vq_code,
        )

    def decode(
        self,
        audio_codes,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Xcodec2DecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, 1, codes_length)`):
                Discrete code indices computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            Decoded audio values of shape `(batch_size, 1, num_samples)` obtained using the decoder part of Xcodec2.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vq_post_emb = self.decoder.quantizer.get_output_from_indices(
            audio_codes.transpose(1, 2)
            if isinstance(audio_codes, torch.Tensor)
            else audio_codes.audio_codes.transpose(1, 2)
        )
        vq_post_emb = vq_post_emb.transpose(1, 2)
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)
        recon_audio = self.decoder(vq_post_emb.transpose(1, 2), vq=False)[0]

        if not return_dict:
            return recon_audio

        return Xcodec2DecoderOutput(
            audio_values=recon_audio,
        )

    @add_start_docstrings_to_model_forward(XCODEC2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Xcodec2Output, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: torch.Tensor,
        audio_codes: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Xcodec2Output]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, Xcodec2Model

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "bezzam/xcodec2"
        >>> model = Xcodec2Model.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        length = input_values.shape[-1]

        if audio_codes is None:
            audio_codes = self.encode(input_values, return_dict=False)

        audio_values = self.decode(audio_codes, return_dict=False)[0][..., :length]

        if not return_dict:
            return (audio_codes, audio_values)

        return Xcodec2Output(
            audio_values=audio_values,
            audio_codes=audio_codes,
        )


__all__ = ["Xcodec2Model", "Xcodec2PreTrainedModel"]
