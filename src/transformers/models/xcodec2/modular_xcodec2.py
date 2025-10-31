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
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import int32, nn, sin, sinc
from torch.amp import autocast
from torch.nn import Module, Parameter

from transformers.activations import ACT2FN

from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)
from .configuration_xcodec2 import Xcodec2Config


# General docstring
_CONFIG_FOR_DOC = "Xcodec2Config"


@dataclass
class Xcodec2Output(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
            Decoded audio waveform values in the time domain, obtained using the decoder
            part of Xcodec2. These represent the reconstructed audio signal.
        audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`. These are the quantized
            representations of the input audio used for further processing or generation.
        quantized_representation (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
            Quantized continuous representation of input's embedding.
        codes_padding_mask (`torch.int32` of shape `(batch_size, 1, codes_length)`, *optional*):
            Downsampled `padding_mask` for indicating valid audio codes in `audio_codes`.
    """

    audio_values: Optional[torch.FloatTensor] = None
    audio_codes: Optional[torch.LongTensor] = None
    quantized_representation: Optional[torch.Tensor] = None
    codes_padding_mask: Optional[torch.Tensor] = None


@dataclass
class Xcodec2EncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`. These represent
            the compressed, quantized form of the input audio signal that can be
            used for storage, transmission, or generation.
        quantized_representation (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
            Quantized continuous representation of input's embedding.
        codes_padding_mask (`torch.int32` of shape `(batch_size, 1, codes_length)`, *optional*):
            Downsampled `padding_mask` for indicating valid audio codes in `audio_codes`.

    """

    audio_codes: Optional[torch.LongTensor] = None
    quantized_representation: Optional[torch.Tensor] = None
    codes_padding_mask: Optional[torch.Tensor] = None


@dataclass
class Xcodec2DecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, segment_length)`, *optional*):
            Decoded audio waveform values in the time domain, obtained by converting
            the discrete codes back into continuous audio signals. This represents
            the reconstructed audio that can be played back.
    """

    audio_values: Optional[torch.FloatTensor] = None


if is_torch_flex_attn_available():
    pass


logger = logging.get_logger(__name__)


# Default `unsqueeze_dim=2`
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 2):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# See here for their attention implementation:
# https://huggingface.co/HKUSTAudio/xcodec2/blob/main/vq/bs_roformer5.py
class Xcodec2Attention(LlamaAttention):
    def __init__(self, config: Xcodec2Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.is_causal = False


class Xcodec2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


# Unlike LlamaMLP, does not have `gate_proj` layer
class Xcodec2MLP(nn.Module):
    def __init__(self, config: Xcodec2Config):
        super().__init__()
        dim = config.hidden_size

        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# Original Transformer block: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/vq/bs_roformer5.py#L91
class Xcodec2DecoderLayer(LlamaDecoderLayer):
    pass


class Xcodec2SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude and frequency of the periodic components.

    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input

    Parameters:
        - dim: The number of input features
        - logscale: Whether to use logarithmic scaling for alpha and beta (default: False)

    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195

    Examples:
        >>> activation = Xcodec2SnakeBeta(256)
        >>> x = torch.randn(1, 256, 100)  # (batch, channels, time)
        >>> y = activation(x)
    """

    def __init__(self, dim, logscale=True):
        """
        Args:
            dim: Shape of the input features (channels dimension)
            logscale: Whether to use log scale initialization (default: True)
                            If True, alpha and beta are initialized to zeros
                            If False, alpha and beta are initialized to ones

        Note:
            - alpha controls the frequency of the periodic components
            - beta controls the magnitude of the periodic components
        """
        super().__init__()
        self.dim = dim

        # initialize alpha
        self.logscale = logscale
        if self.logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(dim))
            self.beta = Parameter(torch.zeros(dim))
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(dim))
            self.beta = Parameter(torch.ones(dim))
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """
    Creates a Kaiser-windowed sinc filter for low-pass filtering.

    Args:
        cutoff: Normalized cutoff frequency (0 to 0.5, where 0.5 is the Nyquist frequency)
        half_width: Transition bandwidth parameter controlling filter roll-off
        kernel_size: Size of the filter kernel

    Returns:
        torch.Tensor: A 1D filter kernel of shape (1, 1, kernel_size)
    """
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
    """
    Used by `DownSample1d`.
    """

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
    """
    1D upsampling module using transposed convolution with a sinc filter.

    This module performs anti-aliased upsampling by first padding the input signal,
    then applying a transposed convolution with a Kaiser-windowed sinc filter,
    and finally trimming the resulting signal to remove boundary effects.

    Args:
        ratio (int): Upsampling ratio (default: 2)
        kernel_size (int, optional): Size of the filter kernel.
            If None, it's automatically calculated as 6 * ratio.
    """

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
    """
    1D downsampling module using a low-pass filter.

    This module performs anti-aliased downsampling by applying a low-pass filter
    with an appropriate cutoff frequency followed by strided convolution.

    Args:
        ratio (int): Downsampling ratio (default: 2)
        kernel_size (int, optional): Size of the filter kernel.
            If None, it's automatically calculated as 6 * ratio.
    """

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
    """
    A module that applies activation function with up-sampling and down-sampling.

    This module first up-samples the input signal, applies an activation function,
    and then down-samples the result. This architecture allows for more complex
    non-linear transformations of the signal.

    Args:
        activation (`torch.nn.Module`):
            The activation function module to apply (e.g., nn.ReLU(), nn.LeakyReLU()).
        up_ratio (`int`, *optional*, defaults to 2):
            The up-sampling ratio for increasing temporal resolution before activation.
        down_ratio (`int`, *optional*, defaults to 2):
            The down-sampling ratio for decreasing temporal resolution after activation.
        up_kernel_size (`int`, *optional*, defaults to 12):
            The kernel size used in the up-sampling operation.
        down_kernel_size (`int`, *optional*, defaults to 12):
            The kernel size used in the down-sampling operation.
    """

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
    """
    A residual unit that combines dilated convolutions with activation functions.

    This unit performs a series of operations while maintaining a residual connection:
    1. First activation with up/down-sampling
    2. Dilated convolution with kernel size 7
    3. Second activation with up/down-sampling
    4. 1x1 convolution for channel mixing
    5. Addition of the input (residual connection)

    Args:
        dim (`int`, *optional*, defaults to 16):
            The number of channels in the input and output tensors.
        dilation (`int`, *optional*, defaults to 1):
            The dilation rate for the main convolution operation. Higher values
            increase the receptive field without increasing parameter count.
    """

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.activation1 = Activation1d(activation=Xcodec2SnakeBeta(dim, logscale=True))
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.activation2 = Activation1d(activation=Xcodec2SnakeBeta(dim, logscale=True))
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        return residual + x


class EncoderBlock(nn.Module):
    """
    Encoder block for the Xcodec2 acoustic encoder.

    This block consists of:
    1. A series of residual units with different dilation rates
    2. An activation function with upsampling and downsampling
    3. A strided convolution to change the dimension and downsample if stride > 1

    Args:
        dim (int): Dimension of the output features. Input dimension is assumed to be dim // 2
        stride (int): Stride for the final convolution, controls downsampling factor
        dilations (tuple): Dilation rates for the residual units
    """

    def __init__(self, dim: int = 16, stride: int = 1, dilations=(1, 3, 9)):
        super().__init__()
        self.residual_units = nn.ModuleList([ResidualUnit(dim // 2, dilation=d) for d in dilations])
        self.activation = Activation1d(activation=Xcodec2SnakeBeta(dim // 2, logscale=True))
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
    """
    The encoder component of the Xcodec2 model for audio encoding.

    This encoder transforms raw audio waveforms into latent representations:
    1. Initial convolution to process the raw audio input
    2. Series of encoder blocks with progressively increasing feature dimensions and downsampling
    3. Final activation and projection to the target hidden dimension

    Args:
        d_model (`int`, *optional*, defaults to 48):
            Initial dimension of the model. This will be doubled at each encoder block.
        downsampling_ratios (`List[int]`, *optional*, defaults to [2, 2, 4, 4, 5]):
            List of downsampling ratios for each encoder block. The product of these values
            determines the total temporal compression factor.
        dilations (`tuple`, *optional*, defaults to (1, 3, 9)):
            Tuple of dilation rates used in the residual units within each encoder block.
        hidden_dim (`int`, *optional*, defaults to 1024):
            Dimension of the final latent representation after encoding.
    """

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

        self.final_activation = Activation1d(activation=Xcodec2SnakeBeta(d_model, logscale=True))
        self.final_conv = nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the Xcodec2 encoder.

        Args:
            x (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Input audio waveform tensor. Expected to be a mono audio signal with shape
                (batch_size, 1, sequence_length).

        Returns:
            `torch.Tensor` of shape `(batch_size, compressed_length, hidden_dim)`:
                Encoded audio representation with temporal dimension compressed by the product
                of all downsampling_ratios. The output tensor is transposed to have the
                sequence dimension second (batch_size, seq_len, features) for compatibility
                with transformer-based models.
        """
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


class ResNetBlock(nn.Module):
    """
    A basic residual block for 1D convolutional networks.

    This block consists of:
    1. GroupNorm + SiLU-like activation (x * sigmoid(x)) + Conv1d
    2. GroupNorm + SiLU-like activation + Dropout + Conv1d
    3. Residual connection, with optional projection if input and output channels differ

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (defaults to in_channels if None)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, *, in_channels, out_channels=None, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = h * torch.sigmoid(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = h * torch.sigmoid(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Xcodec2VocosBackbone(nn.Module):
    """
    Hybrid ResNet-Transformer architecture.

    Architecture overview:
    - Input embedding: A 1D convolution (kernel size 7, padding 3) projects raw features
      into the model's hidden dimension.
    - Prior ResNet blocks: Two ResNet-style residual blocks operate on the hidden states
      to provide early nonlinear feature processing.
    - Rotary embeddings: Rotary positional embeddings (RoPE) are initialized to provide
      position-dependent information to the Transformer layers.
    - Transformer stack: A sequence of Xcodec2DecoderLayer modules applies self-attention
      with rotary embeddings, interleaved with MLPs, to model long-range dependencies.
    - Normalization: A final LayerNorm is applied to stabilize training and outputs.
    - Post ResNet blocks: Two additional ResNet-style blocks refine the representation
      after the Transformer stack.

    This design combines convolutional inductive biases (via ResNet-style blocks and the
    initial Conv1d embedding) with the global sequence modeling capabilities of
    Transformers, making it suitable for sequence data such as audio or other temporal
    signals.
    """

    def __init__(self, config: Xcodec2Config):
        super().__init__()

        self.embed = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=7, padding=3)
        block_in = config.hidden_size
        dropout = config.resnet_dropout

        self.prior_blocks = nn.ModuleList(
            [
                ResNetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout),
                ResNetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout),
            ]
        )

        # Initialize rotary embeddings
        self.position_ids = torch.arange(config.num_attention_heads).unsqueeze(0)
        self.rotary_emb = Xcodec2RotaryEmbedding(config=config)

        # Create transformer layers
        self.transformers = nn.ModuleList(
            [Xcodec2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.post_blocks = nn.ModuleList(
            [
                ResNetBlock(
                    in_channels=config.hidden_size,
                    out_channels=config.hidden_size,
                    dropout=dropout,
                ),
                ResNetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Xcodec2VocosBackbone.

        The processing flow includes:
        1. Initial embedding via 1D convolution
        2. Processing through ResNet blocks
        3. Transformer processing with rotary position embeddings
        4. Final ResNet blocks and normalization

        Args:
            x (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input tensor containing features to process.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`:
                Processed features with the same shape as the input but transformed
                through the backbone architecture.
        """
        # Handle initial transformations
        x = x.transpose(1, 2)
        x = self.embed(x)

        # Process through prior_blocks
        for block in self.prior_blocks:
            x = block(x)

        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Generate rotary embeddings
        position_embeddings = self.rotary_emb(x, self.position_ids.to(x.device))

        # Apply transformer layers with position embeddings
        for layer in self.transformers:
            x = layer(
                x,
                position_embeddings=position_embeddings,
            )

        # Handle final transformations
        x = x.transpose(1, 2)

        # Process through post_blocks
        for block in self.post_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)

        return x


class Xcodec2ISTFT(nn.Module):
    """
    As in original Vocos code:
    https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/spectral_ops.py#L7

    Custom ISTFT implementation to support "same" padding as in Vocos.
    """

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        if config.istft_padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = config.istft_padding
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = getattr(config, "win_length", config.n_fft)
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                  is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if spec.dim() != 3:
            raise ValueError("Expected a 3D tensor as input")

        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)

        elif self.padding == "same":
            # Custom implementation from Vocos codebase
            pad = (self.win_length - self.hop_length) // 2
            n_frames = spec.shape[-1]

            # Inverse FFT
            ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
            ifft = ifft * self.window[None, :, None]

            # Overlap and Add
            output_size = (n_frames - 1) * self.hop_length + self.win_length
            y = F.fold(
                ifft,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )[:, 0, 0, pad:-pad]

            # Window envelope
            window_sq = self.window.square().expand(1, n_frames, -1).transpose(1, 2)
            window_envelope = F.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            ).squeeze()[pad:-pad]

            # Normalize
            if not (window_envelope > 1e-11).all():
                raise ValueError("Window envelope values are too small (<=1e-11)")
            return y / window_envelope

        else:
            raise ValueError("Padding must be 'center' or 'same'.")


class Xcodec2ISTFTHead(nn.Module):
    """
    As in original Vocos code:
    https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/heads.py#L26
    - Projects the hidden states to STFT coefficients (magnitude and phase)
    - Applies ISTFT to reconstruct the time-domain audio signal
    """

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.out = torch.nn.Linear(config.hidden_size, config.n_fft + 2)
        self.istft = Xcodec2ISTFT(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
            Tensor: Predicted STFT coefficients of shape (B, L, N+2), where N is the number of frequency bins.
        """
        x_pred = self.out(x).transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        spectrogram_real = torch.cos(p)
        spectrogram_imag = torch.sin(p)
        spectrogram_complex = mag * (spectrogram_real + 1j * spectrogram_imag)
        audio = self.istft(spectrogram_complex)
        return audio


def round_ste(z):
    """
    Round with straight through gradients.
    Used in `Xcodec2FSQ`.
    """
    zhat = z.round()
    return z + (zhat - z).detach()


def floor_ste(z):
    """
    Floor with straight through gradients.
    Used in `Xcodec2FSQ`.
    """
    zhat = z.floor()
    return z + (zhat - z).detach()


def maybe(fn):
    """
    Used in `Xcodec2FSQ`.
    """

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if x is None:
            return x
        return fn(x, *args, **kwargs)

    return inner


def get_maybe_sync_seed(device, max_size=10_000):
    """
    Used in `Xcodec2ResidualFSQ`.
    """
    rand_int = torch.randint(0, max_size, (), device=device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(rand_int)

    return rand_int.item()


class Xcodec2FSQ(Module):
    """
    Copied from https://github.com/lucidrains/vector-quantize-pytorch/blob/fe903ce2ae9c125ace849576aa6d09c5cec21fe4/vector_quantize_pytorch/finite_scalar_quantization.py#L61
    Simply changed asserts to raise ValueErrors.
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
        if num_codebooks > 1 and not keep_num_codebooks_dim:
            raise ValueError("When num_codebooks > 1, keep_num_codebooks_dim must be True")
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
        if zhat.shape[-1] != self.codebook_dim:
            raise ValueError(f"Expected zhat to have shape [-1, {self.codebook_dim}], but got {zhat.shape}")
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings"""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        if indices is None:
            raise ValueError("indices cannot be None")

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

        if z.shape[-1] != self.dim:
            raise ValueError(f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}")

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
    Simply changed asserts to raise ValueErrors.
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

        if quantize_dropout_cutoff_index < 0:
            raise ValueError("quantize_dropout_cutoff_index must be greater than or equal to 0")

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
            if not self.quantize_dropout > 0.0:
                raise ValueError(
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
        self.quantizer = Xcodec2ResidualFSQ(
            dim=config.vq_dim, levels=config.vq_levels, num_quantizers=config.num_quantizers
        )
        self.backbone = Xcodec2VocosBackbone(config=config)
        self.head = Xcodec2ISTFTHead(config=config)

    def quantize(self, x):
        """
        Quantize input features using the vector quantizer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, L) where B is batch size,
                              D is feature dimension, and L is sequence length.

        Returns:
            Tuple of (quantized_tensor, quantization_indices)
            - quantized_tensor: The quantized representation with shape (B, L, D)
            - quantization_indices: The indices in the codebook with shape (B, L, Q)
              where Q is the number of quantizers
        """
        x = x.permute(0, 2, 1)
        x, q = self.quantizer(x)
        x = x.permute(0, 2, 1)
        q = q.permute(0, 2, 1)
        return x, q

    def forward(self, x):
        """
        x (torch.Tensor):
            Projected audio codes to reconstruct as audio.

        Returns:
            audio_waveform: The reconstructed audio waveform with shape (B, 1, T).
        """
        x = self.backbone(x)
        x = self.head(x).unsqueeze(1)
        return x


class Xcodec2SemanticEncoder(nn.Module):
    """
    Maps input features of pre-trained semantic model to semantic embedding.
    """

    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        encode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()

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
        x = self.initial_conv(x)
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = x + residual
        x = self.final_conv(x)
        return x


class Xcodec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = Xcodec2Config
    base_model_prefix = "xcodec2"
    main_input_name = "audio"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Xcodec2SnakeBeta):
            # Initialize alpha and beta based on the logscale setting
            if module.logscale:
                # Log scale alphas initialized to zeros
                module.alpha.data.zero_()
                module.beta.data.zero_()
            else:
                # Linear scale alphas initialized to ones
                module.alpha.data.fill_(1.0)
                module.beta.data.fill_(1.0)


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
        audio (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`):
            Audio waveform.
        audio_spectrogram (`torch.FloatTensor` of shape `(batch_size, mel_bins, time_steps)`):
            Mel spectrogram.
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

        self.hop_length = config.hop_length  # needed for padding
        self.semantic_model = AutoModel.from_config(config.semantic_model_config).eval()
        # Adaptor of semantic model embedding
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
        audio: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Xcodec2EncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            audio (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Input audio waveform.
            audio_spectrogram (`torch.Tensor` of shape `(batch_size, mel_bins, time_steps)`):
                Input audio mel spectrogram for semantic encoding.
            padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Padding mask used to pad `audio`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `audio_codes` of shape `[batch_size, 1, frames]`, the discrete encoded codes for the input audio waveform.
            `quantized_representation` of shape `[batch_size, hidden_size, frames]`, the continuous quantized
                representation after quantization.
            `codes_padding_mask` of shape `[batch_size, 1, frames]`, downsampled `padding_mask` for indicating valid
                audio codes in `audio_codes`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 1) Semantic embedding: 16th layer of pretrained model: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L64
        semantic_output = self.semantic_model(audio_spectrogram, output_hidden_states=True)
        semantic_hidden_16 = semantic_output.hidden_states[16]
        semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)
        semantic_encoded = self.semantic_encoder(semantic_hidden_16)

        # 2) Get acoustic embedding
        vq_emb = self.acoustic_encoder(audio)
        vq_emb = vq_emb.transpose(1, 2)

        # 3) Concat embeddings and apply final layers
        if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
            vq_emb = vq_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]
        concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # 4) Get codes for decoder
        quantized_representation, audio_codes = self.decoder.quantize(concat_emb)

        # If provided, compute corresponding padding mask for audio codes
        codes_padding_mask = None
        if padding_mask is not None:
            # Expected token length, as in: https://github.com/zhenye234/X-Codec-2.0/blob/ccbbf340ff143dfa6a0ea7cd61ec34a8ba2f1c3d/inference_save_code.py#L89
            audio_length = padding_mask.sum(dim=-1, keepdim=True).cpu()
            token_length = audio_length // self.hop_length
            codes_padding_mask = torch.zeros(audio_codes.shape, dtype=padding_mask.dtype)
            idx = torch.arange(audio_codes.shape[-1]).view(1, -1)
            codes_padding_mask = (idx < token_length).to(padding_mask.dtype).to(padding_mask.device)

        if not return_dict:
            return audio_codes, quantized_representation, codes_padding_mask

        return Xcodec2EncoderOutput(
            audio_codes=audio_codes,
            quantized_representation=quantized_representation,
            codes_padding_mask=codes_padding_mask,
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

        vq_post_emb = self.decoder.quantizer.get_output_from_indices(audio_codes.transpose(1, 2))
        vq_post_emb = self.fc_post_a(vq_post_emb)
        recon_audio = self.decoder(vq_post_emb)

        if not return_dict:
            return recon_audio

        return Xcodec2DecoderOutput(
            audio_values=recon_audio,
        )

    @add_start_docstrings_to_model_forward(XCODEC2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Xcodec2Output, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        audio: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Xcodec2Output]:
        r"""
        Returns:
            `Xcodec2Output` or `tuple(torch.FloatTensor, torch.LongTensor, torch.FloatTensor)`:
            - `audio_values` (`torch.FloatTensor` of shape `(batch_size, 1, num_samples)`):
                Reconstructed audio waveform.
            - `audio_codes` (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`):
                Discrete code indices computed using `model.encode`.
            - `quantized_representation` (`torch.FloatTensor` of shape `(batch_size, hidden_size, frames)`):
                The continuous quantized representation after quantization.
            - `codes_padding_mask` (`torch.int32` of shape `(batch_size, 1, codes_length)`):
                Downsampled `padding_mask` for indicating valid audio codes in `audio_codes`.

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, Xcodec2Model

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> audio = dataset["train"]["audio"][0]["array"]

        >>> model_id = "hf-audio/xcodec2"
        >>> model = Xcodec2Model.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # for truncating output audio to original length
        length = audio.shape[-1]

        audio_codes, quantized_representation, codes_padding_mask = self.encode(
            audio, audio_spectrogram=audio_spectrogram, padding_mask=padding_mask, return_dict=False
        )
        audio_values = self.decode(audio_codes, return_dict=False)[..., :length]

        if not return_dict:
            return (audio_values, audio_codes, quantized_representation, codes_padding_mask)

        return Xcodec2Output(
            audio_values=audio_values,
            audio_codes=audio_codes,
            quantized_representation=quantized_representation,
            codes_padding_mask=codes_padding_mask,
        )


__all__ = ["Xcodec2Model", "Xcodec2PreTrainedModel"]
