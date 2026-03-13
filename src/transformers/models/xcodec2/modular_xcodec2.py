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
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import int32, nn, sin, sinc
from torch.amp import autocast
from torch.nn import Module, Parameter

from transformers.activations import ACT2FN

from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    replace_return_docstrings,
)
from ..auto import AutoModel
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    rotate_half
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


# RoPE is applied on the attention head rather than sequence dimension
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


class Xcodec2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Xcodec2MLP(nn.Module):
    def __init__(self, config: Xcodec2Config):
        super().__init__()
        dim = config.hidden_size

        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Xcodec2DecoderLayer(LlamaDecoderLayer):
    pass


class Xcodec2SnakeBeta(nn.Module):
    """A modified Snake function from https://arxiv.org/abs/2006.08195"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = Parameter(torch.zeros(dim))
        self.beta = Parameter(torch.zeros(dim))
        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(hidden_states * alpha), 2)

        return hidden_states


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Creates a Kaiser-windowed sinc filter for low-pass filtering."""
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


class UpSample1d(nn.Module):
    """1D upsampling module using transposed convolution with a sinc filter."""

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.cutoff = 0.5 / ratio
        self.half_width = 0.6 / ratio
        filter = kaiser_sinc_filter1d(cutoff=self.cutoff, half_width=self.half_width, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    def forward(self, hidden_states):
        _, channels, _ = hidden_states.shape

        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]

        return hidden_states


class DownSample1d(nn.Module):
    """1D downsampling module using a low-pass filter.
    """

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        
        # Low-pass filter setup
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        even = self.kernel_size % 2 == 0
        self.pad_left = self.kernel_size // 2 - int(even)
        self.pad_right = self.kernel_size // 2
        
        filter = kaiser_sinc_filter1d(cutoff, half_width, self.kernel_size)
        self.register_buffer("filter", filter)

    def forward(self, hidden_states):
        _, channels, _ = hidden_states.shape
        
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        hidden_states = F.conv1d(hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        
        return hidden_states


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

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states


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
        self.activation1 = Activation1d(activation=Xcodec2SnakeBeta(dim))
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.activation2 = Activation1d(activation=Xcodec2SnakeBeta(dim))
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.activation1(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation2(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return residual + hidden_states


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
        self.activation = Activation1d(activation=Xcodec2SnakeBeta(dim // 2))
        self.conv = nn.Conv1d(
            dim // 2,
            dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2 + stride % 2,
        )

    def forward(self, hidden_states):
        for residual_unit in self.residual_units:
            hidden_states = residual_unit(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Xcodec2AcousticEncoder(nn.Module):
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
        for _, stride in enumerate(downsampling_ratios):
            d_model *= 2
            self.encoder_blocks.append(EncoderBlock(d_model, stride=stride, dilations=dilations))

        self.final_activation = Activation1d(activation=Xcodec2SnakeBeta(d_model))
        self.final_conv = nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states):
        """
        Forward pass of the Xcodec2 encoder.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
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
        hidden_states = self.initial_conv(hidden_states)

        # Apply all encoder blocks
        for encoder_block in self.encoder_blocks:
            hidden_states = encoder_block(hidden_states)

        # Final processing
        hidden_states = self.final_activation(hidden_states)
        hidden_states = self.final_conv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states


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

    def forward(self, hidden_states):
        residual = hidden_states
        residual = self.norm1(residual)
        residual = residual * torch.sigmoid(residual)
        residual = self.conv1(residual)
        residual = self.norm2(residual)
        residual = residual * torch.sigmoid(residual)
        residual = self.dropout(residual)
        residual = self.conv2(residual)

        if self.in_channels != self.out_channels:
            hidden_states = self.nin_shortcut(hidden_states)

        return hidden_states + residual





class Xcodec2FSQ(Module):
    """
    Finite Scalar Quantization module.

    Based on https://github.com/lucidrains/vector-quantize-pytorch/blob/fe903ce2ae9c125ace849576aa6d09c5cec21fe4/vector_quantize_pytorch/finite_scalar_quantization.py#L61
    """

    def __init__(self, levels: list[int]):
        super().__init__()

        self.levels = levels
        self.codebook_dim = len(levels)

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.codebook_size = math.prod(levels)
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, hidden_states, eps: float = 1e-3):
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (hidden_states + shift).tanh() * half_l - offset

    def _indices_to_codes(self, indices):
        """Convert codebook indices to normalized codes."""
        # Convert to level indices
        indices = indices.unsqueeze(-1)
        level_indices = (indices // self._basis) % self._levels
        # Scale and shift inverse to get normalized codes
        half_width = self._levels // 2
        codes = (level_indices - half_width) / half_width
        return codes

    def codes_to_indices(self, codes):
        """Converts a `code` to an index in the codebook."""
        if codes.shape[-1] != self.codebook_dim:
            raise ValueError(f"Expected codes to have shape [-1, {self.codebook_dim}], but got {codes.shape}")
        # Scale and shift codes
        half_width = self._levels // 2
        codes = (codes * half_width) + half_width
        return (codes * self._basis).sum(dim=-1).to(int32)

    def forward(self, hidden_states):
        if hidden_states.shape[-1] != self.codebook_dim:
            raise ValueError(
                f"expected dimension of {self.codebook_dim} but found dimension of {hidden_states.shape[-1]}"
            )

        with autocast("cuda", enabled=False):
            orig_dtype = hidden_states.dtype

            if orig_dtype not in (torch.float32, torch.float64):
                hidden_states = hidden_states.float()

            # Quantize: bound and round with straight-through gradient
            half_width = self._levels // 2
            bounded = self.bound(hidden_states)
            rounded = bounded.round()
            codes = bounded + (rounded - bounded).detach()
            codes = codes / half_width
            
            indices = self.codes_to_indices(codes)
            codes = codes.to(orig_dtype)

        return codes, indices


class Xcodec2ISTFT(nn.Module):
    """
    Inverse Short-Time Fourier Transform (ISTFT) module.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spectrogram_complex):
        if spectrogram_complex.dim() != 3:
            raise ValueError("Expected a 3D tensor as input")

        pad = (self.win_length - self.hop_length) // 2
        n_frames = spectrogram_complex.shape[-1]

        # Inverse FFT
        ifft = torch.fft.irfft(spectrogram_complex, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (n_frames - 1) * self.hop_length + self.win_length
        audio = F.fold(
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
        audio = audio / window_envelope

        return audio.unsqueeze(1)


class Xcodec2ISTFTHead(nn.Module):
    """
    Head for converting decoder outputs to waveform via STFT projection and ISTFT.
    """

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.out = nn.Linear(config.hidden_size, config.n_fft + 2)
        self.istft = Xcodec2ISTFT(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=getattr(config, "win_length", config.n_fft),
        )

    def forward(self, hidden_states):
        stft_pred = self.out(hidden_states).transpose(1, 2)
        mag, phase = stft_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        spectrogram_complex = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        return self.istft(spectrogram_complex)


class Xcodec2Quantizer(Module):
    """
    Finite Scalar Quantization wrapper with projection layers.

    This module wraps `Xcodec2FSQ` and provides projection layers when the codebook
    dimension differs from the model's hidden dimension. Xcodec2 uses only a single codebook.

    Args:
        levels: List of quantization levels for each dimension (e.g., [4, 4, 4, 4, 4, 4, 4, 4])
        dim: Model hidden dimension. If None, uses codebook_dim (len(levels))

    Based on https://github.com/lucidrains/vector-quantize-pytorch/blob/fe903ce2ae9c125ace849576aa6d09c5cec21fe4/vector_quantize_pytorch/residual_fsq.py#L49
    """

    def __init__(
        self,
        levels: list[int],
        dim: int,
    ):
        super().__init__()
        self.project_in = nn.Linear(dim, len(levels))
        self.project_out = nn.Linear(len(levels), dim)
        self.fsq = Xcodec2FSQ(levels=levels)

    def get_output_from_indices(self, indices):
        """Convert codebook indices back to embeddings with projection applied."""
        indices = indices.squeeze(-1)  # (batch, seq, 1) -> (batch, seq)
        codes = self.fsq.implicit_codebook[indices]
        return self.project_out(codes)

    def forward(self, hidden_states):
        hidden_states = self.project_in(hidden_states)
        bounded = self.fsq.bound(hidden_states)

        with autocast("cuda", enabled=False):
            quantized_out, indices = self.fsq(bounded)

        quantized_out = self.project_out(quantized_out.to(hidden_states.dtype))

        # Add trailing dimension for single codebook: (batch, seq) -> (batch, seq, 1)
        indices = indices.unsqueeze(-1)

        return quantized_out, indices


class Xcodec2Decoder(nn.Module):
    """Vocos-based decoder with ResNet, Transformer, and ISTFT head for audio reconstruction."""

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.embed = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=7, padding=3)
        self.prior_blocks = nn.ModuleList(
            [
                ResNetBlock(in_channels=config.hidden_size, out_channels=config.hidden_size, dropout=config.resnet_dropout),
                ResNetBlock(in_channels=config.hidden_size, out_channels=config.hidden_size, dropout=config.resnet_dropout),
            ]
        )

        self.num_attention_heads = config.num_attention_heads
        self.rotary_emb = Xcodec2RotaryEmbedding(config=config)
        self.transformers = nn.ModuleList(
            [Xcodec2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.post_blocks = nn.ModuleList(
            [
                ResNetBlock(in_channels=config.hidden_size, out_channels=config.hidden_size, dropout=config.resnet_dropout),
                ResNetBlock(in_channels=config.hidden_size, out_channels=config.hidden_size, dropout=config.resnet_dropout),
            ]
        )
        
        self.head = Xcodec2ISTFTHead(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.embed(hidden_states)
        
        for block in self.prior_blocks:
            hidden_states = block(hidden_states)
        
        hidden_states = hidden_states.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        position_ids = torch.arange(self.num_attention_heads).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids.to(hidden_states.device))
        
        for layer in self.transformers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
        
        hidden_states = hidden_states.transpose(1, 2)
        
        for block in self.post_blocks:
            hidden_states = block(hidden_states)
        
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.final_layer_norm(hidden_states)
        
        return self.head(hidden_states)


class Xcodec2SemanticAdapter(nn.Module):
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

    def forward(self, hidden_states):
        hidden_states = self.initial_conv(hidden_states)
        residual = hidden_states
        hidden_states = self.act1(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act2(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.final_conv(hidden_states)
        return hidden_states


class Xcodec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = Xcodec2Config
    base_model_prefix = "xcodec2"
    main_input_name = "audio"

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Xcodec2SnakeBeta):
            init.zeros_(module.alpha)
            init.zeros_(module.beta)
        elif isinstance(module, Xcodec2FSQ):
            levels = module.levels
            init.copy_(module._levels, torch.tensor(levels, dtype=int32))
            init.copy_(module._basis, torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32))
            init.copy_(module.implicit_codebook, module._indices_to_codes(torch.arange(module.codebook_size)))


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
        self.semantic_adapter = Xcodec2SemanticAdapter(
            config.semantic_hidden_size, config.semantic_hidden_size, config.semantic_hidden_size
        )

        self.acoustic_encoder = Xcodec2AcousticEncoder(
            downsampling_ratios=config.downsampling_ratios, hidden_dim=config.encoder_hidden_size
        )
        self.decoder = Xcodec2Decoder(config=config)
        self.quantizer = Xcodec2Quantizer(dim=config.vq_dim, levels=config.vq_levels)
        self.fc_prior = nn.Linear(config.intermediate_size, config.intermediate_size)
        self.fc_post_a = nn.Linear(config.intermediate_size, config.decoder_hidden_size)

        self.post_init()

    def quantize(self, hidden_states):
        """
        Quantize input features using the vector quantizer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, D, L) where B is batch size,
                              D is feature dimension, and L is sequence length.

        Returns:
            Tuple of (quantized_tensor, quantization_indices)
            - quantized_tensor: The quantized representation with shape (B, L, D)
            - quantization_indices: The codebook indices with shape (B, L, 1)
        """
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states, codes = self.quantizer(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        codes = codes.permute(0, 2, 1)
        return hidden_states, codes

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
    ) -> tuple | Xcodec2EncoderOutput:
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
        semantic_encoded = self.semantic_adapter(semantic_hidden_16)

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
        quantized_representation, audio_codes = self.quantize(concat_emb)

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
    ) -> tuple | Xcodec2DecoderOutput:
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

        vq_post_emb = self.quantizer.get_output_from_indices(audio_codes.transpose(1, 2))
        vq_post_emb = self.fc_post_a(vq_post_emb)
        recon_audio = self.decoder(vq_post_emb)

        if not return_dict:
            return recon_audio

        return Xcodec2DecoderOutput(
            audio_values=recon_audio,
        )

    @can_return_tuple
    @add_start_docstrings_to_model_forward(XCODEC2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Xcodec2Output, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        audio: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Xcodec2Output:
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
        # for truncating output audio to original length
        length = audio.shape[-1]

        audio_codes, quantized_representation, codes_padding_mask = self.encode(
            audio, audio_spectrogram=audio_spectrogram, padding_mask=padding_mask, return_dict=False
        )
        audio_values = self.decode(audio_codes, return_dict=False)[..., :length]

        return Xcodec2Output(
            audio_values=audio_values,
            audio_codes=audio_codes,
            quantized_representation=quantized_representation,
            codes_padding_mask=codes_padding_mask,
        )


__all__ = ["Xcodec2Model", "Xcodec2PreTrainedModel"]
