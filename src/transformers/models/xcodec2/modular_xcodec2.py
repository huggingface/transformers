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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...integrations import use_kernel_func_from_hub
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ...utils.generic import maybe_autocast
from ..auto import AutoModel
from ..clip.modeling_clip import CLIPMLP
from ..dac.modeling_dac import DacEncoder, DacEncoderBlock, DacResidualUnit
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, rotate_half
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAntiAliasedActivation1d,
    Qwen2_5OmniDownSample1d,
    Qwen2_5OmniSnakeBeta,
    Qwen2_5OmniUpSample1d,
    kaiser_sinc_filter1d,
)
from ..voxtral.modeling_voxtral import VoxtralPreTrainedModel
from .configuration_xcodec2 import Xcodec2Config


@dataclass
@auto_docstring
class Xcodec2Output(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
            Decoded audio waveform values in the time domain, obtained using the decoder
            part of Xcodec2. These represent the reconstructed audio signal.
        audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`. These are the quantized
            representations of the input audio used for further processing or generation.
        latents (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
            Quantized continuous representation of input's embedding.
        audio_codes_mask (`torch.int32` of shape `(batch_size, 1, codes_length)`, *optional*):
            Downsampled `padding_mask` for indicating valid audio codes in `audio_codes`.
    """

    audio_values: torch.FloatTensor | None = None
    audio_codes: torch.LongTensor | None = None
    latents: torch.Tensor | None = None
    audio_codes_mask: torch.Tensor | None = None


@dataclass
@auto_docstring
class Xcodec2EncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`. These represent
            the compressed, quantized form of the input audio signal that can be
            used for storage, transmission, or generation.
        latents (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
            Quantized continuous representation of input's embedding.
        audio_codes_mask (`torch.int32` of shape `(batch_size, 1, codes_length)`, *optional*):
            Downsampled `padding_mask` for indicating valid audio codes in `audio_codes`.

    """

    audio_codes: torch.LongTensor | None = None
    latents: torch.Tensor | None = None
    audio_codes_mask: torch.Tensor | None = None


@dataclass
@auto_docstring
class Xcodec2DecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, segment_length)`, *optional*):
            Decoded audio waveform values in the time domain, obtained by converting
            the discrete codes back into continuous audio signals. This represents
            the reconstructed audio that can be played back.
    """

    audio_values: torch.FloatTensor | None = None


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 2):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
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


class Xcodec2MLP(CLIPMLP):
    def __init__(self, config: Xcodec2Config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)


class Xcodec2DecoderLayer(LlamaDecoderLayer):
    pass


class Xcodec2SnakeBeta(Qwen2_5OmniSnakeBeta):
    pass


class Xcodec2DownSample1d(Qwen2_5OmniDownSample1d):
    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(
            hidden_states,
            self.filter.to(hidden_states.dtype).expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return out


class Xcodec2UpSample1d(Qwen2_5OmniUpSample1d):
    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states,
            self.filter.to(hidden_states.dtype).expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]
        return hidden_states


class Xcodec2AntiAliasedActivation1d(Qwen2_5OmniAntiAliasedActivation1d):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__(
            activation=activation,
            up_ratio=up_ratio,
            down_ratio=down_ratio,
            up_kernel_size=up_kernel_size,
            down_kernel_size=down_kernel_size,
        )
        self.upsample = Xcodec2UpSample1d(up_ratio, up_kernel_size)
        self.downsample = Xcodec2DownSample1d(down_ratio, down_kernel_size)


class Xcodec2ResidualUnit(DacResidualUnit):
    def __init__(self, dimension, dilation):
        super().__init__(dimension, dilation)
        self.snake1 = Xcodec2AntiAliasedActivation1d(activation=Xcodec2SnakeBeta(dimension))
        self.snake2 = Xcodec2AntiAliasedActivation1d(activation=Xcodec2SnakeBeta(dimension))


class Xcodec2EncoderBlock(DacEncoderBlock):
    def __init__(self, config: Xcodec2Config, stride: int = 1, stride_index: int = 1):
        super().__init__(config, stride, stride_index)
        dimension = config.encoder_hidden_size * 2**stride_index
        self.snake1 = Xcodec2AntiAliasedActivation1d(activation=Xcodec2SnakeBeta(dimension // 2))


class Xcodec2Encoder(DacEncoder):
    def __init__(self, config: Xcodec2Config):
        super().__init__(config)
        d_model = config.encoder_hidden_size * 2 ** len(config.downsampling_ratios)
        self.snake1 = Xcodec2AntiAliasedActivation1d(activation=Xcodec2SnakeBeta(d_model))


class Xcodec2ResNetBlock(nn.Module):
    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=config.hidden_size, eps=1e-6, affine=True)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=config.hidden_size, eps=1e-6, affine=True)
        self.activation2 = nn.SiLU()
        self.activation_dropout = config.activation_dropout
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation1(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.conv2(hidden_states)
        return hidden_states + residual


class Xcodec2FiniteScalarQuantization(nn.Module):
    """
    Finite Scalar Quantization (FSQ) module that quantizes continuous latent representations into discrete codes.
    Original code: https://github.com/lucidrains/vector-quantize-pytorch/blob/353d46027888dfb140c3c65a67a7356f1492d71d/vector_quantize_pytorch/finite_scalar_quantization.py#L64

    Original modeling uses `ResidualFSQ` with a single quantizer: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/vq/codec_decoder_vocos.py#L389
    But we can directly use FSQ since a main feature of Xcodec2 is that it uses a single codebook.
    """

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.quantization_levels = list(config.quantization_levels)
        levels, basis, codebook = self._compute_buffers()
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("codebook", codebook, persistent=False)

    def _compute_buffers(self, device=None):
        levels = torch.tensor(self.quantization_levels, dtype=torch.int32, device=device)
        basis = torch.cumprod(
            torch.tensor([1] + self.quantization_levels[:-1], device=device), dim=0, dtype=torch.int32
        )
        indices = torch.arange(int(np.prod(self.quantization_levels)), device=device).unsqueeze(-1)
        level_indices = (indices // basis) % levels
        half_width = levels // 2
        codebook = (level_indices - half_width) / half_width
        return levels, basis, codebook

    def _indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert integer codebook indices to normalized per-dimension codes in [-1, 1].
        """
        indices = indices.unsqueeze(-1)
        level_indices = (indices // self.basis) % self.levels
        half_width = self.levels // 2
        codes = (level_indices - half_width) / half_width
        return codes

    def bound(self, hidden_states: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        Constrain `hidden_states` to the valid quantization range for each dimension.

        Uses a scaled tanh to soft-clip values into the interval
        $[-(L-1)/2, (L-1)/2]$ (offset by 0.5 for even-level dimensions), where $L$ is
        the number of quantization levels. The small `eps` margin prevents values from
        saturating exactly at the boundary, which would zero out gradients.

        Args:
            hidden_states (`torch.Tensor`): Continuous input to be bounded.
            eps (`float`, *optional*, defaults to `1e-3`):
                Small margin added to the level range to avoid gradient saturation at boundaries.

        Returns:
            `torch.Tensor`: Bounded values in the valid quantization range.
        """
        half_range = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_range).atanh()
        return (hidden_states + shift).tanh() * half_range - offset

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = hidden_states.dtype
        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            hidden_states = hidden_states.float()
            half_width = self.levels // 2
            # Quantize: bound and round with straight-through gradient
            hidden_states = self.bound(hidden_states)
            rounded = hidden_states.round()
            codes = hidden_states + (rounded - hidden_states).detach()
            codes = codes / half_width
            # Code to indices
            code_scaled = (codes * half_width) + half_width
            indices = (code_scaled * self.basis).sum(dim=-1).to(torch.int32)
        return codes.to(original_dtype), indices


class Xcodec2ISTFTHead(nn.Module):
    """
    Head for converting decoder outputs to waveform via STFT projection and ISTFT.

    Uses custom "same" padding ISTFT from Vocos:
    https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/spectral_ops.py#L47
    """

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.n_fft + 2)
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.n_fft
        self.padding = (self.win_length - self.hop_length) // 2
        window = torch.hann_window(config.n_fft)
        self.register_buffer("window", window, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        stft_pred = self.linear(hidden_states).transpose(1, 2)
        magnitude, phase = stft_pred.chunk(2, dim=1)
        # Clamp like original: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/vq/codec_decoder_vocos.py#L138
        magnitude = torch.exp(magnitude).clamp(max=1e2)
        spectrogram_complex = magnitude * torch.exp(1j * phase)

        # Back to audio (ISTFT with "same" padding)
        time_frames = torch.fft.irfft(spectrogram_complex, self.n_fft, dim=1, norm="backward")
        time_frames = time_frames * self.window[None, :, None]
        num_frames = spectrogram_complex.shape[-1]
        output_size = (num_frames - 1) * self.hop_length + self.win_length
        audio = F.fold(
            time_frames,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, self.padding : -self.padding]

        # Normalize
        window_envelope = F.fold(
            self.window.square().expand(1, num_frames, -1).transpose(1, 2),
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[self.padding : -self.padding]
        # Clamp as expected by original: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/vq/codec_decoder_vocos.py#L82
        window_envelope = window_envelope.clamp(min=1e-11)
        audio = audio / window_envelope
        return audio.unsqueeze(1)


class Xcodec2Quantizer(nn.Module):
    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.quantizer = Xcodec2FiniteScalarQuantization(config)
        self.project_in = nn.Linear(config.quantization_dim, len(config.quantization_levels))
        self.project_out = nn.Linear(len(config.quantization_levels), config.quantization_dim)

    def from_codes(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.squeeze(-1)  # Remove channel dimension
        codes = self.quantizer.codebook[indices]
        return self.project_out(codes)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.project_in(hidden_states)
        original_dtype = hidden_states.dtype
        hidden_states = self.quantizer.bound(hidden_states)  # For consistency with original checkpoint
        quantized_out, indices = self.quantizer(hidden_states)
        quantized_out = self.project_out(quantized_out.to(original_dtype))
        indices = indices.unsqueeze(-1)  # Add channel dimension for single codebook
        return quantized_out, indices


class Xcodec2Decoder(nn.Module):
    """Vocos-based decoder with ResNet, Transformer, and ISTFT head for audio reconstruction."""

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size + config.semantic_model_config.hidden_size, config.hidden_size)
        self.embed = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=7, padding=3)
        self.prior_net = nn.ModuleList([Xcodec2ResNetBlock(config), Xcodec2ResNetBlock(config)])
        self.num_attention_heads = config.num_attention_heads
        self.rotary_emb = Xcodec2RotaryEmbedding(config=config)
        self.layers = nn.ModuleList(
            [Xcodec2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_net = nn.ModuleList([Xcodec2ResNetBlock(config), Xcodec2ResNetBlock(config)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.head = Xcodec2ISTFTHead(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.fc(hidden_states)

        # Conv ResNet: (batch, hidden, time)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.embed(hidden_states)
        for layer in self.prior_net:
            hidden_states = layer(hidden_states)

        # Transformer: (batch, time, hidden)
        # position_ids uses num_attention_heads so that RoPE produces cos/sin of shape (batch, num_heads, head_dim),
        # which broadcasts correctly against q/k of shape (batch, num_heads, seq_len, head_dim) via unsqueeze_dim=2
        # in `apply_rotary_pos_emb`
        hidden_states = hidden_states.transpose(1, 2)
        position_ids = torch.arange(self.num_attention_heads, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, **kwargs)

        # Conv ResNet: (batch, hidden, time)
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.post_net:
            hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return self.head(self.norm(hidden_states))


class Xcodec2SemanticAdapter(nn.Module):
    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=config.semantic_model_config.hidden_size,
            out_channels=config.semantic_model_config.hidden_size,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            config.semantic_model_config.hidden_size,
            config.semantic_model_config.hidden_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv1d(
            config.semantic_model_config.hidden_size,
            config.semantic_model_config.hidden_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv4 = nn.Conv1d(
            in_channels=config.semantic_model_config.hidden_size,
            out_channels=config.semantic_model_config.hidden_size,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act1(hidden_states)
        residual = hidden_states
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.act2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.conv4(hidden_states)
        return hidden_states


class Xcodec2PreTrainedModel(VoxtralPreTrainedModel):
    base_model_prefix = "xcodec2"
    main_input_name = "audio"
    input_modalities = "audio"
    _can_record_outputs = {
        "hidden_states": Xcodec2DecoderLayer,
        "attentions": Xcodec2DecoderLayer,
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, Xcodec2SnakeBeta):
            init.zeros_(module.alpha)
            init.zeros_(module.beta)
        elif isinstance(module, Xcodec2ISTFTHead):
            window = torch.hann_window(module.n_fft)
            init.copy_(module.window, window)
        elif isinstance(module, Xcodec2FiniteScalarQuantization):
            levels, basis, codebook = module._compute_buffers(device=module.levels.device)
            init.copy_(module.levels, levels)
            init.copy_(module.basis, basis)
            init.copy_(module.codebook, codebook)
        elif isinstance(module, Xcodec2UpSample1d):
            filter_tensor = kaiser_sinc_filter1d(0.5 / module.ratio, 0.6 / module.ratio, module.kernel_size)
            init.copy_(module.filter, filter_tensor)
        elif isinstance(module, Xcodec2DownSample1d):
            filter_tensor = kaiser_sinc_filter1d(module.cutoff, module.half_width, module.kernel_size)
            init.copy_(module.filter, filter_tensor)


@auto_docstring(custom_intro="Xcodec2 neural audio codec model.")
class Xcodec2Model(Xcodec2PreTrainedModel):
    config_class = Xcodec2Config

    def __init__(self, config: Xcodec2Config):
        super().__init__(config)

        self.hop_length = config.hop_length
        self.semantic_encoder = AutoModel.from_config(config.semantic_model_config)
        self.semantic_adapter = Xcodec2SemanticAdapter(config)
        self.acoustic_encoder = Xcodec2Encoder(config)
        self.fc_encoder = nn.Linear(
            config.hidden_size + config.semantic_model_config.hidden_size,
            config.hidden_size + config.semantic_model_config.hidden_size,
        )
        self.quantizer = Xcodec2Quantizer(config)
        self.acoustic_decoder = Xcodec2Decoder(config)

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def encode(
        self,
        audio: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spectrogram_mask: torch.Tensor | None = None,
        output_latents: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Xcodec2EncoderOutput:
        r"""
        audio (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        audio_spectrogram (`torch.Tensor` of shape `(batch_size, mel_bins, time_steps)`):
            Input audio mel spectrogram for semantic encoding.
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `audio`.
        spectrogram_mask (`torch.Tensor` of shape `(batch_size, time_steps)`, *optional*):
            Attention mask for the spectrogram input to the semantic encoder. `1` for valid frames, `0` for padding.
        output_latents (`bool`, *optional*, defaults to `False`):
            Whether to return the continuous latent representation from the quantizer.
        """

        # Semantic embedding
        encoder_param = next(iter(self.semantic_encoder.parameters()), None)
        semantic_dtype = encoder_param.dtype if encoder_param is not None else torch.float32
        with torch.no_grad():
            semantic_output = self.semantic_encoder(
                audio_spectrogram.to(semantic_dtype), attention_mask=spectrogram_mask
            )
        semantic_hidden_states = semantic_output.last_hidden_state.to(audio.dtype).transpose(1, 2)
        semantic_hidden_states = self.semantic_adapter(semantic_hidden_states)

        # Acoustic embedding and concatenate
        acoustic_hidden_states = self.acoustic_encoder(audio)
        hidden_states = torch.cat([semantic_hidden_states, acoustic_hidden_states], dim=1)
        hidden_states = self.fc_encoder(hidden_states.transpose(1, 2))

        # Quantize
        latents, audio_codes = self.quantizer(hidden_states)
        latents = latents.transpose(1, 2)
        audio_codes = audio_codes.transpose(1, 2)

        # If provided, compute corresponding padding mask for audio codes
        audio_codes_mask = None
        if padding_mask is not None:
            audio_length = padding_mask.sum(dim=-1, keepdim=True)
            token_length = audio_length // self.hop_length
            idx = torch.arange(audio_codes.shape[-1], device=padding_mask.device).view(1, -1)
            audio_codes_mask = (idx < token_length).to(padding_mask.dtype)

        return Xcodec2EncoderOutput(
            audio_codes=audio_codes,
            latents=latents if output_latents else None,
            audio_codes_mask=audio_codes_mask,
        )

    @auto_docstring
    @can_return_tuple
    def decode(
        self,
        audio_codes: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Xcodec2DecoderOutput:
        r"""
        audio_codes (`torch.LongTensor`  of shape `(batch_size, 1, codes_length)`):
            Discrete code indices computed using `model.encode`.
        latents (torch.Tensor of shape `(batch_size, dimension, time_steps)`, *optional*):
            Quantized continuous representation of input.
        """
        if latents is None and audio_codes is None:
            raise ValueError("Either `latents` or `audio_codes` must be provided.")

        if audio_codes is not None:
            latents = self.quantizer.from_codes(audio_codes.transpose(1, 2))
        else:
            latents = latents.transpose(1, 2)

        recon_audio = self.acoustic_decoder(latents, **kwargs)
        return Xcodec2DecoderOutput(audio_values=recon_audio)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        audio: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spectrogram_mask: torch.Tensor | None = None,
        output_latents: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Xcodec2Output:
        r"""
        audio (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        audio_spectrogram (`torch.Tensor` of shape `(batch_size, mel_bins, time_steps)`):
            Input audio mel spectrogram for semantic encoding.
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `audio`.
        spectrogram_mask (`torch.Tensor` of shape `(batch_size, time_steps)`, *optional*):
            Attention mask for the spectrogram input to the semantic encoder. `1` for valid frames, `0` for padding.
        output_latents (`bool`, *optional*, defaults to `False`):
            Whether to return the continuous latent representation from the quantizer.

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, Xcodec2Model

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> audio = dataset["train"]["audio"][0]["array"]

        >>> model_id = "bezzam/xcodec2"
        >>> model = Xcodec2Model.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        # for truncating output audio to original length
        length = audio.shape[-1]

        encoder_outputs = self.encode(
            audio,
            audio_spectrogram=audio_spectrogram,
            padding_mask=padding_mask,
            spectrogram_mask=spectrogram_mask,
            output_latents=True,
            return_dict=True,
        )
        audio_values = self.decode(latents=encoder_outputs.latents, return_dict=True, **kwargs)[0][..., :length]

        return Xcodec2Output(
            audio_values=audio_values,
            audio_codes=encoder_outputs.audio_codes,
            latents=encoder_outputs.latents if output_latents else None,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
        )


__all__ = ["Xcodec2Model", "Xcodec2PreTrainedModel"]
