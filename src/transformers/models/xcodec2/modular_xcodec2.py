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

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from transformers.activations import ACT2FN

from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ..auto import AutoModel
from ..dac.modeling_dac import DacEncoder, DacEncoderBlock, DacResidualUnit
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, rotate_half
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    DownSample1d,
    SnakeBeta,
    TorchActivation1d,
    UpSample1d,
    kaiser_sinc_filter1d,
)
from .configuration_xcodec2 import Xcodec2Config


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

    audio_values: torch.FloatTensor | None = None
    audio_codes: torch.LongTensor | None = None
    quantized_representation: torch.Tensor | None = None
    codes_padding_mask: torch.Tensor | None = None


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

    audio_codes: torch.LongTensor | None = None
    quantized_representation: torch.Tensor | None = None
    codes_padding_mask: torch.Tensor | None = None


@dataclass
class Xcodec2DecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, segment_length)`, *optional*):
            Decoded audio waveform values in the time domain, obtained by converting
            the discrete codes back into continuous audio signals. This represents
            the reconstructed audio that can be played back.
    """

    audio_values: torch.FloatTensor | None = None


# RoPE is applied on the attention head rather than sequence dimension
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
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
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=False)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Xcodec2DecoderLayer(LlamaDecoderLayer):
    pass


class SnakeBeta(SnakeBeta):
    pass


class TorchActivation1d(TorchActivation1d):
    pass


class Xcodec2ResidualUnit(DacResidualUnit):
    def __init__(self, dimension, dilation):
        super().__init__(dimension, dilation)
        self.snake1 = TorchActivation1d(activation=SnakeBeta(dimension))
        self.snake2 = TorchActivation1d(activation=SnakeBeta(dimension))


class Xcodec2EncoderBlock(DacEncoderBlock):
    def __init__(self, config: Xcodec2Config, stride: int = 1, stride_index: int = 1):
        super().__init__(config, stride, stride_index)
        dimension = config.encoder_hidden_size * 2**stride_index
        self.snake1 = TorchActivation1d(activation=SnakeBeta(dimension // 2))


class Xcodec2Encoder(DacEncoder):
    def __init__(self, config: Xcodec2Config):
        super().__init__(config)
        d_model = config.encoder_hidden_size * 2 ** len(config.downsampling_ratios)
        self.snake1 = TorchActivation1d(activation=SnakeBeta(d_model))


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

    def forward(self, hidden_states):
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
    def __init__(self, config: Xcodec2Config):
        super().__init__()

        self.quantization_levels = list(config.quantization_levels)
        levels = torch.tensor(self.quantization_levels, dtype=torch.int32)
        self.register_buffer("levels", levels, persistent=False)

        basis = torch.cumprod(torch.tensor([1] + self.quantization_levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("basis", basis, persistent=False)

        codebook = self._indices_to_codes(torch.arange(int(np.prod(self.quantization_levels))))
        self.register_buffer("codebook", codebook, persistent=False)

    def _indices_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        level_indices = (indices // self.basis) % self.levels
        half_width = self.levels // 2
        codes = (level_indices - half_width) / half_width
        return codes

    def bound(self, hidden_states, eps: float = 1e-3):
        half_range = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_range).atanh()
        return (hidden_states + shift).tanh() * half_range - offset

    def forward(self, hidden_states):
        with autocast("cuda", enabled=False):
            orig_dtype = hidden_states.dtype
            if orig_dtype not in (torch.float32, torch.float64):
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
            codes = codes.to(orig_dtype)

        return codes, indices


class Xcodec2ISTFTHead(nn.Module):
    """
    Head for converting decoder outputs to waveform via STFT projection and ISTFT.

    Uses custom "same" padding ISTFT from:
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

    def forward(self, hidden_states):
        stft_pred = self.linear(hidden_states).transpose(1, 2)
        magnitude, phase = stft_pred.chunk(2, dim=1)
        magnitude = torch.exp(magnitude).clamp(max=1e2)
        spectrogram_complex = torch.polar(magnitude, phase)

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
        window_envelope = window_envelope.clamp(min=1e-11)
        audio = audio / window_envelope
        return audio.unsqueeze(1)


class Xcodec2Quantizer(nn.Module):
    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.finite_scalar_quantization = Xcodec2FiniteScalarQuantization(config)
        self.project_in = nn.Linear(config.quantization_dim, len(config.quantization_levels))
        self.project_out = nn.Linear(len(config.quantization_levels), config.quantization_dim)

    def from_codes(self, indices):
        indices = indices.squeeze(-1)  # Remove channel dimension
        codes = self.finite_scalar_quantization.codebook[indices]
        return self.project_out(codes)

    def forward(self, hidden_states):
        hidden_states = self.project_in(hidden_states)
        hidden_states = self.finite_scalar_quantization.bound(
            hidden_states
        )  # For consistency with original checkpoint
        quantized_out, indices = self.finite_scalar_quantization(hidden_states)
        quantized_out = self.project_out(quantized_out.to(hidden_states.dtype))
        indices = indices.unsqueeze(-1)  # Add channel dimension for single codebook
        return quantized_out, indices


class Xcodec2Decoder(nn.Module):
    """Vocos-based decoder with ResNet, Transformer, and ISTFT head for audio reconstruction."""

    def __init__(self, config: Xcodec2Config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size + config.semantic_model_config.hidden_size, config.hidden_size)
        self.embed = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=7, padding=3)
        self.prior_resnet_block1 = Xcodec2ResNetBlock(config)
        self.prior_resnet_block2 = Xcodec2ResNetBlock(config)
        self.num_attention_heads = config.num_attention_heads
        self.rotary_emb = Xcodec2RotaryEmbedding(config=config)
        self.layers = nn.ModuleList(
            [Xcodec2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.post_resnet_block1 = Xcodec2ResNetBlock(config)
        self.post_resnet_block2 = Xcodec2ResNetBlock(config)
        self.head = Xcodec2ISTFTHead(config)

    def forward(self, hidden_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.embed(hidden_states)
        hidden_states = self.prior_resnet_block1(hidden_states)
        hidden_states = self.prior_resnet_block2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        position_ids = torch.arange(self.num_attention_heads).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids.to(hidden_states.device))
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.post_resnet_block1(hidden_states)
        hidden_states = self.post_resnet_block2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.final_layer_norm(hidden_states)
        return self.head(hidden_states)


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
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            config.semantic_model_config.hidden_size,
            config.semantic_model_config.hidden_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.act2 = nn.ReLU(inplace=True)
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

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        residual = hidden_states
        hidden_states = self.act1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.act2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.conv4(hidden_states)
        return hidden_states


class Xcodec2PreTrainedModel(PreTrainedModel):
    config_class = Xcodec2Config
    base_model_prefix = "xcodec2"
    main_input_name = "audio"

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, SnakeBeta):
            init.zeros_(module.alpha)
            init.zeros_(module.beta)
        elif isinstance(module, Xcodec2ISTFTHead):
            window = torch.hann_window(module.n_fft)
            init.copy_(module.window, window)
        elif isinstance(module, Xcodec2FiniteScalarQuantization):
            quantization_levels = module.quantization_levels
            device = module.levels.device
            init.copy_(module.levels, torch.tensor(quantization_levels, dtype=torch.int32))
            init.copy_(
                module.basis, torch.cumprod(torch.tensor([1] + quantization_levels[:-1]), dim=0, dtype=torch.int32)
            )
            init.copy_(
                module.codebook,
                module._indices_to_codes(torch.arange(math.prod(quantization_levels), device=device)),
            )
        elif isinstance(module, UpSample1d):
            filter_tensor = kaiser_sinc_filter1d(0.5 / module.ratio, 0.6 / module.ratio, module.kernel_size)
            init.copy_(module.filter, filter_tensor)
        elif isinstance(module, DownSample1d):
            filter_tensor = kaiser_sinc_filter1d(module.cutoff, module.half_width, module.kernel_size)
            init.copy_(module.filter, filter_tensor)


@auto_docstring(custom_intro="Xcodec2 neural audio codec model.")
class Xcodec2Model(Xcodec2PreTrainedModel):
    config_class = Xcodec2Config

    def __init__(self, config: Xcodec2Config):
        super().__init__(config)

        self.hop_length = config.hop_length
        self.semantic_model = AutoModel.from_config(config.semantic_model_config).eval()
        self.semantic_adapter = Xcodec2SemanticAdapter(config)
        self.acoustic_encoder = Xcodec2Encoder(config)
        self.fc_encoder = nn.Linear(
            config.hidden_size + config.semantic_model_config.hidden_size,
            config.hidden_size + config.semantic_model_config.hidden_size,
        )
        self.quantizer = Xcodec2Quantizer(config)
        self.decoder = Xcodec2Decoder(config)

        self.post_init()

    def apply_weight_norm(self, legacy=True):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.acoustic_encoder.conv1)
        for encoder_block in self.acoustic_encoder.block:
            weight_norm(encoder_block.res_unit1.conv1)
            weight_norm(encoder_block.res_unit1.conv2)
            weight_norm(encoder_block.res_unit2.conv1)
            weight_norm(encoder_block.res_unit2.conv2)
            weight_norm(encoder_block.res_unit3.conv1)
            weight_norm(encoder_block.res_unit3.conv2)
            weight_norm(encoder_block.conv1)
        weight_norm(self.acoustic_encoder.conv2)

    def remove_weight_norm(self, legacy=True):
        remove_weight_norm = nn.utils.remove_weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            remove_weight_norm = nn.utils.parametrize.remove_parametrizations

        remove_weight_norm(self.acoustic_encoder.conv1)
        for encoder_block in self.acoustic_encoder.block:
            remove_weight_norm(encoder_block.res_unit1.conv1)
            remove_weight_norm(encoder_block.res_unit1.conv2)
            remove_weight_norm(encoder_block.res_unit2.conv1)
            remove_weight_norm(encoder_block.res_unit2.conv2)
            remove_weight_norm(encoder_block.res_unit3.conv1)
            remove_weight_norm(encoder_block.res_unit3.conv2)
            remove_weight_norm(encoder_block.conv1)
        remove_weight_norm(self.acoustic_encoder.conv2)

    @auto_docstring
    @can_return_tuple
    def encode(
        self,
        audio: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Xcodec2EncoderOutput:
        r"""
        audio (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        audio_spectrogram (`torch.Tensor` of shape `(batch_size, mel_bins, time_steps)`):
            Input audio mel spectrogram for semantic encoding.
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `audio`.
        """

        # Semantic embedding (16th layer of Wav2Vec2Bert)
        semantic_output = self.semantic_model(audio_spectrogram, output_hidden_states=True)
        semantic_hidden_16 = semantic_output.hidden_states[16]
        semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)
        semantic_hidden_states = self.semantic_adapter(semantic_hidden_16)

        # Acoustic embedding
        acoustic_hidden_states = self.acoustic_encoder(audio)

        # Concatenate embeddings
        if acoustic_hidden_states.shape[-1] != semantic_hidden_states.shape[-1]:
            min_len = min(acoustic_hidden_states.shape[-1], semantic_hidden_states.shape[-1])
            acoustic_hidden_states = acoustic_hidden_states[:, :, :min_len]
            semantic_hidden_states = semantic_hidden_states[:, :, :min_len]
        hidden_states = torch.cat([semantic_hidden_states, acoustic_hidden_states], dim=1)
        hidden_states = self.fc_encoder(hidden_states.transpose(1, 2))

        # Quantize
        quantized_representation, audio_codes = self.quantizer(hidden_states)
        quantized_representation = quantized_representation.transpose(1, 2)
        audio_codes = audio_codes.transpose(1, 2)

        # If provided, compute corresponding padding mask for audio codes
        codes_padding_mask = None
        if padding_mask is not None:
            audio_length = padding_mask.sum(dim=-1, keepdim=True)
            token_length = audio_length // self.hop_length
            idx = torch.arange(audio_codes.shape[-1], device=padding_mask.device).view(1, -1)
            codes_padding_mask = (idx < token_length).to(padding_mask.dtype)

        return Xcodec2EncoderOutput(
            audio_codes=audio_codes,
            quantized_representation=quantized_representation,
            codes_padding_mask=codes_padding_mask,
        )

    @auto_docstring
    @can_return_tuple
    def decode(
        self,
        quantized_representation: torch.Tensor | None = None,
        audio_codes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Xcodec2DecoderOutput:
        r"""
        quantized_representation (torch.Tensor of shape `(batch_size, dimension, time_steps)`, *optional*):
            Quantized continuous representation of input.
        audio_codes (`torch.LongTensor`  of shape `(batch_size, 1, codes_length)`):
            Discrete code indices computed using `model.encode`.
        """
        if quantized_representation is None and audio_codes is None:
            raise ValueError("Either `quantized_representation` or `audio_codes` must be provided.")

        if audio_codes is not None:
            quantized_representation = self.quantizer.from_codes(audio_codes.transpose(1, 2))
        else:
            quantized_representation = quantized_representation.transpose(1, 2)

        recon_audio = self.decoder(quantized_representation)
        return Xcodec2DecoderOutput(audio_values=recon_audio)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        audio: torch.Tensor,
        audio_spectrogram: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Xcodec2Output:
        r"""
        audio (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        audio_spectrogram (`torch.Tensor` of shape `(batch_size, mel_bins, time_steps)`):
            Input audio mel spectrogram for semantic encoding.
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `audio`.

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
            audio, audio_spectrogram=audio_spectrogram, padding_mask=padding_mask, return_dict=True
        )
        audio_values = self.decode(
            quantized_representation=encoder_outputs.quantized_representation, return_dict=True
        )[0][..., :length]

        return Xcodec2Output(
            audio_values=audio_values,
            audio_codes=encoder_outputs.audio_codes,
            quantized_representation=encoder_outputs.quantized_representation,
            codes_padding_mask=encoder_outputs.codes_padding_mask,
        )


__all__ = ["Xcodec2Model", "Xcodec2PreTrainedModel"]
