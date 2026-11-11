# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""WavTokenizer model, an acoustic discrete codec tokenizer.

Ported from the original implementation by Ji et al. (MIT license):
*WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling*
(https://arxiv.org/abs/2408.16532, ICLR 2025), https://github.com/jishengpeng/WavTokenizer.

The encoder and quantizer follow EnCodec's SEANet encoder + single-codebook vector quantization; the decoder is a
Vocos-style backbone (ConvNeXt blocks with adaptive layer norm and a positional conv/attention net) with an ISTFT head.

This is an inference-only port: it covers encoding audio to discrete codes and decoding codes back to audio. The
original training stack (GAN discriminators, loss modules, differentiable quantization) is not included.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ..encodec.modeling_encodec import (
    EncodecConv1d,
    EncodecEncoder,
    EncodecEuclideanCodebook,
    EncodecLSTM,
    EncodecPreTrainedModel,
    EncodecResnetBlock,
    EncodecVectorQuantization,
)
from ..xcodec2.modeling_xcodec2 import Xcodec2ISTFTHead
from .configuration_wavtokenizer import WavTokenizerConfig, WavTokenizerEncoderConfig


@auto_docstring
@dataclass
class WavTokenizerOutput(ModelOutput):
    r"""
    audio_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
        Decoded audio waveform values in the time domain, obtained using the decoder part of WavTokenizer.
    audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*):
        Discrete code indices computed using `model.encode`.
    audio_codes_mask (`torch.Tensor` of shape `(batch_size, 1, codes_length)`, *optional*):
        Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.
    """

    audio_values: torch.Tensor | None = None
    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.Tensor | None = None


@auto_docstring
@dataclass
class WavTokenizerEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*):
        Discrete code indices computed using `model.encode`. These represent the compressed, quantized form of the
        input audio signal (one codebook, `ceil(num_samples / hop_length)` codes).
    audio_codes_mask (`torch.Tensor` of shape `(batch_size, 1, codes_length)`, *optional*):
        Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.
    """

    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.Tensor | None = None


@auto_docstring
@dataclass
class WavTokenizerDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
        Decoded audio waveform values in the time domain (`codes_length * hop_length` samples).
    """

    audio_values: torch.Tensor | None = None


class WavTokenizerConv1d(EncodecConv1d):
    pass


class WavTokenizerLSTM(EncodecLSTM):
    pass


class WavTokenizerResnetBlock(EncodecResnetBlock):
    pass


class WavTokenizerEncoder(EncodecEncoder):
    pass


class WavTokenizerEuclideanCodebook(EncodecEuclideanCodebook):
    pass


class WavTokenizerVectorQuantization(EncodecVectorQuantization):
    pass


class WavTokenizerAdaLayerNorm(nn.Module):
    """Vocos adaptive layer norm using learned scale and shift condition embeddings."""

    def __init__(self, config: WavTokenizerConfig):
        super().__init__()
        self.dim = config.decoder_hidden_size
        self.eps = config.norm_eps
        self.scale = nn.Embedding(config.adanorm_num_embeddings, config.decoder_hidden_size)
        self.shift = nn.Embedding(config.adanorm_num_embeddings, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        hidden_states = F.layer_norm(hidden_states, (self.dim,), eps=self.eps)
        return hidden_states * scale + shift


class WavTokenizerConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for 1D audio signals (Vocos `ConvNeXtBlock` with adaptive layer norm)."""

    def __init__(self, config: WavTokenizerConfig):
        super().__init__()
        dim = config.decoder_hidden_size
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = WavTokenizerAdaLayerNorm(config)
        self.pwconv1 = nn.Linear(dim, config.decoder_intermediate_size)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(config.decoder_intermediate_size, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1.0 / config.decoder_num_layers))

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states, cond_embedding_id)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        return residual + hidden_states


class WavTokenizerVocosResnetBlock(nn.Module):
    """Residual block of the decoder positional net (GroupNorm + SiLU + Conv1d, twice)."""

    def __init__(self, config: WavTokenizerConfig):
        super().__init__()
        dim = config.decoder_hidden_size
        self.norm1 = nn.GroupNorm(config.decoder_attention_num_groups, dim, eps=config.norm_eps)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(config.decoder_attention_num_groups, dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return residual + hidden_states


class WavTokenizerVocosAttention(nn.Module):
    """Single-head self-attention over time with 1x1 convolutions (decoder positional net)."""

    def __init__(self, config: WavTokenizerConfig):
        super().__init__()
        dim = config.decoder_hidden_size
        self.norm = nn.GroupNorm(config.decoder_attention_num_groups, dim, eps=config.norm_eps)
        self.q = nn.Conv1d(dim, dim, kernel_size=1)
        self.k = nn.Conv1d(dim, dim, kernel_size=1)
        self.v = nn.Conv1d(dim, dim, kernel_size=1)
        self.proj_out = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        batch_size, channels, seq_len = query.shape
        attn_weights = torch.bmm(query.permute(0, 2, 1), key) * channels**-0.5
        attn_weights = F.softmax(attn_weights, dim=2)
        hidden_states = torch.bmm(value, attn_weights.permute(0, 2, 1))
        hidden_states = self.proj_out(hidden_states)
        return residual + hidden_states


class WavTokenizerVocosBackbone(nn.Module):
    """Vocos-style decoder backbone: embedding conv, positional conv/attention net, and ConvNeXt blocks."""

    def __init__(self, config: WavTokenizerConfig):
        super().__init__()
        self.embed = nn.Conv1d(config.hidden_size, config.decoder_hidden_size, kernel_size=7, padding=3)
        self.pos_net = nn.ModuleList(
            [
                WavTokenizerVocosResnetBlock(config),
                WavTokenizerVocosResnetBlock(config),
                WavTokenizerVocosAttention(config),
                WavTokenizerVocosResnetBlock(config),
                WavTokenizerVocosResnetBlock(config),
                nn.GroupNorm(config.decoder_attention_num_groups, config.decoder_hidden_size, eps=config.norm_eps),
            ]
        )
        self.norm = WavTokenizerAdaLayerNorm(config)
        self.convnext = nn.ModuleList([WavTokenizerConvNeXtBlock(config) for _ in range(config.decoder_num_layers)])
        self.final_layer_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states: torch.Tensor, bandwidth_id: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed(hidden_states)
        for layer in self.pos_net:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states.transpose(1, 2), bandwidth_id)
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.convnext:
            hidden_states = layer(hidden_states, bandwidth_id)
        return self.final_layer_norm(hidden_states.transpose(1, 2))


class WavTokenizerISTFTHead(Xcodec2ISTFTHead):
    def __init__(self, config: WavTokenizerConfig):
        super().__init__(config)
        # WavTokenizer's decoder backbone dimension differs from the encoder latent dimension
        self.linear = nn.Linear(config.decoder_hidden_size, config.n_fft + 2)


class WavTokenizerPreTrainedModel(EncodecPreTrainedModel):
    config: WavTokenizerConfig
    base_model_prefix = "encoder_model"
    input_modalities = ("audio",)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, WavTokenizerAdaLayerNorm):
            init.ones_(module.scale.weight)
            init.zeros_(module.shift.weight)
        elif isinstance(module, WavTokenizerConvNeXtBlock):
            init.constant_(module.gamma, 1.0 / self.config.decoder_num_layers)
        elif isinstance(module, WavTokenizerISTFTHead):
            window = torch.hann_window(module.n_fft)
            init.copy_(module.window, window)


@auto_docstring(
    custom_intro="""
    Encoder and quantizer of WavTokenizer for converting audio waveforms to discrete codes.
    """
)
class WavTokenizerEncoderModel(WavTokenizerPreTrainedModel):
    config: WavTokenizerEncoderConfig
    _keys_to_ignore_on_load_unexpected = [
        r"^backbone\.",
        r"^head\.",
    ]

    def __init__(self, config: WavTokenizerEncoderConfig):
        super().__init__(config)
        self.hop_length = config.hop_length
        self.encoder = WavTokenizerEncoder(config)
        self.quantizer = WavTokenizerVectorQuantization(config)
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | WavTokenizerEncoderOutput:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform. Arbitrary non-zero lengths are supported; the encoder pads internally and emits
            `ceil(sequence_length / hop_length)` codes.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*):
            Mask describing padding in `input_values`; used to compute `audio_codes_mask`.
        """
        embeddings = self.encoder(input_values)
        audio_codes = self.quantizer.encode(embeddings)
        audio_codes = audio_codes.unsqueeze(1)  # single codebook: (batch_size, 1, codes_length)

        audio_codes_mask = None
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1)
            audio_length = padding_mask.sum(dim=-1, keepdim=True)
            token_length = (audio_length + self.hop_length - 1) // self.hop_length
            idx = torch.arange(audio_codes.shape[-1], device=padding_mask.device).view(1, 1, -1)
            right_padded_codes_mask = idx < token_length
            left_padded_codes_mask = idx >= audio_codes.shape[-1] - token_length
            is_left_padded = padding_mask[..., :1] == 0
            audio_codes_mask = torch.where(is_left_padded, left_padded_codes_mask, right_padded_codes_mask).to(
                padding_mask.dtype
            )

        return WavTokenizerEncoderOutput(audio_codes=audio_codes, audio_codes_mask=audio_codes_mask)

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | WavTokenizerEncoderOutput:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*):
            Padding mask used to compute `audio_codes_mask`.
        """
        return self.encode(input_values, padding_mask=padding_mask, **kwargs)


@auto_docstring(
    custom_intro="""
    Inference-only, single-codebook WavTokenizer neural audio codec.
    """
)
class WavTokenizerModel(WavTokenizerPreTrainedModel):
    def __init__(self, config: WavTokenizerConfig):
        super().__init__(config)
        self.hop_length = config.hop_length

        self.encoder_model = WavTokenizerEncoderModel(config.encoder_config)
        self.backbone = WavTokenizerVocosBackbone(config)
        self.head = WavTokenizerISTFTHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | WavTokenizerEncoderOutput:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform. Arbitrary non-zero lengths are supported; the encoder pads internally and emits
            `ceil(sequence_length / hop_length)` codes.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*):
            Mask describing padding in `input_values`; used to compute `audio_codes_mask`.
        """
        return self.encoder_model.encode(input_values, padding_mask=padding_mask, **kwargs)

    @auto_docstring
    @can_return_tuple
    def decode(
        self,
        audio_codes: torch.Tensor,
        bandwidth_id: int = 0,
        **kwargs,
    ) -> tuple | WavTokenizerDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`):
            Discrete code indices computed using `model.encode`. One or more codes are supported; each code decodes
            to `hop_length` audio samples.
        bandwidth_id (`int`, *optional*, defaults to 0):
            Condition embedding id of the decoder's adaptive layer norms. Always 0 at inference.
        """
        quantized = self.encoder_model.quantizer.decode(audio_codes.squeeze(1))
        bandwidth_id = torch.tensor([bandwidth_id], device=audio_codes.device)
        hidden_states = self.backbone(quantized, bandwidth_id)
        audio_values = self.head(hidden_states)
        return WavTokenizerDecoderOutput(audio_values=audio_values)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | WavTokenizerOutput:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*):
            Mask indicating valid audio samples (1) versus padding (0), used to compute `audio_codes_mask`.
        """
        length = input_values.shape[-1]
        encoder_outputs = self.encode(input_values, padding_mask=padding_mask)
        audio_values = self.decode(encoder_outputs.audio_codes).audio_values[..., :length]
        return WavTokenizerOutput(
            audio_values=audio_values,
            audio_codes=encoder_outputs.audio_codes,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
        )


__all__ = [
    "WavTokenizerEncoderModel",
    "WavTokenizerModel",
    "WavTokenizerPreTrainedModel",
]
