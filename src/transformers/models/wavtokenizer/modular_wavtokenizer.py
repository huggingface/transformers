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

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
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


@auto_docstring(checkpoint="swiss-ai/wavtokenizer-large-unify-40token")
@strict
class WavTokenizerConfig(PreTrainedConfig):
    r"""
    sampling_rate (`int`, *optional*, defaults to 24000):
        The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
    audio_channels (`int`, *optional*, defaults to 1):
        Number of channels in the audio data. Only mono (1) is supported.
    num_filters (`int`, *optional*, defaults to 32):
        Number of convolution kernels of the first `WavTokenizerConv1d` down sampling layer.
    upsampling_ratios (`list[int]`, *optional*, defaults to `[6, 5, 5, 4]`):
        Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
        will use the ratios in the reverse order to the ones specified here that must match the decoder order.
        The product (600 by default) is the hop length: the number of audio samples represented by a single code.
    num_residual_layers (`int`, *optional*, defaults to 1):
        Number of residual layers in each encoder stage.
    dilation_growth_rate (`int`, *optional*, defaults to 2):
        How much to increase the dilation with each residual layer in the encoder.
    compress (`int`, *optional*, defaults to 2):
        Reduced dimensionality in residual branches.
    use_conv_shortcut (`bool`, *optional*, defaults to `True`):
        Whether to use a convolutional layer as the (skip) shortcut in the residual blocks. If False, an identity
        function will be used, giving a generic residual connection.
    use_causal_conv (`bool`, *optional*, defaults to `False`):
        Whether to use fully causal convolution.
    pad_mode (`str`, *optional*, defaults to `"reflect"`):
        Padding mode for the convolutions.
    norm_type (`str`, *optional*, defaults to `"weight_norm"`):
        Normalization method for the encoder convolutions. Should be in `["weight_norm", "time_group_norm"]`.
    kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for the initial encoder convolution.
    last_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size for the last encoder convolution.
    residual_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the residual layers of the encoder.
    num_lstm_layers (`int`, *optional*, defaults to 2):
        Number of LSTM layers at the end of the encoder.
    hidden_size (`int`, *optional*, defaults to 512):
        Dimensionality of the encoder output (the latent space of the quantizer).
    codebook_size (`int`, *optional*, defaults to 4096):
        Number of discrete codes that make up the single VQ codebook.
    codebook_dim (`int`, *optional*, defaults to 512):
        Dimension of the codebook vectors. Must match `hidden_size` (no projections are used).
    decoder_hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the Vocos-style decoder backbone.
    decoder_intermediate_size (`int`, *optional*, defaults to 2304):
        Dimensionality of the pointwise convolutions inside the decoder ConvNeXt blocks.
    decoder_num_layers (`int`, *optional*, defaults to 12):
        Number of ConvNeXt blocks in the decoder backbone.
    adanorm_num_embeddings (`int`, *optional*, defaults to 4):
        Number of condition embeddings of the decoder's adaptive layer norms. Conditioning is a training-time
        feature (one embedding per bandwidth); inference always uses embedding 0.
    decoder_attention_num_groups (`int`, *optional*, defaults to 32):
        Number of groups of the decoder GroupNorm layers.
    norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the decoder normalization layers.

    Example:

    ```python
    >>> from transformers import WavTokenizerConfig, WavTokenizerModel

    >>> # Initializing configuration
    >>> configuration = WavTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = WavTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "wavtokenizer"

    sampling_rate: int = 24000
    audio_channels: int = 1
    num_filters: int = 32
    upsampling_ratios: list[int] | tuple[int, ...] = (6, 5, 5, 4)
    num_residual_layers: int = 1
    dilation_growth_rate: int = 2
    compress: int = 2
    use_conv_shortcut: bool = True
    use_causal_conv: bool = False
    pad_mode: str = "reflect"
    norm_type: str = "weight_norm"
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    num_lstm_layers: int = 2
    hidden_size: int = 512
    codebook_size: int = 4096
    codebook_dim: int = 512
    decoder_hidden_size: int = 768
    decoder_intermediate_size: int = 2304
    decoder_num_layers: int = 12
    adanorm_num_embeddings: int = 4
    decoder_attention_num_groups: int = 32
    norm_eps: float = 1e-6

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(f'`norm_type` must be one of `"weight_norm"`, `"time_group_norm"`, got {self.norm_type}')
        if self.codebook_dim != self.hidden_size:
            raise ValueError(
                "WavTokenizer uses no projections around the quantizer, so `codebook_dim` "
                f"({self.codebook_dim}) must equal `hidden_size` ({self.hidden_size})."
            )

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.upsampling_ratios))

    @property
    def n_fft(self) -> int:
        return self.hop_length * 4

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sampling_rate / self.hop_length)


@auto_docstring
@dataclass
class WavTokenizerOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
        Decoded audio waveform values in the time domain, obtained using the decoder part of WavTokenizer.
    audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*):
        Discrete code indices computed using `model.encode`.
    audio_codes_mask (`torch.Tensor` of shape `(batch_size, 1, codes_length)`, *optional*):
        Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.
    """

    audio_values: torch.FloatTensor | None = None
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
    audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
        Decoded audio waveform values in the time domain (`codes_length * hop_length` samples).
    """

    audio_values: torch.FloatTensor | None = None


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
    _keys_to_ignore_on_load_unexpected = [
        r"^backbone\.",
        r"^head\.",
    ]

    def __init__(self, config: WavTokenizerConfig):
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

        self.encoder_model = WavTokenizerEncoderModel(config)
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
            Padding mask used to pad `input_values`.
        """
        length = input_values.shape[-1]
        encoder_outputs = self.encode(input_values, padding_mask=padding_mask, return_dict=True)
        audio_values = self.decode(encoder_outputs.audio_codes, return_dict=True).audio_values[..., :length]
        return WavTokenizerOutput(
            audio_values=audio_values,
            audio_codes=encoder_outputs.audio_codes,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
        )


__all__ = [
    "WavTokenizerConfig",
    "WavTokenizerEncoderModel",
    "WavTokenizerModel",
    "WavTokenizerPreTrainedModel",
]
