# coding=utf-8
# Copyright 2025 OpenMOSS and HuggingFace Inc. teams. All rights reserved.
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
"""PyTorch XY-Tokenizer model."""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np

from ...activations import ACT2FN
from ...feature_extraction_utils import BatchFeature
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, is_torch_available, logging
from .configuration_xy_tokenizer import XYTokenizerConfig
from .feature_extraction_xy_tokenizer import ExtractorIterator, XYTokenizerFeatureExtractor


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.parametrizations import weight_norm

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "XYTokenizerConfig"


@dataclass
class XYTokenizerEncoderOutput(ModelOutput):
    """
    Output type of [`XYTokenizer.encode`].

    Args:
        quantized_representation (`torch.FloatTensor` of shape `(batch_size, hidden_dim, sequence_length)`):
            The quantized continuous representation of the input audio. This is the output of the quantizer.
        audio_codes (`torch.LongTensor` of shape `(num_codebooks, batch_size, sequence_length)`):
            The discrete codes from the quantizer for each codebook.
        codes_lengths (`torch.LongTensor` of shape `(batch_size,)`):
            The valid length of each sequence in `audio_codes`.
        commit_loss (`torch.FloatTensor`, *optional*):
            The commitment loss from the vector quantizer.
        overlap_seconds (`int`, *optional*):
            The duration of the overlap in seconds between adjacent audio chunks.
    """

    quantized_representation: torch.FloatTensor = None
    audio_codes: torch.LongTensor = None
    codes_lengths: torch.LongTensor = None
    commit_loss: Optional[torch.FloatTensor] = None
    overlap_seconds: Optional[int] = None


@dataclass
class XYTokenizerDecoderOutput(ModelOutput):
    """
    Output type of [`XYTokenizer.decode`].

    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
            The reconstructed audio waveform.
        output_length (`torch.LongTensor` of shape `(batch_size,)`):
            The valid length of each sequence in `audio_values`.
    """

    audio_values: torch.FloatTensor = None
    output_length: Optional[torch.LongTensor] = None


@dataclass
class XYTokenizerOutput(ModelOutput):
    """
    Output type of [`XYTokenizer`]'s forward pass.

    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
            The reconstructed audio waveform.
        output_length (`torch.LongTensor` of shape `(batch_size,)`):
            The valid length of each sequence in `audio_values`.
        quantized_representation (`torch.FloatTensor` of shape `(batch_size, hidden_dim, sequence_length)`):
            The quantized continuous representation of the input audio. This is the output of the quantizer.
        audio_codes (`torch.LongTensor` of shape `(num_codebooks, batch_size, sequence_length)`):
            The discrete codes from the quantizer for each codebook.
        codes_lengths (`torch.LongTensor` of shape `(batch_size,)`):
            The valid length of each sequence in `audio_codes`.
        commit_loss (`torch.FloatTensor`, *optional*):
            The commitment loss from the vector quantizer.
    """

    audio_values: torch.FloatTensor = None
    output_length: torch.LongTensor = None
    quantized_representation: torch.FloatTensor = None
    audio_codes: torch.LongTensor = None
    codes_lengths: torch.LongTensor = None
    commit_loss: Optional[torch.FloatTensor] = None


@dataclass
class VectorQuantizerConfig:
    """Configuration for the VectorQuantize module."""

    commitment: float = 1.0
    decay: float = 0.99
    epsilon: float = 1e-5
    threshold_ema_dead: int = 2
    kmeans_init: bool = True
    kmeans_iters: int = 10


def sinusoids(length, channels, max_timescale=10000, device=None):
    if channels % 2 != 0:
        raise ValueError("channels must be an even number for sinusoidal embeddings")
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length, device=device)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len, device=inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    return sequence_mask


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class XYTokenizerAttention(nn.Module):
    """Multi-headed attention for XY-Tokenizer model."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _create_attention_mask(self, seq_len, max_len, device, dtype):
        bsz = seq_len.size(0)
        mask = torch.ones(bsz, 1, max_len, max_len, device=device, dtype=dtype)
        seq_indices = torch.arange(max_len, device=device).unsqueeze(0)
        seq_len_expanded = seq_len.unsqueeze(1)
        valid_mask = seq_indices < seq_len_expanded.unsqueeze(-1)
        mask = mask * (valid_mask.unsqueeze(2) & valid_mask.unsqueeze(3)).to(dtype)
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(max_len, max_len, device=device, dtype=torch.bool), diagonal=1)
            mask = mask * (~causal_mask.unsqueeze(0).unsqueeze(1)).to(dtype)
        mask = mask + (1.0 - mask) * torch.finfo(dtype).min
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_len: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Initialize length_mask as None
        length_mask = None

        # Create attention mask based on sequence lengths if provided
        if seq_len is not None:
            # Apply causal mask if needed
            attn_mask = self._create_attention_mask(seq_len, tgt_len, hidden_states.device, attn_weights.dtype)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        elif attention_mask is not None:
            # Handle externally provided attention mask
            if attention_mask.dim() == 2:
                attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype, tgt_len)
            attention_mask = attention_mask.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Apply length mask for non-causal attention after softmax
        if length_mask is not None:
            attn_weights = attn_weights * length_mask

        if output_attentions:
            # this operation is a bit awkward, but it's the only way to
            # reuse the parameterized conditioning apply_rotary_pos_emb
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_size` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class XYTokenizerMLP(nn.Module):
    """MLP as used in XY-Tokenizer."""

    def __init__(self, hidden_size, intermediate_size, dropout, activation_function):
        super().__init__()
        self.activation_fn = ACT2FN[activation_function]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class XYTokenizerTransformerLayer(nn.Module):
    """Transformer layer for XY-Tokenizer."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        activation_function,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = XYTokenizerAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            is_causal=is_causal,
        )

        self.mlp = XYTokenizerMLP(hidden_size, intermediate_size, dropout, activation_function)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_len: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            seq_len (`torch.Tensor`, *optional*): sequence lengths for variable length inputs
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_len=seq_len,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Clamp values for numerical stability in half precision
        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class XYTokenizerEncoder(nn.Module):
    def __init__(
        self,
        num_mel_bins=128,
        sampling_rate=16000,
        hop_length=160,
        stride_size=2,
        kernel_size=3,
        d_model=1280,
        scale_embedding=True,
        max_audio_seconds=30,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        activation_function="gelu",
        attn_type="varlen",
    ):
        super().__init__()
        self.max_source_positions = (max_audio_seconds * sampling_rate // hop_length) // stride_size
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.num_mel_bins, self.d_model, self.stride_size = (
            num_mel_bins,
            d_model,
            stride_size,
        )
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=stride_size, padding=1)
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, d_model))
        # Create a config object for the transformer layers
        layer_config = {
            "hidden_size": d_model,
            "num_attention_heads": encoder_attention_heads,
            "intermediate_size": encoder_ffn_dim,
            "activation_function": activation_function,
        }
        self.layers = nn.ModuleList(
            [XYTokenizerTransformerLayer(**layer_config, is_causal=False) for _ in range(encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_features, input_length, output_hidden_states=False):
        input_features = input_features.to(self.conv1.weight.dtype)
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        output_length = (input_length // self.stride_size).long()
        hidden_states = inputs_embeds.permute(0, 2, 1)
        bsz, tgt_len, _ = hidden_states.size()
        pos_embed = (
            self.positional_embedding[:tgt_len]
            if tgt_len < self.positional_embedding.shape[0]
            else self.positional_embedding
        )
        hidden_states = (hidden_states.to(torch.float32) + pos_embed).to(hidden_states.dtype)
        attention_mask = get_sequence_mask(hidden_states, output_length)
        all_hidden = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                seq_len=output_length,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        hidden_states = torch.where(attention_mask, hidden_states, 0).transpose(1, 2)
        if not output_hidden_states:
            return hidden_states, output_length
        return hidden_states, output_length, all_hidden


class XYTokenizerDecoder(nn.Module):
    def __init__(
        self,
        num_mel_bins=128,
        sampling_rate=16000,
        hop_length=160,
        stride_size=2,
        kernel_size=3,
        d_model=1280,
        scale_embedding=True,
        max_audio_seconds=30,
        decoder_layers=32,
        decoder_attention_heads=20,
        decoder_ffn_dim=5120,
        activation_function="gelu",
        attn_type="varlen",
    ):
        super().__init__()
        self.max_source_positions = (max_audio_seconds * sampling_rate // hop_length) // stride_size
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.num_mel_bins, self.d_model, self.stride_size = (
            num_mel_bins,
            d_model,
            stride_size,
        )
        self.deconv1 = nn.ConvTranspose1d(d_model, d_model, kernel_size, stride_size, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose1d(d_model, num_mel_bins, kernel_size, stride=1, padding=0)
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, d_model))
        # Create a config object for the transformer layers
        layer_config = {
            "hidden_size": d_model,
            "num_attention_heads": decoder_attention_heads,
            "intermediate_size": decoder_ffn_dim,
            "activation_function": activation_function,
        }
        self.layers = nn.ModuleList(
            [XYTokenizerTransformerLayer(**layer_config, is_causal=False) for _ in range(decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, input_length):
        hidden_states = hidden_states.transpose(1, 2)
        bsz, tgt_len, _ = hidden_states.size()
        pos_embed = (
            self.positional_embedding[:tgt_len]
            if tgt_len < self.positional_embedding.shape[0]
            else self.positional_embedding
        )
        hidden_states = (hidden_states.to(torch.float32) + pos_embed).to(hidden_states.dtype)
        attention_mask = get_sequence_mask(hidden_states, input_length)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                seq_len=input_length,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.where(attention_mask, hidden_states, 0).permute(0, 2, 1)
        output_features = F.gelu(self.deconv1(hidden_states))
        output_features = F.gelu(self.deconv2(output_features))
        expected_length = tgt_len * self.stride_size
        if output_features.size(2) > expected_length:
            output_features = output_features[:, :, :expected_length]
        output_length = input_length * self.stride_size
        return output_features, output_length


class ResidualDownConv(nn.Module):
    def __init__(self, d_model=1280, avg_pooler=4):
        super().__init__()
        self.d_model, self.avg_pooler = d_model, avg_pooler
        self.intermediate_dim = d_model * avg_pooler
        self.gate_proj = nn.Conv1d(d_model, self.intermediate_dim, avg_pooler, avg_pooler, bias=False)
        self.up_proj = nn.Conv1d(d_model, self.intermediate_dim, avg_pooler, avg_pooler, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.intermediate_dim, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)

    def forward(self, x, input_length):
        output_length = input_length // self.avg_pooler
        x = x.transpose(1, 2)
        batch_size, seq_len, _ = x.shape
        if seq_len % self.avg_pooler != 0:
            pad_size = self.avg_pooler - seq_len % self.avg_pooler
            x = F.pad(x, (0, 0, 0, pad_size), "constant", 0)  # Pad sequence dim
        xt = x.permute(0, 2, 1)
        g, u = self.gate_proj(xt).permute(0, 2, 1), self.up_proj(xt).permute(0, 2, 1)
        x = x.reshape(batch_size, -1, self.intermediate_dim)
        c = self.down_proj(self.act_fn(g) * u)
        res = self.layer_norm(c + x).transpose(1, 2)
        return res, output_length


class UpConv(nn.Module):
    def __init__(self, d_model=1280, stride=4):
        super().__init__()
        self.d_model, self.stride = d_model, stride
        self.up_conv = nn.ConvTranspose1d(self.stride * d_model, d_model, stride, stride, bias=False)

    def forward(self, x, input_length):
        res = self.up_conv(x)
        output_length = input_length * self.stride
        return res, output_length


class XYTokenizerTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1280,
        d_model=1280,
        output_dim=1280,
        max_source_positions=1500,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        activation_function="gelu",
        attn_type="varlen",
    ):
        super().__init__()
        self.input_dim, self.d_model, self.output_dim, self.max_source_positions = (
            input_dim,
            d_model,
            output_dim,
            max_source_positions,
        )
        self.proj = nn.Linear(input_dim, d_model, bias=True) if input_dim != d_model else None
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, d_model))
        # Create a config object for the transformer layers
        layer_config = {
            "hidden_size": d_model,
            "num_attention_heads": encoder_attention_heads,
            "intermediate_size": encoder_ffn_dim,
            "activation_function": activation_function,
        }
        self.layers = nn.ModuleList(
            [XYTokenizerTransformerLayer(**layer_config, is_causal=False) for _ in range(encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, output_dim, bias=True) if output_dim != d_model else None

    def forward(self, input_features, input_length, output_hidden_states=False):
        output_length = input_length.long()
        hidden_states = self.proj(input_features.permute(0, 2, 1)).permute(0, 2, 1) if self.proj else input_features
        hidden_states = hidden_states.permute(0, 2, 1)
        bsz, tgt_len, _ = hidden_states.size()
        pos_embed = (
            self.positional_embedding[:tgt_len]
            if tgt_len < self.positional_embedding.shape[0]
            else self.positional_embedding
        )
        hidden_states = (hidden_states.to(torch.float32) + pos_embed).to(hidden_states.dtype)
        attention_mask = get_sequence_mask(hidden_states, output_length)
        all_hidden = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                seq_len=output_length,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        hidden_states = torch.where(attention_mask, hidden_states, 0).transpose(1, 2)
        if self.out_proj:
            hidden_states = self.out_proj(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        if not output_hidden_states:
            return hidden_states, output_length
        return hidden_states, output_length, all_hidden


# Note: The other helper classes like STFT, ISTFT, Vocos, VectorQuantize, etc.,
# would be placed here. For brevity, they are omitted but are required dependencies.
# Assuming they are defined in the same way as the user provided code.
# The code below will assume these classes are defined in the current scope.
# ... [Paste all other helper classes here] ...
class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding, self.n_fft, self.hop_length, self.win_length = (
            padding,
            n_fft,
            hop_length,
            win_length,
        )
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")
        B, N, T = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
        output_size = (T - 1) * self.hop_length + self.win_length

        y = F.fold(ifft, (1, output_size), (1, self.win_length), stride=(1, self.hop_length))[:, 0, 0, pad:-pad]
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]
        if not (window_envelope > 1e-11).all():
            raise ValueError("Window envelope contains near-zero values leading to instability")
        return y / window_envelope


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length, n_fft, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clip(max=1e2)
        s = mag.float() * (torch.cos(p).float() + 1j * torch.sin(p).float())
        return self.istft(s).to(x.dtype)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init_value):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id=None):
        res = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = res + x.transpose(1, 2)
        return x


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels,
        dim,
        intermediate_dim,
        num_layers,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels, self.embed = input_channels, nn.Conv1d(input_channels, dim, 7, 1, 3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim,
                    intermediate_dim,
                    layer_scale_init_value or 1 / num_layers,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        for block in self.convnext:
            x = block(x, kwargs.get("bandwidth_id"))
        return self.final_layer_norm(x.transpose(1, 2))


class Vocos(nn.Module):
    def __init__(
        self,
        input_channels=128,
        dim=512,
        intermediate_dim=4096,
        num_layers=30,
        n_fft=640,
        hop_size=160,
        padding="same",
    ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels,
            dim,
            intermediate_dim,
            num_layers,
        )
        self.head = ISTFTHead(dim, n_fft, hop_size, padding)
        self.hop_size = hop_size

    def forward(self, x, input_length):
        x = self.backbone(x)
        x = self.head(x)
        return x[:, None, :], input_length * self.hop_size


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new.float(), alpha=(1 - decay))


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    indices = (
        torch.randperm(num_samples, device=device)[:num]
        if num_samples >= num
        else torch.randint(0, num_samples, (num,), device=device)
    )
    return samples[indices].float()


def kmeans(samples, num_clusters, num_iters=10):
    dim, means = samples.shape[-1], sample_vectors(samples, num_clusters).float()
    for _ in range(num_iters):
        dists = -(
            samples.float().pow(2).sum(1, keepdim=True)
            - 2 * samples.float() @ means.t()
            + means.t().float().pow(2).sum(0, keepdim=True)
        )
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = (
            buckets.new_zeros(num_clusters, dim, dtype=torch.float32).scatter_add_(
                0, buckets.unsqueeze(1).expand(-1, dim), samples.float()
            )
            / bins_min_clamped[..., None]
        )
        means = torch.where(zero_mask[..., None], means, new_means)
    dists = -(
        samples.float().pow(2).sum(1, keepdim=True)
        - 2 * samples.float() @ means.t()
        + means.t().float().pow(2).sum(0, keepdim=True)
    )
    return (
        means,
        torch.bincount(dists.max(dim=-1).indices, minlength=num_clusters).float(),
    )


class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=1.0,
        decay=0.99,
        epsilon=1e-5,
        threshold_ema_dead=2,
        kmeans_init=True,
        kmeans_iters=10,
    ):
        super().__init__()
        self.input_dim, self.codebook_size, self.codebook_dim = (
            input_dim,
            codebook_size,
            codebook_dim,
        )
        self.commitment, self.decay, self.epsilon, self.threshold_ema_dead = (
            commitment,
            decay,
            epsilon,
            threshold_ema_dead,
        )
        self.kmeans_init, self.kmeans_iters = kmeans_init, kmeans_iters
        self.register_buffer(
            "codebook",
            (torch.zeros(codebook_size, codebook_dim) if kmeans_init else torch.randn(codebook_size, codebook_dim)),
        )
        self.register_buffer("inited", torch.tensor(not kmeans_init, dtype=torch.bool))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", self.codebook.clone())

    def ema_update(self, encodings, embed_onehot):
        encodings, embed_onehot = encodings.float(), embed_onehot.float()
        cluster_size_new, embed_sum = embed_onehot.sum(0), encodings.t() @ embed_onehot
        if dist.is_initialized():
            dist.all_reduce(cluster_size_new)
            dist.all_reduce(embed_sum)
        ema_inplace(self.cluster_size, cluster_size_new, self.decay)
        ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
        cluster_size = (
            (self.cluster_size + self.epsilon)
            / (self.cluster_size.sum() + self.codebook_size * self.epsilon)
            * self.cluster_size.sum()
        )
        self.codebook.copy_(self.embed_avg / cluster_size.unsqueeze(1))

    def replace_dead_codes(self, encodings):
        if self.threshold_ema_dead == 0:
            return
        dead_mask = self.cluster_size < self.threshold_ema_dead
        if dead_mask.any():
            samples = (
                sample_vectors(encodings.float(), self.codebook_size)
                if not dist.is_initialized() or dist.get_rank() == 0
                else torch.zeros_like(self.codebook)
            )
            if dist.is_initialized():
                dist.broadcast(samples, src=0)
            self.codebook[dead_mask] = samples[: dead_mask.sum()].to(self.codebook.dtype)

    def init_codebook(self, encodings):
        if self.inited.item():
            return
        if not dist.is_initialized() or dist.get_rank() == 0:
            embed, cluster_sizes = kmeans(encodings.float(), self.codebook_size, self.kmeans_iters)
        else:
            embed, cluster_sizes = (
                torch.zeros(self.codebook_size, self.codebook_dim, device=encodings.device),
                torch.zeros(self.codebook_size, device=encodings.device),
            )
        if dist.is_initialized():
            dist.broadcast(embed, src=0)
            dist.broadcast(cluster_sizes, src=0)
        self.codebook.copy_(embed)
        self.embed_avg.copy_(embed.clone())
        self.cluster_size.copy_(cluster_sizes)
        self.inited.fill_(True)

    def forward(self, z):
        z_e = z.float()
        encodings = z_e.permute(0, 2, 1).reshape(-1, z_e.size(1))  # b d t -> (b t) d
        if self.kmeans_init and not self.inited.item():
            self.init_codebook(encodings)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ self.codebook.float().t()
            + self.codebook.float().pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(z.size(0), -1)  # (b t) -> b t
        z_q_emb = F.embedding(indices, self.codebook.float()).transpose(1, 2)  # Get embeddings without projection
        commit_loss = F.mse_loss(z_e, z_q_emb.detach(), reduction="none").mean([1, 2]) * self.commitment
        if self.training and torch.is_grad_enabled():
            self.ema_update(encodings, F.one_hot(indices.view(-1), self.codebook_size))
            self.replace_dead_codes(encodings)
        z_q = z_e + (z_q_emb - z_e).detach()
        return z_q, commit_loss, torch.tensor(0.0, device=z.device), indices, z_e

    def decode_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.float()).transpose(1, 2)


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        rvq_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        skip_rvq_ratio: float = 0.0,
        vq_config: VectorQuantizerConfig = None,
        **kwargs,
    ):
        super().__init__()
        self.input_dim, self.rvq_dim, self.output_dim = (
            input_dim,
            rvq_dim,
            output_dim or input_dim,
        )
        self.num_quantizers, self.codebook_size, self.codebook_dim = (
            num_quantizers,
            codebook_size,
            codebook_dim,
        )
        self.quantizer_dropout, self.skip_rvq_ratio = quantizer_dropout, skip_rvq_ratio
        self.input_proj = WNConv1d(input_dim, rvq_dim, 1) if input_dim != rvq_dim else nn.Identity()
        self.output_proj = WNConv1d(rvq_dim, self.output_dim, 1) if rvq_dim != self.output_dim else nn.Identity()
        if vq_config is None:
            vq_config = VectorQuantizerConfig()
        quantizer_kwargs = asdict(vq_config)
        # Remove overlapping parameters from kwargs to avoid conflicts
        for key in quantizer_kwargs.keys():
            kwargs.pop(key, None)
        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(rvq_dim, codebook_size, codebook_dim, **quantizer_kwargs, **kwargs)
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z, input_length, n_quantizers: Optional[int] = None):
        z = self.input_proj(z)

        with torch.autocast("cuda", enabled=False):
            batch_size, _, max_time = z.shape
            device = z.device
            mask = torch.arange(max_time, device=device).expand(batch_size, max_time) < input_length.unsqueeze(1)

            quantized_out = torch.zeros_like(z)
            residual = z.clone().float()

            all_commit_losses = []
            all_indices = []
            all_quantized = []

            # --- Complexity Reduction Start ---
            # 1. Extracted logic for determining quantizer numbers and skip mask
            n_q_tensor = self._get_n_quantizers_tensor(batch_size, device, n_quantizers)
            skip_mask = self._get_skip_mask(batch_size, device)
            # --- Complexity Reduction End ---

            max_q_to_run = self.num_quantizers if self.training else (n_quantizers or self.num_quantizers)

            for i, quantizer in enumerate(self.quantizers[:max_q_to_run]):
                # Create a mask for which batch items are active in this iteration
                active_in_iteration_mask = i < n_q_tensor

                # Skip quantization for items that are not active
                if not active_in_iteration_mask.any():
                    # If no items are active, we can add placeholders and continue
                    # This branch is less common but handles the case where all items have dropped out
                    all_commit_losses.append(torch.tensor(0.0, device=device))
                    all_indices.append(torch.zeros(batch_size, max_time, dtype=torch.long, device=device))
                    all_quantized.append(torch.zeros_like(z))
                    continue

                masked_residual = residual * mask.unsqueeze(1)

                # --- Complexity Reduction Start ---
                # 2. Extracted quantization step logic
                z_q_i, commit_loss_i, indices_i = self._quantize_step(quantizer, masked_residual, skip_mask)
                # --- Complexity Reduction End ---

                # Create a mask for updating tensors (batch items active in this iteration AND within valid length)
                update_mask = active_in_iteration_mask.view(-1, 1, 1) & mask.unsqueeze(1)

                quantized_out += z_q_i * update_mask
                residual -= z_q_i * update_mask

                # Calculate average commitment loss only for active items
                commit_loss_i = (
                    commit_loss_i[active_in_iteration_mask].mean()
                    if active_in_iteration_mask.any()
                    else torch.tensor(0.0, device=device)
                )

                all_commit_losses.append(commit_loss_i)
                all_indices.append(indices_i)
                all_quantized.append(z_q_i)

            # Pad the outputs if the loop was exited early (e.g., in eval mode with n_quantizers)
            num_loops_done = len(all_commit_losses)
            if num_loops_done < self.num_quantizers:
                remaining = self.num_quantizers - num_loops_done
                all_commit_losses.extend([torch.tensor(0.0, device=device)] * remaining)
                all_indices.extend([torch.zeros(batch_size, max_time, dtype=torch.long, device=device)] * remaining)
                all_quantized.extend([torch.zeros_like(z)] * remaining)

        quantized_out = self.output_proj(quantized_out)
        all_indices_tensor = torch.stack(all_indices)
        all_commit_losses_tensor = torch.stack(all_commit_losses)
        all_quantized_tensor = torch.stack(all_quantized)

        return (
            quantized_out,
            all_indices_tensor,
            all_commit_losses_tensor,
            all_quantized_tensor,
            input_length,
        )

    def decode_codes(self, codes):
        nq, B, T = codes.shape
        # If output_proj is nn.Identity, use float32 as default
        if isinstance(self.output_proj, nn.Identity):
            output_dtype = torch.float32
        else:
            output_dtype = next(self.output_proj.parameters()).dtype
        emb = torch.zeros(B, self.rvq_dim, T, device=codes.device, dtype=output_dtype)
        for i, quantizer in enumerate(self.quantizers[:nq]):
            emb += quantizer.decode_code(codes[i])
        return self.output_proj(emb)

    def _get_n_quantizers_tensor(
        self,
        batch_size: int,
        device: torch.device,
        n_quantizers_override: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Determines the number of quantizers to use for each item in the batch,
        applying dropout during training.
        """
        # If not training or dropout is disabled, use the override or default number of quantizers
        is_training = self.training and torch.is_grad_enabled()
        if not is_training or self.quantizer_dropout == 0:
            num_q = n_quantizers_override or self.num_quantizers
            return torch.full((batch_size,), num_q, dtype=torch.long, device=device)

        # During training, apply quantizer dropout
        n_q_tensor = torch.full((batch_size,), self.num_quantizers, device=device)
        n_dropout = int(batch_size * self.quantizer_dropout)
        if n_dropout > 0:
            dropout_indices = torch.randperm(batch_size, device=device)[:n_dropout]
            dropout_values = torch.randint(1, self.num_quantizers + 1, (n_dropout,), device=device)
            n_q_tensor[dropout_indices] = dropout_values

        return n_q_tensor

    def _get_skip_mask(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """Generates a mask for skipping RVQ during training if skip_rvq_ratio > 0."""
        is_training = self.training and torch.is_grad_enabled()
        if not is_training or self.skip_rvq_ratio <= 0:
            return None

        skip_mask = torch.rand(batch_size, device=device) < self.skip_rvq_ratio
        # Ensure at least one sample is not skipped to avoid errors in modules like DDP
        if skip_mask.all():
            skip_mask[0] = False
        return skip_mask

    def _quantize_step(self, quantizer, residual, skip_mask):
        """Helper to perform one step of quantization, handling the skip logic."""
        # The main logic is for non-skipped samples
        z_q_i, commit_loss_i, _, indices_i, z_e_i = quantizer(residual.float())

        # If skipping is active, overwrite the results for the masked samples
        if skip_mask is not None:
            # For skipped samples, the "quantized" output is the residual itself
            # and the loss is zero.
            skip_mask_expanded = skip_mask.view(-1, 1, 1)
            z_q_i = torch.where(skip_mask_expanded, residual, z_q_i)
            commit_loss_i = torch.where(skip_mask, torch.zeros_like(commit_loss_i), commit_loss_i)

        return z_q_i, commit_loss_i, indices_i


# ----------------------------------------------- #
#    PreTrainedModel Base Class                   #
# ----------------------------------------------- #

XY_TOKENIZER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`XYTokenizerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XY_TOKENIZER_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of the input audio waveform.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        n_quantizers (`int`, *optional*):
            The number of quantizers to use for encoding. If not specified, all quantizers are used.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare XY-Tokenizer Model outputting raw hidden-states without any specific head on top.",
    XY_TOKENIZER_START_DOCSTRING,
)
class XYTokenizerPreTrainedModel(PreTrainedAudioTokenizerBase):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XYTokenizerConfig
    base_model_prefix = "xy_tokenizer"
    main_input_name = "input_values"
    _supports_grad_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (XYTokenizerEncoder, XYTokenizerDecoder, XYTokenizerTransformer)):
            module.gradient_checkpointing = value


# ----------------------------------------------- #
#    Main Model Class                             #
# ----------------------------------------------- #


@add_start_docstrings(
    "The XY-Tokenizer Model for encoding and decoding audio.",
    XY_TOKENIZER_START_DOCSTRING,
)
class XYTokenizer(XYTokenizerPreTrainedModel):
    def __init__(self, config: XYTokenizerConfig):
        super().__init__(config)
        # Reconstruct the nested parameter dictionaries from the flat config
        # This is a bit of a boilerplate but necessary to reuse the original module code.
        # A more integrated approach would refactor the sub-modules to accept the flat config directly.
        self.config = config

        params = config.params
        self.semantic_encoder = XYTokenizerEncoder(**params["semantic_encoder_kwargs"])
        self.semantic_encoder_adapter = XYTokenizerTransformer(**params["semantic_encoder_adapter_kwargs"])
        self.acoustic_encoder = XYTokenizerEncoder(**params["acoustic_encoder_kwargs"])
        self.pre_rvq_adapter = XYTokenizerTransformer(**params["pre_rvq_adapter_kwargs"])
        self.downsample = ResidualDownConv(**params["downsample_kwargs"])
        self.quantizer = ResidualVQ(**params["quantizer_kwargs"])
        self.post_rvq_adapter = XYTokenizerTransformer(**params["post_rvq_adapter_kwargs"])
        self.upsample = UpConv(**params["upsample_kwargs"])
        self.acoustic_decoder = XYTokenizerDecoder(**params["acoustic_decoder_kwargs"])
        self.enhanced_vocos = Vocos(**params["vocos_kwargs"])
        self.feature_extractor = XYTokenizerFeatureExtractor(**params["feature_extractor_kwargs"])
        # Store some config values for easier access
        self.encoder_downsample_rate = config.encoder_downsample_rate
        self.nq = params["quantizer_kwargs"]["num_quantizers"]
        # Prefer new canonical names but expose deprecated ones for compatibility
        self.input_sampling_rate = getattr(config, "input_sampling_rate", getattr(config, "input_sample_rate", 16000))
        self.sampling_rate = getattr(config, "sampling_rate", getattr(config, "output_sample_rate", 16000))
        self.input_sample_rate = self.input_sampling_rate
        self.output_sample_rate = self.sampling_rate

        # Initialize weights and apply final processing
        self.post_init()

    def scale_window_size(self, boundaries, scaling_factor):
        scaling_range = []
        scaling_boundaries = []
        for left_boundary, right_boundary in boundaries:
            scaling_left_boundary = left_boundary // scaling_factor
            scaling_right_boundary = right_boundary // scaling_factor
            scaling_range.append(scaling_right_boundary - scaling_left_boundary)
            scaling_boundaries.append(slice(scaling_left_boundary, scaling_right_boundary))
        return scaling_range, scaling_boundaries

    @add_start_docstrings_to_model_forward(XY_TOKENIZER_INPUTS_DOCSTRING)
    @torch.no_grad()
    def encode(
        self,
        features: Union[BatchFeature, ExtractorIterator],
        n_quantizers: Optional[int] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[XYTokenizerEncoderOutput, tuple]:
        r"""
        Encodes the input audio waveform into discrete codes.

        Args:
            features (`BatchFeature` or `ExtractorIterator`):
                A single batch of features or an iterator that yields batches of chunks for long audio files.
                The iterator is expected to yield `BatchFeature` dicts which must contain a `sequence_ids`
                tensor of shape `(batch_size,)` mapping each item in the chunk to its original sequence.
            n_quantizers (`int`, *optional*):
                The number of quantizers to use. If not specified, all quantizers are used.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
            [`XYTokenizerEncoderOutput`] or `tuple(torch.FloatTensor)`
        """
        if not isinstance(features, (BatchFeature, ExtractorIterator)):
            raise TypeError("features must be a BatchFeature or ExtractorIterator")
        # Handle single batch case
        if isinstance(features, BatchFeature):
            return self._encode(features, n_quantizers, return_dict)

        # Handle streaming/chunked case
        else:
            # Use a dictionary to group chunks by their original sequence ID
            encodings = defaultdict(lambda: {"zq": [], "codes": [], "length": 0})
            commit_losses = []
            total_frames = 0

            # 1. Iterate through chunks and store intermediate results
            for chunk_features in features:
                # Always use return_dict=True for easier access to named outputs
                chunk_output = self._encode(chunk_features, n_quantizers, return_dict=True)
                valid_code_lengths, valid_code_ranges = self.scale_window_size(
                    chunk_features["input_lengths"], self.encoder_downsample_rate
                )

                # Accumulate weighted commit loss
                chunk_length = chunk_output.codes_lengths.sum().item()
                valid_chunk_length = sum(valid_code_lengths)
                if chunk_output.commit_loss is not None and valid_chunk_length > 0:
                    commit_loss = chunk_output.commit_loss / chunk_length * valid_chunk_length
                    commit_losses.append((commit_loss.cpu(), valid_chunk_length))
                    total_frames += valid_chunk_length

                # Group results by original sequence ID
                for i, seq_id in enumerate(chunk_features["chunk_seq_no"].tolist()):
                    valid_code_range = valid_code_ranges[i]
                    if valid_code_range.stop > 0:
                        encodings[seq_id]["zq"].append(
                            chunk_output.quantized_representation[i : i + 1, :, valid_code_range]
                        )
                        encodings[seq_id]["codes"].append(chunk_output.audio_codes[:, i : i + 1, valid_code_range])
                        # Add the valid length of this chunk to the total for this sequence
                        encodings[seq_id]["length"] += valid_code_lengths[i]

            final_outputs = []
            for seq_id, seq_data in encodings.items():
                final_outputs.append(
                    {
                        "zq": torch.cat(seq_data["zq"], dim=2),
                        "codes": torch.cat(seq_data["codes"], dim=2),
                        "length": seq_data["length"],
                    }
                )

            # 3. Pad all sequences to the same length and stack into a batch
            max_len = max(seq["zq"].shape[2] for seq in final_outputs)

            batch_zq = []
            batch_codes = []
            batch_lengths = []

            for seq in final_outputs:
                pad_amount = max_len - seq["zq"].shape[2]
                # Pad on the right side of the last dimension (time)
                padded_zq = F.pad(seq["zq"], (0, pad_amount))
                padded_codes = F.pad(seq["codes"], (0, pad_amount))

                batch_zq.append(padded_zq)
                batch_codes.append(padded_codes)
                batch_lengths.append(seq["length"])

            # Stack the list of tensors into a single batch tensor
            quantized_representation = torch.cat(batch_zq, dim=0)
            audio_codes = torch.cat(batch_codes, dim=0)
            codes_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=self.device)

            # 4. Calculate final commit loss
            if total_frames > 0:
                # Weighted average of commit losses
                commit_loss = sum(loss * length for loss, length in commit_losses) / total_frames
                commit_loss = commit_loss.to(self.device)
            else:
                commit_loss = torch.tensor(0.0, device=self.device)

            if not return_dict:
                return (
                    quantized_representation,
                    audio_codes,
                    codes_lengths,
                    commit_loss,
                )

            return XYTokenizerEncoderOutput(
                quantized_representation=quantized_representation,
                audio_codes=audio_codes,
                codes_lengths=codes_lengths,
                commit_loss=commit_loss,
                overlap_seconds=features.overlap_seconds,
            )

    def _encode(
        self,
        features: BatchFeature,
        n_quantizers: Optional[int] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[XYTokenizerEncoderOutput, tuple]:
        input_mel = features["input_features"].to(self.device, dtype=self.dtype)
        mel_attention_mask = features["attention_mask"].to(self.device)
        mel_output_length = mel_attention_mask.sum(dim=-1).long()

        # --- Encoder Path ---
        semantic_encoder_output, semantic_encoder_output_length = self.semantic_encoder(input_mel, mel_output_length)
        semantic_adapter_output, _ = self.semantic_encoder_adapter(
            semantic_encoder_output, semantic_encoder_output_length
        )
        acoustic_encoder_output, acoustic_encoder_output_length = self.acoustic_encoder(input_mel, mel_output_length)

        concated_channel = torch.cat([semantic_adapter_output, acoustic_encoder_output], dim=1)

        pre_rvq_adapter_output, pre_rvq_adapter_output_length = self.pre_rvq_adapter(
            concated_channel, acoustic_encoder_output_length
        )
        downsample_output, downsample_output_length = self.downsample(
            pre_rvq_adapter_output, pre_rvq_adapter_output_length
        )

        n_quantizers = n_quantizers or self.quantizer.num_quantizers
        zq, codes, vq_loss, _, quantizer_output_length = self.quantizer(
            downsample_output, downsample_output_length, n_quantizers=n_quantizers
        )

        if not return_dict:
            return (zq, codes, quantizer_output_length, vq_loss)

        return XYTokenizerEncoderOutput(
            quantized_representation=zq,
            audio_codes=codes,
            codes_lengths=quantizer_output_length,
            commit_loss=vq_loss.mean(),
        )

    @torch.no_grad()
    def decode(
        self,
        audio_codes: Union[torch.Tensor, XYTokenizerEncoderOutput],
        overlap_seconds: int = 10,
        return_dict: Optional[bool] = True,
    ) -> Union[XYTokenizerDecoderOutput, tuple]:
        r"""
        Decodes discrete codes back into an audio waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(num_codebooks, batch_size, sequence_length)`):
                The discrete codes from the quantizer for each codebook.
            codes_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The valid length of each sequence in `audio_codes`. If not provided, it's assumed to be the full length.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
            [`XYTokenizerDecoderOutput`] or `tuple(torch.FloatTensor)`
        """
        if isinstance(audio_codes, tuple):
            raise ValueError("try to set param `return_dict=True` for `codec.encode()` function")
        if not isinstance(audio_codes, (torch.Tensor, XYTokenizerEncoderOutput)):
            raise TypeError("only accept `torch.Tensor` or `XYTokenizerEncoderOutput` for `codec.decode()` function")
        if isinstance(audio_codes, XYTokenizerEncoderOutput):
            audio_codes = audio_codes.audio_codes
            if hasattr(audio_codes, "overlap_seconds"):
                overlap_seconds = audio_codes.overlap_seconds
        if overlap_seconds is None:
            overlap_seconds = 0
        chunk_length = self.feature_extractor.chunk_length
        duration_seconds = chunk_length - overlap_seconds
        chunk_code_length = int(
            chunk_length * self.feature_extractor.sampling_rate // self.config.encoder_downsample_rate
        )  # Maximum code length per chunk
        duration_code_length = int(
            duration_seconds * self.feature_extractor.sampling_rate // self.config.encoder_downsample_rate
        )  # Valid code length per chunk
        duration_wav_length = (
            duration_code_length * self.config.decoder_upsample_rate
        )  # Valid waveform length per chunk

        # Get maximum code length
        batch_size = audio_codes.shape[1]
        codes_list = [audio_codes[:, i, :] for i in range(batch_size)]
        max_code_length = max(codes.shape[-1] for codes in codes_list)
        batch_size = len(codes_list)
        codes_tensor = torch.zeros(self.nq, batch_size, max_code_length, device=self.device, dtype=torch.long)
        code_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i, codes in enumerate(codes_list):
            codes_tensor[:, i, : codes.shape[-1]] = codes.to(self.device)
            code_lengths[i] = codes.shape[-1]  # (B,)

        # Calculate number of chunks needed
        max_chunks = (max_code_length + duration_code_length - 1) // duration_code_length
        wav_list = []

        # Process the entire batch in chunks
        for chunk_idx in range(max_chunks):
            start = chunk_idx * duration_code_length
            end = min(start + chunk_code_length, max_code_length)
            chunk_codes = codes_tensor[:, :, start:end]  # (nq, B, T')
            chunk_code_lengths = torch.clamp(code_lengths - start, 0, end - start)  # (B,)

            # Skip empty chunks
            if chunk_code_lengths.max() == 0:
                continue

            # Decode
            result = self._decode(chunk_codes, chunk_code_lengths)  # {"y": (B, 1, T'), "output_length": (B,)}
            chunk_wav = result["audio_values"]  # (B, 1, T')
            chunk_wav_lengths = result["output_length"]  # (B,)

            # Extract valid portion
            valid_wav_lengths = torch.clamp(chunk_wav_lengths, 0, duration_wav_length)  # (B,)
            valid_chunk_wav = torch.zeros(batch_size, 1, duration_wav_length, device=self.device)
            for b in range(batch_size):
                if valid_wav_lengths[b] > 0:
                    valid_chunk_wav[b, :, : valid_wav_lengths[b]] = chunk_wav[
                        b, :, : valid_wav_lengths[b]
                    ]  # (B, 1, valid_wav_length)

            wav_list.append(valid_chunk_wav)  # (B, 1, valid_wav_length)

        # Concatenate all chunks
        if wav_list:
            wav_tensor = torch.cat(wav_list, dim=-1)  # (B, 1, T_total)
            syn_wav_list = [
                wav_tensor[i, :, : code_lengths[i] * self.config.decoder_upsample_rate] for i in range(batch_size)
            ]  # B * (1, T,)
        else:
            syn_wav_list = [torch.zeros(1, 0, device=self.device) for _ in range(batch_size)]  # B * (1, 0,)

        if not return_dict:
            return (syn_wav_list,)

        return XYTokenizerDecoderOutput(audio_values=syn_wav_list)

    def _decode(
        self,
        audio_codes: torch.Tensor,
        codes_lengths: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[XYTokenizerDecoderOutput, tuple]:
        r"""
        Decodes discrete codes back into an audio waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(num_codebooks, batch_size, sequence_length)`):
                The discrete codes from the quantizer for each codebook.
            codes_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The valid length of each sequence in `audio_codes`. If not provided, it's assumed to be the full length.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
            [`XYTokenizerDecoderOutput`] or `tuple(torch.FloatTensor)`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if codes_lengths is None:
            codes_lengths = torch.full((audio_codes.shape[1],), audio_codes.shape[2], device=self.device)

        # --- Decoder Path ---
        zq = self.quantizer.decode_codes(audio_codes)

        post_rvq_adapter_output, post_rvq_adapter_output_length = self.post_rvq_adapter(zq, codes_lengths)
        upsample_output, upsample_output_length = self.upsample(
            post_rvq_adapter_output, post_rvq_adapter_output_length
        )
        acoustic_decoder_output, acoustic_decoder_output_length = self.acoustic_decoder(
            upsample_output, upsample_output_length
        )
        y, vocos_output_length = self.enhanced_vocos(acoustic_decoder_output, acoustic_decoder_output_length)

        if not return_dict:
            return (y, vocos_output_length)

        return XYTokenizerDecoderOutput(audio_values=y, output_length=vocos_output_length)

    @add_start_docstrings_to_model_forward(XY_TOKENIZER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_quantizers: Optional[int] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[XYTokenizerOutput, tuple]:
        r"""
        The forward method that handles the full encoding and decoding process.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            n_quantizers (`int`, *optional*):
                The number of quantizers to use for encoding. If not specified, all quantizers are used.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Examples:

        ```python
        >>> from transformers import AutoModel, AutoFeatureExtractor
        >>> from datasets import load_dataset, Audio
        >>> import torch

        >>> # This is a placeholder model name, replace with the actual one on the Hub
        >>> model_id = "your-namespace/xy-tokenizer-model"
        >>> model = AutoModel.from_pretrained(model_id)
        >>> # The feature extractor config is part of the model config, so it can be loaded this way
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> # Load a dummy audio dataset
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> audio_sample = ds[0]["audio"]["array"]
        >>> sampling_rate = ds[0]["audio"]["sampling_rate"]

        >>> # Process audio
        >>> inputs = feature_extractor(audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

        >>> # Encode to get codes
        >>> with torch.no_grad():
        ...     encoder_output = model.encode(inputs["input_values"], attention_mask=inputs["attention_mask"])
        ...     audio_codes = encoder_output.audio_codes

        >>> # Decode from codes
        >>> with torch.no_grad():
        ...     decoder_output = model.decode(audio_codes)
        ...     reconstructed_audio = decoder_output.audio_values

        >>> # Full forward pass
        >>> with torch.no_grad():
        ...     model_output = model(**inputs)
        ...     reconstructed_audio_fwd = model_output.audio_values

        >>> print(reconstructed_audio.shape)
        torch.Size([1, 1, 147200])
        >>> print(torch.allclose(reconstructed_audio, reconstructed_audio_fwd))
        True
        ```

        Returns:
            [`XYTokenizerOutput`] or `tuple(torch.FloatTensor)`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create BatchFeature from input_values and attention_mask for encode method
        # If attention_mask is None, create a default one
        if attention_mask is None:
            batch_size = input_values.shape[0]
            seq_len = input_values.shape[-1]
            attention_mask = torch.ones(batch_size, seq_len, device=input_values.device, dtype=torch.long)

        features = BatchFeature({"input_features": input_values, "attention_mask": attention_mask})

        encoder_outputs = self.encode(
            features=features,
            n_quantizers=n_quantizers,
            return_dict=True,
        )

        decoder_outputs = self.decode(audio_codes=encoder_outputs, return_dict=True)

        if not return_dict:
            return (
                decoder_outputs.audio_values,
                decoder_outputs.output_length,
                encoder_outputs.quantized_representation,
                encoder_outputs.audio_codes,
                encoder_outputs.codes_lengths,
                encoder_outputs.commit_loss,
            )

        return XYTokenizerOutput(
            audio_values=decoder_outputs.audio_values,
            output_length=decoder_outputs.output_length,
            quantized_representation=encoder_outputs.quantized_representation,
            audio_codes=encoder_outputs.audio_codes,
            codes_lengths=encoder_outputs.codes_lengths,
            commit_loss=encoder_outputs.commit_loss,
        )


__all__ = ["XYTokenizer"]
