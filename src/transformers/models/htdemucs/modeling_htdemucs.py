# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch HT Demucs model."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_htdemucs import HtdemucsConfig


@dataclass
class HtdemucsBaseModelOutput(ModelOutput):
    """
    Base class for Htdemuc's model outputs, with potential hidden states and attentions.

    Args:
        last_freq_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of frequency hidden-states at the output of the last layer of the model.
        last_temp_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of temporal hidden-states at the output of the last layer of the model.
        freq_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the frequency branch of the model at the output of each layer plus the optional initial embedding outputs.
        temp_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the frequency branch of the model at the output of each layer plus the optional initial embedding outputs.
        freq_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights for the frequency branch after the attention softmax, used to compute the weighted average
            in the self-attention heads.
        temp_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights for the frequency branch after the attention softmax, used to compute the weighted average
            in the self-attention heads.
    """

    last_freq_hidden_state: torch.FloatTensor = None
    last_temp_hidden_state: torch.FloatTensor = None
    freq_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    temp_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    freq_attentions: Optional[Tuple[torch.FloatTensor]] = None
    temp_attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def inverse_spectrogram(spectrogram, hop_length, length=None):
    spectrogram = torch.view_as_complex(spectrogram.contiguous())
    spectrogram = nn.functional.pad(spectrogram, (0, 0, 0, 1))
    spectrogram = nn.functional.pad(spectrogram, (2, 2))
    padding = hop_length // 2 * 3
    unpadded_length = hop_length * int(math.ceil(length / hop_length)) + 2 * padding

    *other, freqs, frames = spectrogram.shape
    n_fft = 2 * freqs - 2
    spectrogram = spectrogram.view(-1, freqs, frames)

    waveform = torch.istft(
        spectrogram,
        n_fft,
        hop_length,
        window=torch.hann_window(n_fft).to(spectrogram.real),
        win_length=n_fft,
        normalized=True,
        length=unpadded_length,
        center=True,
    )

    unpadded_length = waveform.shape[1]
    waveform = waveform.view(*other, unpadded_length)
    waveform = waveform[..., padding : padding + length]
    return waveform


class HtdemucsAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: HtdemucsConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.head_dim = config.hidden_size // config.num_attention_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_attention_heads`. Got `hidden_size`: {config.hidden_size}"
                f" and `num_attention_heads`: {config.num_attention_heads}."
            )

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
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

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class HtdemucsTransformerBlock(nn.Module):
    def __init__(self, config: HtdemucsConfig, is_cross_attn=False):
        super().__init__()
        self.embed_dim = config.bottom_channels if config.bottom_channels is not None else config.hidden_size
        self.attn = HtdemucsAttention(config)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if is_cross_attn:
            self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.layer_scale_1 = nn.Parameter(torch.ones(self.embed_dim) * config.layer_scale_init_value)
        self.layer_scale_2 = nn.Parameter(torch.ones(self.embed_dim) * config.layer_scale_init_value)

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
                Inputs to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch, 1, tgt_len, src_len)`, *optional*):
                Attention mask, where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`, *optional*):
                Cross attention input to the layer. Only used for cross attention layers.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.attn_layer_norm(hidden_states)

        if encoder_hidden_states is not None:
            encoder_hidden_states = self.cross_attn_layer_norm(encoder_hidden_states)

        # Cross attention
        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.layer_scale_1 * hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.layer_scale_2 * hidden_states
        hidden_states = residual + hidden_states

        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class HtdemucsTransformerLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig, is_cross_attn: bool):
        super().__init__()
        # mix the temporal/frequency branches with cross-attention conditioning when a cross-attention layer
        self.is_cross_attn = is_cross_attn
        self.freq_attn = HtdemucsTransformerBlock(config, is_cross_attn=is_cross_attn)
        self.temp_attn = HtdemucsTransformerBlock(config, is_cross_attn=is_cross_attn)

    def forward(
        self,
        freq_hidden_states: torch.FloatTensor,
        temp_hidden_states: torch.FloatTensor,
        freq_attention_mask: torch.LongTensor = None,
        temp_attention_mask: torch.LongTensor = None,
        output_attentions: Optional[bool] = False,
    ):
        freq_layer_outputs = self.freq_attn(
            freq_hidden_states,
            attention_mask=freq_attention_mask,
            encoder_hidden_states=temp_hidden_states if self.is_cross_attn else None,
            output_attentions=output_attentions,
        )
        temp_layer_outputs = self.temp_attn(
            temp_hidden_states,
            attention_mask=temp_attention_mask,
            encoder_hidden_states=freq_hidden_states if self.is_cross_attn else None,
            output_attentions=output_attentions,
        )

        # freq and temp hidden-states
        output = (freq_layer_outputs[0], temp_layer_outputs[0])

        if output_attentions:
            # freq and temp attentions
            output += freq_layer_outputs[1] + temp_layer_outputs[1]

        return output


class HtdemucsScaledFrequencyEmbedding(nn.Module):
    """
    Boost the learning rate for frequency embeddings by a factor `scale`, and optionally smooth by the
    distribution's standard deviation.
    """

    def __init__(self, config: HtdemucsConfig):
        super().__init__()
        num_embeddings = config.n_fft // (2 * config.stride)
        self.embedding = nn.Embedding(num_embeddings, config.hidden_channels)

        # smooth the frequency embedding weights
        # when summing gaussian, scale raises as sqrt(n), so we normalize by that
        weight = torch.cumsum(self.embedding.weight.data, dim=0)
        smoothing_factor = torch.arange(1, num_embeddings + 1)
        smoothing_factor = smoothing_factor[:, None].to(weight)
        self.embedding.weight.data[:] = weight / smoothing_factor.sqrt()

        self.embedding.weight.data /= config.freq_embedding_lr_scale
        self.lr_scale = config.freq_embedding_lr_scale
        self.weight_scale = config.freq_embedding_scale

    def forward(self, input_features):
        frequencies = torch.arange(input_features.shape[-2], device=input_features.device)
        embeddings = self.lr_scale * self.embedding(frequencies)
        embeddings = embeddings.transpose(1, 0)[None, :, :, None].expand_as(input_features)
        embeddings = self.weight_scale * embeddings
        return embeddings


class HtdemucsResidualConvLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig, out_channels):
        super().__init__()
        residual_hidden_size = int(out_channels / 8)

        self.conv_in = nn.ModuleList([])
        self.norm_in = nn.ModuleList([])
        self.conv_out = nn.ModuleList([])
        self.norm_out = nn.ModuleList([])
        self.layer_scales = nn.ParameterList([])

        for layer_idx in range(config.residual_conv_depth):
            dilation = 2**layer_idx
            self.conv_in.append(
                nn.Conv1d(out_channels, residual_hidden_size, kernel_size=3, dilation=dilation, padding=dilation)
            )
            self.norm_in.append(nn.GroupNorm(1, residual_hidden_size))
            self.conv_out.append(nn.Conv1d(residual_hidden_size, 2 * out_channels, kernel_size=1))
            self.norm_out.append(nn.GroupNorm(1, 2 * out_channels))
            self.layer_scales.append(nn.Parameter(torch.ones(out_channels) * config.layer_scale_init_value))

    def forward(self, hidden_states):
        for idx in range(len(self.conv_in)):
            residual = hidden_states
            hidden_states = self.conv_in[idx](hidden_states)
            hidden_states = self.norm_in[idx](hidden_states)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.conv_out[idx](hidden_states)
            hidden_states = self.norm_out[idx](hidden_states)
            hidden_states = nn.functional.glu(hidden_states, dim=1)
            hidden_states = residual + self.layer_scales[idx][:, None] * hidden_states
        return hidden_states


class HtdemucsTempEncoderLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig, in_channels, out_channels):
        super().__init__()
        self.stride = config.stride
        self.conv_in = nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=self.stride, padding=2)
        self.residual_conv = HtdemucsResidualConvLayer(config, out_channels)
        self.conv_out = nn.Conv1d(out_channels, 2 * out_channels, kernel_size=1)

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[-1]
        if seq_len % self.stride != 0:
            hidden_states = nn.functional.pad(hidden_states, (0, self.stride - (seq_len % self.stride)))

        hidden_states = self.conv_in(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)

        hidden_states = self.residual_conv(hidden_states)

        hidden_states = self.conv_out(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        return hidden_states


class HtdemucsFreqEncoderLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig, in_channels, out_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=(8, 1), stride=(4, 1), padding=(2, 0))
        self.residual_conv = HtdemucsResidualConvLayer(config, out_channels)
        self.conv_out = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)

        bsz, channels, freq, seq_len = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(-1, channels, seq_len)
        hidden_states = self.residual_conv(hidden_states)
        hidden_states = hidden_states.view(bsz, freq, channels, seq_len).permute(0, 2, 1, 3)

        hidden_states = self.conv_out(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        return hidden_states


class HtdemucsTempDecoderLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig, in_channels, out_channels, is_last):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.residual_conv = HtdemucsResidualConvLayer(config, in_channels)
        self.conv_out = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=8, stride=4)
        self.is_last = is_last

    def forward(self, hidden_states, res_hidden_states, length):
        hidden_states = hidden_states + res_hidden_states

        hidden_states = self.conv_in(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        hidden_states = self.residual_conv(hidden_states)

        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states[..., 2 : 2 + length]
        if not self.is_last:
            hidden_states = nn.functional.gelu(hidden_states)

        return hidden_states


class HtdemucsFreqDecoderLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig, in_channels, out_channels, is_last):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.residual_conv = HtdemucsResidualConvLayer(config, in_channels)
        self.conv_out = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(8, 1), stride=(4, 1))
        self.is_last = is_last

    def forward(self, hidden_states, res_hidden_states):
        hidden_states = hidden_states + res_hidden_states

        hidden_states = self.conv_in(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        bsz, channels, freq, seq_len = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(-1, channels, seq_len)
        hidden_states = self.residual_conv(hidden_states)
        hidden_states = hidden_states.view(bsz, freq, channels, seq_len).permute(0, 2, 1, 3)

        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states[..., 2:-2, :]
        if not self.is_last:
            hidden_states = nn.functional.gelu(hidden_states)

        return hidden_states


class HtdemucsPreTrainedModel(PreTrainedModel):
    config_class = HtdemucsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(
            module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)
        ):  # TODO(SG): implement rescale
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HtdemucsTransformer):
            module.gradient_checkpointing = value


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenSinusoidalPositionalEmbedding
class HtdemucsSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        seq_len = input_ids.size(-1)
        # Create the position ids from the input token ids.
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        # expand embeddings if needed
        if seq_len > self.weights.size(0):
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1)).detach()


class Htdemucs2dSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.max_len = config.max_position_embeddings
        self.max_len = 336
        self.d_model = config.hidden_size
        self.num_stems = config.num_stems

        self.pos_emb = None
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        seq_len = x.size(-1)
        # Reset the positional encodings
        if self.pos_emb is not None:
            if self.pos_emb.size(-1) >= seq_len:
                if self.pos_emb.dtype != x.dtype or self.pos_emb.device != x.device:
                    self.pos_emb = self.pos_emb.to(dtype=x.dtype, device=x.device)
                return

        pos_emb = torch.zeros(self.d_model, 2 * self.num_stems, seq_len)
        height_position = torch.arange(0, 2 * self.num_stems, dtype=torch.float32).unsqueeze(1)
        width_position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        # Each dimension uses half of d_model
        d_model = int(self.d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        pos_emb[0:d_model:2, :, :] = (
            torch.sin(width_position * div_term).transpose(0, 1).unsqueeze(1).repeat(1, 2 * self.num_stems, 1)
        )
        pos_emb[1:d_model:2, :, :] = (
            torch.cos(width_position * div_term).transpose(0, 1).unsqueeze(1).repeat(1, 2 * self.num_stems, 1)
        )
        pos_emb[d_model::2, :, :] = (
            torch.sin(height_position * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, seq_len)
        )
        pos_emb[d_model + 1 :: 2, :, :] = (
            torch.cos(height_position * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, seq_len)
        )

        self.pos_emb = pos_emb.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        self.extend_pe(hidden_states)
        seq_len = hidden_states.size(-1)
        relative_position_embeddings = self.pos_emb[:, :, -seq_len:seq_len]
        return relative_position_embeddings


class HtdemucsTransformer(HtdemucsPreTrainedModel):
    def __init__(self, config: HtdemucsConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.layerdrop

        embed_dim = config.bottom_channels if config.bottom_channels is not None else config.hidden_size

        self.layers = nn.ModuleList([])
        for idx in range(config.num_hidden_layers):
            self.layers.append(HtdemucsTransformerLayer(config, is_cross_attn=bool(idx % 2)))

        self.freq_pos_embedding = Htdemucs2dSinusoidalPositionalEmbedding(config)
        self.temp_pos_embedding = HtdemucsSinusoidalPositionalEmbedding(
            config.max_position_embeddings, embedding_dim=embed_dim
        )

        self.freq_layernorm_embedding = nn.LayerNorm(embed_dim)
        self.temp_layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        freq_hidden_states: torch.FloatTensor,
        temp_hidden_states: torch.FloatTensor,
        freq_attention_mask: torch.LongTensor = None,
        temp_attention_mask: torch.LongTensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HtdemucsBaseModelOutput]:
        r"""
        Args:
            freq_hidden_states (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                TODO
            temp_hidden_states (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                TODO
            freq_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on spatial (frequency) padding token indices. Mask values selected
                in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            temp_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on temporal padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        freq_positions = self.freq_pos_embedding(freq_hidden_states)
        temp_positions = self.temp_pos_embedding(temp_hidden_states)

        # TODO(SG): see if we can remove this reshape
        bsz, bottom_channels, freq, seq_len = freq_hidden_states.shape
        freq_hidden_states = freq_hidden_states.reshape(bsz, bottom_channels, freq * seq_len).transpose(1, 2)
        freq_hidden_states = self.freq_layernorm_embedding(freq_hidden_states)
        freq_positions = freq_positions.reshape(bottom_channels, freq * seq_len).transpose(0, 1)

        # TODO(SG): see if we can remove this reshpae
        temp_hidden_states = temp_hidden_states.transpose(1, 2)
        temp_hidden_states = self.temp_layernorm_embedding(temp_hidden_states)

        freq_hidden_states = freq_hidden_states + freq_positions.to(freq_hidden_states.device)
        temp_hidden_states = temp_hidden_states + temp_positions.to(temp_hidden_states.device)

        freq_hidden_states = nn.functional.dropout(freq_hidden_states, p=self.dropout, training=self.training)
        temp_hidden_states = nn.functional.dropout(temp_hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len]
        if freq_attention_mask is not None:
            freq_attention_mask = _expand_mask(freq_attention_mask, freq_hidden_states.dtype)
        if temp_attention_mask is not None:
            temp_attention_mask = _expand_mask(temp_attention_mask, freq_hidden_states.dtype)

        all_freq_hidden_states = (freq_hidden_states,) if output_hidden_states else None
        all_temp_hidden_states = (temp_hidden_states,) if output_hidden_states else None
        all_freq_attentions = () if output_attentions else None
        all_temp_attentions = () if output_attentions else None

        for idx, layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None, (None,), (None,))

            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions, output_hidden_states)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        freq_hidden_states,
                        temp_hidden_states,
                        freq_attention_mask,
                        temp_attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        freq_hidden_states,
                        temp_hidden_states,
                        freq_attention_mask=freq_attention_mask,
                        temp_attention_mask=temp_attention_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                freq_hidden_states = layer_outputs[0]
                temp_hidden_states = layer_outputs[1]

            if output_attentions:
                all_freq_attentions = all_freq_attentions + (layer_outputs[3],)
                all_temp_attentions = all_temp_attentions + (layer_outputs[3],)

            if output_hidden_states:
                all_freq_hidden_states = all_freq_hidden_states + (freq_hidden_states,)
                all_temp_hidden_states = all_temp_hidden_states + (temp_hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [freq_hidden_states, temp_hidden_states, all_freq_hidden_states, all_temp_hidden_states, all_freq_attentions, all_temp_attentions] if v is not None
            )
        return HtdemucsBaseModelOutput(
            last_freq_hidden_state=freq_hidden_states,
            last_temp_hidden_state=temp_hidden_states,
            freq_hidden_states=all_freq_hidden_states,
            temp_hidden_states=all_temp_hidden_states,
            freq_attentions=all_freq_attentions,
            temp_attentions=all_temp_attentions,
        )


class HtdemucsModel(HtdemucsPreTrainedModel):
    def __init__(self, config: HtdemucsConfig):
        super().__init__(config)
        in_channels = config.audio_channels
        out_channels = config.hidden_channels
        num_layers = config.num_conv_layers

        self.bottom_channels = config.bottom_channels
        self.num_stems = config.num_stems
        self.hop_length = config.n_fft // 4

        self.freq_embedding = HtdemucsScaledFrequencyEmbedding(config)

        self.temp_encoder = nn.ModuleList()
        self.freq_encoder = nn.ModuleList()

        self.temp_decoder = nn.ModuleList()
        self.freq_decoder = nn.ModuleList()

        for layer in range(num_layers):
            is_last = layer == 0
            freq_multp = 2 if is_last else 1
            self.temp_encoder.append(HtdemucsTempEncoderLayer(config, in_channels, out_channels))
            self.freq_encoder.append(HtdemucsFreqEncoderLayer(config, freq_multp * in_channels, out_channels))

            in_channels = config.num_stems * in_channels if is_last else in_channels

            self.temp_decoder.insert(0, HtdemucsTempDecoderLayer(config, out_channels, in_channels, is_last))
            self.freq_decoder.insert(
                0, HtdemucsFreqDecoderLayer(config, out_channels, freq_multp * in_channels, is_last)
            )

            in_channels = out_channels
            out_channels = config.channel_growth * out_channels

        hidden_channels = config.hidden_channels * config.channel_growth ** (num_layers - 1)

        self.temp_upsampler = nn.Conv1d(hidden_channels, config.bottom_channels, kernel_size=1)
        self.temp_downsampler = nn.Conv1d(config.bottom_channels, hidden_channels, kernel_size=1)
        self.freq_upsampler = nn.Conv1d(hidden_channels, config.bottom_channels, kernel_size=1)
        self.freq_downsampler = nn.Conv1d(config.bottom_channels, hidden_channels, kernel_size=1)

        self.transformer = HtdemucsTransformer(config)

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_values: torch.FloatTensor,
        labels: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HtdemucsBaseModelOutput]:
        """
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~HtdemucsProcessor.__call__`]
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `torch.FloatTensor`. See [`HtdemucsProcessor.__call__`] for details.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_freq_hidden_states = ()
        all_temp_hidden_states = ()
        temp_lengths = ()

        # prepare the freq branch input
        freq_channels = input_features.size(2)
        freq_mean = input_features.mean(dim=(1, 2, 3), keepdim=True)
        freq_std = input_features.std(dim=(1, 2, 3), keepdim=True)
        freq_hidden_states = (input_features - freq_mean) / (1e-5 + freq_std)

        # prepare the temporal (time) branch input
        temp_mean = input_values.mean(dim=(1, 2), keepdim=True)
        temp_std = input_values.std(dim=(1, 2), keepdim=True)
        temp_hidden_states = (input_values - temp_mean) / (1e-5 + temp_std)

        # down-blocks
        for layer, (temp_encoder, freq_encoder) in enumerate(zip(self.temp_encoder, self.freq_encoder)):
            if layer < len(self.temp_encoder):
                # we have not yet merged branches
                temp_lengths += (temp_hidden_states.shape[-1],)
                temp_hidden_states = temp_encoder(temp_hidden_states)
                # save temp hidden-state for skip connection
                all_temp_hidden_states += (temp_hidden_states,)

            freq_hidden_states = freq_encoder(freq_hidden_states)

            if layer == 0:
                # add frequency embedding to allow for non equivariant convolutions over the frequency axis
                positional_embedding = self.freq_embedding(freq_hidden_states)
                freq_hidden_states = freq_hidden_states + positional_embedding

            # save freq hidden-state for skip connection
            all_freq_hidden_states += (freq_hidden_states,)

        # mid-block
        bsz, channels, freq, seq_len = freq_hidden_states.shape
        freq_hidden_states = freq_hidden_states.reshape(bsz, channels, freq * seq_len)
        freq_hidden_states = self.freq_upsampler(freq_hidden_states)
        # TODO(SG): see if we can remove this reshape
        freq_hidden_states = freq_hidden_states.reshape(bsz, self.config.bottom_channels, freq, seq_len)
        temp_hidden_states = self.temp_upsampler(temp_hidden_states)

        transformer_outputs = self.transformer(
            freq_hidden_states,
            temp_hidden_states,
            output_attention=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        freq_hidden_states = transformer_outputs[0]
        temp_hidden_states = transformer_outputs[1]

        if output_hidden_states:
            # TODO(SG): correct
            all_freq_hidden_states += (freq_hidden_states,)
            all_temp_hidden_states += (temp_hidden_states,)

        freq_hidden_states = freq_hidden_states.transpose(1, 2).reshape(
            bsz, self.config.bottom_channels, freq * seq_len
        )
        freq_hidden_states = self.freq_downsampler(freq_hidden_states)
        freq_hidden_states = freq_hidden_states.reshape(bsz, channels, freq, seq_len)
        temp_hidden_states = temp_hidden_states.transpose(1, 2)
        temp_hidden_states = self.temp_downsampler(temp_hidden_states)

        # up-blocks
        for layer, (temp_decoder, freq_decoder) in enumerate(zip(self.temp_decoder, self.freq_decoder)):
            res_layer = len(self.temp_decoder) - layer - 1
            freq_hidden_states = freq_decoder(freq_hidden_states, all_freq_hidden_states[res_layer])
            temp_hidden_states = temp_decoder(
                temp_hidden_states, all_temp_hidden_states[res_layer], temp_lengths[res_layer]
            )
            if output_hidden_states:
                all_freq_hidden_states += (freq_hidden_states,)
                all_temp_hidden_states += (temp_hidden_states,)

        # un-normalize the frequency branch and post-process (spectrogram -> waveform)
        freq_hidden_states = freq_hidden_states.reshape(bsz, self.num_stems, -1, freq_channels, seq_len)
        freq_hidden_states = freq_hidden_states * freq_std[:, None] + freq_mean[:, None]

        freq_hidden_states = freq_hidden_states.reshape(bsz, self.num_stems, -1, 2, freq_channels, seq_len)
        freq_hidden_states = freq_hidden_states.permute(0, 1, 2, 4, 5, 3)
        freq_hidden_states = inverse_spectrogram(freq_hidden_states, self.hop_length, input_values.shape[-1])

        # un-normalize the temporal branch
        temp_hidden_states = temp_hidden_states.reshape(bsz, self.num_stems, -1, input_values.shape[-1])
        temp_hidden_states = temp_hidden_states * temp_std[:, None] + temp_mean[:, None]

        output_values = temp_hidden_states + freq_hidden_states

        loss = None
        if labels is not None:
           loss_fn = L1Loss()
           loss = loss_fn(output_values, labels, reduction="mean")

        if not return_dict:
            output = (output_values,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return output_values
