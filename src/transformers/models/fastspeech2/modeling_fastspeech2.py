# coding=utf-8
# Copyright 2022 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch FastSpeech2 model."""


import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_fastspeech2 import FastSpeech2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "fastspeech2"
_CONFIG_FOR_DOC = "FastSpeech2Config"
_TOKENIZER_FOR_DOC = "FastSpeech2Tokenizer"

FASTSPEECH2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "fastspeech2",
    # See all FastSpeech2 models at https://huggingface.co/models?filter=fastspeech2
]


# Copied from https://github.com/pytorch/fairseq/blob/main/fairseq/data/data_utils.py
def lengths_to_padding_mask(lengths: torch.LongTensor) -> torch.BoolTensor:
    batch_size, max_lengths = lengths.size(0), torch.max(lengths).item()
    mask = torch.arange(max_lengths).to(lengths.device).view(1, max_lengths)
    mask = mask.expand(batch_size, -1) >= lengths.view(batch_size, 1).expand(-1, max_lengths)
    return mask


# Adapted from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_length: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, sequence_length]` to `[batch_size, 1, target_length, source_length]`.
    """
    batch_size, source_length = mask.size()
    target_length = target_length if target_length is not None else source_length
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_length, source_length).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


@dataclass
class FastSpeech2ModelOutput(ModelOutput):
    """
    Output type of [`FastSpeech2Model`].
    Args:
        mel_spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_s)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    """

    mel_spectrogram: torch.FloatTensor = None
    out_lengths: torch.FloatTensor = None
    log_duration: torch.FloatTensor = None
    pitch: torch.FloatTensor = None
    energy: torch.FloatTensor = None


# Copied from transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->FastSpeech2
class FastSpeech2PositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        batch_size, sequence_length = input_ids.size()
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + sequence_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(batch_size, sequence_length, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->FastSpeech2
class FastSpeech2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
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
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, sequence_length: int, batch_size: int):
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

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
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class FastSpeech2PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(hidden_dim, in_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = self.dropout_module = nn.Dropout(dropout)

    def forward(self, hidden):
        # hidden.shape == (batch_size, sequence_length, hidden_size)
        residual = hidden
        hidden = hidden.transpose(1, 2)
        hidden = self.conv2(F.relu(self.conv1(hidden)))
        hidden = hidden.transpose(1, 2)
        hidden = self.dropout(hidden)
        return self.layer_norm(hidden + residual)


class FastSpeech2FFTLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, kernel_size, dropout, attention_dropout):
        super().__init__()
        self.self_attn = FastSpeech2Attention(embed_dim, n_heads, dropout=attention_dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = FastSpeech2PositionwiseFeedForward(embed_dim, hidden_dim, kernel_size, dropout=dropout)

    def forward(self, hidden, padding_mask=None):
        # hidden.shape == (batch_size, sequence_length, hidden_size)
        residual = hidden
        hidden, _, _ = self.self_attn(hidden, attention_mask=padding_mask)
        hidden = self.layer_norm(hidden + residual)
        return self.ffn(hidden)


class FastSpeech2LengthRegulator(nn.Module):
    def forward(self, hidden, durations):
        # hidden.shape == (batch_size, sequence_length, hidden_size)
        out_lengths = durations.sum(dim=1)
        max_length = out_lengths.max()
        batch_size, sequence_length, dim = hidden.size()
        out = hidden.new_zeros((batch_size, max_length, dim))
        device = hidden.device

        for b in range(batch_size):
            indices = []
            for t in range(sequence_length):
                indices.extend([t] * durations[b, t].item())
            indices = torch.tensor(indices, dtype=torch.long, device=device)
            out_len = out_lengths[b].item()
            out[b, :out_len] = hidden[b].index_select(0, indices)

        return out, out_lengths


class FastSpeech2VariancePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.encoder_embed_dim,
            config.var_pred_hidden_dim,
            kernel_size=config.var_pred_kernel_size,
            padding=(config.var_pred_kernel_size - 1) // 2,
        )
        self.ln1 = nn.LayerNorm(config.var_pred_hidden_dim)
        self.dropout_module = nn.Dropout(config.var_pred_dropout)
        self.conv2 = nn.Conv1d(
            config.var_pred_hidden_dim,
            config.var_pred_hidden_dim,
            kernel_size=config.var_pred_kernel_size,
            padding=1,
        )
        self.ln2 = nn.LayerNorm(config.var_pred_hidden_dim)
        self.proj = nn.Linear(config.var_pred_hidden_dim, 1)

    def forward(self, hidden):
        # hidden.shape == (batch_size, sequence_length, hidden_size)
        hidden = F.relu(self.conv1(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.dropout_module(self.ln1(hidden))
        hidden = F.relu(self.conv2(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.dropout_module(self.ln2(hidden))
        out = self.proj(hidden).squeeze(dim=2)
        # out.shape == (batch_size, sequence_length)
        return out


class FastSpeech2VarianceAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.length_regulator = FastSpeech2LengthRegulator()
        self.duration_predictor = FastSpeech2VariancePredictor(config)
        self.pitch_predictor = FastSpeech2VariancePredictor(config)
        self.energy_predictor = FastSpeech2VariancePredictor(config)

        n_bins, steps = self.config.var_pred_n_bins, self.config.var_pred_n_bins - 1
        self.pitch_bins = torch.linspace(config.pitch_min, config.pitch_max, steps)
        self.embed_pitch = nn.Embedding(n_bins, config.encoder_embed_dim)
        self.energy_bins = torch.linspace(config.energy_min, config.energy_max, steps)
        self.embed_energy = nn.Embedding(n_bins, config.encoder_embed_dim)

    def get_pitch_embedding(self, hidden, target=None, factor=1.0):
        out = self.pitch_predictor(hidden)
        bins = self.pitch_bins.to(hidden.device)
        if target is None:
            out = out * factor
            pitch_embedding = self.embed_pitch(torch.bucketize(out, bins))
        else:
            pitch_embedding = self.embed_pitch(torch.bucketize(target, bins))
        return out, pitch_embedding

    def get_energy_embedding(self, hidden, target=None, factor=1.0):
        out = self.energy_predictor(hidden)
        bins = self.energy_bins.to(hidden.device)
        if target is None:
            out = out * factor
            energy_embedding = self.embed_energy(torch.bucketize(out, bins))
        else:
            energy_embedding = self.embed_energy(torch.bucketize(target, bins))
        return out, energy_embedding

    def forward(
        self,
        hidden,
        padding_mask,
        durations=None,
        pitches=None,
        energies=None,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
    ):
        # hidden.shape == (batch_size, sequence_length, hidden_size)
        log_dur_out = self.duration_predictor(hidden)
        dur_out = torch.clamp(torch.round((torch.exp(log_dur_out) - 1) * d_factor).long(), min=0)
        dur_out.masked_fill_(padding_mask, 0)

        pitch_out, pitch_embedding = self.get_pitch_embedding(hidden, pitches, p_factor)
        hidden = hidden + pitch_embedding
        energy_out, energy_embedding = self.get_energy_embedding(hidden, energies, e_factor)
        hidden = hidden + energy_embedding

        hidden, out_lens = self.length_regulator(hidden, dur_out if durations is None else durations)

        return hidden, out_lens, log_dur_out, pitch_out, energy_out


class FastSpeech2Postnet(nn.Module):
    def __init__(self, in_dim, n_channels, kernel_size, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            cur_layers = [
                nn.Conv1d(
                    in_dim if i == 0 else n_channels,
                    n_channels if i < n_layers - 1 else in_dim,
                    kernel_size=kernel_size,
                    padding=((kernel_size - 1) // 2),
                ),
                nn.BatchNorm1d(n_channels if i < n_layers - 1 else in_dim),
            ]
            if i < n_layers - 1:
                cur_layers.append(nn.Tanh())
            cur_layers.append(nn.Dropout(dropout))
            self.layers.extend(cur_layers)

    def forward(self, hidden):
        # hidden.shape == (batch_size, sequence_length, hidden_size)
        hidden = hidden.transpose(1, 2)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden.transpose(1, 2)


class FastSpeech2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_dim = config.mel_dim
        self.padding_idx = config.pad_token_id

        self.spk_emb_proj = None
        self.embed_speaker = None
        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embed_dim)
            self.spk_emb_proj = nn.Linear(
                config.encoder_embed_dim + config.speaker_embed_dim, config.encoder_embed_dim
            )

        self.dropout_module = nn.Dropout(config.dropout)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.encoder_embed_dim, padding_idx=self.padding_idx)

        self.embed_positions = FastSpeech2PositionalEmbedding(
            config.max_source_positions, config.encoder_embed_dim, self.padding_idx
        )
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.dec_pos_emb_alpha = nn.Parameter(torch.ones(1))

        self.encoder_fft_layers = nn.ModuleList(
            FastSpeech2FFTLayer(
                config.encoder_embed_dim,
                config.encoder_attention_heads,
                config.fft_hidden_dim,
                config.fft_kernel_size,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.encoder_layers)
        )

        self.var_adaptor = FastSpeech2VarianceAdaptor(config)

        self.decoder_fft_layers = nn.ModuleList(
            FastSpeech2FFTLayer(
                config.decoder_embed_dim,
                config.decoder_attention_heads,
                config.fft_hidden_dim,
                config.fft_kernel_size,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.decoder_layers)
        )

        self.out_proj = nn.Linear(config.decoder_embed_dim, self.out_dim)

        self.postnet = None
        if config.add_postnet:
            self.postnet = FastSpeech2Postnet(
                self.out_dim,
                config.postnet_conv_dim,
                config.postnet_conv_kernel_size,
                config.postnet_layers,
                config.postnet_dropout,
            )

        if config.use_mean:
            self.register_buffer("mean", torch.zeros(self.out_dim))
            logger.warning(
                "Initializing `mean` to zero. "
                "Please disregard this warning if you are loading a pretrained checkpoint. "
                "Otherwise, call `FastSpeech2Model.set_mean(mean)` to set the cepstral mean."
            )
        else:
            self.mean = None
        if config.use_standard_deviation:
            self.register_buffer("std", torch.zeros(self.out_dim))
            logger.warning(
                "Initializing `std` to zero. "
                "Please disregard this warning if you are loading a pretrained checkpoint. "
                "Otherwise, call `FastSpeech2Model.set_standard_deviation(std)` to set the cepstral variance."
            )
        else:
            self.std = None

    def forward(
        self,
        input_ids,
        speaker_ids=None,
        durations=None,
        pitches=None,
        energies=None,
        return_dict=False,
        **kwargs,
    ):
        hidden = self.embed_tokens(input_ids)

        enc_padding_mask = input_ids.eq(self.padding_idx)
        hidden = hidden + self.pos_emb_alpha * self.embed_positions(enc_padding_mask)
        hidden = self.dropout_module(hidden)

        attention_mask = _expand_mask(1 - enc_padding_mask.int(), hidden.dtype)

        for layer in self.encoder_fft_layers:
            hidden = layer(hidden, attention_mask)

        if self.embed_speaker is not None:
            if speaker_ids is None:
                raise ValueError("`speaker` cannot be `None` for multi-speaker FastSpeech2.")
            batch_size, sequence_length, _ = hidden.size()
            speaker_embedding = self.embed_speaker(speaker_ids).expand(batch_size, sequence_length, -1)
            hidden = self.spk_emb_proj(torch.cat([hidden, speaker_embedding], dim=2))

        hidden, out_lengths, log_dur_out, pitch_out, energy_out = self.var_adaptor(
            hidden, enc_padding_mask, durations, pitches, energies
        )

        dec_padding_mask = lengths_to_padding_mask(out_lengths)
        attention_mask = _expand_mask(1 - dec_padding_mask.int(), hidden.dtype)

        hidden = hidden + self.dec_pos_emb_alpha * self.embed_positions(dec_padding_mask)

        for layer in self.decoder_fft_layers:
            hidden = layer(hidden, attention_mask)

        hidden = self.out_proj(hidden)
        if self.postnet is not None:
            hidden = hidden + self.postnet(hidden)
        if self.std is not None:
            hidden = hidden * self.std.view(1, 1, -1).expand_as(hidden)
        if self.mean is not None:
            hidden = hidden + self.mean.view(1, 1, -1).expand_as(hidden)

        if not return_dict:
            return tuple([hidden, out_lengths, log_dur_out, pitch_out, energy_out])
        return FastSpeech2ModelOutput(hidden, out_lengths, log_dur_out, pitch_out, energy_out)


class FastSpeech2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastSpeech2Config
    base_model_prefix = "fastspeech2"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FastSpeech2Postnet):
            convolutions = [layer for layer in module.layers if isinstance(layer, nn.Conv1d)]
            for i, convolution in enumerate(convolutions):
                nn.init.xavier_uniform_(
                    convolution.weight,
                    nn.init.calculate_gain("tanh" if i < len(convolutions) - 1 else "linear"),
                )
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight, nn.init.calculate_gain("relu"))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2Encoder):
            module.gradient_checkpointing = value


FASTSPEECH2_START_DOCSTRING = r"""
    FastSpeech2 was proposed in [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) by Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`FastSpeech2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FASTSPEECH2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`FastSpeech2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        speaker_ids (`torch.LongTensor`, *optional*):
            Indices of speaker ids. 
        durations (`torch.LongTensor`, *optional*):
            pass
        pitches (`torch.FloatTensor` of shape `()`, *optional*):
            pass
        energies (`torch.FloatTensor` of shape `()`, *optional*):
            pass
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The FastSpeech2 Model that outputs predicted mel-spectrograms.",
    FASTSPEECH2_START_DOCSTRING,
)
class FastSpeech2Model(FastSpeech2PreTrainedModel):

    config_class = FastSpeech2Config
    base_model_prefix = "fastspeech2"
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = FastSpeech2Encoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def set_mean(self, mean: torch.Tensor) -> None:
        """
        Set the mean for mel-cepstral denormalization.
        """
        valid_shape = self.encoder.mean.shape
        if mean.shape != valid_shape:
            raise ValueError(f"`mean` should be of shape {valid_shape}, but got {mean.shape} instead.")
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        self.encoder.mean = mean.to(self.encoder.mean.device, dtype=torch.float32)

    def set_standard_deviation(self, standard_deviation: torch.Tensor) -> None:
        """
        Set the standard deviation for mel-cepstral scaling.
        """
        valid_shape = self.encoder.standard_deviation.shape
        if standard_deviation.shape != valid_shape:
            raise ValueError(
                f"`standard_deviation` should be of shape {valid_shape}, but got {standard_deviation.shape} instead."
            )
        if isinstance(standard_deviation, np.ndarray):
            standard_deviation = torch.from_numpy(standard_deviation)
        self.encoder.std = standard_deviation.to(self.encoder.std.device, dtype=torch.float32)

    @add_start_docstrings_to_model_forward(FASTSPEECH2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=FastSpeech2ModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        pitches: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = False,
        **kwargs,
    ):
        return self.encoder(
            input_ids=input_ids,
            speaker_ids=speaker_ids,
            durations=durations,
            pitches=pitches,
            energies=energies,
            return_dict=return_dict,
            **kwargs,
        )
