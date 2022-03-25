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

from ...file_utils import (
    ModelOutput,
    # add_code_sample_docstrings,
    # add_start_docstrings,
    # add_start_docstrings_to_model_forward,
    # replace_return_docstrings,
)
from ...modeling_utils import (
    PreTrainedModel,
    # apply_chunking_to_forward,
    # find_pruneable_heads_and_indices,
    # prune_linear_layer,
)
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


def lengths_to_padding_mask(lens: torch.LongTensor) -> torch.BoolTensor:
    # from https://github.com/pytorch/fairseq/blob/main/fairseq/data/data_utils.py
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


@dataclass
class FastSpeech2ModelOutput(ModelOutput):
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
        bsz, seq_len = input_ids.size()
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Args:
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(
                in_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                in_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = self.dropout_module = nn.Dropout(dropout)

    def forward(self, x):
        # B x T x C
        residual = x
        x = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class FFTLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, kernel_size, dropout, attention_dropout):
        super().__init__()
        self.self_attn = FastSpeech2Attention(embed_dim, n_heads, dropout=attention_dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = PositionwiseFeedForward(embed_dim, hidden_dim, kernel_size, dropout=dropout)

    def forward(self, x, padding_mask=None):
        # B x T x C
        residual = x
        x, _, _ = self.self_attn(x, attention_mask=padding_mask)
        x = self.layer_norm(x + residual)
        return self.ffn(x)


class LengthRegulator(nn.Module):
    def forward(self, x, durations):
        # x: B x T x C
        out_lens = durations.sum(dim=1)
        max_len = out_lens.max()
        bsz, seq_len, dim = x.size()
        out = x.new_zeros((bsz, max_len, dim))

        for b in range(bsz):
            indices = []
            for t in range(seq_len):
                indices.extend([t] * durations[b, t].item())
            indices = torch.tensor(indices, dtype=torch.long).to(x.device)
            out_len = out_lens[b].item()
            out[b, :out_len] = x[b].index_select(0, indices)

        return out, out_lens


class VariancePredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.encoder_embed_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=(args.var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.dropout_module = nn.Dropout(args.var_pred_dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                args.var_pred_hidden_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.proj = nn.Linear(args.var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln1(x))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln2(x))
        return self.proj(x).squeeze(dim=2)


class VarianceAdaptor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.length_regulator = LengthRegulator()
        self.duration_predictor = VariancePredictor(args)
        self.pitch_predictor = VariancePredictor(args)
        self.energy_predictor = VariancePredictor(args)

        n_bins, steps = self.args.var_pred_n_bins, self.args.var_pred_n_bins - 1
        self.pitch_bins = torch.linspace(args.pitch_min, args.pitch_max, steps)
        self.embed_pitch = nn.Embedding(n_bins, args.encoder_embed_dim)
        self.energy_bins = torch.linspace(args.energy_min, args.energy_max, steps)
        self.embed_energy = nn.Embedding(n_bins, args.encoder_embed_dim)

    def get_pitch_emb(self, x, tgt=None, factor=1.0):
        out = self.pitch_predictor(x)
        bins = self.pitch_bins.to(x.device)
        if tgt is None:
            out = out * factor
            emb = self.embed_pitch(torch.bucketize(out, bins))
        else:
            emb = self.embed_pitch(torch.bucketize(tgt, bins))
        return out, emb

    def get_energy_emb(self, x, tgt=None, factor=1.0):
        out = self.energy_predictor(x)
        bins = self.energy_bins.to(x.device)
        if tgt is None:
            out = out * factor
            emb = self.embed_energy(torch.bucketize(out, bins))
        else:
            emb = self.embed_energy(torch.bucketize(tgt, bins))
        return out, emb

    def forward(
        self,
        x,
        padding_mask,
        durations=None,
        pitches=None,
        energies=None,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
    ):
        # x: B x T x C
        log_dur_out = self.duration_predictor(x)
        dur_out = torch.clamp(torch.round((torch.exp(log_dur_out) - 1) * d_factor).long(), min=0)
        dur_out.masked_fill_(padding_mask, 0)

        pitch_out, pitch_emb = self.get_pitch_emb(x, pitches, p_factor)
        x = x + pitch_emb
        energy_out, energy_emb = self.get_energy_emb(x, energies, e_factor)
        x = x + energy_emb

        x, out_lens = self.length_regulator(x, dur_out if durations is None else durations)

        return x, out_lens, log_dur_out, pitch_out, energy_out


class Postnet(nn.Module):
    def __init__(self, in_dim, n_channels, kernel_size, n_layers, dropout):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        assert kernel_size % 2 == 1
        for i in range(n_layers):
            cur_layers = (
                [
                    nn.Conv1d(
                        in_dim if i == 0 else n_channels,
                        n_channels if i < n_layers - 1 else in_dim,
                        kernel_size=kernel_size,
                        padding=((kernel_size - 1) // 2),
                    ),
                    nn.BatchNorm1d(n_channels if i < n_layers - 1 else in_dim),
                ]
                + ([nn.Tanh()] if i < n_layers - 1 else [])
                + [nn.Dropout(dropout)]
            )
            nn.init.xavier_uniform_(
                cur_layers[0].weight,
                torch.nn.init.calculate_gain("tanh" if i < n_layers - 1 else "linear"),
            )
            self.convolutions.append(nn.Sequential(*cur_layers))

    def forward(self, x):
        x = x.transpose(1, 2)  # B x T x C -> B x C x T
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)


class FastSpeech2Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.padding_idx = args.pad_token_id
        self.n_frames_per_step = args.n_frames_per_step
        self.out_dim = args.output_frame_dim * args.n_frames_per_step

        self.spk_emb_proj = None
        self.embed_speaker = None
        if args.num_speakers > 1:
            self.embed_speaker = nn.Embedding(args.num_speakers, args.speaker_embed_dim)
            self.spk_emb_proj = nn.Linear(args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim)

        self.dropout_module = nn.Dropout(args.dropout)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.encoder_embed_dim, padding_idx=self.padding_idx)

        self.embed_positions = FastSpeech2PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.dec_pos_emb_alpha = nn.Parameter(torch.ones(1))

        self.encoder_fft_layers = nn.ModuleList(
            FFTLayer(
                args.encoder_embed_dim,
                args.encoder_attention_heads,
                args.fft_hidden_dim,
                args.fft_kernel_size,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
            )
            for _ in range(args.encoder_layers)
        )

        self.var_adaptor = VarianceAdaptor(args)

        self.decoder_fft_layers = nn.ModuleList(
            FFTLayer(
                args.decoder_embed_dim,
                args.decoder_attention_heads,
                args.fft_hidden_dim,
                args.fft_kernel_size,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
            )
            for _ in range(args.decoder_layers)
        )

        self.out_proj = nn.Linear(args.decoder_embed_dim, self.out_dim)

        self.postnet = None
        if args.add_postnet:
            self.postnet = Postnet(
                self.out_dim,
                args.postnet_conv_dim,
                args.postnet_conv_kernel_size,
                args.postnet_layers,
                args.postnet_dropout,
            )

        if args.mean:
            self.register_buffer("mean", torch.zeros(self.out_dim))
        else:
            self.mean = None
        if args.std:
            self.register_buffer("std", torch.zeros(self.out_dim))
        else:
            self.std = None

    def forward(
        self,
        src_tokens,
        speaker=None,
        durations=None,
        pitches=None,
        energies=None,
    ):
        x = self.embed_tokens(src_tokens)

        enc_padding_mask = src_tokens.eq(self.padding_idx)
        x += self.pos_emb_alpha * self.embed_positions(enc_padding_mask)
        x = self.dropout_module(x)

        attention_mask = _expand_mask(1 - enc_padding_mask.int(), x.dtype)

        for layer in self.encoder_fft_layers:
            x = layer(x, attention_mask)

        if self.embed_speaker is not None:
            if speaker is None:
                raise ValueError("`speaker` cannot be `None` for multi-speaker FastSpeech2.")
            bsz, seq_len, _ = x.size()
            emb = self.embed_speaker(speaker).expand(bsz, seq_len, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        x, out_lens, log_dur_out, pitch_out, energy_out = self.var_adaptor(
            x, enc_padding_mask, durations, pitches, energies
        )

        dec_padding_mask = lengths_to_padding_mask(out_lens)
        attention_mask = _expand_mask(1 - dec_padding_mask.int(), x.dtype)

        x += self.dec_pos_emb_alpha * self.embed_positions(dec_padding_mask)

        for layer in self.decoder_fft_layers:
            x = layer(x, attention_mask)

        x = self.out_proj(x)
        if self.postnet is not None:
            x = x + self.postnet(x)
        if self.std is not None:
            x = x * self.std.view(1, 1, -1).expand_as(x)
        if self.mean is not None:
            x = x + self.mean.view(1, 1, -1).expand_as(x)
        return FastSpeech2ModelOutput(x, out_lens, log_dur_out, pitch_out, energy_out)


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
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight, nn.init.calculate_gain("relu"))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2Encoder):
            module.gradient_checkpointing = value


class FastSpeech2Model(FastSpeech2PreTrainedModel):
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

    def set_mean(self, mean):
        valid_shape = self.encoder.mean.shape
        if mean.shape != valid_shape:
            raise ValueError(f"`mean` should be of shape {valid_shape}, but got {mean.shape} instead")
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        self.encoder.mean = mean.to(self.encoder.mean.device, dtype=torch.float32)

    def set_std(self, std):
        valid_shape = self.encoder.std.shape
        if std.shape != valid_shape:
            raise ValueError(f"`std` should be of shape {valid_shape}, but got {std.shape} instead")
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std)
        self.encoder.std = std.to(self.encoder.std.device, dtype=torch.float32)

    def forward(
        self,
        input_ids,
        speaker=None,
        durations=None,
        pitches=None,
        energies=None,
        **kwargs,
    ):
        return self.encoder(
            src_tokens=input_ids,
            speaker=speaker,
            durations=durations,
            pitches=pitches,
            energies=energies,
            **kwargs,
        )
