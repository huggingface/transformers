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
""" PyTorch FastSpeech2 model. """


import math
import os
from typing import Optional

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
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


def model_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    return m


def lengths_to_padding_mask(lens):
    # lens: torch.LongTensor
    # returns: torch.BoolTensor
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


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
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=attention_dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = PositionwiseFeedForward(embed_dim, hidden_dim, kernel_size, dropout=dropout)

    def forward(self, x, padding_mask=None):
        # B x T x C
        residual = x
        x = x.transpose(0, 1)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=padding_mask, need_weights=False)
        x = x.transpose(0, 1)
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
        self.embed_pitch = Embedding(n_bins, args.encoder_embed_dim)
        self.energy_bins = torch.linspace(args.energy_min, args.energy_max, steps)
        self.embed_energy = Embedding(n_bins, args.encoder_embed_dim)

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
        # super().__init__(src_dict)
        super().__init__()
        self.args = args
        self.padding_idx = args.pad_token_id
        self.n_frames_per_step = args.n_frames_per_step
        self.out_dim = args.output_frame_dim * args.n_frames_per_step

        # NOTE: modified from original fairseq code
        self.spk_emb_proj = None
        self.embed_speaker = None
        if args.num_speakers > 1:
            self.embed_speaker = nn.Embedding(args.num_speakers, args.speaker_embed_dim)
            self.spk_emb_proj = nn.Linear(args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim)

        self.dropout_module = nn.Dropout(args.dropout)
        # len(src_dict) = 75
        # self.embed_tokens = Embedding(len(src_dict), args.encoder_embed_dim, padding_idx=self.padding_idx)
        self.embed_tokens = Embedding(args.vocab_size, args.encoder_embed_dim, padding_idx=self.padding_idx)

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

        self.apply(model_init)

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        speaker=None,
        durations=None,
        pitches=None,
        energies=None,
        **kwargs,
    ):
        x = self.embed_tokens(src_tokens)

        enc_padding_mask = src_tokens.eq(self.padding_idx)
        x += self.pos_emb_alpha * self.embed_positions(enc_padding_mask)
        x = self.dropout_module(x)

        for layer in self.encoder_fft_layers:
            x = layer(x, enc_padding_mask)

        if self.embed_speaker is not None:
            bsz, seq_len, _ = x.size()
            emb = self.embed_speaker(speaker).expand(bsz, seq_len, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        x, out_lens, log_dur_out, pitch_out, energy_out = self.var_adaptor(
            x, enc_padding_mask, durations, pitches, energies
        )

        dec_padding_mask = lengths_to_padding_mask(out_lens)
        x += self.dec_pos_emb_alpha * self.embed_positions(dec_padding_mask)
        for layer in self.decoder_fft_layers:
            x = layer(x, dec_padding_mask)

        x = self.out_proj(x)
        x_post = None
        if self.postnet is not None:
            x_post = x + self.postnet(x)
        return x, x_post, out_lens, log_dur_out, pitch_out, energy_out


class FastSpeech2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = FastSpeech2Config
    # load_tf_weights = load_tf_weights_in_fastspeech2
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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2Encoder):
            module.gradient_checkpointing = value


class FastSpeech2Model(FastSpeech2PreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # self.embeddings = FastSpeech2Embeddings(config)
        self.encoder = FastSpeech2Encoder(config)

        # Initialize weights and apply final processing
        # self.post_init()

    # def get_input_embeddings(self):
    #     return self.embeddings.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        src_tokens,
        src_lengths=None,
        speaker=None,
        durations=None,
        pitches=None,
        energies=None,
        **kwargs,
    ):
        return self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            speaker=speaker,
            durations=durations,
            pitches=pitches,
            energies=energies,
            **kwargs,
        )
