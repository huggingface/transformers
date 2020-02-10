# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BART model, ported from the fairseq repo."""

import logging
import random
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .configuration_bart import BartConfig
from .file_utils import add_start_docstrings
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)


BART_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/bart-large-pytorch_model.bin",
    "bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/bart-large-mnli-pytorch_model.bin",
}


BART_START_DOCSTRING = r"""  TODO(SS): FIXME"""


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    pretrained_model_archive_map = BART_PRETRAINED_MODEL_ARCHIVE_MAP

    def _init_weights(self, module):
        std = self.init_std  # used by init_params

        # called init_bert_params in fairseq
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Public API


@add_start_docstrings(
    "The bare BART Model model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
    "FIXME(SS)",
)
class BartModel(PretrainedBartModel,):
    """FIXME(SS)"""

    def __init__(self, config: BartConfig):  # should take config
        super().__init__(config)
        self.config = config
        self._is_generation_fast = False
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_std = config.init_std
        self.init_weights()
        self._is_generation_fast = False  # TODO(SS): this might need deletion

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def forward(self, input_ids: torch.LongTensor = None, return_for_head=False, **kwargs):
        input_ids = input_ids if input_ids is not None else kwargs["encoder_input_ids"]  # TODO(SS): decide on API
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.size(-1) > min(self.max_positions()):
            raise ValueError(
                "input_ids exceeds maximum length: {} > {}".format(input_ids.size(-1), self.max_positions())
            )
        encoder_out = self.encoder(input_ids)
        prev_output_tokens = self.shift_tokens_left(input_ids, self.config.pad_token_id)
        dec_features, dec_hidden, dec_attn = self.decoder(prev_output_tokens, encoder_out=encoder_out,)
        if return_for_head:  # split encoder and decoder outputs nicely
            return (
                _filter_out_nones(dec_features, dec_hidden, dec_attn),
                _filter_out_nones(encoder_out.encoder_out, encoder_out.encoder_states, encoder_out.encoder_attn),
            )
        if self.output_hidden_states and self.output_attentions:
            return (
                dec_features,
                dec_hidden,
                dec_attn,
                encoder_out.encoder_out,
                encoder_out.encoder_states,
                encoder_out.encoder_attn,
            )
        elif self.output_hidden_states:
            return (
                dec_features,
                dec_hidden,
                encoder_out.encoder_out,
                encoder_out.encoder_states,
            )
        elif self.output_attentions:
            return (dec_features, dec_attn, encoder_out.encoder_out, encoder_out.encoder_attn)
        else:
            return (dec_features, encoder_out.encoder_out)

    @staticmethod
    def shift_tokens_left(input_ids, pad_token_id):
        """Shift input ids one token to the left"""
        prev_output_tokens = input_ids.clone()
        prev_output_tokens[:, 0] = input_ids.gather(
            1, (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1),
        ).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())


class BartWithLMHeadModel(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)

    def forward(self, *args, **kwargs):  # TODO(SS): maybe use split kwargs?
        kwargs["return_for_head"] = True
        lm_labels = kwargs.pop("lm_labels", None)
        decoder_outputs, encoder_outputs = self.model(*args, **kwargs)
        lm_logits = self.output_layer(decoder_outputs[0])
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:  # TODO(SS): do we need to know mask_token_ids?
            loss = self.calc_lm_loss(lm_labels, lm_logits)
            decoder_outputs = (loss,) + decoder_outputs
        return decoder_outputs + encoder_outputs

    @staticmethod
    def calc_lm_loss(lm_labels, lm_logits):
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def output_layer(self, x):
        return F.linear(x, self.model.shared.weight)


class BartForSequenceClassification(PretrainedBartModel):
    eos_token = 2

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BARTClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, *args, **kwargs):

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        kwargs["return_for_head"] = True
        labels = kwargs.pop("labels", None)
        decoder_outputs, encoder_outputs = self.model(input_ids, *args, **kwargs)
        x = decoder_outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.eos_token)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        # Prepend logits
        decoder_outputs = (logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs + encoder_outputs


# Encoder and Decoder


class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, attention_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attention_mask (ByteTensor): binary tensor of shape (seq_len, seq_len)
                attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if attention_mask is not None:
            attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e8)  # unused, asked why!
        # anything in original attention_mask = 1, becomes -1e8
        # anything in original attention_mask = 0, becomes 0
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=self.output_attentions,
            attn_mask=attention_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = x
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        y = x  # TODO(SS): why
        x, self_attn_weights = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_attn,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)

        x, encoder_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_out,  # could be None
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=False,  # not returning it so why compute it
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, self_attn_weights  # just self_attn weights for now, following t5


EncoderOut = namedtuple(
    "TransformerEncoderOut",
    [
        "encoder_out",  # T x B x C
        "encoder_padding_mask",  # B x T
        "encoder_embedding",  # B x T x C
        "encoder_states",  # List[T x B x C]
        "encoder_attn",
    ],
)


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx,)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim)

    def forward_embedding(self, input_ids):
        # embed tokens and positions
        embedded_tokens = self.embed_tokens(input_ids)
        x = embedded_tokens + self.embed_positions(input_ids)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embedded_tokens

    def forward(self, input_ids, **unused):  # TODO(SS): this will need more
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(input_ids)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = input_ids.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states, all_attentions = [], []

        # encoder layers
        for layer in self.layers:

            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = layer(x, encoder_padding_mask)

            if self.output_attentions:
                all_attentions.append(attn)
        if self.output_hidden_states:
            encoder_states.append(x)

        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            encoder_attn=all_attentions,  # TODO(SS): document types
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return min(self.max_source_positions, self.embed_positions.max_positions)

    # Unused
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(encoder_out=encoder_out.encoder_out.index_select(1, new_order))
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.register_buffer("version", torch.Tensor([3]))  # TODO(SS): could delete this and pop from state_dict
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, full_context_alignment=False,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # if alignment_layer is None: alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state)
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)

        if positions is not None:
            x += positions

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        for layer in self.layers:
            # layer: DecoderLayer
            encoder_state = None if encoder_out is None else encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                # TODO(SS): check this
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.layerdrop):
                x, layer_self_attn = layer(  # TODO(SS): remove forward
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=self.output_attentions,  # (i == alignment_layer),
                )
                if self.output_hidden_states:
                    all_hidden_states += (x,)
                if self.output_attentions:
                    all_self_attns += (layer_self_attn,)  # .float?
                # if layer_self_attn is not None and i == alignment_layer:
                #    attn = layer_self_attn.float()

        # T x B x C -> B x T x C
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)

        return x, all_hidden_states, list(all_self_attns)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        """Upper triangular matrix filled with negative inf for masking."""
        # TODO(SS): identical to encoder code
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


# Helper Modules


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self, input_dim, inner_dim, num_classes, pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x, **unused):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # WHY?
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.max_positions = num_embeddings - self.padding_idx - 1

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            # Without the int() cast, it doesn't work in some cases when exporting to ONNX
            positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
        else:
            positions = self.make_positions(input, self.padding_idx)
        return super().forward(positions)

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class SelfAttention(nn.Module):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim  # True for all BART

        assert self.encoder_decoder_attention or qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        q = self.q_proj(query) * self.scaling
        if self.encoder_decoder_attention:
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)  # .view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, key_padding_mask, v = self._use_and_update_saved_state(
                k, v, saved_state, key_padding_mask, incremental_state, static_kv, bsz
            )
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float, p=self.dropout, training=self.training,)
        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_weights

    def _use_and_update_saved_state(self, k, v, saved_state, key_padding_mask, incremental_state, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        prev_key_padding_mask = None  # type: Optional[Tensor]
        if "prev_key_padding_mask" in saved_state:
            prev_key_padding_mask = saved_state["prev_key_padding_mask"]
        assert k is not None and v is not None
        key_padding_mask = SelfAttention._append_prev_key_padding_mask(
            key_padding_mask=key_padding_mask,
            prev_key_padding_mask=prev_key_padding_mask,
            batch_size=bsz,
            src_len=k.size(1),
            static_kv=static_kv,
        )
        saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_key_padding_mask"] = key_padding_mask
        # In this branch incremental_state is never None
        assert incremental_state is not None
        self._set_input_buffer(incremental_state, saved_state)
        return k, key_padding_mask, v

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:

            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1))
            if key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order):
        """Reorder buffered internal state (for incremental generation)."""
        # TODO(SS): Where is this used?
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        empty_dict_annotated = {}  # type: Dict[str, Optional[Tensor]]
        if incremental_state is None:
            return empty_dict_annotated
        full_key = _get_full_incremental_state_key(self, "attn_state")
        if full_key not in incremental_state:
            return empty_dict_annotated
        return incremental_state[full_key]

    def _set_input_buffer(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]],
    ):
        full_key = _get_full_incremental_state_key(self, "attn_state")
        incremental_state[full_key] = buffer


def _get_full_incremental_state_key(module_instance, key: str) -> str:
    return "{}.{}.{}".format(module_instance.module_name, module_instance._fairseq_instance_id, key)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _filter_out_nones(*tup):
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)
