# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
"""
PyTorch DilBERT model.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import sys
from io import open

import itertools
import numpy as np

import torch
import torch.nn as nn

from pytorch_transformers.modeling_utils import PretrainedConfig, PreTrainedModel, add_start_docstrings

import logging
logger = logging.getLogger(__name__)


DILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'dilbert-base-uncased': None, # TODO(Victor)
}

DILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'dilbert-base-uncased': None, #TODO(Victor)
}


class DilBertconfig(PretrainedConfig):
    pretrained_config_archive_map = DILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 max_position_embeddings=512,
                 sinusoidal_pos_embds=True,
                 n_layers=6,
                 n_heads=12,
                 dim=768,
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation='gelu',
                 initializer_range=0.02,
                 tie_weights=True,
                 **kwargs):
        super(DilBertconfig, self).__init__(**kwargs)

        if isintance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.max_position_embeddings = max_position_embeddings
            self.sinusoidal_pos_embds = sinusoidal_pos_embds
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dim = dim
            self.dropout = dropout
            self.attention_dropout = attention_dropout
            self.activation = activation
            self.initializer_range = initializer_range
            self.tie_weights = tie_weights
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False

class Embeddings(nn.Module):
    def __init__(self,
                 config):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings,
                                         dim=config.dim,
                                         out=self.position_embeddings.weight)

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length) - The token ids to embed.
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                      # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)                   # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)        # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 config):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=dim, out_features=dim)
        self.k_lin = nn.Linear(in_features=dim, out_features=dim)
        self.v_lin = nn.Linear(in_features=dim, out_features=dim)
        self.out_lin = nn.Linear(in_features=dim, out_features=dim)

    def forward(self,
                query: torch.tensor,
                key: torch.tensor,
                value: torch.tensor,
                mask: torch.tensor):
        """
        Classic Self Attention. I don't understand the one of PyTorch...

        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Return
        ------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        assert key.size() == value.size()

        dim_per_head = dim // self.n_heads

        assert 2 <= mask.dim() <= 3
        causal = (mask.dim() == 3)
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, dim)

        q = shape(self.q_lin(query))           # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))             # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))           # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)                     # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2,3))          # (bs, n_heads, q_length, k_length)
        mask = (mask==0).view(mask_reshp).expand_as(scores) # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float('inf'))            # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)        # (bs, n_heads, q_length, k_length)
        context = torch.matmul(weights, v)     # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)             # (bs, q_length, dim)
        context = self.out_lin(context)        # (bs, q_length, dim)

        if self.output_attentions:
            return context, weights
        else:
            return context

class FFN(nn.Module):
    def __init__(self,
                 config):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert activation in ['relu', 'gelu'], ValueError(f"activation ({config.activation}) must be in ['relu', 'gelu']")
        self.activation = gelu if activation == 'gelu' else nn.ReLU()

    def forward(self,
                input: torch.tensor):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 config):
        super(TransformerBlock, self).__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(p=config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions

        assert dim % n_heads == 0

        self.attention = MultiHeadSelfAttention(dim=config.dim,
                                                n_heads=config.n_heads,
                                                dropout=config.attention_dropout,
                                                output_attentions=config.output_attentions)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(in_dim=config.dim,
                       hidden_dim=config.hidden_dim,
                       out_dim=config.dim,
                       dropout=config.dropout,
                       activation=config.activation)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self,
                x: torch.tensor,
                attn_mask: torch.tensor = None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)
        """
        # Self-Attention
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask)
        if self.output_attentions:
            sa_output, sa_weights = sa_output                  # (bs, seq_length, dim)
        sa_output = self.sa_layer_norm(sa_output + x)          # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)                             # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        if self.output_attentions:
            return sa_weights, ffn_output
        else:
            return ffn_output

class Transformer(nn.Module):
    def __init__(self,
                 config):
        super(Transformer, self).__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions

        layer = TransformerBlock(n_heads=config.n_heads,
                                 dim=config.dim,
                                 hidden_dim=config.hidden_dim,
                                 dropout=config.dropout,
                                 attention_dropout=config.attention_dropout,
                                 activation=config.activation,
                                 output_attentions=config.output_attentions)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self,
                x: torch.tensor,
                attn_mask: torch.tensor = None,
                output_all_encoded_layers: bool = True):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)
        output_all_encoded_layers: bool
        """
        all_encoder_layers = []
        all_attentions = []

        for _, layer_module in enumerate(self.layer):
            x = layer_module(x=x, attn_mask=attn_mask)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            all_encoder_layers.append(x)

        if not output_all_encoded_layers:
            all_encoder_layers = all_encoder_layers[-1]

        if self.output_attentions:
            return all_attentions, all_encoder_layers
        else:
            return all_encoder_layers



# TODO(Victor)
# class DilBertWithLMHeadModel(DilBertPreTrainedModel):
# class DilBertForSequenceClassification(DilBertPretrainedModel):


class DilBertForQuestionAnswering(DilBertPreTrainedModel):
    def __init__(self, config):
        super(DilBertForQuestionAnswering, self).__init__(config)

        self.dilbert = DilBertModel(config)
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        self.apply(self.init_weights)
        
    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor = None,
                start_positions: torch.tensor = None,
                end_positions: torch.tensor = None):
        _, _, hidden_states = self.dilbert(input_ids=input_ids,
                                           attention_mask=attention_mask) # _, _, (bs, max_query_len, dim)
        
        hidden_states = self.dropout(hidden_states)                       # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)                           # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)                           # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)                               # (bs, max_query_len)

        outputs = (start_logits, end_logits,) + (hidden_states,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, hidden_states