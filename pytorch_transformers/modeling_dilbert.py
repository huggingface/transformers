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
import copy
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


class DilBertConfig(PretrainedConfig):
    pretrained_config_archive_map = DILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 max_position_embeddings=512,
                 sinusoidal_pos_embds=True,
                 n_layers=6,
                 n_heads=12,
                 dim=768,
                 hidden_dim=4*768,
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation='gelu',
                 initializer_range=0.02,
                 tie_weights_=True,
                 **kwargs):
        super(DilBertConfig, self).__init__(**kwargs)

        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
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
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.attention_dropout = attention_dropout
            self.activation = activation
            self.initializer_range = initializer_range
            self.tie_weights_ = tie_weights_
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


### UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE ###
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
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings,
                                         dim=config.dim,
                                         out=self.position_embeddings.weight)

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                      # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)                   # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)        # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)             # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)               # (bs, max_seq_length, dim)
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

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

    def forward(self,
                query: torch.tensor,
                key: torch.tensor,
                value: torch.tensor,
                mask: torch.tensor):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
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
            return (context, weights)
        else:
            return (context,)

class FFN(nn.Module):
    def __init__(self,
                 config):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ['relu', 'gelu'], ValueError(f"activation ({config.activation}) must be in ['relu', 'gelu']")
        self.activation = gelu if config.activation == 'gelu' else nn.ReLU()

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

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self,
                x: torch.tensor,
                attn_mask: torch.tensor = None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask)
        if self.output_attentions:
            sa_output, sa_weights = sa_output                  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else: # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)          # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)                             # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output

class Transformer(nn.Module):
    def __init__(self,
                 config):
        super(Transformer, self).__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self,
                x: torch.tensor,
                attn_mask: torch.tensor = None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()

        hidden_state = x
        for _, layer_module in enumerate(self.layer):
            hidden_state = layer_module(x=hidden_state, attn_mask=attn_mask)
            if self.output_attentions:
                attentions, hidden_state = hidden_state
                all_attentions = all_attentions + (attentions,)
            else: # To handle these `output_attention` or `output_hidden_states` cases returning tuples
                assert type(hidden_state) == tuple
                hidden_state = hidden_state[0]
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


### INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL ###
class DilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = DilBertConfig
    pretrained_model_archive_map = DILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "dilbert"

    def __init__(self, *inputs, **kwargs):
        super(DilBertPreTrainedModel, self).__init__(*inputs, **kwargs)
    
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


DILBERT_START_DOCSTRING = r"""
    Smaller, faster, cheaper, lighter: DilBERT

    For more information on DilBERT, you should check TODO(Victor): Link to Medium

    Parameters:
        config (:class:`~pytorch_transformers.DilBertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DILBERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**L ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices oof input sequence tokens in the vocabulary.
            The input sequences should start with `[CLS]` and `[SEP]` tokens.
            
            For now, ONLY BertTokenizer(`bert-base-uncased`) is supported and you should use this tokenizer when using DilBERT.
        **attention_mask**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

@add_start_docstrings("The bare DilBERT encoder/transformer outputing raw hidden-states without any specific head on top.",
                      DILBERT_START_DOCSTRING, DILBERT_INPUTS_DOCSTRING)
class DilBertModel(DilBertPreTrainedModel):
    r"""
        Parameters
        ----------
        input_ids: torch.tensor(bs, seq_length)
            Sequences of token ids.
        attention_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequences. Optional: If None, it's like there was no padding.
        
        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        pooled_output: torch.tensor(bs, dim)
            Pooled output: for DilBert, the pooled output is simply the hidden state of the [CLS] token.
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
    """
    def __init__(self, config):
        super(DilBertModel, self).__init__(config)

        self.embeddings = Embeddings(config)   # Embeddings
        self.transformer = Transformer(config) # Encoder

        self.apply(self.init_weights)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) # (bs, seq_length)

        embedding_output = self.embeddings(input_ids)   # (bs, seq_length, dim)
        tfmr_output = self.transformer(x=embedding_output,
                                       attn_mask=attention_mask)
        hidden_state = tfmr_output[0]
        pooled_output = hidden_state[:, 0]
        output = (hidden_state, pooled_output) + tfmr_output[1:]

        return output # hidden_state, pooled_output, (hidden_states), (attentions)

@add_start_docstrings("""DilBert Model with a `masked language modeling` head on top. """,
                      DILBERT_START_DOCSTRING, DILBERT_INPUTS_DOCSTRING)
class DilBertForMaskedLM(DilBertPreTrainedModel):
    r"""
        Parameters
        ----------
        input_ids: torch.tensor(bs, seq_length)
            Token ids.
        attention_mask: torch.tensor(bs, seq_length)
            Attention mask. Optional: If None, it's like there was no padding.
        masked_lm_labels: torch.tensor(bs, seq_length)
            The masked language modeling labels. Optional: If None, no loss is computed.

        Outputs
        -------
        mlm_loss: torch.tensor(1,)
            Masked Language Modeling loss to optimize. 
            Optional: only if `masked_lm_labels` is not None
        prediction_logits: torch.tensor(bs, seq_length, voc_size)
            Token prediction logits
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if `output_hidden_states`=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if `output_attentions`=True
    """
    def __init__(self, config):
        super(DilBertForMaskedLM, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.dilbert = DilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.apply(self.init_weights)
        self.tie_weights()

        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def tie_weights(self):
        """
        Tying the weights of the vocabulary projection to the base token embeddings.
        """
        if self.config.tie_weights_:
            self.vocab_projector.weight = self.dilbert.embeddings.word_embeddings.weight

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor = None,
                masked_lm_labels: torch.tensor = None):
        dlbrt_output = self.dilbert(input_ids=input_ids,
                                    attention_mask=attention_mask)
        hidden_states = dlbrt_output[0]                              # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)      # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)                  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits) # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        outputs = (prediction_logits, ) + dlbrt_output[2:]
        if masked_lm_labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)),
                                         masked_lm_labels.view(-1))
            outputs = (mlm_loss,) + outputs     

        return outputs # (mlm_loss), prediction_logits, (hidden_states), (attentions)

@add_start_docstrings("""DilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                         the pooled output) e.g. for GLUE tasks. """,
                      DILBERT_START_DOCSTRING, DILBERT_INPUTS_DOCSTRING)
class DilBertForSequenceClassification(DilBertPreTrainedModel):
    r"""
        Parameters
        ----------
        input_ids: torch.tensor(bs, seq_length)
            Token ids.
        attention_mask: torch.tensor(bs, seq_length)
            Attention mask. Optional: If None, it's like there was no padding.
        labels: torch.tensor(bs,)
            Classification Labels: Optional: If None, no loss will be computed.
        
        Outputs
        -------
        loss: torch.tensor(1)
            Sequence classification loss.
            Optional: Is computed only if `labels` is not None.
        logits: torch.tensor(bs, seq_length)
            Classification (or regression if config.num_labels==1) scores
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if `output_hidden_states`=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if `output_attentions`=True        
    """
    def __init__(self, config):
        super(DilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.dilbert = DilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.apply(self.init_weights)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor = None,
                labels: torch.tensor = None):
        dilbert_output = self.dilbert(input_ids=input_ids,
                                      attention_mask=attention_mask)
        pooled_output = dilbert_output[1]                    # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)
        pooled_output = self.dropout(pooled_output)         # (bs, dim)
        logits = self.classifier(pooled_output)              # (bs, dim)

        outputs = (logits,) + dilbert_output[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

@add_start_docstrings("""DilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
                         the hidden-states output to compute `span start logits` and `span end logits`). """,
                      DILBERT_START_DOCSTRING, DILBERT_INPUTS_DOCSTRING)
class DilBertForQuestionAnswering(DilBertPreTrainedModel):
    r"""
        Parameters
        ----------
        input_ids: torch.tensor(bs, seq_length)
            Token ids.
        attention_mask: torch.tensor(bs, seq_length)
            Attention mask. Optional: If None, it's like there was no padding.
        start_positions: torch,tensor(bs)
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
            Optional: if None, no loss is computed.
        end_positions: torch,tensor(bs)
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
            Optional: if None, no loss is computed.

        Outputs
        -------
        loss: torch.tensor(1)
            Question answering loss.
            Optional: Is computed only if `start_positions` and `end_positions` are not None.
        start_logits: torch.tensor(bs, seq_length)
            Span-start scores.
        end_logits: torch.tensor(bs, seq_length)
            Spand-end scores.
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if `output_hidden_states`=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if `output_attentions`=True
    """
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
        dilbert_output = self.dilbert(input_ids=input_ids,
                                      attention_mask=attention_mask)
        hidden_states = dilbert_output[0]                                 # (bs, max_query_len, dim)

        hidden_states = self.dropout(hidden_states)                       # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)                           # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)                           # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)                               # (bs, max_query_len)

        outputs = (start_logits, end_logits,) + dilbert_output[2:]
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

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
