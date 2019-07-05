# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
""" PyTorch XLM model.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import math
import itertools
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from .file_utils import cached_path
from .model_utils import (CONFIG_NAME, WEIGHTS_NAME, PretrainedConfig, PreTrainedModel,
                          prune_linear_layer, SequenceSummary, SQuADHead)

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-pytorch_model.bin",
}
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-config.json",
}


class XLMConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `XLMModel`.
    """
    pretrained_config_archive_map = PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30145,
                 n_special=0,
                 emb_dim=2048,
                 n_layers=12,
                 n_heads=16,
                 dropout=0.1,
                 attention_dropout=0.1,
                 gelu_activation=True,
                 sinusoidal_embeddings=False,
                 causal=False,
                 asm=False,
                 n_langs=1,
                 max_position_embeddings=512,
                 embed_init_std=2048 ** -0.5,
                 layer_norm_eps=1e-12,
                 init_std=0.02,
                 bos_index=0,
                 eos_index=1,
                 pad_index=2,
                 unk_index=3,
                 mask_index=5,
                 is_encoder=True,

                 finetuning_task=None,
                 num_labels=2,
                 summary_type='last',
                 summary_use_proj=True,
                 summary_activation='tanh',
                 summary_dropout=0.1,
                 start_n_top=5,
                 end_n_top=5,
                 **kwargs):
        """Constructs XLMConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `XLMModel`.
            d_model: Size of the encoder layers and the pooler layer.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            d_inner: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            ff_activation: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            untie_r: untie relative position biases
            attn_type: 'bi' for XLM, 'uni' for Transformer-XL

            dropout: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            dropatt: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.

            dropout: float, dropout rate.
            dropatt: float, dropout rate on attention probabilities.
            init: str, the initialization scheme, either "normal" or "uniform".
            init_range: float, initialize the parameters with a uniform distribution
                in [-init_range, init_range]. Only effective when init="uniform".
            init_std: float, initialize the parameters with a normal distribution
                with mean 0 and stddev init_std. Only effective when init="normal".
            mem_len: int, the number of tokens to cache.
            reuse_len: int, the number of tokens in the currect batch to be cached
                and reused in the future.
            bi_data: bool, whether to use bidirectional input pipeline.
                Usually set to True during pretraining and False during finetuning.
            clamp_len: int, clamp all relative distances larger than clamp_len.
                -1 means no clamping.
            same_length: bool, whether to use the same attention length for each token.
        """
        super(XLMConfig, self).__init__(**kwargs)

        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_words = vocab_size_or_config_json_file
            self.n_special = n_special
            self.emb_dim = emb_dim
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dropout = dropout
            self.attention_dropout = attention_dropout
            self.gelu_activation = gelu_activation
            self.sinusoidal_embeddings = sinusoidal_embeddings
            self.causal = causal
            self.asm = asm
            self.n_langs = n_langs
            self.layer_norm_eps = layer_norm_eps
            self.bos_index = bos_index
            self.eos_index = eos_index
            self.pad_index = pad_index
            self.unk_index = unk_index
            self.mask_index = mask_index
            self.is_encoder = is_encoder
            self.max_position_embeddings = max_position_embeddings
            self.embed_init_std = embed_init_std
            self.init_std = init_std
            self.finetuning_task = finetuning_task
            self.num_labels = num_labels
            self.summary_type = summary_type
            self.summary_use_proj = summary_use_proj
            self.summary_activation = summary_activation
            self.summary_dropout = summary_dropout
            self.start_n_top = start_n_top
            self.end_n_top = end_n_top
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @property
    def total_tokens_embeddings(self):
        return self.n_words + self.n_special

    @property
    def hidden_size(self):
        return self.emb_dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-transformers/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    bs = lengths.size(0)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super(MultiHeadAttention, self).__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.output_attentions = config.output_attentions
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads

    def forward(self, input, mask, kv=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))                                          # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))                                          # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))                                          # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)                                       # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))                           # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if self.output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super(TransformerFFN, self).__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = gelu if config.gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class XLMPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLMConfig
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(XLMPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights. """
        if isinstance(module, nn.Embedding):
            if self.config is not None and self.config.embed_init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)
        if isinstance(module, nn.Linear):
            if self.config is not None and self.config.init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class XLMModel(XLMPreTrainedModel):

    ATTRIBUTES = ['encoder', 'eos_index', 'pad_index',  # 'with_output', 
                  'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 
                  'hidden_dim', 'dropout', 'attention_dropout', 'asm',
                  'asm_cutoffs', 'asm_div_value']

    def __init__(self, config):  #, dico, is_encoder, with_output):
        """ XLM model from: "Cross-lingual Language Model Pretraining" by Guillaume Lample, Alexis Conneau
            Paper: https://arxiv.org/abs/1901.07291
            Original code: https://github.com/facebookresearch/XLM

        Params:
            `config`: a XLMConfig class instance with the configuration to build a new model
            `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
            `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
                This can be used to compute head importance metrics. Default: False

        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
                with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
                `run_bert_extract_features.py`, `run_bert_classifier.py` and `run_bert_squad.py`)
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
                types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
                a `sentence B` token (see XLM paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
                input sequence length in the current batch. It's the mask that we typically use for attention when
                a batch has varying length sentences.
            `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


        Outputs: Tuple of (encoded_layers, pooled_output)
            `encoded_layers`: controled by `output_all_encoded_layers` argument:
                - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                    of each attention block (i.e. 12 full sequences for XLM-base, 24 for XLM-large), each
                    encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
                - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                    to the last attention block of shape [batch_size, sequence_length, hidden_size],
            `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
                classifier pretrained on top of the hidden state associated to the first character of the
                input (`CLS`) to train on the Next-Sentence task (see XLM's paper).

        Example usage:
        ```python
        # Already been converted into WordPiece token ids
        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        config = modeling.XLMConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        model = modeling.XLMModel(config=config)
        all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
        ```
        """
        super(XLMModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError("Currently XLM can only be used as an encoder")
        # self.with_output = with_output
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        # self.dico = dico
        # self.id2lang = config.id2lang
        # self.lang2id = config.lang2id
        # assert len(self.dico) == self.n_words
        # assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = config.emb_dim       # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = config.n_heads   # 8 by default
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        if config.sinusoidal_embeddings:
            create_sinusoidal_embeddings(config.max_position_embeddings, self.dim, out=self.position_embeddings.weight)
        if config.n_langs > 1:
            self.lang_embeddings = nn.Embedding(self.n_langs, self.dim)
        self.embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=config.layer_norm_eps)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        # if self.is_decoder:
        #     self.layer_norm15 = nn.ModuleList()
        #     self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, config=config))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            # if self.is_decoder:
            #     self.layer_norm15.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            #     self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, config=config))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))

        self.apply(self.init_weights)

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    def forward(self, input_ids, lengths=None, positions=None, langs=None,
                token_type_ids=None, attention_mask=None, cache=None, head_mask=None):  # src_enc=None, src_len=None, 
        """
        Inputs:
            `input_ids` LongTensor(bs, slen), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `positions` LongTensor(bs, slen), containing word positions
            `langs` LongTensor(bs, slen), containing language IDs
            `token_type_ids` LongTensor (bs, slen) same as `langs` used for compatibility
        """
        if lengths is None:
            lengths = (input_ids != self.pad_index).sum(dim=1).long()
        # mask = input_ids != self.pad_index

        # check inputs
        bs, slen = input_ids.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # positions
        if positions is None:
            positions = input_ids.new((slen,)).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (bs, slen)  # (slen, bs)
            # positions = positions.transpose(0, 1)

        # langs
        assert langs is None or token_type_ids is None, "You can only use one among langs and token_type_ids"
        if token_type_ids is not None:
            langs = token_type_ids
        if langs is not None:
            assert langs.size() == (bs, slen)  # (slen, bs)
            # langs = langs.transpose(0, 1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x qlen x klen]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.n_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layers

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            input_ids = input_ids[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        tensor = self.embeddings(input_ids)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        hidden_states = ()
        attentions = ()
        for i in range(self.n_layers):
            if self.output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attentions[i](tensor, attn_mask, cache=cache, head_mask=head_mask[i])
            attn = attn_outputs[0]
            if self.output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = F.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        outputs = (tensor,)
        if self.output_hidden_states:
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            outputs = outputs + (attentions,)
        return outputs  # outputs, (hidden_states), (attentions)


class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, config):
        super(XLMPredLayer, self).__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim

        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y=None):
        """ Compute the loss, and optionally the scores.
        """
        outputs = ()
        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            outputs = (scores,) + outputs
            if y is not None:
                loss = F.cross_entropy(scores, y, reduction='elementwise_mean')
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs

        return outputs


class XLMWithLMHeadModel(XLMPreTrainedModel):
    """ XLM model from: "Cross-lingual Language Model Pretraining" by Guillaume Lample, Alexis Conneau
        Paper: https://arxiv.org/abs/1901.07291
        Original code: https://github.com/facebookresearch/XLM

    Params:
        `config`: a XLMConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `run_bert_extract_features.py`, `run_bert_classifier.py` and `run_bert_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see XLM paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for XLM-base, 24 for XLM-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see XLM's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.XLMConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.XLMModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(XLMWithLMHeadModel, self).__init__(config)
        self.torchscript = config.torchscript

        self.transformer = XLMModel(config)
        self.pred_layer = XLMPredLayer(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the embeddings
        """
        if self.torchscript:
            self.pred_layer.proj.weight = nn.Parameter(self.transformer.embeddings.weight.clone())
        else:
            self.pred_layer.proj.weight = self.transformer.embeddings.weight

    def forward(self, input_ids, lengths=None, positions=None, langs=None, token_type_ids=None,
                attention_mask=None, cache=None, labels=None, head_mask=None):
        """
        Args:
            inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            input_mask: float32 Tensor in shape [bsz, len], the input mask.
                0 for real tokens and 1 for padding.
            mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
                from previous batches. The length of the list equals n_layer.
                If None, no memory is used.
            perm_mask: float32 Tensor in shape [bsz, len, len].
                If perm_mask[k, i, j] = 0, i attend to j in batch k;
                if perm_mask[k, i, j] = 1, i does not attend to j in batch k.
                If None, each position attends to all the others.
            target_mapping: float32 Tensor in shape [bsz, num_predict, len].
                If target_mapping[k, i, j] = 1, the i-th predict in batch k is
                on the j-th token.
                Only used during pretraining for partial prediction.
                Set to None during finetuning.
            inp_q: float32 Tensor in shape [bsz, len].
                1 for tokens with losses and 0 for tokens without losses.
                Only used during pretraining for two-stream attention.
                Set to None during finetuning.

            summary_type: str, "last", "first", "mean", or "attn". The method
                to pool the input to get a vector representation.
        """
        transformer_outputs = self.transformer(input_ids, lengths=lengths, positions=positions, token_type_ids=token_type_ids,
                                               langs=langs, attention_mask=attention_mask, cache=cache, head_mask=head_mask)

        output = transformer_outputs[0]
        outputs = self.pred_layer(output, labels)
        outputs = outputs + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        return outputs


class XLMForSequenceClassification(XLMPreTrainedModel):
    """XLM model ("XLM: Generalized Autoregressive Pretraining for Language Understanding").

    Params:
        `config`: a XLMConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
        `summary_type`: str, "last", "first", "mean", or "attn". The method
            to pool the input to get a vector representation. Default: last

    Inputs:
        inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
        token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
        input_mask: float32 Tensor in shape [bsz, len], the input mask.
            0 for real tokens and 1 for padding.
        attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
            but with 1 for real tokens and 0 for padding.
            Added for easy compatibility with the XLM model (which uses this negative masking).
            You can only uses one among `input_mask` and `attention_mask`
        mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
            from previous batches. The length of the list equals n_layer.
            If None, no memory is used.
        perm_mask: float32 Tensor in shape [bsz, len, len].
            If perm_mask[k, i, j] = 0, i attend to j in batch k;
            if perm_mask[k, i, j] = 1, i does not attend to j in batch k.
            If None, each position attends to all the others.
        target_mapping: float32 Tensor in shape [bsz, num_predict, len].
            If target_mapping[k, i, j] = 1, the i-th predict in batch k is
            on the j-th token.
            Only used during pretraining for partial prediction.
            Set to None during finetuning.
        inp_q: float32 Tensor in shape [bsz, len].
            1 for tokens with losses and 0 for tokens without losses.
            Only used during pretraining for two-stream attention.
            Set to None during finetuning.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


    Outputs: Tuple of (logits or loss, mems)
        `logits or loss`:
            if labels is None:
                Token logits with shape [batch_size, sequence_length] 
            else:
                CrossEntropy loss with the targets
        `new_mems`: list (num layers) of updated mem states at the entry of each layer
            each mem state is a torch.FloatTensor of size [self.config.mem_len, batch_size, self.config.d_model]
            Note that the first two dimensions are transposed in `mems` with regards to `input_ids` and `labels`

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.XLMConfig(vocab_size_or_config_json_file=32000, d_model=768,
        n_layer=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.XLMModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(XLMForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, lengths=None, positions=None, langs=None, token_type_ids=None,
                attention_mask=None, cache=None, labels=None, head_mask=None):
        """
        Args:
            inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            input_mask: float32 Tensor in shape [bsz, len], the input mask.
                0 for real tokens and 1 for padding.
            attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
                but with 1 for real tokens and 0 for padding.
                Added for easy compatibility with the XLM model (which uses this negative masking).
                You can only uses one among `input_mask` and `attention_mask`
            mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
                from previous batches. The length of the list equals n_layer.
                If None, no memory is used.
            perm_mask: float32 Tensor in shape [bsz, len, len].
                If perm_mask[k, i, j] = 0, i attend to j in batch k;
                if perm_mask[k, i, j] = 1, i does not attend to j in batch k.
                If None, each position attends to all the others.
            target_mapping: float32 Tensor in shape [bsz, num_predict, len].
                If target_mapping[k, i, j] = 1, the i-th predict in batch k is
                on the j-th token.
                Only used during pretraining for partial prediction.
                Set to None during finetuning.
            inp_q: float32 Tensor in shape [bsz, len].
                1 for tokens with losses and 0 for tokens without losses.
                Only used during pretraining for two-stream attention.
                Set to None during finetuning.
        """
        transformer_outputs = self.transformer(input_ids, lengths=lengths, positions=positions, token_type_ids=token_type_ids,
                                               langs=langs, attention_mask=attention_mask, cache=cache, head_mask=head_mask)

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class XLMForQuestionAnswering(XLMPreTrainedModel):
    """XLM model for Question Answering (span extraction).
    This module is composed of the XLM model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a XLMConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `run_bert_extract_features.py`, `run_bert_classifier.py` and `run_bert_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see XLM paper for more details).
        `attention_mask`: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
            but with 1 for real tokens and 0 for padding.
            Added for easy compatibility with the XLM model (which uses this negative masking).
            You can only uses one among `input_mask` and `attention_mask`
        `input_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = XLMConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = XLMForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(XLMForQuestionAnswering, self).__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = SQuADHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, lengths=None, positions=None, langs=None, token_type_ids=None,
                attention_mask=None, cache=None, start_positions=None, end_positions=None,
                cls_index=None, is_impossible=None, p_mask=None, head_mask=None):

        transformer_outputs = self.transformer(input_ids, lengths=lengths, positions=positions, token_type_ids=token_type_ids,
                                               langs=langs, attention_mask=attention_mask, cache=cache, head_mask=head_mask)

        output = transformer_outputs[0]

        outputs = self.qa_outputs(output, start_positions=start_positions, end_positions=end_positions,
                                  cls_index=cls_index, is_impossible=is_impossible, p_mask=p_mask)

        outputs = outputs + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        return outputs
