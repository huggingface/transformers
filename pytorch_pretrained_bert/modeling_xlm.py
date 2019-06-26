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

import copy
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
from .model_utils import CONFIG_NAME, WEIGHTS_NAME, PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-pytorch_model.bin",
}
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-config.json",
}

DECODER_ONLY_PARAMS = [
    'layer_norm15.%i.weight', 'layer_norm15.%i.bias',
    'encoder_attn.%i.q_lin.weight', 'encoder_attn.%i.q_lin.bias',
    'encoder_attn.%i.k_lin.weight', 'encoder_attn.%i.k_lin.bias',
    'encoder_attn.%i.v_lin.weight', 'encoder_attn.%i.v_lin.bias',
    'encoder_attn.%i.out_lin.weight', 'encoder_attn.%i.out_lin.bias'
]

TRANSFORMER_LAYER_PARAMS = [
    'attentions.%i.q_lin.weight', 'attentions.%i.q_lin.bias',
    'attentions.%i.k_lin.weight', 'attentions.%i.k_lin.bias',
    'attentions.%i.v_lin.weight', 'attentions.%i.v_lin.bias',
    'attentions.%i.out_lin.weight', 'attentions.%i.out_lin.bias',
    'layer_norm1.%i.weight', 'layer_norm1.%i.bias',
    'ffns.%i.lin1.weight', 'ffns.%i.lin1.bias',
    'ffns.%i.lin2.weight', 'ffns.%i.lin2.bias',
    'layer_norm2.%i.weight', 'layer_norm2.%i.bias'
]

class XLMConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `XLMModel`.
    """
    pretrained_config_archive_map = PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file,
                 d_model=1024,
                 n_layer=24,
                 n_head=16,
                 d_inner=4096,
                 ff_activation="gelu",
                 untie_r=True,
                 attn_type="bi",

                 max_position_embeddings=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,

                 dropout=0.1,
                 dropatt=0.1,
                 init="normal",
                 init_range=0.1,
                 init_std=0.02,
                 mem_len=None,
                 reuse_len=None,
                 bi_data=False,
                 clamp_len=-1,
                 same_length=False):
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
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_token = vocab_size_or_config_json_file
            self.d_model = d_model
            self.n_layer = n_layer
            self.n_head = n_head
            assert d_model % n_head == 0
            self.d_head = d_model // n_head
            self.ff_activation = ff_activation
            self.d_inner = d_inner
            self.untie_r = untie_r
            self.attn_type = attn_type

            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps

            self.init = init
            self.init_range = init_range
            self.init_std = init_std
            self.dropout = dropout
            self.dropatt = dropatt
            self.mem_len = mem_len
            self.reuse_len = reuse_len
            self.bi_data = bi_data
            self.clamp_len = clamp_len
            self.same_length = same_length
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as XLMLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class XLMLayerNorm(nn.Module):
        def __init__(self, d_model, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(XLMLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    # nn.init.normal_(m.weight, mean=0, std=1)
    # nn.init.xavier_uniform_(m.weight)
    # nn.init.constant_(m.bias, 0.)
    return m


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
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
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

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None):
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
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
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
        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)

        return self.out_lin(context)


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty


class XLMPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLMConfig
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    pretrained_config_archive_map = PRETRAINED_CONFIG_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "xlm"

    def __init__(self, *inputs, **kwargs):
        super(XLMPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLMLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLMRelativeAttention):
            for param in [module.q, module.k, module.v, module.o, module.r,
                          module.r_r_bias, module.r_s_bias, module.r_w_bias,
                          module.seg_embed]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class XLMModel(XLMPreTrainedModel):

    ATTRIBUTES = ['encoder', 'eos_index', 'pad_index',  # 'with_output', 
                  'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 
                  'hidden_dim', 'dropout', 'attention_dropout', 'asm',
                  'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, output_attentions=False, keep_multihead_output=False):  #, dico, is_encoder, with_output):
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
        super(XLMModel, self).__init__(params)
        self.output_attentions = output_attentions

        # encoder / decoder, output layer
        # self.is_encoder = is_encoder
        # self.is_decoder = not is_encoder
        # self.with_output = with_output

        # dictionary / languages
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        # self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        # assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = params.emb_dim       # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads   # 8 by default
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = Embedding(params.max_position_embeddings, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(params.max_position_embeddings, self.dim, out=self.position_embeddings.weight)
        if params.n_langs > 1:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        # output layer
        # if self.with_output:
        #     self.pred_layer = PredLayer(params)
        #     if params.share_inout_emb:
        #         self.pred_layer.proj.weight = self.embeddings.weight

    # def forward(self, mode, **kwargs):
    #     """
    #     Forward function with different forward modes.
    #     ### Small hack to handle PyTorch distributed.
    #     """
    #     if mode == 'fwd':
    #         return self.fwd(**kwargs)
    #     elif mode == 'predict':
    #         return self.predict(**kwargs)
    #     else:
    #         raise Exception("Unknown mode: %s" % mode)

    def forward(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        for i in range(self.n_layers):

            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # input batch
        bs = len(src_len)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)       # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)    # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # language IDs
        langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        cache = {'slen': 0}

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :]               # (bs, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)                   # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # language IDs
        langs = positions.clone().fill_(tgt_lang_id)

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]               # (bs * beam_size, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores, dim=-1)       # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len


class XLMModel(XLMPreTrainedModel):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLMModel, self).__init__(config)
        self.output_attentions = output_attentions
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len

        self.word_embedding = nn.Embedding(config.n_token, config.d_model)
        self.mask_emb = nn.Parameter(torch.Tensor(1, 1, config.d_model))
        layer = XLMLayer(config, output_attentions=output_attentions,
                                   keep_multihead_output=keep_multihead_output)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """ Gather all multi-head outputs.
            Return: list (layers) of multihead module outputs with gradients
        """
        return [layer.attention.self.multihead_output for layer in self.layer]

    def create_mask(self, qlen, mlen):
        """ create causal attention mask.
            float mask where 1.0 indicate masked, 0.0 indicated not-masked.
             same_length=False:      same_length=True:
             <mlen > <  qlen >       <mlen > <  qlen >
          ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
            [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
       qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
            [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
          v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        """
        attn_mask = torch.ones([qlen, qlen])
        mask_up = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(next(self.parameters()))
        return ret

    def cache_mem(self, curr_out, prev_mem):
        """cache hidden states into memory."""
        if self.mem_len is None or self.mem_len == 0:
            return None
        else:
            if self.reuse_len is not None and self.reuse_len > 0:
                curr_out = curr_out[:self.reuse_len]

            if prev_mem is None:
                new_mem = curr_out[-self.mem_len:]
            else:
                new_mem = torch.cat([prev_mem, curr_out], dim=0)[-self.mem_len:]

        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """create relative positional encoding."""
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / (10000 ** (freq_seq / self.d_model))

        if self.attn_type == 'bi':
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(next(self.parameters()))
        return pos_emb

    def forward(self, inp_k, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                output_all_encoded_layers=True, head_mask=None):
        """
        Args:
            inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            input_mask: [optional] float32 Tensor in shape [bsz, len], the input mask.
                0 for real tokens and 1 for padding.
            attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
                but with 1 for real tokens and 0 for padding.
                Added for easy compatibility with the XLM model (which uses this negative masking).
                You can only uses one among `input_mask` and `attention_mask`
            mems: [optional] a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
                from previous batches. The length of the list equals n_layer.
                If None, no memory is used.
            perm_mask: [optional] float32 Tensor in shape [bsz, len, len].
                If perm_mask[k, i, j] = 0, i attend to j in batch k;
                if perm_mask[k, i, j] = 1, i does not attend to j in batch k.
                If None, each position attends to all the others.
            target_mapping: [optional] float32 Tensor in shape [bsz, num_predict, len].
                If target_mapping[k, i, j] = 1, the i-th predict in batch k is
                on the j-th token.
                Only used during pretraining for partial prediction.
                Set to None during finetuning.
            inp_q: [optional] float32 Tensor in shape [bsz, len].
                1 for tokens with losses and 0 for tokens without losses.
                Only used during pretraining for two-stream attention.
                Set to None during finetuning.

            mem_len: int, the number of tokens to cache.
            reuse_len: int, the number of tokens in the currect batch to be cached
                and reused in the future.
            bi_data: bool, whether to use bidirectional input pipeline.
                Usually set to True during pretraining and False during finetuning.
            clamp_len: int, clamp all relative distances larger than clamp_len.
                -1 means no clamping.
            same_length: bool, whether to use the same attention length for each token.
            summary_type: str, "last", "first", "mean", or "attn". The method
                to pool the input to get a vector representation.
        """
        # the original code for XLM uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        inp_k = inp_k.transpose(0, 1).contiguous()
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None
        inp_q = inp_q.transpose(0, 1).contiguous() if inp_q is not None else None

        qlen, bsz = inp_k.shape[0], inp_k.shape[1]
        mlen = mems[0].shape[0] if mems is not None else 0
        klen = mlen + qlen

        dtype_float = next(self.parameters()).dtype
        device = next(self.parameters()).device

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with XLM). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
            data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        ##### Word embeddings and prepare h & g hidden states
        word_emb_k = self.word_embedding(inp_k)
        output_h = self.dropout(word_emb_k)
        if inp_q is not None:
            if target_mapping is not None:
                word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            else:
                inp_q_ext = inp_q[:, :, None]
                word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        ##### Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
            cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        ##### Head mask if needed (for bertology/pruning)
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [n_layer x num_heads]
        # and head_mask is converted to shape [n_layer x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        new_mems = []
        if mems is None:
            mems = [None] * len(self.layer)

        hidden_states = []
        for i, layer_module in enumerate(self.layer):
            # cache new mems
            new_mems.append(self.cache_mem(output_h, mems[i]))

            output_h, output_g = layer_module(output_h, output_g,
                                              attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
                                              r=pos_emb, seg_mat=seg_mat,
                                              mems=mems[i], target_mapping=target_mapping,
                                              head_mask=head_mask)
            hidden_states.append(output_h)
        output = self.dropout(output_g if output_g is not None else output_h)

        # We transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2).contiguous()
        hidden_states = [hs.permute(1, 0, 2).contiguous() for hs in hidden_states]

        return output, hidden_states, new_mems


class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim

        if params.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=params.n_words,
                cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0

        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction='elementwise_mean')
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x) if get_scores else None

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class XLMLMHeadModel(XLMPreTrainedModel):
    """XLM model ("XLM: Generalized Autoregressive Pretraining for Language Understanding").

    Params:
        `config`: a XLMConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
        token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
        attention_mask: [optional] float32 Tensor in shape [bsz, len], the input mask.
            0 for real tokens and 1 for padding.
        mems: [optional] a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
            from previous batches. The length of the list equals n_layer.
            If None, no memory is used.
        perm_mask: [optional] float32 Tensor in shape [bsz, len, len].
            If perm_mask[k, i, j] = 0, i attend to j in batch k;
            if perm_mask[k, i, j] = 1, i does not attend to j in batch k.
            If None, each position attends to all the others.
        target_mapping: [optional] float32 Tensor in shape [bsz, num_predict, len].
            If target_mapping[k, i, j] = 1, the i-th predict in batch k is
            on the j-th token.
            Only used during pretraining for partial prediction.
            Set to None during finetuning.
        inp_q: [optional] float32 Tensor in shape [bsz, len].
            1 for tokens with losses and 0 for tokens without losses.
            Only used during pretraining for two-stream attention.
            Set to None during finetuning.


    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for XLM-base, 24 for XLM-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, d_model],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, d_model],
        `pooled_output`: a torch.FloatTensor of size [batch_size, d_model] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see XLM's paper).

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
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLMLMHeadModel, self).__init__(config)
        self.output_attentions = output_attentions
        self.attn_type = config.attn_type
        self.same_length = config.same_length

        self.transformer = XLMModel(config, output_attentions=output_attentions,
                                              keep_multihead_output=keep_multihead_output)
        self.lm_loss = nn.Linear(config.d_model, config.n_token, bias=True)

        # Tie weights

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_loss.weight = self.transformer.word_embedding.weight

    def forward(self, inp_k, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                labels=None, output_all_encoded_layers=True, head_mask=None):
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
        output, hidden_states, new_mems = self.transformer(inp_k, token_type_ids, input_mask, attention_mask,
                                            mems, perm_mask, target_mapping, inp_q,
                                            output_all_encoded_layers, head_mask)

        logits = self.lm_loss(output)

        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            labels.view(-1))
            return loss, new_mems

        # if self.output_attentions:
        #     all_attentions, encoded_layers = encoded_layers
        # sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        # if not output_all_encoded_layers:
        #     encoded_layers = encoded_layers[-1]
        # if self.output_attentions:
        return logits, new_mems
        #     return all_attentions, encoded_layers, pooled_output


class XLMSequenceSummary(nn.Module):
    def __init__(self, config, summary_type="last", use_proj=True,
                 output_attentions=False, keep_multihead_output=False):
        super(XLMSequenceSummary, self).__init__()
        self.summary_type = summary_type
        if use_proj:
            self.summary = nn.Linear(config.d_model, config.d_model)
        else:
            self.summary = None
        if summary_type == 'attn':
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """ hidden_states: float Tensor in shape [bsz, seq_len, d_model], the hidden-states of the last layer."""
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif summary_type == 'attn':
            raise NotImplementedError

        output = self.summary(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output


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
    def __init__(self, config, summary_type="last", use_proj=True, num_labels=2,
                 output_attentions=False, keep_multihead_output=False):
        super(XLMForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.summary_type = summary_type
        self.num_labels = num_labels

        self.transformer = XLMModel(config, output_attentions=output_attentions,
                                              keep_multihead_output=keep_multihead_output)

        self.sequence_summary = XLMSequenceSummary(config, summary_type=summary_type,
                                                     use_proj=use_proj, output_attentions=output_attentions,
                                                     keep_multihead_output=keep_multihead_output)
        self.logits_proj = nn.Linear(config.d_model, num_labels)
        self.apply(self.init_weights)

    def forward(self, inp_k, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                labels=None, output_all_encoded_layers=True, head_mask=None):
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
        output, _, new_mems = self.transformer(inp_k, token_type_ids, input_mask, attention_mask,
                                               mems, perm_mask, target_mapping, inp_q,
                                               output_all_encoded_layers, head_mask)

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, new_mems

        # if self.output_attentions:
        #     all_attentions, encoded_layers = encoded_layers
        # sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        # if not output_all_encoded_layers:
        #     encoded_layers = encoded_layers[-1]
        # if self.output_attentions:
        return logits, new_mems
        #     return all_attentions, encoded_layers, pooled_output


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
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLMForQuestionAnswering, self).__init__(config)
        self.output_attentions = output_attentions
        self.transformer = XLMModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, inp_k, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                start_positions=None, end_positions=None,
                output_all_encoded_layers=True, head_mask=None):
        output, _, new_mems = self.transformer(inp_k, token_type_ids, input_mask, attention_mask,
                                            mems, perm_mask, target_mapping, inp_q,
                                            output_all_encoded_layers, head_mask)

        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

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

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        elif self.output_attentions:
            return all_attentions, start_logits, end_logits
        return start_logits, end_logits
