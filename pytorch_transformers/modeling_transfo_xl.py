# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" PyTorch Transformer XL model.
    Adapted from https://github.com/kimiyoung/transformer-xl.
    In particular https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import math
import logging
import collections
import sys
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

from .modeling_bert import BertLayerNorm as LayerNorm
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax, sample_logits
from .modeling_utils import (PretrainedConfig, PreTrainedModel, add_start_docstrings)

logger = logging.getLogger(__name__)

TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'transfo-xl-wt103': "https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-pytorch_model.bin",
}
TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'transfo-xl-wt103': "https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json",
}

def build_tf_to_pytorch_map(model, config):
    """ A map of modules from TF to PyTorch.
        This time I use a map to keep the PyTorch model as identical to the original PyTorch model as possible.
    """
    tf_to_pt_map = {}

    if hasattr(model, 'transformer'):
        # We are loading in a TransfoXLLMHeadModel => we will load also the Adaptive Softmax
        tf_to_pt_map.update({
            "transformer/adaptive_softmax/cutoff_0/cluster_W": model.crit.cluster_weight,
            "transformer/adaptive_softmax/cutoff_0/cluster_b": model.crit.cluster_bias})
        for i, (out_l, proj_l, tie_proj) in enumerate(zip(
                                model.crit.out_layers,
                                model.crit.out_projs,
                                config.tie_projs)):
            layer_str = "transformer/adaptive_softmax/cutoff_%d/" % i
            if config.tie_weight:
                tf_to_pt_map.update({
                    layer_str + 'b': out_l.bias})
            else:
                raise NotImplementedError
                # I don't think this is implemented in the TF code
                tf_to_pt_map.update({
                    layer_str + 'lookup_table': out_l.weight,
                    layer_str + 'b': out_l.bias})
            if not tie_proj:
                tf_to_pt_map.update({
                    layer_str + 'proj': proj_l
                    })
        # Now load the rest of the transformer
        model = model.transformer

    # Embeddings
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers, model.word_emb.emb_projs)):
        layer_str = "transformer/adaptive_embed/cutoff_%d/" % i
        tf_to_pt_map.update({
            layer_str + 'lookup_table': embed_l.weight,
            layer_str + 'proj_W': proj_l
            })

    # Transformer blocks
    for i, b in enumerate(model.layers):
        layer_str = "transformer/layer_%d/" % i
        tf_to_pt_map.update({
            layer_str + "rel_attn/LayerNorm/gamma": b.dec_attn.layer_norm.weight,
            layer_str + "rel_attn/LayerNorm/beta": b.dec_attn.layer_norm.bias,
            layer_str + "rel_attn/o/kernel": b.dec_attn.o_net.weight,
            layer_str + "rel_attn/qkv/kernel": b.dec_attn.qkv_net.weight,
            layer_str + "rel_attn/r/kernel": b.dec_attn.r_net.weight,
            layer_str + "ff/LayerNorm/gamma": b.pos_ff.layer_norm.weight,
            layer_str + "ff/LayerNorm/beta": b.pos_ff.layer_norm.bias,
            layer_str + "ff/layer_1/kernel": b.pos_ff.CoreNet[0].weight,
            layer_str + "ff/layer_1/bias": b.pos_ff.CoreNet[0].bias,
            layer_str + "ff/layer_2/kernel": b.pos_ff.CoreNet[3].weight,
            layer_str + "ff/layer_2/bias": b.pos_ff.CoreNet[3].bias,
        })

    # Relative positioning biases
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map.update({
        'transformer/r_r_bias': r_r_list,
        'transformer/r_w_bias': r_w_list})
    return tf_to_pt_map

def load_tf_weights_in_transfo_xl(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_to_pytorch_map(model, config)

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    for name, pointer in tf_to_pt_map.items():
        assert name in tf_weights
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if 'kernel' in name or 'proj' in name:
            array = np.transpose(array)
        if ('r_r_bias' in name or 'r_w_bias' in name) and len(pointer) > 1:
            # Here we will split the TF weigths
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info("Initialize PyTorch weight {} for layer {}".format(name, i))
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)

    logger.info("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model


class TransfoXLConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `TransfoXLModel`.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `TransfoXLModel` or a configuration json file.
            cutoffs: cutoffs for the adaptive softmax
            d_model: Dimensionality of the model's hidden states.
            d_embed: Dimensionality of the embeddings
            d_head: Dimensionality of the model's heads.
            div_val: divident value for adapative input and softmax
            pre_lnorm: apply LayerNorm to the input instead of the output
            d_inner: Inner dimension in FF
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            tgt_len: number of tokens to predict
            ext_len: length of the extended context
            mem_len: length of the retained previous heads
            same_length: use the same attn length for all tokens
            proj_share_all_but_first: True to share all but first projs, False not to share.
            attn_type: attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
            clamp_len: use the same pos embeddings after clamp_len
            sample_softmax: number of samples in sampled softmax
            adaptive: use adaptive softmax
            tie_weight: tie the word embedding and softmax weights
            dropout: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            dropatt: The dropout ratio for the attention probabilities.
            untie_r: untie relative position biases
            embd_pdrop: The dropout ratio for the embeddings.
            init: parameter initializer to use
            init_range: parameters initialized by U(-init_range, init_range).
            proj_init_std: parameters initialized by N(0, init_std)
            init_std: parameters initialized by N(0, init_std)
    """
    pretrained_config_archive_map = TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=267735,
                 cutoffs=[20000, 40000, 200000],
                 d_model=1024,
                 d_embed=1024,
                 n_head=16,
                 d_head=64,
                 d_inner=4096,
                 div_val=4,
                 pre_lnorm=False,
                 n_layer=18,
                 tgt_len=128,
                 ext_len=0,
                 mem_len=1600,
                 clamp_len=1000,
                 same_length=True,
                 proj_share_all_but_first=True,
                 attn_type=0,
                 sample_softmax=-1,
                 adaptive=True,
                 tie_weight=True,
                 dropout=0.1,
                 dropatt=0.0,
                 untie_r=True,
                 init="normal",
                 init_range=0.01,
                 proj_init_std=0.01,
                 init_std=0.02,
                 **kwargs):
        """Constructs TransfoXLConfig.
        """
        super(TransfoXLConfig, self).__init__(**kwargs)

        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_token = vocab_size_or_config_json_file
            self.cutoffs = []
            self.cutoffs.extend(cutoffs)
            self.tie_weight = tie_weight
            if proj_share_all_but_first:
                self.tie_projs = [False] + [True] * len(self.cutoffs)
            else:
                self.tie_projs = [False] + [False] * len(self.cutoffs)
            self.d_model = d_model
            self.d_embed = d_embed
            self.d_head = d_head
            self.d_inner = d_inner
            self.div_val = div_val
            self.pre_lnorm = pre_lnorm
            self.n_layer = n_layer
            self.n_head = n_head
            self.tgt_len = tgt_len
            self.ext_len = ext_len
            self.mem_len = mem_len
            self.same_length = same_length
            self.attn_type = attn_type
            self.clamp_len = clamp_len
            self.sample_softmax = sample_softmax
            self.adaptive = adaptive
            self.dropout = dropout
            self.dropatt = dropatt
            self.untie_r = untie_r
            self.init = init
            self.init_range = init_range
            self.proj_init_std = proj_init_std
            self.init_std = init_std
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")

    @property
    def max_position_embeddings(self):
        return self.tgt_len + self.ext_len + self.mem_len

    @property
    def vocab_size(self):
        return self.n_token

    @vocab_size.setter
    def vocab_size(self, value):
        self.n_token = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]



class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output



class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False, r_r_bias=None, r_w_bias=None, output_attentions=False):
        super(MultiHeadAttn, self).__init__()

        self.output_attentions = output_attentions
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None: # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def forward(self, h, attn_mask=None, mems=None, head_mask=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            outputs = [h + attn_out]
        else:
            ##### residual connection + layer normalization
            outputs = [self.layer_norm(h + attn_out)]

        if self.output_attentions:
            outputs.append(attn_prob)

        return outputs

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                 r_r_bias=None, r_w_bias=None, output_attentions=False):
        super(RelMultiHeadAttn, self).__init__()

        self.output_attentions = output_attentions
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None: # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + self.r_w_bias                                    # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -1e30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -1e30).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            outputs = [w + attn_out]
        else:
            ##### residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]

        if self.output_attentions:
            outputs.append(attn_prob)

        return outputs

class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None, head_mask=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            outputs = [w + attn_out]
        else:
            ##### residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]

        if self.output_attentions:
            outputs.append(attn_prob)

        return outputs



class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None, head_mask=None):

        attn_outputs = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None, head_mask=None):

        attn_outputs = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None):

        attn_outputs = self.dec_attn(dec_inp, r,
                               attn_mask=dec_attn_mask,
                               mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs



class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)

        embed.mul_(self.emb_scale)

        return embed


class TransfoXLPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = TransfoXLConfig
    pretrained_model_archive_map = TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_transfo_xl
    base_model_prefix = "transformer"

    def _init_weight(self, weight):
        if self.config.init == 'uniform':
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == 'normal':
            nn.init.normal_(weight, 0.0, self.config.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        """ Initialize the weights.
        """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self._init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, 'r_emb'):
                self._init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self._init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self._init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self._init_bias(m.r_bias)

    def set_num_special_tokens(self, num_special_tokens):
        pass


TRANSFO_XL_START_DOCSTRING = r"""    The Transformer-XL model was proposed in
    `Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context`_
    by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
    It's a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can reuse
    previously computed hidden-states to attend to longer context (memory).
    This model also uses adaptive softmax inputs and outputs (tied).

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context`:
        https://arxiv.org/abs/1901.02860

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.TransfoXLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            Transformer-XL is a model with relative position embeddings so you can either pad the inputs on
            the right or on the left.
            Indices can be obtained using :class:`pytorch_transformers.TransfoXLTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **mems**: (`optional`)
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` output below). Can be used to speed up sequential decoding and attend to longer context.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare Bert Model transformer outputing raw hidden-states without any specific head on top.",
                      TRANSFO_XL_START_DOCSTRING, TRANSFO_XL_INPUTS_DOCSTRING)
class TransfoXLModel(TransfoXLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **mems**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` input above). Can be used to speed up sequential decoding and attend to longer context.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states, mems = outputs[:2]

    """
    def __init__(self, config):
        super(TransfoXLModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.n_token = config.n_token

        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head

        self.word_emb = AdaptiveEmbedding(config.n_token, config.d_embed, config.d_model, config.cutoffs, 
                                          div_val=config.div_val)

        self.drop = nn.Dropout(config.dropout)

        self.n_layer = config.n_layer

        self.tgt_len = config.tgt_len
        self.mem_len = config.mem_len
        self.ext_len = config.ext_len
        self.max_klen = config.tgt_len + config.ext_len + config.mem_len

        self.attn_type = config.attn_type

        if not config.untie_r:
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.layers = nn.ModuleList()
        if config.attn_type == 0: # the default attention
            for i in range(config.n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout,
                        tgt_len=config.tgt_len, ext_len=config.ext_len, mem_len=config.mem_len,
                        dropatt=config.dropatt, pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if config.untie_r else self.r_w_bias,
                        r_r_bias=None if config.untie_r else self.r_r_bias,
                        output_attentions=self.output_attentions)
                )
        elif config.attn_type == 1: # learnable embeddings
            for i in range(config.n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout,
                        tgt_len=config.tgt_len, ext_len=config.ext_len, mem_len=config.mem_len,
                        dropatt=config.dropatt, pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if config.untie_r else self.r_w_bias,
                        r_r_bias=None if config.untie_r else self.r_r_bias,
                        output_attentions=self.output_attentions)
                )
        elif config.attn_type in [2, 3]: # absolute embeddings
            for i in range(config.n_layer):
                self.layers.append(
                    DecoderLayer(
                        config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout,
                        dropatt=config.dropatt, pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if config.untie_r else self.r_w_bias,
                        r_r_bias=None if config.untie_r else self.r_r_bias,
                        output_attentions=self.output_attentions)
                )

        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.FloatTensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.FloatTensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.FloatTensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        return self.word_emb

    def backward_compatible(self):
        self.sample_softmax = -1

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def _prune_heads(self, heads):
        logger.info("Head pruning is not implemented for Transformer-XL model")
        pass

    def init_mems(self, data):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(self.mem_len, data.size(1), self.config.d_model,
                                    dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, head_mask=None):
        qlen, bsz = dec_inp.size()

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]

        hids = []
        attentions = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(core_out, pos_emb, dec_attn_mask=dec_attn_mask,
                                      mems=mems_i, head_mask=head_mask[i])
                core_out = layer_outputs[0]
                if self.output_attentions:
                    attentions.append(layer_outputs[1])
        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(core_out, r_emb, self.r_w_bias[i],
                                      r_bias, dec_attn_mask=dec_attn_mask,
                                      mems=mems_i, head_mask=head_mask[i])
                core_out = layer_outputs[0]
                if self.output_attentions:
                    attentions.append(layer_outputs[1])
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                layer_outputs = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i, head_mask=head_mask[i])
                core_out = layer_outputs[0]
                if self.output_attentions:
                    attentions.append(layer_outputs[1])
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                layer_outputs = layer(core_out, dec_attn_mask=dec_attn_mask,
                                      mems=mems_i, head_mask=head_mask[i])
                core_out = layer_outputs[0]
                if self.output_attentions:
                    attentions.append(layer_outputs[1])

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        # We transpose back here to shape [bsz, len, hidden_dim]
        outputs = [core_out.transpose(0, 1).contiguous(), new_mems]
        if self.output_hidden_states:
            # Add last layer and transpose to library standard shape [bsz, len, hidden_dim]
            hids.append(core_out)
            hids = list(t.transpose(0, 1).contiguous() for t in hids)
            outputs.append(hids)
        if self.output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = list(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs.append(attentions)
        return outputs  # last hidden state, new_mems, (all hidden states), (all attentions)

    def forward(self, input_ids, mems=None, head_mask=None):
        # the original code for Transformer-XL used shapes [len, bsz] but we want a unified interface in the library
        # so we transpose here from shape [bsz, len] to shape [len, bsz]
        input_ids = input_ids.transpose(0, 1).contiguous()

        if mems is None:
            mems = self.init_mems(input_ids)
        outputs = self._forward(input_ids, mems=mems, head_mask=head_mask)

        return outputs  # last hidden state, new_mems, (all hidden states), (all attentions)


@add_start_docstrings("""The Transformer-XL Model with a language modeling head on top
    (adaptive softmax with weights tied to the adaptive input embeddings)""",
    TRANSFO_XL_START_DOCSTRING, TRANSFO_XL_INPUTS_DOCSTRING)
class TransfoXLLMHeadModel(TransfoXLPreTrainedModel):
    r"""
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``None`` if ``lm_labels`` is provided else ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            We don't output them when the loss is computed to speedup adaptive softmax decoding.
        **mems**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` input above). Can be used to speed up sequential decoding and attend to longer context.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, mems = outputs[:2]

    """
    def __init__(self, config):
        super(TransfoXLLMHeadModel, self).__init__(config)
        self.transformer = TransfoXLModel(config)
        self.sample_softmax = config.sample_softmax
        # use sampled softmax
        if config.sample_softmax > 0:
            self.out_layer = nn.Linear(config.d_model, config.n_token)
            self.sampler = LogUniformSampler(config.n_token, config.sample_softmax)
        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(config.n_token, config.d_embed, config.d_model, 
                                                    config.cutoffs, div_val=config.div_val)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """
        # sampled softmax
        if self.sample_softmax > 0:
            if self.config.tie_weight:
                self.out_layer.weight = self.transformer.word_emb.weight
        # adaptive softmax (including standard softmax)
        else:
            if self.config.tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self._tie_or_clone_weights(self.crit.out_layers[i],
                                               self.transformer.word_emb.emb_layers[i])
            if self.config.tie_projs:
                for i, tie_proj in enumerate(self.config.tie_projs):
                    if tie_proj and self.config.div_val == 1 and self.config.d_model != self.config.d_embed:
                        if self.config.torchscript:
                            self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[0].clone())
                        else:
                            self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[0]
                    elif tie_proj and self.config.div_val != 1:
                        if self.config.torchscript:
                            self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[i].clone())
                        else:
                            self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[i]

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.transformer.reset_length(tgt_len, ext_len, mem_len)

    def init_mems(self, data):
        return self.transformer.init_mems(data)

    def forward(self, input_ids, labels=None, mems=None, head_mask=None):
        bsz = input_ids.size(0)
        tgt_len = input_ids.size(1)

        transformer_outputs = self.transformer(input_ids, mems=mems, head_mask=head_mask)

        last_hidden = transformer_outputs[0]
        pred_hid = last_hidden[:, -tgt_len:]
        outputs = transformer_outputs[1:]
        if self.sample_softmax > 0 and self.training:
            assert self.config.tie_weight
            logit = sample_logits(self.transformer.word_emb, self.out_layer.bias, labels, pred_hid, self.sampler)
            softmax_output = -F.log_softmax(logit, -1)[:, :, 0]
            outputs = [softmax_output] + outputs
            if labels is not None:
                # TODO: This is not implemented
                raise NotImplementedError
        else:
            softmax_output = self.crit(pred_hid.view(-1, pred_hid.size(-1)), labels)
            if labels is None:
                softmax_output = softmax_output.view(bsz, tgt_len, -1)
                outputs = [softmax_output] + outputs
            else:
                softmax_output = softmax_output.view(bsz, tgt_len)
                outputs = [softmax_output, None] + outputs

        return outputs  # (loss), logits or None if labels is not None (speed up adaptive softmax), new_mems, (all hidden states), (all attentions)
