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
""" PyTorch XLNet model.
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

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from .file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlnet-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-pytorch_model.bin",
}
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlnet-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-config.json",
}
XLNET_CONFIG_NAME = 'xlnet_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'


def build_tf_xlnet_to_pytorch_map(model, config):
    """ A map of modules from TF to PyTorch.
        I use a map to keep the PyTorch model as
        identical to the original PyTorch model as possible.
    """

    tf_to_pt_map = {}

    if hasattr(model, 'transformer'):
        # We are loading pre-trained weights in a XLNetLMHeadModel => we will load also the output bias
        tf_to_pt_map['model/lm_loss/bias'] = model.lm_loss.bias
        # Now load the rest of the transformer
        model = model.transformer

    # Embeddings and output
    tf_to_pt_map.update({'model/transformer/word_embedding/lookup_table': model.word_embedding.weight,
                    'model/transformer/mask_emb/mask_emb': model.mask_emb})

    # Transformer blocks
    for i, b in enumerate(model.layer):
        layer_str = "model/transformer/layer_%d/" % i
        tf_to_pt_map.update({
            layer_str + "rel_attn/LayerNorm/gamma": b.rel_attn.layer_norm.weight,
            layer_str + "rel_attn/LayerNorm/beta": b.rel_attn.layer_norm.bias,
            layer_str + "rel_attn/o/kernel": b.rel_attn.o,
            layer_str + "rel_attn/q/kernel": b.rel_attn.q,
            layer_str + "rel_attn/k/kernel": b.rel_attn.k,
            layer_str + "rel_attn/r/kernel": b.rel_attn.r,
            layer_str + "rel_attn/v/kernel": b.rel_attn.v,
            layer_str + "ff/LayerNorm/gamma": b.ff.layer_norm.weight,
            layer_str + "ff/LayerNorm/beta": b.ff.layer_norm.bias,
            layer_str + "ff/layer_1/kernel": b.ff.layer_1.weight,
            layer_str + "ff/layer_1/bias": b.ff.layer_1.bias,
            layer_str + "ff/layer_2/kernel": b.ff.layer_2.weight,
            layer_str + "ff/layer_2/bias": b.ff.layer_2.bias,
        })

    # Relative positioning biases
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update({
        'model/transformer/r_r_bias': r_r_list,
        'model/transformer/r_w_bias': r_w_list,
        'model/transformer/r_s_bias': r_s_list,
        'model/transformer/seg_embed': seg_embed_list})
    return tf_to_pt_map

def load_tf_weights_in_xlnet(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config)

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    for name, pointer in tf_to_pt_map.items():
        print("Importing {}".format(name))
        assert name in tf_weights
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if 'kernel' in name and 'ff' in name:
            print("Transposing")
            array = np.transpose(array)
        if isinstance(pointer, list):
            # Here we will split the TF weigths
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                print("Initialize PyTorch weight {} for layer {}".format(name, i))
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            print("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)

    print("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model


def gelu(x):
    """ Implementation of the gelu activation function.
        XLNet is using OpenAI GPT's gelu (not exactly the same as BERT)
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class XLNetBaseConfig(object):
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `XLNetBaseConfig` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `XLNetBaseConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def update(self, other):
        dict_b = other.to_dict()
        for key, value in dict_b.items():
            self.__dict__[key] = value

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class XLNetConfig(XLNetBaseConfig):
    """Configuration class to store the configuration of a `XLNetModel`.
    """
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
        """Constructs XLNetConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `XLNetModel`.
            d_model: Size of the encoder layers and the pooler layer.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            d_inner: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            ff_activation: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            untie_r: untie relative position biases
            attn_type: 'bi' for XLNet, 'uni' for Transformer-XL

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


class XLNetRunConfig(XLNetBaseConfig):
    """XLNetRunConfig contains hyperparameters that could be different
    between pretraining and finetuning.
    These hyperparameters can also be changed from run to run.
    We store them separately from XLNetConfig for flexibility.
    """
    def __init__(self, 
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
        """
        Args:
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

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as XLNetLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class XLNetLayerNorm(nn.Module):
        def __init__(self, d_model, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(XLNetLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class XLNetRelativeAttention(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLNetRelativeAttention, self).__init__()
        self.output_attentions = output_attentions
        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.n_head))
        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output
        self.multihead_output = None

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.Tensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.Tensor(2, self.n_head, self.d_head))

        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = x[:, 0:klen, :, :]

        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=ac.shape[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(self, h, g,
                      attn_mask_h, attn_mask_g,
                      r, seg_mat,
                      mems=None, target_mapping=None, head_mask=None):
        if g is not None:
            ###### Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content-based key head
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)

            # content-based value head
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)

            # position-based key head
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)

            ##### h-stream
            # content-stream query head
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h)

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            ##### g-stream
            # query-stream query head
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g)
                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g)

            # post processing
            output_g = self.post_attention(g, attn_vec_g)
        else:
            ###### Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content heads
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)

            # positional heads
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h)

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None


        # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)
        # if self.keep_multihead_output:
        #     self.multihead_output = context_layer
        #     self.multihead_output.retain_grad()

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)

        # if self.output_attentions:
        #     attentions, self_output = self_output
        # if self.output_attentions:
        #     return attentions, attention_output
        return output_h, output_g

class XLNetFeedForward(nn.Module):
    def __init__(self, config):
        super(XLNetFeedForward, self).__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str) or (sys.version_info[0] == 2 and isinstance(config.ff_activation, unicode)):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output

class XLNetLayer(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLNetLayer, self).__init__()
        self.output_attentions = output_attentions
        self.rel_attn = XLNetRelativeAttention(config, output_attentions=output_attentions,
                                               keep_multihead_output=keep_multihead_output)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g,
                attn_mask_h, attn_mask_g,
                r, seg_mat,
                mems=None, target_mapping=None, head_mask=None):
        output_h, output_g = self.rel_attn(output_h, output_g,
                                           attn_mask_h, attn_mask_g,
                                           r, seg_mat,
                                           mems=mems, target_mapping=target_mapping, head_mask=head_mask)
        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        # if self.output_attentions:
        #     return attentions, layer_output
        return output_h, output_g

class XLNetPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(XLNetPreTrainedModel, self).__init__()
        if not isinstance(config, XLNetBaseConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `XLNetBaseConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLNetLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a XLNetPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `xlnet-large-cased`
                - a path or url to a pretrained model archive containing:
                    . `config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a XLNetForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `xlnet_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific XLNet class
                (ex: num_labels for XLNetForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME)
                config_file = os.path.join(pretrained_model_name_or_path, XLNET_CONFIG_NAME)
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                        archive_file))
            return None
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_CONFIG_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                        config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
                        config_file))
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info("loading weights file {}".format(archive_file))
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = XLNetConfig.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config {}".format(config))

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return load_tf_weights_in_xlnet(model, config, resolved_archive_file)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'transformer') and any(s.startswith('transformer') for s in state_dict.keys()):
            start_prefix = 'transformer.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        if isinstance(model, XLNetLMHeadModel):
            model.tie_weights()  # make sure word embedding weights are still tied
        return model


class XLNetModel(XLNetPreTrainedModel):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLNetModel, self).__init__(config)
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
        layer = XLNetLayer(config, output_attentions=output_attentions,
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

    def forward(self, inp_k, token_type_ids=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                output_all_encoded_layers=True, head_mask=None):
        """
        Args:
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
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        inp_k = inp_k.transpose(0, 1).contiguous()
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
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
        if attention_mask is not None and perm_mask is not None:
            data_mask = attention_mask[None] + perm_mask
        elif attention_mask is not None and perm_mask is None:
            data_mask = attention_mask[None]
        elif attention_mask is None and perm_mask is not None:
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


class XLNetLMHeadModel(XLNetPreTrainedModel):
    """XLNet model ("XLNet: Generalized Autoregressive Pretraining for Language Understanding").

    Params:
        `config`: a XLNetConfig class instance with the configuration to build a new model
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
                of each attention block (i.e. 12 full sequences for XLNet-base, 24 for XLNet-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, d_model],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, d_model],
        `pooled_output`: a torch.FloatTensor of size [batch_size, d_model] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see XLNet's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.XLNetConfig(vocab_size_or_config_json_file=32000, d_model=768,
        n_layer=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.XLNetModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, attention_mask)
    ```
    """
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLNetLMHeadModel, self).__init__(config)
        self.output_attentions = output_attentions
        self.attn_type = config.attn_type
        self.same_length = config.same_length

        self.transformer = XLNetModel(config, output_attentions=output_attentions,
                                              keep_multihead_output=keep_multihead_output)
        self.lm_loss = nn.Linear(config.d_model, config.n_token, bias=True)

        # Tie weights

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_loss.weight = self.transformer.word_embedding.weight

    def forward(self, inp_k, token_type_ids=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                target=None, output_all_encoded_layers=True, head_mask=None):
        """
        Args:
            inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            attention_mask: float32 Tensor in shape [bsz, len], the input mask.
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
        output, hidden_states, new_mems = self.transformer(inp_k, token_type_ids, attention_mask,
                                            mems, perm_mask, target_mapping, inp_q,
                                            output_all_encoded_layers, head_mask)

        logits = self.lm_loss(output)

        if target is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            target.view(-1))
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

class XLNetSequenceSummary(nn.Module):
    def __init__(self, config, summary_type="last", use_proj=True,
                 output_attentions=False, keep_multihead_output=False):
        super(XLNetSequenceSummary, self).__init__()
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


class XLNetForSequenceClassification(XLNetPreTrainedModel):
    """XLNet model ("XLNet: Generalized Autoregressive Pretraining for Language Understanding").

    Params:
        `config`: a XLNetConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
        `summary_type`: str, "last", "first", "mean", or "attn". The method
            to pool the input to get a vector representation. Default: last

    Inputs:
        inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
        token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
        attention_mask: float32 Tensor in shape [bsz, len], the input mask.
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
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


    Outputs: Tuple of (logits or loss, mems)
        `logits or loss`:
            if target is None:
                Token logits with shape [batch_size, sequence_length] 
            else:
                CrossEntropy loss with the targets
        `new_mems`: list (num layers) of updated mem states at the entry of each layer
            each mem state is a torch.FloatTensor of size [self.config.mem_len, batch_size, self.config.d_model]
            Note that the first two dimensions are transposed in `mems` with regards to `input_ids` and `target`

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.XLNetConfig(vocab_size_or_config_json_file=32000, d_model=768,
        n_layer=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.XLNetModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, attention_mask)
    ```
    """
    def __init__(self, config, summary_type="last", use_proj=True, num_labels=2,
                 is_regression=False, output_attentions=False, keep_multihead_output=False):
        super(XLNetForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.summary_type = summary_type
        self.is_regression = is_regression

        self.transformer = XLNetModel(config, output_attentions=output_attentions,
                                              keep_multihead_output=keep_multihead_output)

        self.sequence_summary = XLNetSequenceSummary(config, summary_type=summary_type,
                                                     use_proj=use_proj, output_attentions=output_attentions,
                                                     keep_multihead_output=keep_multihead_output)
        self.loss_proj = nn.Linear(config.d_model, num_labels if not is_regression else 1)
        self.apply(self.init_weights)

    def forward(self, inp_k, token_type_ids=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                target=None, output_all_encoded_layers=True, head_mask=None):
        """
        Args:
            inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            attention_mask: float32 Tensor in shape [bsz, len], the input mask.
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
        """
        output, _, new_mems = self.transformer(inp_k, token_type_ids, attention_mask,
                                            mems, perm_mask, target_mapping, inp_q,
                                            output_all_encoded_layers, head_mask)

        output = self.sequence_summary(output)
        logits = self.loss_proj(output)

        if target is not None:
            if self.is_regression:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), target.view(-1))
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))
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

class XLNetForQuestionAnswering(XLNetPreTrainedModel):
    """XLNet model for Question Answering (span extraction).
    This module is composed of the XLNet model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a XLNetConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see XLNet paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
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
    attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = XLNetConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = XLNetForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
    ```
    """
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(XLNetForQuestionAnswering, self).__init__(config)
        self.output_attentions = output_attentions
        self.transformer = XLNetModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, inp_k, token_type_ids=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                start_positions=None, end_positions=None,
                output_all_encoded_layers=True, head_mask=None):
        output, _, new_mems = self.transformer(inp_k, token_type_ids, attention_mask,
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
