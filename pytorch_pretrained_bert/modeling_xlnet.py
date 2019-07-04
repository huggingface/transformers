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

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from .file_utils import cached_path
from .model_utils import (CONFIG_NAME, WEIGHTS_NAME, PretrainedConfig, PreTrainedModel,
                          SequenceSummary, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits)


logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlnet-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-pytorch_model.bin",
}
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlnet-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-config.json",
}


def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """ A map of modules from TF to PyTorch.
        I use a map to keep the PyTorch model as
        identical to the original PyTorch model as possible.
    """

    tf_to_pt_map = {}

    if hasattr(model, 'transformer'):
        if hasattr(model, 'lm_loss'):
            # We will load also the output bias
            tf_to_pt_map['model/lm_loss/bias'] = model.lm_loss.bias
        if hasattr(model, 'sequence_summary') and 'model/sequnece_summary/summary/kernel' in tf_weights:
            # We will load also the sequence summary
            tf_to_pt_map['model/sequnece_summary/summary/kernel'] = model.sequence_summary.summary.weight
            tf_to_pt_map['model/sequnece_summary/summary/bias'] = model.sequence_summary.summary.bias
        if hasattr(model, 'logits_proj') and config.finetuning_task is not None \
                and 'model/regression_{}/logit/kernel'.format(config.finetuning_task) in tf_weights:
            tf_to_pt_map['model/regression_{}/logit/kernel'.format(config.finetuning_task)] = model.logits_proj.weight
            tf_to_pt_map['model/regression_{}/logit/bias'.format(config.finetuning_task)] = model.logits_proj.bias

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
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        print("Importing {}".format(name))
        if name not in tf_weights:
            print("{} not in tf pre-trained weights, skipping".format(name))
            continue
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if 'kernel' in name and ('ff' in name or 'summary' in name or 'logit' in name):
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


class XLNetConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `XLNetModel`.
    """
    pretrained_config_archive_map = PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=32000,
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
                 same_length=False,

                 finetuning_task=None,
                 num_labels=2,
                 summary_type='last',
                 summary_use_proj=True,
                 summary_activation='tanh',
                 summary_dropout=0.1,
                 start_n_top=5,
                 end_n_top=5,
                 **kwargs):
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
            finetuning_task: name of the glue task on which the model was fine-tuned if any
        """
        super(XLNetConfig, self).__init__(**kwargs)

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
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


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
    def __init__(self, config):
        super(XLNetRelativeAttention, self).__init__()
        self.output_attentions = config.output_attentions

        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.n_head))

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
        # x = x[:, 0:klen, :, :]
        x = torch.index_select(x, 1, torch.arange(klen))

        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):
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

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # attention output
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        if self.output_attentions:
            return attn_vec, attn_prob

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
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)

            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            ##### g-stream
            # query-stream query head
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

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
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)

            if self.output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs

class XLNetFeedForward(nn.Module):
    def __init__(self, config):
        super(XLNetFeedForward, self).__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str) or \
                (sys.version_info[0] == 2 and isinstance(config.ff_activation, unicode)):
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
    def __init__(self, config):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g,
                attn_mask_h, attn_mask_g,
                r, seg_mat, mems=None, target_mapping=None, head_mask=None):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g,
                                r, seg_mat, mems=mems, target_mapping=target_mapping,
                                head_mask=head_mask)
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs


class XLNetPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLNetConfig
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_xlnet
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(XLNetPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, XLNetLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLNetRelativeAttention):
            for param in [module.q, module.k, module.v, module.o, module.r,
                          module.r_r_bias, module.r_s_bias, module.r_w_bias,
                          module.seg_embed]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLNetModel):
                module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)


class XLNetModel(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNetModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.n_token, config.d_model)
        self.mask_emb = nn.Parameter(torch.Tensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        self.apply(self.init_weights)

    def _prune_heads(self, heads_to_prune):
        logger.info("Head pruning is not implemented for XLNet")
        pass

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
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

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

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None, head_mask=None):
        """
        Args:
            input_ids: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            input_mask: [optional] float32 Tensor in shape [bsz, len], the input mask.
                0 for real tokens and 1 for padding.
            attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
                but with 1 for real tokens and 0 for padding.
                Added for easy compatibility with the BERT model (which uses this negative masking).
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
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        input_ids = input_ids.transpose(0, 1).contiguous()
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None
        inp_q = inp_q.transpose(0, 1).contiguous() if inp_q is not None else None

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
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
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
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
        word_emb_k = self.word_embedding(input_ids)
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

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []
        for i, layer_module in enumerate(self.layer):
            # cache new mems
            new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(output_h, output_g, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
                                   r=pos_emb, seg_mat=seg_mat, mems=mems[i], target_mapping=target_mapping,
                                   head_mask=head_mask[i])
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = (output.permute(1, 0, 2).contiguous(), new_mems)
        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, new_mems, (hidden_states), (attentions)


class XLNetLMHeadModel(XLNetPreTrainedModel):
    """XLNet model ("XLNet: Generalized Autoregressive Pretraining for Language Understanding").

    Params:
        `config`: a XLNetConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        input_ids: int32 Tensor in shape [bsz, len], the input token IDs.
        token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
        input_mask: [optional] float32 Tensor in shape [bsz, len], the input mask.
            0 for real tokens and 1 for padding.
        attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
            but with 1 for real tokens and 0 for padding.
            Added for easy compatibility with the BERT model (which uses this negative masking).
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
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.XLNetConfig(vocab_size_or_config_json_file=32000, d_model=768,
        n_layer=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.XLNetModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(XLNetLMHeadModel, self).__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.torchscript = config.torchscript

        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.n_token, bias=True)

        # Tie weights

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the embeddings
        """
        if self.torchscript:
            self.lm_loss.weight = nn.Parameter(self.transformer.word_embedding.weight.clone())
        else:
            self.lm_loss.weight = self.transformer.word_embedding.weight

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                labels=None, head_mask=None):
        """
        Args:
            input_ids: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            input_mask: float32 Tensor in shape [bsz, len], the input mask.
                0 for real tokens and 1 for padding.
            attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
                but with 1 for real tokens and 0 for padding.
                Added for easy compatibility with the BERT model (which uses this negative masking).
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

            summary_type: str, "last", "first", "mean", or "attn". The method
                to pool the input to get a vector representation.
        """
        transformer_outputs = self.transformer(input_ids, token_type_ids, input_mask, attention_mask,
                                               mems, perm_mask, target_mapping, inp_q, head_mask)

        logits = self.lm_loss(transformer_outputs[0])

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, (mems), (hidden states), (attentions)


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
        input_ids: int32 Tensor in shape [bsz, len], the input token IDs.
        token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
        input_mask: float32 Tensor in shape [bsz, len], the input mask.
            0 for real tokens and 1 for padding.
        attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
            but with 1 for real tokens and 0 for padding.
            Added for easy compatibility with the BERT model (which uses this negative masking).
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

    config = modeling.XLNetConfig(vocab_size_or_config_json_file=32000, d_model=768,
        n_layer=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.XLNetModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(XLNetForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                labels=None, head_mask=None):
        """
        Args:
            input_ids: int32 Tensor in shape [bsz, len], the input token IDs.
            token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            input_mask: float32 Tensor in shape [bsz, len], the input mask.
                0 for real tokens and 1 for padding.
            attention_mask: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
                but with 1 for real tokens and 0 for padding.
                Added for easy compatibility with the BERT model (which uses this negative masking).
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
        transformer_outputs = self.transformer(input_ids, token_type_ids, input_mask, attention_mask,
                                               mems, perm_mask, target_mapping, inp_q, head_mask)
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, (mems), (hidden states), (attentions)


class XLNetForQuestionAnswering(XLNetPreTrainedModel):
    """ XLNet model for Question Answering (span extraction).
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
            `run_bert_extract_features.py`, `run_bert_classifier.py` and `run_bert_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see XLNet paper for more details).
        `attention_mask`: [optional] float32 Tensor, SAME FUNCTION as `input_mask`
            but with 1 for real tokens and 0 for padding.
            Added for easy compatibility with the BERT model (which uses this negative masking).
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

    config = XLNetConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = XLNetForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(XLNetForQuestionAnswering, self).__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                start_positions=None, end_positions=None, cls_index=None, is_impossible=None, p_mask=None,
                head_mask=None):
        transformer_outputs = self.transformer(input_ids, token_type_ids, input_mask, attention_mask,
                                               mems, perm_mask, target_mapping, inp_q, head_mask)
        hidden_states = transformer_outputs[0]
        start_logits = self.start_logits(hidden_states, p_mask)

        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is
                # comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5
                outputs = (total_loss, start_logits, end_logits, cls_logits) + outputs
            else:
                outputs = (total_loss, start_logits, end_logits) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1) # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1) # shape (bsz, start_n_top)
            start_top_index = start_top_index.unsqueeze(-1).expand(-1, -1, hsz) # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index) # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1) # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states) # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1) # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1) # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits, mems, (hidden states), (attentions)
        # or (if labels are provided) total_loss, start_logits, end_logits, (cls_logits), mems, (hidden states), (attentions)
        return outputs
