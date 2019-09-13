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
""" Utilities for PyTorch Transformer XL model.
        Directly adapted from https://github.com/kimiyoung/transformer-xl.
"""

from collections import defaultdict

import numpy as np

import tensorflow as tf

from .modeling_tf_utils import shape_list

class TFAdaptiveSoftmaxMask(tf.keras.layers.Layer):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False, **kwargs):
        super(TFAdaptiveSoftmaxMask, self).__init__(**kwargs)

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        self.keep_order = keep_order

        self.out_layers = []
        self.out_projs = []

    def build(self, input_shape):
        if self.n_clusters > 0:
            self.cluster_weight = self.add_weight(shape=(self.n_clusters, self.d_embed),
                                                  initializer='zeros',
                                                  trainable=True,
                                                  name='cluster_weight')
            self.cluster_bias = self.add_weight(shape=(self.n_clusters,),
                                                initializer='zeros',
                                                trainable=True,
                                                name='cluster_bias')

        if self.div_val == 1:
            for i in range(len(self.cutoffs)):
                if self.d_proj != self.d_embed:
                    weight = self.add_weight(shape=(self.d_embed, self.d_proj),
                                             initializer='zeros',
                                             trainable=True,
                                             name='out_projs_._{}'.format(i))
                    self.out_projs.append(weight)
                else:
                    self.out_projs.append(None)
                weight = self.add_weight(shape=(self.n_token, self.d_embed,),
                                         initializer='zeros',
                                         trainable=True,
                                         name='out_layers_._{}_._weight'.format(i))
                bias = self.add_weight(shape=(self.n_token,),
                                         initializer='zeros',
                                         trainable=True,
                                         name='out_layers_._{}_._bias'.format(i))
                self.out_layers.append((weight, bias))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = self.d_embed // (self.div_val ** i)

                weight = self.add_weight(shape=(d_emb_i, self.d_proj),
                                         initializer='zeros',
                                         trainable=True,
                                         name='out_projs_._{}'.format(i))
                self.out_projs.append(weight)
                weight = self.add_weight(shape=(r_idx-l_idx, d_emb_i,),
                                         initializer='zeros',
                                         trainable=True,
                                         name='out_layers_._{}_._weight'.format(i))
                bias = self.add_weight(shape=(r_idx-l_idx,),
                                         initializer='zeros',
                                         trainable=True,
                                         name='out_layers_._{}_._bias'.format(i))
                self.out_layers.append((weight, bias))
        super(TFAdaptiveSoftmaxMask, self).build(input_shape)

    @staticmethod
    def _logit(x, W, b, proj=None):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    @staticmethod
    def _gather_logprob(logprob, target):
        lp_size = tf.shape(logprob)
        r = tf.range(lp_size[0])
        idx = tf.stack([r, target], 1)
        return tf.gather_nd(logprob, idx)

    def call(self, inputs, return_mean=True, training=False):
        hidden, target = inputs
        head_logprob = 0
        if self.n_clusters == 0:
            softmax_b = tf.get_variable('bias', [n_token], initializer=tf.zeros_initializer())
            output = self._logit(hidden, self.out_layers[0][0], self.out_layers[0][1], self.out_projs[0])
            if target is not None:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
            out = tf.nn.log_softmax(output, axis=-1)
        else:
            hidden_sizes = shape_list(hidden)
            out = []
            loss = tf.zeros(hidden_sizes[:2], dtype=tf.float32)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                if target is not None:
                    mask = (target >= l_idx) & (target < r_idx)
                    mask_idx = tf.where(mask)
                    cur_target = tf.boolean_mask(target, mask) - l_idx

                if self.div_val == 1:
                    cur_W = self.out_layers[0][0][l_idx:r_idx]
                    cur_b = self.out_layers[0][1][l_idx:r_idx]
                else:
                    cur_W = self.out_layers[i][0]
                    cur_b = self.out_layers[i][1]

                if i == 0:
                    cur_W = tf.concat([cur_W, self.cluster_weight], 0)
                    cur_b = tf.concat([cur_b, self.cluster_bias], 0)

                    head_logit = self._logit(hidden, cur_W, cur_b, self.out_projs[0])
                    head_logprob = tf.nn.log_softmax(head_logit)
                    out.append(head_logprob[..., :self.cutoffs[0]])
                    if target is not None:
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_logprob = self._gather_logprob(cur_head_logprob, cur_target)
                else:
                    tail_logit = self._logit(hidden, cur_W, cur_b, self.out_projs[i])
                    tail_logprob = tf.nn.log_softmax(tail_logit)
                    cluster_prob_idx = self.cutoffs[0] + i - 1  # No probability for the head cluster
                    logprob_i = head_logprob[..., cluster_prob_idx, None] + tail_logprob
                    out.append(logprob_i)
                    if target is not None:
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_tail_logprob = tf.boolean_mask(tail_logprob, mask)
                        cur_logprob = self._gather_logprob(cur_tail_logprob, cur_target)
                        cur_logprob += cur_head_logprob[:, self.cutoff_ends[1] + i - 1]
                if target is not None:
                    loss += tf.scatter_nd(mask_idx, -cur_logprob, tf.cast(tf.shape(loss), dtype=tf.int64))
            out = tf.concat(out, axis=-1)

        if target is not None:
            if return_mean:
                loss = tf.reduce_mean(loss)
            # Add the training-time loss value to the layer using `self.add_loss()`.
            self.add_loss(loss)

            # Log the loss as a metric (we could log arbitrary metrics,
            # including different metrics for training and inference.
            self.add_metric(loss, name=self.name, aggregation='mean' if return_mean else '')

        return out


def mul_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs,
                                                        params, tie_projs,
                                                        initializer=None, proj_initializer=None,
                                                        div_val=1, perms=None, proj_same_dim=True,
                                                        scope='adaptive_softmax',
                                                        **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if x.shape.ndims == 3:
            if proj is not None:
                y = tf.einsum('ibd,ed->ibe', y, proj)
            return tf.einsum('ibd,nd->ibn', y, W) + b
        else:
            if proj is not None:
                y = tf.einsum('id,ed->ie', y, proj)
            return tf.einsum('id,nd->in', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = tf.get_variable('bias', [n_token],
                                                                    initializer=tf.zeros_initializer())
            output = _logit(hidden, params_W, softmax_b, params_projs)
            nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                                                                     logits=output)
            nll = tf.reduce_mean(nll)
        else:
            total_loss, total_cnt = 0, 0
            cutoff_ends = [0] + cutoffs + [n_token]
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]

                    cur_d_embed = d_embed // (div_val ** i)

                    if div_val == 1:
                        cur_W = params_W[l_idx: r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = tf.get_variable('b', [r_idx - l_idx],
                                                                    initializer=tf.zeros_initializer())
                    if tie_projs[i]:
                        if div_val == 1:
                            cur_proj = params_projs
                        else:
                            cur_proj = params_projs[i]
                    else:
                        if (div_val == 1 or not proj_same_dim) and d_proj == cur_d_embed:
                            cur_proj = None
                        else:
                            cur_proj = tf.get_variable('proj', [cur_d_embed, d_proj],
                                                                                 initializer=proj_initializer)

                    if i == 0:
                        cluster_W = tf.get_variable('cluster_W', [len(cutoffs), d_embed],
                                                                                initializer=tf.zeros_initializer())
                        cluster_b = tf.get_variable('cluster_b', [len(cutoffs)],
                                                                                initializer=tf.zeros_initializer())
                        cur_W = tf.concat([cur_W, cluster_W], 0)
                        cur_b = tf.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)

                        head_target = kwargs.get("head_target")
                        head_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=head_target,
                                logits=head_logit)

                        masked_loss = head_nll * perms[i]
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(perms[i])

                        # head_logprob = tf.nn.log_softmax(head_logit)

                        # final_logprob = head_logprob * perms[i][:, :, None]
                        # final_target = tf.one_hot(target, tf.shape(head_logprob)[2])
                        # total_loss -= tf.einsum('ibn,ibn->', final_logprob, final_target)
                        # total_cnt += tf.reduce_sum(perms[i])
                    else:
                        cur_head_nll = tf.einsum('ib,ibk->k', head_nll, perms[i])

                        cur_hidden = tf.einsum('ibd,ibk->kd', hidden, perms[i])
                        tail_logit = _logit(cur_hidden, cur_W, cur_b, cur_proj)

                        tail_target = tf.einsum('ib,ibk->k', tf.to_float(target - l_idx),
                                                                        perms[i])
                        tail_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=tf.to_int32(tail_target),
                                logits=tail_logit)

                        sum_nll = cur_head_nll + tail_nll
                        mask = tf.reduce_sum(perms[i], [0, 1])

                        masked_loss = sum_nll * mask
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(mask)

            nll = total_loss / total_cnt

    return nll