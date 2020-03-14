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
""" A TF 2.0 Adaptive Softmax for Transformer XL model.
"""


import tensorflow as tf

from .modeling_tf_utils import shape_list


class TFAdaptiveSoftmaxMask(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_embed, d_proj, cutoffs, div_val=1, keep_order=False, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [vocab_size]
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
            self.cluster_weight = self.add_weight(
                shape=(self.n_clusters, self.d_embed), initializer="zeros", trainable=True, name="cluster_weight"
            )
            self.cluster_bias = self.add_weight(
                shape=(self.n_clusters,), initializer="zeros", trainable=True, name="cluster_bias"
            )

        if self.div_val == 1:
            for i in range(len(self.cutoffs)):
                if self.d_proj != self.d_embed:
                    weight = self.add_weight(
                        shape=(self.d_embed, self.d_proj),
                        initializer="zeros",
                        trainable=True,
                        name="out_projs_._{}".format(i),
                    )
                    self.out_projs.append(weight)
                else:
                    self.out_projs.append(None)
                weight = self.add_weight(
                    shape=(self.vocab_size, self.d_embed,),
                    initializer="zeros",
                    trainable=True,
                    name="out_layers_._{}_._weight".format(i),
                )
                bias = self.add_weight(
                    shape=(self.vocab_size,),
                    initializer="zeros",
                    trainable=True,
                    name="out_layers_._{}_._bias".format(i),
                )
                self.out_layers.append((weight, bias))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = self.d_embed // (self.div_val ** i)

                weight = self.add_weight(
                    shape=(d_emb_i, self.d_proj), initializer="zeros", trainable=True, name="out_projs_._{}".format(i)
                )
                self.out_projs.append(weight)
                weight = self.add_weight(
                    shape=(r_idx - l_idx, d_emb_i,),
                    initializer="zeros",
                    trainable=True,
                    name="out_layers_._{}_._weight".format(i),
                )
                bias = self.add_weight(
                    shape=(r_idx - l_idx,),
                    initializer="zeros",
                    trainable=True,
                    name="out_layers_._{}_._bias".format(i),
                )
                self.out_layers.append((weight, bias))
        super().build(input_shape)

    @staticmethod
    def _logit(x, W, b, proj=None):
        y = x
        if proj is not None:
            y = tf.einsum("ibd,ed->ibe", y, proj)
        return tf.einsum("ibd,nd->ibn", y, W) + b

    @staticmethod
    def _gather_logprob(logprob, target):
        lp_size = shape_list(logprob)
        r = tf.range(lp_size[0])
        idx = tf.stack([r, target], 1)
        return tf.gather_nd(logprob, idx)

    def call(self, inputs, return_mean=True, training=False):
        hidden, target = inputs
        head_logprob = 0
        if self.n_clusters == 0:
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
                    out.append(head_logprob[..., : self.cutoffs[0]])
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
                    loss += tf.scatter_nd(mask_idx, -cur_logprob, tf.cast(shape_list(loss), dtype=tf.int64))
            out = tf.concat(out, axis=-1)

        if target is not None:
            if return_mean:
                loss = tf.reduce_mean(loss)
            # Add the training-time loss value to the layer using `self.add_loss()`.
            self.add_loss(loss)

            # Log the loss as a metric (we could log arbitrary metrics,
            # including different metrics for training and inference.
            self.add_metric(loss, name=self.name, aggregation="mean" if return_mean else "")

        return out


# TODO: (PVP) add tests to compare to PT!
class TFLogUniformSampler(object):
    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        self.range_max = range_max
        log_indices = tf.math.log(tf.range(1.0, range_max + 2.0))
        self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]

        self.log_q = tf.cast(
            tf.math.log(-tf.math.expm1(-tf.math.log1p(tf.cast(self.dist, tf.double)) * 2 * n_sample)), tf.float32
        )

        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        n_sample = self.n_sample
        n_tries = 2 * n_sample

        neg_samples = tf.unique(tf.squeeze(tf.random.categorical(tf.expand_dims(self.dist, 0), n_tries), 0))[0]
        true_log_probs = tf.gather_nd(self.log_q, labels)
        samp_log_probs = tf.gather(self.log_q, neg_samples)
        return true_log_probs, samp_log_probs, neg_samples
