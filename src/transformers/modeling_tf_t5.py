# coding=utf-8
# Copyright 2018 T5 Authors and The HuggingFace Inc. team.
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
""" TF 2.0 T5 model. """


import copy
import itertools
import logging
import math

import tensorflow as tf

from .configuration_t5 import T5Config
from .file_utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings
from .modeling_tf_utils import TFPreTrainedModel, TFSharedEmbeddings, shape_list


logger = logging.getLogger(__name__)

TF_T5_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "t5-small": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-small-tf_model.h5",
    "t5-base": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-base-tf_model.h5",
    "t5-large": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-large-tf_model.h5",
    "t5-3b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-3b-tf_model.h5",
    "t5-11b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-11b-tf_model.h5",
}

####################################################
# TF 2.0 Models are constructed using Keras imperative API by sub-classing
# - tf.keras.layers.Layer for the layers and
# - TFPreTrainedModel for the models (it-self a sub-class of tf.keras.Model)
####################################################


class TFT5LayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__(**kwargs)
        self.variance_epsilon = epsilon

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)

    def call(self, x):
        variance = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class TFT5DenseReluDense(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.wi = tf.keras.layers.Dense(config.d_ff, use_bias=False, name="wi")
        self.wo = tf.keras.layers.Dense(config.d_model, use_bias=False, name="wo")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.act = tf.keras.activations.relu

    def call(self, hidden_states, training=False):
        h = self.wi(hidden_states)
        h = self.act(h)
        h = self.dropout(h, training=training)
        h = self.wo(h)
        return h


class TFT5LayerFF(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.DenseReluDense = TFT5DenseReluDense(config, name="DenseReluDense")
        self.layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, hidden_states, training=False):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x, training=training)
        layer_output = hidden_states + self.dropout(y, training=training)
        return layer_output


class TFT5Attention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFT5Attention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.output_attentions = config.output_attentions
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="q")
        self.k = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="k")
        self.v = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="v")
        self.o = tf.keras.layers.Dense(self.d_model, use_bias=False, name="o")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = tf.keras.layers.Embedding(
                self.relative_attention_num_buckets, self.n_heads, name="relative_attention_bias"
            )
        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += tf.dtypes.cast(tf.math.less(n, 0), tf.int32) * num_buckets
            n = tf.math.abs(n)
        else:
            n = tf.math.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(n, max_exact)
        val_if_large = max_exact + tf.dtypes.cast(
            tf.math.log(tf.dtypes.cast(n, tf.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            tf.int32,
        )
        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        ret += tf.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = tf.range(qlen)[:, None]
        memory_position = tf.range(klen)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position, bidirectional=not self.is_decoder, num_buckets=self.relative_attention_num_buckets
        )
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)  # shape (1, num_heads, qlen, klen)
        return values

    def call(self, input, mask=None, kv=None, position_bias=None, cache=None, head_mask=None, training=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = shape_list(input)
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = shape_list(kv)[1]

        def shape(x):
            """  projection """
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, self.d_kv)), perm=(0, 2, 1, 3))

        def unshape(x):
            """  compute context """
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.inner_dim))

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = tf.concat([k_, k], axis=2)  # (bs, n_heads, klen, dim_per_head)
                    v = tf.concat([v_, v], axis=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        # q = q / math.sqrt(dim_per_head)                                     # No scaling in T5
        # scores = tf.matmul(q, k, transpose_b=True)                            # (bs, n_heads, qlen, klen)
        scores = tf.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(qlen, klen)
            if mask is not None:
                position_bias = position_bias + mask
                # mask = (mask == 0).expand_as(scores)                              # (bs, n_heads, qlen, klen)
                # scores.masked_fill_(mask, -float('inf'))                          # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights, training=training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,)
        if self.output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class TFT5LayerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.SelfAttention = TFT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, name="SelfAttention"
        )
        self.layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, hidden_states, attention_mask=None, position_bias=None, head_mask=None, training=False):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x, mask=attention_mask, position_bias=position_bias, head_mask=head_mask, training=training
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y, training=training)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TFT5LayerCrossAttention(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.EncDecAttention = TFT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, name="EncDecAttention"
        )
        self.layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, hidden_states, kv, attention_mask=None, position_bias=None, head_mask=None, training=False):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x, mask=attention_mask, kv=kv, position_bias=position_bias, head_mask=head_mask, training=training
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y, training=training)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TFT5Block(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.is_decoder = config.is_decoder
        self.layer = []
        self.layer.append(
            TFT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, name="layer_._0")
        )
        if self.is_decoder:
            self.layer.append(
                TFT5LayerCrossAttention(
                    config, has_relative_attention_bias=has_relative_attention_bias, name="layer_._1"
                )
            )
            self.layer.append(TFT5LayerFF(config, name="layer_._2"))
        else:
            self.layer.append(TFT5LayerFF(config, name="layer_._1"))

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        training=False,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            training=training,
        )
        hidden_states = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if not self.is_decoder:
            hidden_states = self.layer[1](hidden_states, training=training)
        else:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                training=training,
            )
            hidden_states = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
            hidden_states = self.layer[2](hidden_states, training=training)

        outputs = (hidden_states,) + outputs  # add attentions if we output them
        return outputs  # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class _NoLayerEmbedTokens(object):
    """
     this class wraps a the TFSharedEmbeddingTokens layer into a python 'no-keras-layer'
     class to avoid problem with weight restoring. Also it makes sure that the layer is
     called from the correct scope to avoid problem with saving/storing the correct weights
    """

    def __init__(self, layer, abs_scope_name=None):
        self._layer = layer
        self._abs_scope_name = abs_scope_name

    def call(self, inputs, mode="embedding"):
        if self._abs_scope_name is None:
            return self._layer.call(inputs, mode)

        # if an abs scope name is given to the embedding variable, call variable from absolute scope
        with tf.compat.v1.variable_scope(self._abs_scope_name, auxiliary_name_scope=False) as abs_scope_name:
            with tf.name_scope(abs_scope_name.original_name_scope):
                return self._layer.call(inputs, mode)

    def __call__(self, inputs, mode="embedding"):
        if self._abs_scope_name is None:
            return self._layer(inputs, mode)

        # if an abs scope name is given to the embedding variable, call variable from absolute scope
        with tf.compat.v1.variable_scope(self._abs_scope_name, auxiliary_name_scope=False) as abs_scope_name:
            with tf.name_scope(abs_scope_name.original_name_scope):
                return self._layer(inputs, mode)


####################################################
# The full model without a specific pretrained or finetuning head is
# provided as a tf.keras.layers.Layer usually called "TFT5MainLayer"
####################################################
class TFT5MainLayer(tf.keras.layers.Layer):
    def __init__(self, config, embed_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.config = config
        self.num_hidden_layers = config.num_layers

        self.block = [
            TFT5Block(config, has_relative_attention_bias=bool(i == 0), name="block_._{}".format(i))
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="final_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def call(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        training=False,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = tf.fill((batch_size, seq_length), 1)
        if self.is_decoder and encoder_attention_mask is None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = tf.fill((batch_size, encoder_seq_length), 1)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        num_dims_attention_mask = len(shape_list(attention_mask))
        if num_dims_attention_mask == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif num_dims_attention_mask == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                seq_ids = tf.range(seq_length)
                causal_mask = tf.less_equal(
                    tf.tile(seq_ids[None, None, :], (batch_size, seq_length, 1)), seq_ids[None, :, None]
                )
                causal_mask = tf.cast(causal_mask, dtype=tf.float32)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposistion
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
        # extended_attention_mask = tf.math.equal(extended_attention_mask,
        #                                         tf.transpose(extended_attention_mask, perm=(-1, -2)))

        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if self.is_decoder:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=tf.float32)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposistion
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = inputs_embeds
        for i, layer_module in enumerate(self.block):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                training=training,
            )
            hidden_states = layer_outputs[0]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2 if self.output_attentions else 1]
                if self.is_decoder:
                    encoder_decoder_position_bias = layer_outputs[4 if self.output_attentions else 2]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


####################################################
# TFT5PreTrainedModel is a sub-class of tf.keras.Model
# which take care of loading and saving pretrained weights
# and various common utilities.
# Here you just need to specify a few (self-explanatory)
# pointers for your model.
####################################################
class TFT5PreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    pretrained_model_archive_map = TF_T5_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = tf.constant(DUMMY_INPUTS)
        input_mask = tf.constant(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs


T5_START_DOCSTRING = r"""    The T5 model was proposed in
    `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_
    by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    It's an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

    This model is a tf.keras.Model `tf.keras.Model`_ sub-class. Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. _`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`:
        https://arxiv.org/abs/1910.10683

    .. _`tf.keras.Model`:
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model

    Note on the model inputs:
        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is usefull when using `tf.keras.Model.fit()` method which currently requires having all the tensors in the first argument of the model call function: `model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the first positional argument :

        - a single Tensor with input_ids only and nothing else: `model(inputs_ids)
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
            `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associaed to the input names given in the docstring:
            `model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, T5 input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``


            T5 is a model with relative position embeddings so you should be able to pad the inputs on
            the right or the left.

            Indices can be obtained using :class:`transformers.T5Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
    T5_INPUTS_DOCSTRING,
)
class TFT5Model(TFT5PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import T5Tokenizer, TFT5Model

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = TFT5Model.from_pretrained('t5-small')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, name="shared")

        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("shared") as shared_abs_scope_name:
            pass

        embed_tokens = _NoLayerEmbedTokens(self.shared, abs_scope_name=shared_abs_scope_name)

        encoder_config = copy.deepcopy(config)
        self.encoder = TFT5MainLayer(encoder_config, embed_tokens, name="encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = TFT5MainLayer(decoder_config, embed_tokens, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def get_output_embeddings(self):
        return self.shared

    def call(self, decoder_input_ids, **kwargs):

        if isinstance(decoder_input_ids, dict):
            kwargs.update(decoder_input_ids)
        else:
            kwargs["decoder_input_ids"] = decoder_input_ids

        # retrieve arguments
        input_ids = kwargs.get("input_ids", None)
        decoder_input_ids = kwargs.get("decoder_input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        encoder_outputs = kwargs.get("encoder_outputs", None)
        decoder_attention_mask = kwargs.get("decoder_attention_mask", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        decoder_inputs_embeds = kwargs.get("decoder_inputs_embeds", None)
        head_mask = kwargs.get("head_mask", None)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
        )

        return decoder_outputs + encoder_outputs


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING, T5_INPUTS_DOCSTRING)
class TFT5ForConditionalGeneration(TFT5PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **prediction_scores**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import T5Tokenizer, TFT5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids=input_ids)
        prediction_scores = outputs[0]

    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model_dim = config.d_model

        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, name="shared")

        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("shared") as shared_abs_scope_name:
            pass

        embed_tokens = _NoLayerEmbedTokens(self.shared, abs_scope_name=shared_abs_scope_name)

        encoder_config = copy.deepcopy(config)
        self.encoder = TFT5MainLayer(encoder_config, embed_tokens, name="encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = TFT5MainLayer(decoder_config, embed_tokens, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def get_output_embeddings(self):
        return self.shared

    def get_encoder(self):
        return self.encoder

    def call(self, decoder_input_ids, **kwargs):

        if isinstance(decoder_input_ids, dict):
            kwargs.update(decoder_input_ids)
        else:
            kwargs["decoder_input_ids"] = decoder_input_ids

        # retrieve arguments
        input_ids = kwargs.get("input_ids", None)
        decoder_input_ids = kwargs.get("decoder_input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        encoder_outputs = kwargs.get("encoder_outputs", None)
        decoder_attention_mask = kwargs.get("decoder_attention_mask", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        decoder_inputs_embeds = kwargs.get("decoder_inputs_embeds", None)
        head_mask = kwargs.get("head_mask", None)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = decoder_outputs[0] * (self.model_dim ** -0.5)
        embed_tokens = self.get_output_embeddings()
        lm_logits = embed_tokens(sequence_output, mode="linear")
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]

        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs = past
        else:
            encoder_outputs = (past,)

        return {
            "inputs": input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past, beam_idx):
        # past does not have to be re-ordered for T5.
        return past
