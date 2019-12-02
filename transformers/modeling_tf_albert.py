# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
""" TF 2.0 ALBERT model. """
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys

import tensorflow as tf

from .configuration_albert import AlbertConfig
from .modeling_tf_utils import TFPreTrainedModel, get_initializer, shape_list
from .modeling_tf_bert import ACT2FN, TFBertSelfAttention
from .file_utils import add_start_docstrings

import logging

logger = logging.getLogger(__name__)

TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'albert-base-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-tf_model.h5",
    'albert-large-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v1-tf_model.h5",
    'albert-xlarge-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v1-tf_model.h5",
    'albert-xxlarge-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v1-tf_model.h5",
    'albert-base-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-tf_model.h5",
    'albert-large-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-tf_model.h5",
    'albert-xlarge-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-tf_model.h5",
    'albert-xxlarge-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-tf_model.h5",
}


class TFAlbertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(TFAlbertEmbeddings, self).__init__(**kwargs)

        self.config = config
        self.position_embeddings = tf.keras.layers.Embedding(config.max_position_embeddings,
                                                             config.embedding_size,
                                                             embeddings_initializer=get_initializer(
                                                                 self.config.initializer_range),
                                                             name='position_embeddings')
        self.token_type_embeddings = tf.keras.layers.Embedding(config.type_vocab_size,
                                                               config.embedding_size,
                                                               embeddings_initializer=get_initializer(
                                                                   self.config.initializer_range),
                                                               name='token_type_embeddings')

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=get_initializer(self.config.initializer_range))
        super(TFAlbertEmbeddings, self).build(input_shape)

    def call(self, inputs, mode="embedding", training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, embedding_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]
        x = tf.reshape(inputs, [-1, self.config.embedding_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)
        return tf.reshape(logits, [batch_size, length, self.config.vocab_size])


class TFAlbertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFAlbertSelfAttention, self).__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(
                                               config.initializer_range),
                                           name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size,
                                         kernel_initializer=get_initializer(
                                             config.initializer_range),
                                         name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size,
                                           kernel_initializer=get_initializer(
                                               config.initializer_range),
                                           name='value')

        self.dropout = tf.keras.layers.Dropout(
            config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # scale attention_scores
        dk = tf.cast(shape_list(key_layer)[-1], tf.float32)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFAlbertModel call() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,
                                   (batch_size, -1, self.all_head_size))  # (batch_size, seq_len_q, all_head_size)

        outputs = (context_layer, attention_probs) if self.output_attentions else (
            context_layer,)
        return outputs


class TFAlbertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFAlbertSelfOutput, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size,
                                           kernel_initializer=get_initializer(
                                               config.initializer_range),
                                           name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFAlbertAttention(TFBertSelfAttention):
    def __init__(self, config, **kwargs):
        super(TFAlbertAttention, self).__init__(config, **kwargs)

        self.hidden_size = config.hidden_size
        self.dense = tf.keras.layers.Dense(config.hidden_size,
                                           kernel_initializer=get_initializer(
                                               config.initializer_range),
                                           name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name='LayerNorm')
        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        input_tensor, attention_mask, head_mask = inputs

        batch_size = shape_list(input_tensor)[0]
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # scale attention_scores
        dk = tf.cast(shape_list(key_layer)[-1], tf.float32)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,
                                   (batch_size, -1, self.all_head_size))  # (batch_size, seq_len_q, all_head_size)

        self_outputs = (context_layer, attention_probs) if self.output_attentions else (
            context_layer,)

        hidden_states = self_outputs[0]

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        attention_output = self.LayerNorm(hidden_states + input_tensor)

        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class TFAlbertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFAlbertLayer, self).__init__(**kwargs)
        self.attention = TFAlbertAttention(config, name='attention')

        self.ffn = tf.keras.layers.Dense(config.intermediate_size, kernel_initializer=get_initializer(
            config.initializer_range), name='ffn')

        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

        self.ffn_output = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(
            config.initializer_range), name='ffn_output')
        self.full_layer_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name='full_layer_layer_norm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask, head_mask], training=training)
        ffn_output = self.ffn(attention_outputs[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.full_layer_layer_norm(
            ffn_output + attention_outputs[0])

        # add attentions if we output them
        outputs = (hidden_states,) + attention_outputs[1:]
        return outputs


class TFAlbertLayerGroup(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFAlbertLayerGroup, self).__init__(**kwargs)

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = [TFAlbertLayer(config, name="albert_layers_._{}".format(
            i)) for i in range(config.inner_group_num)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(
                [hidden_states, attention_mask, head_mask[layer_index]], training=training)
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        # last-layer hidden state, (layer hidden states), (layer attentions)
        return outputs


class TFAlbertTransformer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFAlbertTransformer, self).__init__(**kwargs)

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(
            config.initializer_range), name='embedding_hidden_mapping_in')
        self.albert_layer_groups = [TFAlbertLayerGroup(
            config, name="albert_layer_groups_._{}".format(i)) for i in range(config.num_hidden_groups)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(
                self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(
                i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                [hidden_states, attention_mask, head_mask[group_idx*layers_per_group:(group_idx+1)*layers_per_group]], training=training)
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class TFAlbertPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = AlbertConfig
    pretrained_model_archive_map = TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "albert"


class TFAlbertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super(TFAlbertMLMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size

        self.dense = tf.keras.layers.Dense(config.embedding_size,
                                           kernel_initializer=get_initializer(
                                               config.initializer_range),
                                           name='dense')
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name='LayerNorm')

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')
        self.decoder_bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='decoder/bias')
        super(TFAlbertMLMHead, self).build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states, mode="linear") + self.decoder_bias
        hidden_states = hidden_states + self.bias
        return hidden_states


ALBERT_START_DOCSTRING = r"""    The ALBERT model was proposed in
    `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations`_
    by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. It presents
    two parameter-reduction techniques to lower memory consumption and increase the trainig speed of BERT.

    This model is a tf.keras.Model `tf.keras.Model`_ sub-class. Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. _`ALBERT: A Lite BERT for Self-supervised Learning of Language Representations`:
        https://arxiv.org/abs/1909.11942

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
        config (:class:`~transformers.AlbertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ALBERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, ALBERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``

            Albert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.AlbertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `ALBERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare Albert Model transformer outputing raw hidden-states without any specific head on top.",
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class TFAlbertModel(TFAlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``tf.Tensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Albert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import AlbertTokenizer, TFAlbertModel

        tokenizer = AlbertTokenizer.from_pretrained('bert-base-uncased')
        model = TFAlbertModel.from_pretrained('bert-base-uncased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config, **kwargs):
        super(TFAlbertModel, self).__init__(config, **kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.embeddings = TFAlbertEmbeddings(config, name="embeddings")
        self.encoder = TFAlbertTransformer(config, name="encoder")
        self.pooler = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(
            config.initializer_range), activation='tanh', name='pooler')

    def get_input_embeddings(self):
        return self.embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(self, inputs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if not head_mask is None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        embedding_output = self.embeddings(
            [input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
        encoder_outputs = self.encoder(
            [embedding_output, extended_attention_mask, head_mask], training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output[:, 0])

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


@add_start_docstrings("""Albert Model with a `language modeling` head on top. """,
                      ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel):
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
        from transformers import AlbertTokenizer, TFAlbertForMaskedLM

        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = TFAlbertForMaskedLM.from_pretrained('albert-base-v2')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        prediction_scores = outputs[0]

    """

    def __init__(self, config, *inputs, **kwargs):
        super(TFAlbertForMaskedLM, self).__init__(config, *inputs, **kwargs)

        self.albert = TFAlbertModel(config, name='albert')
        self.predictions = TFAlbertMLMHead(
            config, self.albert.embeddings, name='predictions')

    def get_output_embeddings(self):
        return self.albert.embeddings

    def call(self, inputs, **kwargs):
        outputs = self.albert(inputs, **kwargs)

        sequence_output = outputs[0]
        prediction_scores = self.predictions(
            sequence_output, training=kwargs.get('training', False))

        # Add hidden states and attention if they are here
        outputs = (prediction_scores,) + outputs[2:]

        return outputs  # prediction_scores, (hidden_states), (attentions)


@add_start_docstrings("""Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING)
class TFAlbertForSequenceClassification(TFAlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import AlbertTokenizer, TFAlbertForSequenceClassification

        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFAlbertForSequenceClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.albert = TFAlbertModel(config, name='albert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier')

    def call(self, inputs, **kwargs):
        outputs = self.albert(inputs, **kwargs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)