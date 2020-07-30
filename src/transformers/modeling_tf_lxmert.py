# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" TF 2.0 LXMERT model. """


import logging

import numpy as np
import tensorflow as tf

from .configuration_lxmert import LxmertConfig
from .file_utils import add_start_docstrings
from .modeling_tf_utils import (
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    shape_list,
)


logger = logging.getLogger(__name__)


TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
]


def gelu(x):
    """ Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    return x * tf.sigmoid(x)


ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
}


class TFVisualFeatEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # Object feature encoding
        self.visn_fc = tf.keras.layers.Dense(
            feat_dim, kernel_initializer=get_initializer(config.initializer_range), name="visn_fc"
        )
        self.visual_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="visual_layer_norm"
        )

        # Box position encoding
        self.box_fc = tf.keras.layers.Dense(
            pos_dim, kernel_initializer=get_initializer(config.initializer_range), name="visn_fc"
        )
        self.box_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="box_layer_norm")

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, visn_input):
        feats, boxes = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output


class TFLxmertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="token_type_embeddings",
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

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
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


class TFLxmertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        hidden_states, context, attention_mask = inputs

        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], tf.float32)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)
        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)

        return context_layer


class TFLxmertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TFLxmertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFLxmertAttOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFLxmertSelfattLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self = TFLxmertAttention(config, name="self")
        self.output = TFLxmertAttOutput(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self((input_tensor, input_tensor, attention_mask))
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class TFLxmertCrossattLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.att = TFLxmertAttention(config, "att")
        self.output = TFLxmertAttOutput(config, "output")

    def call(self, inputs, training=False):
        input_tensor, ctx_tensor, ctx_att_mask = inputs
        output = self.att(inputs, training)
        attention_output = self.output((output, input_tensor), training)
        return attention_output


class TFLxmertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFLxmertAttention(config, name="attention")
        self.intermediate = TFLxmertIntermediate(config, name="intermediate")
        self.transformer_output = TFLxmertOutput(config, name="output")

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        attention_outputs = self.attention([hidden_states, attention_mask, head_mask], training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.transformer_output([intermediate_output, attention_output], training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class TFLxmertXLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFLxmertAttention(config, name="attention")
        self.visual_attention = TFLxmertCrossattLayer(config, name="visual_attention")

        # Self-attention Layers
        self.lang_self_att = TFLxmertSelfattLayer(config, name="lang_self_att")
        self.visn_self_att = TFLxmertSelfattLayer(config, name="visn_self_att")

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = TFLxmertIntermediate(config, name="lang_inter")
        self.lang_output = TFLxmertOutput(config, name="lang_output")
        self.visn_inter = TFLxmertIntermediate(config, name="visn_inter")
        self.visn_output = TFLxmertOutput(config, name="visn_output")

    def cross_att(self, inputs, training=False):
        # Cross Attention
        lang_input, lang_attention_mask, visn_input, visn_attention_mask = inputs
        lang_att_output = self.visual_attention((lang_input, visn_input, visn_attention_mask), training)
        visn_att_output = self.visual_attention((visn_input, lang_input, lang_attention_mask), training)
        return lang_att_output, visn_att_output

    def self_att(self, inputs, training=False):
        # Self Attention
        lang_input, lang_attention_mask, visn_input, visn_attention_mask = inputs
        lang_att_output = self.lang_self_att((lang_input, lang_attention_mask), training)
        visn_att_output = self.visn_self_att((visn_input, visn_attention_mask), training)
        return lang_att_output, visn_att_output

    def output_fc(self, inputs, training=False):
        # FC layers
        lang_input, visn_input = inputs
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output((lang_inter_output, lang_input), training)
        visn_output = self.visn_output((visn_inter_output, visn_input), training)
        return lang_output, visn_output

    def call(self, inputs, training=False):
        lang_feats, lang_attention_mask, visn_feats, visn_attention_mask = inputs
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(
            (lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask), training
        )
        lang_att_output, visn_att_output = self.self_att(
            (lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask), training
        )
        lang_output, visn_output = self.output_fc((lang_att_output, visn_att_output), training)

        return lang_output, visn_output


class TFLxmertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.visn_fc = TFVisualFeatEncoder(config, name="visn_fc")

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.x_layers = [TFLxmertXLayer(config, name="x_layer_._{}".format(i)) for i in range(self.x_layers)]
        self.r_layers = [TFLxmertLayer(config, name="r_layer_._{}".format(i)) for i in range(self.x_layers)]
        self.l_layers = [TFLxmertLayer(config, name="layer_._{}".format(i)) for i in range(self.x_layers)]

    def call(self, inputs, training=False):
        lang_feats, lang_attention_mask, visn_feats, visn_attention_mask = inputs

        visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask, visn_feats, visn_attention_mask)

        return lang_feats, visn_feats


@keras_serializable
class TFLxmertMainLayer(tf.keras.layers.Layer):
    config_class = LxmertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        self.initializer_range = config.initializer_range
        self.embeddings = TFLxmertEmbeddings(config, name="embeddings")
        self.encoder = TFLxmertEncoder(config, name="encoder")
        self.pooler = TFLxmertPooler(config, name="pooler")

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def call(self, inputs, token_type_ids=None, attention_mask=None, visual_feats=None, visual_attention_mask=None):
        # We allow three types of multi-inputs:
        # - traditional keyword arguments in the call method
        # - all the arguments provided as a dict in the first positional argument of call
        # - all the arguments provided as a list/tuple (ordered) in the first positional argument of call
        # The last two options are useful to use the tf.keras fit() method.

        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            token_type_ids = inputs[1] if len(inputs) > 1 else token_type_ids
            attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
            visual_feats = inputs[3] if len(inputs) > 3 else visual_feats
            visual_attention_mask = inputs[4] if len(inputs) > 4 else visual_attention_mask
            assert len(inputs) <= 5, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            attention_mask = inputs.get("attention_mask", attention_mask)
            visual_feats = inputs.get("visual_feats", visual_feats)
            visual_attention_mask = inputs.get("attention_mask", visual_attention_mask)
            assert len(inputs) <= 5, "Too many inputs."
        else:
            input_ids = inputs

        if attention_mask is None:
            attention_mask = tf.fill(shape_list(input_ids), 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(shape_list(input_ids), 0)

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

        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask[:, tf.newaxis, tf.newaxis, :]

            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run Lxmert encoder
        lang_feats, visn_feats = self.encoder(
            [embedding_output, extended_attention_mask, visual_feats, extended_visual_attention_mask]
        )
        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats), pooled_output


class TFLxmertPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = LxmertConfig
    base_model_prefix = "Lxmert"


LXMERT_START_DOCSTRING = r"""    The LXMERT model was proposed in
    `LXMERT: Learning Cross-Modality Encoder Representations from Transformers
    by Hao Tan and Mohit Bansal. It's a vision and language transformer model,
    pre-trained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome,
    using a combination of masked language modeling, region of interest feature regression,
    cross entropy loss for question answering attribute prediction, and object tag predicition.

    This model is a tf.keras.Model `tf.keras.Model`_ sub-class. Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. _`LXMERT: Learning Cross-Modality Encoder Representations from Transformers`
        https://arxiv.org/pdf/1908.07490.pdf

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
        - a dictionary with one or several input Tensors associaed to the input names given in the docstring:

    Parameters:
        config (:class:`~transformers.LxmertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

LXMERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, LXMERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:
            Indices can be obtained using :class:`transformers.LxmertTokenizer`.
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
        **visual_feats**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, features_length, features_dim)``:
            Pre-trained and instance-level vision features ROI-aligned and pooled from bounding boxes.
        **visual_attention_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, features_length)``:
            Mask to avoid performing attention on padding token indices for visual features.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Lxmert Model transformer outputing raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
    LXMERT_INPUTS_DOCSTRING,
)
class TFLxmertModel(TFLxmertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:

        **(lang_feats, visn_feats)**: ``Tuple[Numpy array or tf.Tensor, Tuple[Numpy array or tf.Tensor]]`` of shapes
        ``(batch_size, sequence_length, hidden_size)`` and  each element of the nested tuple being of shape
        ``(batch_size, num_features, visual_feat_dim)`` and ``(batch_size, num_features, visual_pos_dim)`` respectively

        **pooler_output**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence from the cross modality encoder(classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFLxmertMainLayer(config, name="transformer")

    def call(self, inputs, **kwargs):
        outputs = self.transformer(inputs, **kwargs)
        return outputs


class TFLxmertForFeatureExtraction(TFLxmertPreTrainedModel):
    """
    Lxmert model for extracting features
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFLxmertModel(config, name="bert")
        self.mode = config.mode

    def call(self, inputs, **kwargs):

        feat_seq, pooled_output = self.bert(inputs, **kwargs)

        if "x" == self.mode:
            return pooled_output
        elif "x" in self.mode and ("l" in self.mode or "r" in self.mode):
            return feat_seq, pooled_output
        elif "l" in self.mode or "r" in self.mode:
            return feat_seq


class TFLxmertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class TFLxmertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TFLxmertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.transform = TFLxmertPredictionHeadTransform(config, name="transform")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


class TFLxmertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TFLxmertLMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TFLxmertPreTrainingHeads(tf.keras.layers.Layer):
    def __init__(self, config, bert_model_embedding_weights, **kwargs):
        super().__init__(**kwargs)
        self.predictions = TFLxmertLMPredictionHead(config, bert_model_embedding_weights, name="predictions")

        self.seq_relationship = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="seq_relationship"
        )

    def call(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class TFLxmertVisualAnswerHead(tf.keras.layers.Layer):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = tf.keras.Sequential(layers=None, name="logit_fc")
        self.logit_fc.add(tf.keras.layers.Dense(hid_dim, kernel_initializer=get_initializer(config.initializer_range)))
        self.logit_fc.add(tf.keras.layers.Activation(tf.nn.relu))
        self.logit_fc.add(tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps))
        self.logit_fc.add(
            tf.keras.layers.Dense(hid_dim * 2, kernel_initializer=get_initializer(config.initializer_range),)
        )

    def call(self, hidden_states):
        return self.logit_fc(hidden_states)


class TFLxmertVisualObjHead(tf.keras.layers.Layer):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = TFLxmertPredictionHeadTransform(config, "transform")

        # Decide the use of visual losses
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.n_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.n_attr_labels}
        if config.visual_obj_loss:
            visual_losses["feat"] = {"shape": (-1, 2048), "num": config.visual_feat_dim}
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = {
            key: tf.keras.layers.Dense(
                config.hidden_states, kernel_initializer=get_initializer(config.initializer_range), name=key
            )
            for key in self.visual_losses
        }

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


@add_start_docstrings(
    """Lxmert Model with a `language modeling` head on top. """, LXMERT_START_DOCSTRING, LXMERT_INPUTS_DOCSTRING
)
class TFLxmertForPretraining(TFLxmertPreTrainedModel):
    r"""
    **input_ids**: ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
        Indices of input sequence tokens in the vocabulary.
        To match pre-training, LXMERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:
        Indices can be obtained using :class:`transformers.LxmertTokenizer`.
        See :func:`transformers.PreTrainedTokenizer.encode` and
        :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
    **attention_mask**: (`optional`) ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
        Mask to avoid performing attention on padding token indices.
        Mask values selected in ``[0, 1]``:
        ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
    **token_type_ids**: (`optional`) ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
        Segment token indices to indicate first and second portions of the inputs.
        Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        corresponds to a `sentence B` token
    **visual_feats**: (`optional`) ``tf.Tensor`` of shape ``(batch_size, features_length, features_dim)``:
        Pre-trained and instance-level vision features ROI-aligned and pooled from bounding boxes.
    **pos**: (`optional`) ``tf.Tensor`` of shape ``(batch_size, features_length, visual_pos_dim)``:
        the bounding box coordinate for each respective visual feature instance.
    **masked_lm_labels**: (`optional`) ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
        Labels for computing the masked language modeling loss.
        Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
        Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
        in ``[0, ..., config.vocab_size]``
    **obj_labels**: (`optional`): ``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``
        each key is named after each one of the visual losses and each element of the tuple is of the shape
        ``(batch_size, num_features, num_objects/num_attrs)`` and ``(batch_size, num_features)``
        for each the label id and the label score respectively
    **matched_label**: (`optional`) ``Torch.Tensor`` of shape ``(batch_size,)`` an int [0,1] that represents
    whether or not this text is correctly matching its associated image
    **ans**: (`optional`) ``Torch.Tensor`` of shape ``(batch_size, num_qa_answers)`` a one hot represntation of the correct answer



    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **total_loss**: ``tf.Tensor`` of shape ``(1,)``:
            total loss as a simple sum from all the modeling objectives.
        **losses**: ``Tuple[tf.Tensor]`` each of shape ``(1,)`` for each indivual loss
        **answer_score**: ``Torch.FloatTensor`` the accuracy of the question answering loss
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFLxmertMainLayer(config, name="bert")
        self.config = config
        self.num_answers = config.num_answers
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # Use of pre-training tasks
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa

        # Pre-training heads
        self.cls = TFLxmertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight, name="cls")
        if self.task_obj_predict:
            self.obj_predict_head = TFLxmertVisualObjHead(config, name="obj_head_predict")
        if self.task_qa:
            self.answer_head = TFLxmertVisualAnswerHead(config, self.num_answers, name="answer_head")

        # Weight initialization
        self.init_weights()

        # Loss functions
        self.loss_fcts = {
            "l2": tf.keras.losses.Huber(delta=1.0, reduction=None, name="huber_loss"),
            "visn_ce": TFMultipleChoiceLoss,
            "ce": TFTokenClassificationLoss,
        }

        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.n_object_labels, "loss": "visn_ce"}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.n_attr_labels, "loss": "visn_ce"}
        if config.visual_obj_loss:
            visual_losses["feat"] = {"shape": (-1, 2048), "num": config.visual_feat_dim, "loss": "l2"}
        self.visual_losses = visual_losses

    def call(self, inputs, **kwargs):

        input_ids, token_type_ids, attention_mask, visual_feats = inputs

        pos = getattr(kwargs, "pos", None)
        obj_labels = getattr(kwargs, "obj_labels", None)
        matched_label = getattr(kwargs, "matched_label", None)
        ans = getattr(kwargs, "ans", None)
        masked_lm_labels = getattr(kwargs, "masked_lm_labels", None)

        visual_feats = (visual_feats, pos)
        inputs = (input_ids, token_type_ids, attention_mask, visual_feats)

        (lang_output, visn_output), pooled_output = self.bert(inputs)

        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            # This answer_score would not be used anywhere,
            # just to keep a constant return function signature.
            answer_score = pooled_output[0][0]

        total_loss = 0.0
        losses = ()
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)
        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fcts["ce"](cross_relationship_score.view(-1, 2), matched_label.view(-1))
            total_loss += matched_loss
            losses += (matched_loss.detach(),)
        if obj_labels is not None and self.task_obj_predict:
            total_visn_loss = 0.0
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key, key_info in self.visual_losses.items():
                label, mask_conf = obj_labels[key]
                output_dim = key_info["num"]
                loss_fct_name = key_info["loss"]
                label_shape = key_info["shape"]
                weight = self.visual_loss_normalizer
                visn_loss_fct = self.loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                if key != "feat":
                    visn_loss = visn_loss_fct.compute_loss(
                        visn_prediction_scores.view(-1, output_dim), label.view(*label_shape),
                    )
                else:
                    visn_loss = visn_loss_fct(visn_prediction_scores.view(-1, output_dim), label.view(*label_shape),)

                if visn_loss.dim() > 1:  # Regression Losses
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
                losses += (visn_loss.detach(),)
            total_loss += total_visn_loss
        if ans is not None and self.task_qa:
            answer_loss = self.loss_fcts["ce"](answer_score.view(-1, self.num_answers), ans.view(-1))
            # exclude "*2" here to match the effect of QA losses.
            # Previous: (loss *0) for 6 epochs, (loss *2) for 6 epochs.   (Used 10 instead of 6 in EMNLP paper)
            # Now     : (loss *1) for 12 epochs
            #
            # * 2       # Multiply by 2 because > half of the data will not have label
            total_loss += answer_loss
            losses += (answer_loss.detach(),)
        return total_loss, tf.stack(losses)[tf.new_axis, ...], answer_score.detach()
