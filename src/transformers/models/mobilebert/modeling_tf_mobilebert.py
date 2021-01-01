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
""" TF 2.0 MobileBERT model. """

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_mobilebert import MobileBertConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MobileBertConfig"
_TOKENIZER_FOR_DOC = "MobileBertTokenizer"

TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilebert-uncased",
    # See all MobileBERT models at https://huggingface.co/models?filter=mobilebert
]


class TFMobileBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(config.intermediate_size, name="dense")

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TFLayerNorm(tf.keras.layers.LayerNormalization):
    def __init__(self, feat_size, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TFNoNorm(tf.keras.layers.Layer):
    def __init__(self, feat_size, epsilon=None, **kwargs):
        super().__init__(**kwargs)
        self.feat_size = feat_size

    def build(self, input_shape):
        self.bias = self.add_weight("bias", shape=[self.feat_size], initializer="zeros")
        self.weight = self.add_weight("weight", shape=[self.feat_size], initializer="ones")

    def call(self, inputs: tf.Tensor):
        return inputs * self.weight + self.bias


NORM2FN = {"layer_norm": TFLayerNorm, "no_norm": TFNoNorm}


class TFMobileBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
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

        self.embedding_transformation = tf.keras.layers.Dense(config.hidden_size, name="embedding_transformation")

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        mode="embedding",
        training=False,
    ):
        """
        Get token embeddings of inputs.

        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".

        Returns:
            outputs: If mode == "embedding", output embedding tensor, float32 with shape [batch_size, length,
            embedding_size]; if mode == "linear", output linear tensor, float32 with shape [batch_size, length,
            vocab_size].

        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        elif mode == "linear":
            return self._linear(input_ids)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids, position_ids, token_type_ids, inputs_embeds, training=False):
        """Applies embedding based on inputs tensor."""
        assert not (input_ids is None and inputs_embeds is None)

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

        if self.trigram_input:
            # From the paper MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
            # Devices (https://arxiv.org/abs/2004.02984)
            #
            # The embedding table in BERT models accounts for a substantial proportion of model size. To compress
            # the embedding layer, we reduce the embedding dimension to 128 in MobileBERT.
            # Then, we apply a 1D convolution with kernel size 3 on the raw token embedding to produce a 512
            # dimensional output.
            inputs_embeds = tf.concat(
                [
                    tf.pad(inputs_embeds[:, 1:], ((0, 0), (0, 1), (0, 0))),
                    inputs_embeds,
                    tf.pad(inputs_embeds[:, :-1], ((0, 0), (1, 0), (0, 0))),
                ],
                axis=2,
            )

        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

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


class TFMobileBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
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

    def call(
        self, query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=False
    ):
        batch_size = shape_list(attention_mask)[0]
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)
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

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TFMobileBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.dense = tf.keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        if not self.use_bottleneck:
            self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, residual_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + residual_tensor)
        return hidden_states


class TFMobileBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self = TFMobileBertSelfAttention(config, name="self")
        self.mobilebert_output = TFMobileBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        layer_input,
        attention_mask,
        head_mask,
        output_attentions,
        training=False,
    ):
        self_outputs = self.self(
            query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=training
        )

        attention_output = self.mobilebert_output(self_outputs[0], layer_input, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TFOutputBottleneck(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, residual_tensor, training=False):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs, training=training)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class TFMobileBertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.dense = tf.keras.layers.Dense(
            config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        if not self.use_bottleneck:
            self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = TFOutputBottleneck(config, name="bottleneck")

    def call(self, hidden_states, residual_tensor_1, residual_tensor_2, training=False):
        hidden_states = self.dense(hidden_states)
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
        else:
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
            hidden_states = self.bottleneck(hidden_states, residual_tensor_2)
        return hidden_states


class TFBottleneckLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.intra_bottleneck_size, name="dense")
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.intra_bottleneck_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )

    def call(self, inputs):
        hidden_states = self.dense(inputs)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TFBottleneck(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.bottleneck_input = TFBottleneckLayer(config, name="input")
        if self.key_query_shared_bottleneck:
            self.attention = TFBottleneckLayer(config, name="attention")

    def call(self, hidden_states):
        # This method can return three different tuples of values. These different values make use of bottlenecks,
        # which are linear layers used to project the hidden states to a lower-dimensional vector, reducing memory
        # usage. These linear layer have weights that are learned during training.
        #
        # If `config.use_bottleneck_attention`, it will return the result of the bottleneck layer four times for the
        # key, query, value, and "layer input" to be used by the attention layer.
        # This bottleneck is used to project the hidden. This last layer input will be used as a residual tensor
        # in the attention self output, after the attention scores have been computed.
        #
        # If not `config.use_bottleneck_attention` and `config.key_query_shared_bottleneck`, this will return
        # four values, three of which have been passed through a bottleneck: the query and key, passed through the same
        # bottleneck, and the residual layer to be applied in the attention self output, through another bottleneck.
        #
        # Finally, in the last case, the values for the query, key and values are the hidden states without bottleneck,
        # and the residual layer will be this value passed through a bottleneck.

        bottlenecked_hidden_states = self.bottleneck_input(hidden_states)
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        else:
            return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)


class TFFFNOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.true_hidden_size, name="dense")
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.true_hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm"
        )

    def call(self, hidden_states, residual_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + residual_tensor)
        return hidden_states


class TFFFNLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.intermediate = TFMobileBertIntermediate(config, name="intermediate")
        self.mobilebert_output = TFFFNOutput(config, name="output")

    def call(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.mobilebert_output(intermediate_output, hidden_states)
        return layer_outputs


class TFMobileBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks
        self.attention = TFMobileBertAttention(config, name="attention")
        self.intermediate = TFMobileBertIntermediate(config, name="intermediate")
        self.mobilebert_output = TFMobileBertOutput(config, name="output")

        if self.use_bottleneck:
            self.bottleneck = TFBottleneck(config, name="bottleneck")
        if config.num_feedforward_networks > 1:
            self.ffn = [
                TFFFNLayer(config, name="ffn.{}".format(i)) for i in range(config.num_feedforward_networks - 1)
            ]

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions,
            training=training,
        )

        attention_output = attention_outputs[0]
        s = (attention_output,)

        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.mobilebert_output(intermediate_output, attention_output, hidden_states, training=training)

        outputs = (
            (layer_output,)
            + attention_outputs[1:]
            + (
                tf.constant(0),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )  # add attentions if we output them

        return outputs


class TFMobileBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [TFMobileBertLayer(config, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class TFMobileBertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.do_activate = config.classifier_activation
        if self.do_activate:
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
        if not self.do_activate:
            return first_token_tensor
        else:
            pooled_output = self.dense(first_token_tensor)
            return pooled_output


class TFMobileBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = NORM2FN["layer_norm"](config.hidden_size, epsilon=config.layer_norm_eps, name="LayerNorm")

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TFMobileBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.transform = TFMobileBertPredictionHeadTransform(config, name="transform")
        self.vocab_size = config.vocab_size
        self.config = config

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        self.dense = self.add_weight(
            shape=(self.config.hidden_size - self.config.embedding_size, self.vocab_size),
            initializer="zeros",
            trainable=True,
            name="dense/weight",
        )
        self.decoder = self.add_weight(
            shape=(self.config.vocab_size, self.config.embedding_size),
            initializer="zeros",
            trainable=True,
            name="decoder/weight",
        )
        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = tf.matmul(hidden_states, tf.concat([tf.transpose(self.decoder), self.dense], axis=0))
        hidden_states = hidden_states + self.bias
        return hidden_states


class TFMobileBertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.predictions = TFMobileBertLMPredictionHead(config, name="predictions")

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@keras_serializable
class TFMobileBertMainLayer(tf.keras.layers.Layer):
    config_class = MobileBertConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        self.embeddings = TFMobileBertEmbeddings(config, name="embeddings")
        self.encoder = TFMobileBertEncoder(config, name="encoder")
        self.pooler = TFMobileBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(input_shape, 1)

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(input_shape, 0)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = inputs["attention_mask"][:, tf.newaxis, tf.newaxis, :]

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
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(
            inputs["input_ids"],
            inputs["position_ids"],
            inputs["token_type_ids"],
            inputs["inputs_embeds"],
            training=inputs["training"],
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            inputs["head_mask"],
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not inputs["return_dict"]:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFMobileBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    base_model_prefix = "mobilebert"


@dataclass
class TFMobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFMobileBertForPreTraining`.

    Args:
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


MOBILEBERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Parameters:
        config (:class:`~transformers.MobileBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

MOBILEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.MobileBertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.",
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertModel(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mobilebert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return outputs


@add_start_docstrings(
    """
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForPreTraining(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        self.predictions = TFMobileBertMLMHead(config, name="predictions___cls")
        self.seq_relationship = TFMobileBertOnlyNSPHead(2, name="seq_relationship___cls")

    def get_output_embeddings(self):
        return self.predictions.predictions

    def get_output_layer_with_bias(self):
        return self.predictions.predictions

    def get_prefix_bias_name(self):
        return self.name + "/" + self.predictions.name + "/" + self.predictions.predictions.name

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        r"""
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import MobileBertTokenizer, TFMobileBertForPreTraining

            >>> tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
            >>> model = TFMobileBertForPreTraining.from_pretrained('google/mobilebert-uncased')
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
            >>> outputs = model(input_ids)
            >>> prediction_scores, seq_relationship_scores = outputs[:2]

        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mobilebert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)

        if not inputs["return_dict"]:
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        return TFMobileBertForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""MobileBert Model with a `language modeling` head on top. """, MOBILEBERT_START_DOCSTRING)
class TFMobileBertForMaskedLM(TFMobileBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"seq_relationship___cls",
        r"predictions___cls",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        self.mlm = TFMobileBertMLMHead(config, name="mlm___cls")

    def get_output_embeddings(self):
        return self.mlm.predictions

    def get_output_layer_with_bias(self):
        return self.mlm.predictions

    def get_prefix_bias_name(self):
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mobilebert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output, training=inputs["training"])

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], prediction_scores)

        if not inputs["return_dict"]:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TFMobileBertOnlyNSPHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.seq_relationship = tf.keras.layers.Dense(2, name="seq_relationship")

    def call(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


@add_start_docstrings(
    """MobileBert Model with a `next sentence prediction (classification)` head on top. """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForNextSentencePrediction(TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"predictions___cls", r"cls.predictions"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        self.cls = TFMobileBertOnlyNSPHead(config, name="seq_relationship___cls")

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        next_sentence_label=None,
        training=False,
        **kwargs,
    ):
        r"""
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import MobileBertTokenizer, TFMobileBertForNextSentencePrediction

            >>> tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
            >>> model = TFMobileBertForNextSentencePrediction.from_pretrained('google/mobilebert-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

            >>> logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            next_sentence_label=next_sentence_label,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mobilebert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = (
            None
            if inputs["next_sentence_label"] is None
            else self.compute_loss(labels=inputs["next_sentence_label"], logits=seq_relationship_scores)
        )

        if not inputs["return_dict"]:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return TFNextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForSequenceClassification(TFMobileBertPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mobilebert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=inputs["training"])
        logits = self.classifier(pooled_output)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForQuestionAnswering(TFMobileBertPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        training=False,
        **kwargs,
    ):
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_positions=start_positions,
            end_positions=end_positions,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mobilebert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if inputs["start_positions"] is not None and inputs["end_positions"] is not None:
            labels = {"start_position": inputs["start_positions"]}
            labels["end_position"] = inputs["end_positions"]
            loss = self.compute_loss(labels, (start_logits, end_logits))

        if not inputs["return_dict"]:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForMultipleChoice(TFMobileBertPreTrainedModel, TFMultipleChoiceLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mobilebert = TFMobileBertMainLayer(config, name="mobilebert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {"input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)}

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None:
            num_choices = shape_list(inputs["input_ids"])[1]
            seq_length = shape_list(inputs["input_ids"])[2]
        else:
            num_choices = shape_list(inputs["inputs_embeds"])[1]
            seq_length = shape_list(inputs["inputs_embeds"])[2]

        flat_input_ids = tf.reshape(inputs["input_ids"], (-1, seq_length)) if inputs["input_ids"] is not None else None
        flat_attention_mask = (
            tf.reshape(inputs["attention_mask"], (-1, seq_length)) if inputs["attention_mask"] is not None else None
        )
        flat_token_type_ids = (
            tf.reshape(inputs["token_type_ids"], (-1, seq_length)) if inputs["token_type_ids"] is not None else None
        )
        flat_position_ids = (
            tf.reshape(inputs["position_ids"], (-1, seq_length)) if inputs["position_ids"] is not None else None
        )
        flat_inputs_embeds = (
            tf.reshape(inputs["inputs_embeds"], (-1, seq_length, shape_list(inputs["inputs_embeds"])[3]))
            if inputs["inputs_embeds"] is not None
            else None
        )
        outputs = self.mobilebert(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            inputs["head_mask"],
            flat_inputs_embeds,
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=inputs["training"])
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], reshaped_logits)

        if not inputs["return_dict"]:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForTokenClassification(TFMobileBertPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"predictions___cls",
        r"seq_relationship___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mobilebert = TFMobileBertMainLayer(config, add_pooling_layer=False, name="mobilebert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/mobilebert-uncased",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mobilebert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output, training=inputs["training"])
        logits = self.classifier(sequence_output)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
