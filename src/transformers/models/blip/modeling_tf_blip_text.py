# coding=utf-8
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the BSD-3-clause license (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import math

import tensorflow as tf

from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    get_tf_activation,
    keras,
    keras_serializable,
    shape_list,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
from ...utils import add_start_docstrings_to_model_forward, logging
from .configuration_blip import BlipTextConfig


logger = logging.get_logger(__name__)

BLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L52
class TFBlipTextEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        self.position_embeddings = keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )

        # self.LayerNorm is not snake-cased to stick with PyTorch model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

        self.position_ids = tf.expand_dims(tf.range(config.max_position_embeddings), 0)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config

    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, training=None):
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "word_embeddings", None) is not None:
            with tf.name_scope(self.word_embeddings.name):
                self.word_embeddings.build(None)
        if getattr(self, "position_embeddings", None) is not None:
            with tf.name_scope(self.position_embeddings.name):
                self.position_embeddings.build(None)
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L97
class TFBlipTextSelfAttention(keras.layers.Layer):
    def __init__(self, config, is_cross_attention, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = keras.layers.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.is_cross_attention = is_cross_attention

    def transpose_for_scores(self, x):
        new_x_shape = tf.concat(
            [tf.shape(x)[:-1], tf.constant([self.num_attention_heads, self.attention_head_size], dtype=tf.int32)],
            axis=0,
        )
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = shape_list(hidden_states)[1]
            position_ids_l = tf.expand_dims(tf.range(seq_length, dtype=tf.int64, device=hidden_states.device), 1)
            position_ids_r = tf.expand_dims(tf.range(seq_length, dtype=tf.int64, device=hidden_states.device), 0)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = tf.cast(positional_embedding, query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = tf.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = tf.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = tf.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BlipTextModel forward() function)
            attention_scores = attention_scores + tf.cast(attention_mask, attention_scores.dtype)

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = attention_probs_dropped @ value_layer

        context_layer = tf.transpose(context_layer, perm=(0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.all_head_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if self.is_cross_attention:
            if getattr(self, "key", None) is not None:
                with tf.name_scope(self.key.name):
                    self.key.build([None, None, self.config.encoder_hidden_size])
            if getattr(self, "value", None) is not None:
                with tf.name_scope(self.value.name):
                    self.value.build([None, None, self.config.encoder_hidden_size])
        else:
            if getattr(self, "key", None) is not None:
                with tf.name_scope(self.key.name):
                    self.key.build([None, None, self.config.hidden_size])
            if getattr(self, "value", None) is not None:
                with tf.name_scope(self.value.name):
                    self.value.build([None, None, self.config.hidden_size])


class TFBlipTextSelfOutput(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#242
class TFBlipTextAttention(keras.layers.Layer):
    def __init__(self, config, is_cross_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.self = TFBlipTextSelfAttention(config, is_cross_attention, name="self")
        # "output" is a protected attribute on TF models
        self.self_output = TFBlipTextSelfOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        past_key_value: tuple[tuple[tf.Tensor]] | None = None,
        output_attentions: bool | None = False,
        training: bool | None = None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            training=training,
        )
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)


# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->BlipText
class TFBlipTextIntermediate(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFBlipTextOutput(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFBlipTextLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.attention = TFBlipTextAttention(config, name="attention")
        if self.config.is_decoder:
            self.crossattention = TFBlipTextAttention(
                config, is_cross_attention=self.config.is_decoder, name="crossattention"
            )
        self.intermediate = TFBlipTextIntermediate(config, name="intermediate")
        self.self_output = TFBlipTextOutput(config, name="output")

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.self_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L386
@keras_serializable
class TFBlipTextEncoder(keras.layers.Layer):
    config_class = BlipTextConfig

    def __init__(self, config, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.layer = [TFBlipTextLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    @unpack_inputs
    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                training=training,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->BlipText
class TFBlipTextPooler(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform with Bert->BlipText
class TFBlipTextPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFBlipTextLMPredictionHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.transform = TFBlipTextPredictionHeadTransform(config, name="transform")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = keras.layers.Dense(
            config.vocab_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="decoder",
            use_bias=False,
        )
        self.config = config

    def build(self, input_shape=None):
        self.bias = self.add_weight(name="bias", shape=(self.config.vocab_size,), initializer="zeros", trainable=True)

        if self.built:
            return
        self.built = True
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build([None, None, self.config.hidden_size])

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class TFBlipTextOnlyMLMHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.predictions = TFBlipTextLMPredictionHead(config, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L548
class TFBlipTextPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BlipTextConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


# Adapted from https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/med.py#L571
class TFBlipTextModel(TFBlipTextPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, name=None, **kwargs):
        super().__init__(config, name=name, **kwargs)
        self.config = config

        self.embeddings = TFBlipTextEmbeddings(config, name="embeddings")
        self.encoder = TFBlipTextEncoder(config, name="encoder")
        self.pooler = TFBlipTextPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @tf.function
    def get_extended_attention_mask(
        self, attention_mask: tf.Tensor, input_shape: tuple[int], is_decoder: bool
    ) -> tf.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`tf.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`tuple[int]`):
                The shape of the input to the model.
            is_decoder (`bool`):
                Whether the model is used as a decoder.

        Returns:
            `tf.Tensor` The extended attention mask, with the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if not isinstance(attention_mask, tf.Tensor):
            attention_mask = tf.convert_to_tensor(attention_mask)  # Catches NumPy inputs that haven't been cast yet
        if attention_mask.shape.rank == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.shape.rank == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = tf.range(seq_length, dtype=attention_mask.dtype)
                causal_mask = tf.broadcast_to(seq_ids, (batch_size, seq_length, seq_length)) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask

                if shape_list(causal_mask)[1] < shape_list(attention_mask)[1]:
                    prefix_seq_len = tf.shape(attention_mask)[1] - tf.shape(causal_mask)[1]
                    causal_mask = tf.concat(
                        [
                            tf.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (
                    tf.cast(causal_mask[:, None, :, :], attention_mask.dtype) * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        encoder_embeds: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        past_key_values: tuple[tuple[tf.Tensor]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        is_decoder: bool = False,
        training: bool = False,
    ) -> tuple[tf.Tensor] | TFBaseModelOutputWithPoolingAndCrossAttentions:
        r"""
        encoder_hidden_states  (`tf.Tensor`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
            batch_size, seq_length = input_shape
        elif encoder_embeds is not None:
            input_shape = shape_list(encoder_embeds)[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = tf.ones((batch_size, seq_length + past_key_values_length))

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: tf.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = shape_list(encoder_hidden_states[0])
            else:
                encoder_batch_size, encoder_sequence_length, _ = shape_list(encoder_hidden_states)
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = tf.ones(encoder_hidden_shape)
                encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L811
class TFBlipTextLMHeadModel(TFBlipTextPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.bert = TFBlipTextModel(config, add_pooling_layer=False, name="bert")
        self.cls = TFBlipTextOnlyMLMHead(config, name="cls")
        self.label_smoothing = config.label_smoothing

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        training=None,
    ):
        r"""
        encoder_hidden_states (`tf.Tensor`, *optional*): Sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is
            configured as a decoder.
        encoder_attention_mask (`tf.Tensor`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`tf.Tensor`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            training=training,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :]

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            shifted_prediction_scores = tf.reshape(shifted_prediction_scores, (-1, self.config.vocab_size))
            labels = labels[:, 1:]
            labels = tf.reshape(labels, (-1,))
            # Keras won't give us label smoothing for sparse CE, so we de-sparsify things here
            # Use relu to clamp masked labels at 0 to avoid NaN (we will be zeroing those out later anyway)
            one_hot_labels = tf.one_hot(tf.nn.relu(labels), depth=self.config.vocab_size, dtype=tf.float32)
            loss_fct = keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=self.label_smoothing, reduction="none"
            )
            masked_positions = tf.cast(tf.not_equal(labels, -100), dtype=tf.float32)
            lm_loss = loss_fct(one_hot_labels, shifted_prediction_scores)
            lm_loss *= masked_positions
            lm_loss = tf.reduce_sum(lm_loss, axis=0) / tf.math.count_nonzero(masked_positions, dtype=tf.float32)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return TFCausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states"),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask"),
            "is_decoder": True,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)


__all__ = ["TFBlipTextLMHeadModel", "TFBlipTextModel", "TFBlipTextPreTrainedModel"]
