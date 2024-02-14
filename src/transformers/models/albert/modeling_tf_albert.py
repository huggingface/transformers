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
""" TF 2.0 ALBERT model."""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_albert import AlbertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"

TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    # See all ALBERT models at https://huggingface.co/models?filter=albert
]


class TFAlbertPreTrainingLoss:
    """
    Loss function suitable for ALBERT pretraining, that is, the task of pretraining a language model by combining SOP +
    MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    """

    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        if self.config.tf_legacy_loss:
            # make sure only labels that are not equal to -100
            # are taken into account as loss
            masked_lm_active_loss = tf.not_equal(tf.reshape(tensor=labels["labels"], shape=(-1,)), -100)
            masked_lm_reduced_logits = tf.boolean_mask(
                tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(logits[0])[2])),
                mask=masked_lm_active_loss,
            )
            masked_lm_labels = tf.boolean_mask(
                tensor=tf.reshape(tensor=labels["labels"], shape=(-1,)), mask=masked_lm_active_loss
            )
            sentence_order_active_loss = tf.not_equal(
                tf.reshape(tensor=labels["sentence_order_label"], shape=(-1,)), -100
            )
            sentence_order_reduced_logits = tf.boolean_mask(
                tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=sentence_order_active_loss
            )
            sentence_order_label = tf.boolean_mask(
                tensor=tf.reshape(tensor=labels["sentence_order_label"], shape=(-1,)), mask=sentence_order_active_loss
            )
            masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
            sentence_order_loss = loss_fn(y_true=sentence_order_label, y_pred=sentence_order_reduced_logits)
            masked_lm_loss = tf.reshape(tensor=masked_lm_loss, shape=(-1, shape_list(sentence_order_loss)[0]))
            masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss, axis=0)

            return masked_lm_loss + sentence_order_loss

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels["labels"]), y_pred=logits[0])
        # make sure only labels that are not equal to -100
        # are taken into account for the loss computation
        lm_loss_mask = tf.cast(labels["labels"] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)

        sop_logits = tf.reshape(logits[1], (-1, 2))
        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_sop_loss = loss_fn(y_true=tf.nn.relu(labels["sentence_order_label"]), y_pred=sop_logits)
        sop_loss_mask = tf.cast(labels["sentence_order_label"] != -100, dtype=unmasked_sop_loss.dtype)

        masked_sop_loss = unmasked_sop_loss * sop_loss_mask
        reduced_masked_sop_loss = tf.reduce_sum(masked_sop_loss) / tf.reduce_sum(sop_loss_mask)

        return tf.reshape(reduced_masked_lm_loss + reduced_masked_sop_loss, (1,))


class TFAlbertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = config.embedding_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


class TFAlbertAttention(tf.keras.layers.Layer):
    """Contains the complete attention sublayer, including both dropouts and layer norm."""

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.output_attentions = config.output_attentions

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Two different dropout probabilities; see https://github.com/google-research/albert/blob/master/modeling.py#L971-L993
        self.attention_dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.output_dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(input_tensor)[0]
        mixed_query_layer = self.query(inputs=input_tensor)
        mixed_key_layer = self.key(inputs=input_tensor)
        mixed_value_layer = self.value(inputs=input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFAlbertModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, self.all_head_size))
        self_outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        hidden_states = self_outputs[0]
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.output_dropout(inputs=hidden_states, training=training)
        attention_output = self.LayerNorm(inputs=hidden_states + input_tensor)

        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFAlbertLayer(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFAlbertAttention(config, name="attention")
        self.ffn = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="ffn"
        )

        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        self.ffn_output = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="ffn_output"
        )
        self.full_layer_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="full_layer_layer_norm"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        ffn_output = self.ffn(inputs=attention_outputs[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(inputs=ffn_output)
        ffn_output = self.dropout(inputs=ffn_output, training=training)
        hidden_states = self.full_layer_layer_norm(inputs=ffn_output + attention_outputs[0])

        # add attentions if we output them
        outputs = (hidden_states,) + attention_outputs[1:]

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build([None, None, self.config.hidden_size])
        if getattr(self, "ffn_output", None) is not None:
            with tf.name_scope(self.ffn_output.name):
                self.ffn_output.build([None, None, self.config.intermediate_size])
        if getattr(self, "full_layer_layer_norm", None) is not None:
            with tf.name_scope(self.full_layer_layer_norm.name):
                self.full_layer_layer_norm.build([None, None, self.config.hidden_size])


class TFAlbertLayerGroup(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        self.albert_layers = [
            TFAlbertLayer(config, name=f"albert_layers_._{i}") for i in range(config.inner_group_num)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        layer_hidden_states = () if output_hidden_states else None
        layer_attentions = () if output_attentions else None

        for layer_index, albert_layer in enumerate(self.albert_layers):
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

            layer_output = albert_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[layer_index],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

        # Add last layer
        if output_hidden_states:
            layer_hidden_states = layer_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, layer_hidden_states, layer_attentions] if v is not None)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert_layers", None) is not None:
            for layer in self.albert_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFAlbertTransformer(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        # Number of layers in a hidden group
        self.layers_per_group = int(config.num_hidden_layers / config.num_hidden_groups)
        self.embedding_hidden_mapping_in = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="embedding_hidden_mapping_in",
        )
        self.albert_layer_groups = [
            TFAlbertLayerGroup(config, name=f"albert_layer_groups_._{i}") for i in range(config.num_hidden_groups)
        ]
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        all_attentions = () if output_attentions else None
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for i in range(self.num_hidden_layers):
            # Index of the hidden group
            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))
            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[group_idx * self.layers_per_group : (group_idx + 1) * self.layers_per_group],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                training=training,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embedding_hidden_mapping_in", None) is not None:
            with tf.name_scope(self.embedding_hidden_mapping_in.name):
                self.embedding_hidden_mapping_in.build([None, None, self.config.embedding_size])
        if getattr(self, "albert_layer_groups", None) is not None:
            for layer in self.albert_layer_groups:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFAlbertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AlbertConfig
    base_model_prefix = "albert"


class TFAlbertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = config.embedding_size
        self.dense = tf.keras.layers.Dense(
            config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        self.decoder_bias = self.add_weight(
            shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="decoder/bias"
        )

        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])

    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.decoder

    def set_output_embeddings(self, value: tf.Variable):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias, "decoder_bias": self.decoder_bias}

    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.decoder_bias = value["decoder_bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.decoder_bias)

        return hidden_states


@keras_serializable
class TFAlbertMainLayer(tf.keras.layers.Layer):
    config_class = AlbertConfig

    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFAlbertEmbeddings(config, name="embeddings")
        self.encoder = TFAlbertTransformer(config, name="encoder")
        self.pooler = (
            tf.keras.layers.Dense(
                units=config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                activation="tanh",
                name="pooler",
            )
            if add_pooling_layer
            else None
        )

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            training=training,
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(inputs=sequence_output[:, 0]) if self.pooler is not None else None

        if not return_dict:
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
                self.pooler.build([None, None, self.config.hidden_size])


@dataclass
class TFAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFAlbertForPreTraining`].

    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: tf.Tensor = None
    prediction_logits: tf.Tensor = None
    sop_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


ALBERT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`AlbertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ALBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Albert Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class TFAlbertModel(TFAlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.albert = TFAlbertMainLayer(config, name="albert")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        outputs = self.albert(
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
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)


@add_start_docstrings(
    """
    Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order
    prediction` (classification) head.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForPreTraining(TFAlbertPreTrainedModel, TFAlbertPreTrainingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.albert = TFAlbertMainLayer(config, name="albert")
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name="predictions")
        self.sop_classifier = TFAlbertSOPHead(config, name="sop_classifier")

    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        sentence_order_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFAlbertForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        Return:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFAlbertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        >>> model = TFAlbertForPreTraining.from_pretrained("albert-base-v2")

        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> sop_logits = outputs.sop_logits
        ```"""

        outputs = self.albert(
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
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(hidden_states=sequence_output)
        sop_scores = self.sop_classifier(pooled_output=pooled_output, training=training)
        total_loss = None

        if labels is not None and sentence_order_label is not None:
            d_labels = {"labels": labels}
            d_labels["sentence_order_label"] = sentence_order_label
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, sop_scores))

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TFAlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        if getattr(self, "sop_classifier", None) is not None:
            with tf.name_scope(self.sop_classifier.name):
                self.sop_classifier.build(None)


class TFAlbertSOPHead(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        self.config = config

    def call(self, pooled_output: tf.Tensor, training: bool) -> tf.Tensor:
        dropout_pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=dropout_pooled_output)

        return logits

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings("""Albert Model with a `language modeling` head on top.""", ALBERT_START_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name="predictions")

    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFAlbertForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        >>> model = TFAlbertForMaskedLM.from_pretrained("albert-base-v2")

        >>> # add mask_token
        >>> inputs = tokenizer(f"The capital of [MASK] is Paris.", return_tensors="tf")
        >>> logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = tf.where(inputs.input_ids == tokenizer.mask_token_id)[0][1]
        >>> predicted_token_id = tf.math.argmax(logits[0, mask_token_index], axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        'france'
        ```

        ```python
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
        >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
        >>> outputs = model(**inputs, labels=labels)
        >>> round(float(outputs.loss), 2)
        0.81
        ```
        """
        outputs = self.albert(
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
        )
        sequence_output = outputs[0]
        prediction_scores = self.predictions(hidden_states=sequence_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


@add_start_docstrings(
    """
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForSequenceClassification(TFAlbertPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-imdb",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'LABEL_1'",
        expected_loss=0.12,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.albert(
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
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForTokenClassification(TFAlbertPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")
        classifier_dropout_prob = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(rate=classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.albert(
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
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        logits = self.classifier(inputs=sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForQuestionAnswering(TFAlbertPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=12,
        qa_target_end_index=13,
        expected_output="'a nice puppet'",
        expected_loss=7.36,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        outputs = self.albert(
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
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(inputs=sequence_output)
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None

        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForMultipleChoice(TFAlbertPreTrainedModel, TFMultipleChoiceLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = (
            tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        )
        flat_token_type_ids = (
            tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        )
        flat_position_ids = (
            tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        )
        flat_inputs_embeds = (
            tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        outputs = self.albert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
