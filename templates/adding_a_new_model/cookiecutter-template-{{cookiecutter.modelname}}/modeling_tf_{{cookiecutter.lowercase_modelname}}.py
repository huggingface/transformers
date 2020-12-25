# coding=utf-8
# Copyright {{cookiecutter.authors}} and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 {{cookiecutter.modelname}} model. """

{% if cookiecutter.is_encoder_decoder_model == "False" %}


import tensorflow as tf

from transformers.modeling_tf_outputs import TFCausalLMOutput
from ...activations_tf import get_tf_activation
from ...file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
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
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    TFCausalLanguageModelingLoss,
    TFSequenceSummary,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Config"
_TOKENIZER_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Tokenizer"

TF_{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "{{cookiecutter.checkpoint_identifier}}",
    # See all {{cookiecutter.modelname}} models at https://huggingface.co/models?filter={{cookiecutter.lowercase_modelname}}
]


# Copied from transformers.models.bert.modeling_tf_bert.TFBertEmbeddings with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Embeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

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

        position_embeddings = tf.cast(self.position_embeddings(position_ids), inputs_embeds.dtype)
        token_type_embeddings = tf.cast(self.token_type_embeddings(token_type_ids), inputs_embeds.dtype)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [batch_size, length, hidden_size].

        Returns:
            float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.query = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, config.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, config.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, config.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="value",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

    def call(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, training=False):
        query_layer = self.query(inputs=hidden_states)
        key_layer = self.key(inputs=hidden_states)
        value_layer = self.value(inputs=hidden_states)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        dk = tf.cast(x=self.attention_head_size, dtype=query_layer.dtype)
        query_layer = tf.multiply(x=query_layer, y=tf.math.rsqrt(x=dk))
        attention_scores = tf.einsum("aecd,abcd->acbe", key_layer, query_layer)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TF{{cookiecutter.camelcase_modelname}}Model call() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_scores = attention_scores * head_mask

        attention_output = tf.einsum("acbe,aecd->abcd", attention_probs, value_layer)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs


# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}SelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dense = tf.keras.layers.experimental.EinsumDense(
            equation="abcd,cde->abe",
            output_shape=(None, self.all_head_size),
            bias_axes="e",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Attention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TF{{cookiecutter.camelcase_modelname}}SelfAttention(config, name="self")
        self.dense_output = TF{{cookiecutter.camelcase_modelname}}SelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        self_outputs = self.self_attention(
            input_tensor, attention_mask, head_mask, output_attentions, training=training
        )
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Intermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, config.intermediate_size),
            bias_axes="d",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(activation_string=config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(inputs=hidden_states)

        return hidden_states


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Output(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cd->abd",
            bias_axes="d",
            output_shape=(None, config.hidden_size),
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TF{{cookiecutter.camelcase_modelname}}Layer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TF{{cookiecutter.camelcase_modelname}}Attention(config, name="attention")
        self.intermediate = TF{{cookiecutter.camelcase_modelname}}Intermediate(config, name="intermediate")
        self.{{cookiecutter.lowercase_modelname}}_output = TF{{cookiecutter.camelcase_modelname}}Output(config, name="output")

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer.call with bert->{{cookiecutter.lowercase_modelname}}
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.{{cookiecutter.lowercase_modelname}}_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs

# Copied from transformers.models.bert.modeling_tf_bert.TFBertEncoder with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.layer = [TF{{cookiecutter.camelcase_modelname}}Layer(config, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)]

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


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPredictionHead with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}PredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLMPredictionHead with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}LMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.transform = TF{{cookiecutter.camelcase_modelname}}PredictionHeadTransform(config, name="transform")

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


# Copied from transformers.models.bert.modeling_tf_bert.TFBertMLMHead with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}MLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TF{{cookiecutter.camelcase_modelname}}LMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)

        return prediction_scores


@keras_serializable
class TF{{cookiecutter.camelcase_modelname}}MainLayer(tf.keras.layers.Layer):
    config_class = {{cookiecutter.camelcase_modelname}}Config

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TF{{cookiecutter.camelcase_modelname}}Embeddings(config, name="embeddings")
        self.encoder = TF{{cookiecutter.camelcase_modelname}}Encoder(config, name="encoder")
        self.config = config

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
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

        embedding_output = self.embeddings(
            inputs["input_ids"],
            inputs["position_ids"],
            inputs["token_type_ids"],
            inputs["inputs_embeds"],
            training=inputs["training"],
        )

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
        extended_attention_mask = tf.cast(extended_attention_mask, embedding_output.dtype)
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

        if not return_dict:
            return (
                sequence_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TF{{cookiecutter.camelcase_modelname}}PreTrainedModel(TFPreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = {{cookiecutter.camelcase_modelname}}Config
    base_model_prefix = "{{cookiecutter.lowercase_modelname}}"



{{cookiecutter.uppercase_modelname}}_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass.
    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general
    usage and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.{{cookiecutter.camelcase_modelname}}Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

{{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.{{cookiecutter.camelcase_modelname}}Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.__call__` and
            :func:`transformers.PreTrainedTokenizer.encode` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

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
    "The bare {{cookiecutter.modelname}} Model transformer outputing raw hidden-states without any specific head on top.",
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
class TF{{cookiecutter.camelcase_modelname}}Model(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}MainLayer(config, name="{{cookiecutter.lowercase_modelname}}")

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
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
        outputs = self.{{cookiecutter.lowercase_modelname}}(
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


@add_start_docstrings("""{{cookiecutter.modelname}} Model with a `language modeling` head on top. """, {{cookiecutter.uppercase_modelname}}_START_DOCSTRING)
class TF{{cookiecutter.camelcase_modelname}}ForMaskedLM(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel, TFMaskedLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TF{{cookiecutter.camelcase_modelname}}ForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}MainLayer(config, name="{{cookiecutter.lowercase_modelname}}")
        self.mlm = TF{{cookiecutter.camelcase_modelname}}MLMHead(config, self.{{cookiecutter.lowercase_modelname}}.embeddings, name="mlm___cls")
    
    def get_output_embeddings(self):
        return self.{{cookiecutter.lowercase_modelname}}.embeddings

    def get_output_layer_with_bias(self):
        return self.mlm.predictions

    def get_prefix_bias_name(self):
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
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
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
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
        outputs = self.{{cookiecutter.lowercase_modelname}}(
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
            output = (prediction_scores,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """{{cookiecutter.modelname}} Model with a `language modeling` head on top for CLM fine-tuning. """, {{cookiecutter.uppercase_modelname}}_START_DOCSTRING
)
class TF{{cookiecutter.camelcase_modelname}}ForCausalLM(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if not config.is_decoder:
            logger.warning("If you want to use `TF{{cookiecutter.camelcase_modelname}}ForCausalLM` as a standalone, add `is_decoder=True.`")

        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}MainLayer(config, name="{{cookiecutter.lowercase_modelname}}")
        self.mlm = TF{{cookiecutter.camelcase_modelname}}MLMHead(config, self.{{cookiecutter.lowercase_modelname}}.embeddings, name="mlm___cls")

    def get_output_embeddings(self):
        return self.{{cookiecutter.lowercase_modelname}}.embeddings

    def get_output_layer_with_bias(self):
        return self.mlm.predictions

    def get_prefix_bias_name(self):
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
        output_type=TFCausalLMOutput,
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
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
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
        outputs = self.{{cookiecutter.lowercase_modelname}}(
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
        logits = self.mlm(sequence_output, training=inputs["training"])
        loss = None

        if inputs["labels"] is not None:
            # shift labels to the left and cut last logit token
            logits = logits[:, :-1]
            labels = inputs["labels"][:, 1:]
            loss = self.compute_loss(labels, logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class TF{{cookiecutter.camelcase_modelname}}ClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

        self.config = config

    def call(self, inputs, **kwargs):
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_tf_activation(self.config.hidden_act)(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


@add_start_docstrings(
    """{{cookiecutter.modelname}} Model transformer with a sequence classification/regression head on top 
    e.g., for GLUE tasks. """,
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
class TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}MainLayer(config, name="{{cookiecutter.lowercase_modelname}}")
        self.classifier = TF{{cookiecutter.camelcase_modelname}}ClassificationHead(config, name="classifier")

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
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
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
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
        outputs = self.{{cookiecutter.lowercase_modelname}}(
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
        logits = self.classifier(outputs[0], training=inputs["training"])
        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """{{cookiecutter.modelname}} Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
class TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}MainLayer(config, name="{{cookiecutter.lowercase_modelname}}")
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
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

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
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
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
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
        outputs = self.{{cookiecutter.lowercase_modelname}}(
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
        logits = self.sequence_summary(outputs[0], training=inputs["training"])
        logits = self.classifier(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], reshaped_logits)

        if not inputs["return_dict"]:
            output = (reshaped_logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """{{cookiecutter.modelname}} Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
class TF{{cookiecutter.camelcase_modelname}}ForTokenClassification(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel, TFTokenClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}MainLayer(config, name="{{cookiecutter.lowercase_modelname}}")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
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
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
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
        outputs = self.{{cookiecutter.lowercase_modelname}}(
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
        sequence_output = self.dropout(sequence_output, training=inputs["training"])
        logits = self.classifier(sequence_output)
        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """{{cookiecutter.modelname}} Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
class TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}MainLayer(config, name="{{cookiecutter.lowercase_modelname}}")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
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
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
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
        outputs = self.{{cookiecutter.lowercase_modelname}}(
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
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

{% else %}
import random
from typing import Dict, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPast,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# Public API
from ...modeling_tf_utils import (
    DUMMY_INPUTS,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    TFWrappedEmbeddings,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Config"
_TOKENIZER_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Tokenizer"


LARGE_NEGATIVE = -1e8


def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = tf.cast(input_ids, tf.int32)
    shifted_input_ids = tf.roll(shifted_input_ids, 1, axis=-1)
    start_tokens = tf.fill((shape_list(shifted_input_ids)[0], 1), decoder_start_token_id)
    shifted_input_ids = tf.concat([start_tokens, shifted_input_ids[:, 1:]], -1)
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = tf.where(
        shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
    )

    # "Verify that `labels` has only positive values and -100"
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.cast(0, tf.int32))

    # Make sure the assertion op is called by wrapping the result in an identity no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = tf.ones((tgt_len, tgt_len), dtype=tf.float32) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])

    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)
    mask = tf.cast(mask, tf.float32)

    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length), dtype=tf.float32), mask], axis=-1)
    return tf.broadcast_to(mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length))


def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None, past_key_values_length: int = 0):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = shape_list(mask)
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = tf.cast(tf.broadcast_to(mask[:, None, None, :], (bsz, 1, tgt_len, src_len)), tf.float32)

    return (1.0 - expanded_mask) * LARGE_NEGATIVE


class TF{{cookiecutter.camelcase_modelname}}LearnedPositionalEmbedding(TFSharedEmbeddings):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, **kwargs):
        assert padding_idx is not None, "padding_idx cannot be None"
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_shape[:2]

        positions = tf.range(
            past_key_values_length, seq_len + past_key_values_length, delta=1, dtype=tf.int32, name="range"
        )
        return super().call(positions)


class TF{{cookiecutter.camelcase_modelname}}Attention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        training=False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        tf.debugging.assert_equal(
            shape_list(attn_weights),
            [bsz * self.num_heads, tgt_len, src_len],
            message=f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {shape_list(attn_weights)}",
        )

        if attention_mask is not None:
            tf.debugging.assert_equal(
                shape_list(attention_mask),
                [bsz, 1, tgt_len, src_len],
                message=f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {shape_list(attention_mask)}",
            )
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        attn_probs = self.dropout(attn_weights, training=training)

        attn_output = tf.matmul(attn_probs, value_states)

        tf.debugging.assert_equal(
            shape_list(attn_output),
            [bsz * self.num_heads, tgt_len, self.head_dim],
            message=f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {shape_list(attn_output)}",
        )

        attn_output = tf.transpose(
            tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))

        return attn_output, attn_weights, past_key_value


class TF{{cookiecutter.camelcase_modelname}}EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TF{{cookiecutter.camelcase_modelname}}Attention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, training=False):
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        """
        residual = hidden_states
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, self_attn_weights


class TF{{cookiecutter.camelcase_modelname}}DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TF{{cookiecutter.camelcase_modelname}}Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.encoder_attn = TF{{cookiecutter.camelcase_modelname}}Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(
        self,
        hidden_states,
        attention_mask: Optional[tf.Tensor] = None,
        encoder_hidden_states: Optional[tf.Tensor] = None,
        encoder_attention_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        training=False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`tf.Tensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`tf.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_value (:obj:`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, _, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
            )
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return (
            hidden_states,
            self_attn_weights,
            present_key_value,
        )


class TF{{cookiecutter.camelcase_modelname}}PreTrainedModel(TFPreTrainedModel):
    config_class = {{cookiecutter.camelcase_modelname}}Config
    base_model_prefix = "model"

    @property
    def dummy_inputs(self):
        pad_token = 1
        input_ids = tf.cast(tf.constant(DUMMY_INPUTS), tf.int32)
        decoder_input_ids = tf.cast(tf.constant(DUMMY_INPUTS), tf.int32)
        dummy_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": tf.math.not_equal(input_ids, pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


{{cookiecutter.uppercase_modelname}}_START_DOCSTRING = r"""
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

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(input_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.{{cookiecutter.camelcase_modelname}}Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

{{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
        encoder_outputs (:obj:`tf.FloatTensor`, `optional`):
            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            of shape :obj:`(batch_size, sequence_length, hidden_size)` is a sequence of
        past_key_values (:obj:`Tuple[Tuple[tf.Tensor]]` of length :obj:`config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`). Set to :obj:`False` during training, :obj:`True` during generation
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.TFModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@keras_serializable
class TF{{cookiecutter.camelcase_modelname}}Encoder(tf.keras.layers.Layer):
    config_class = {{cookiecutter.camelcase_modelname}}Config
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TF{{cookiecutter.camelcase_modelname}}EncoderLayer`.

    Args:
        config: {{cookiecutter.camelcase_modelname}}Config
    """

    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_positions = TF{{cookiecutter.camelcase_modelname}}LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
            name="embed_positions",
        )
        self.layers = [TF{{cookiecutter.camelcase_modelname}}EncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        """
        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.{{cookiecutter.camelcase_modelname}}Tokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
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

        if inputs["inputs_embeds"] is None:
            inputs_embeds = self.embed_tokens(inputs["input_ids"])
        else:
            inputs_embeds = inputs["inputs_embeds"]

        inputs_embeds = inputs_embeds

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # check attention mask and invert
        if inputs["attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(inputs["attention_mask"])
        else:
            attention_mask = None

        encoder_states = () if inputs["output_hidden_states"] else None
        all_attentions = () if inputs["output_attentions"] else None

        # encoder layers
        for encoder_layer in self.layers:

            if inputs["output_hidden_states"]:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if inputs["training"] and (dropout_probability < self.layerdrop):  # skip the layer
                continue

            hidden_states, attn = encoder_layer(hidden_states, attention_mask)

            if inputs["output_attentions"]:
                all_attentions += (attn,)

        if inputs["output_hidden_states"]:
            encoder_states = encoder_states + (hidden_states,)

        if not inputs["return_dict"]:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


@keras_serializable
class TF{{cookiecutter.camelcase_modelname}}Decoder(tf.keras.layers.Layer):
    config_class = {{cookiecutter.camelcase_modelname}}Config
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TF{{cookiecutter.camelcase_modelname}}DecoderLayer`

    Args:
        config: {{cookiecutter.camelcase_modelname}}Config
        embed_tokens: output embedding
    """

    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TF{{cookiecutter.camelcase_modelname}}LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
            name="embed_positions",
        )
        self.layers = [TF{{cookiecutter.camelcase_modelname}}DecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        r"""
        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.{{cookiecutter.camelcase_modelname}}Tokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`tf.Tensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[tf.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = (
            inputs["past_key_values"][0][0].shape[2] if inputs["past_key_values"] is not None else 0
        )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"])

        hidden_states = inputs["inputs_embeds"]

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(
                tf.ones((input_shape[0], input_shape[1] + past_key_values_length)), tgt_len=input_shape[-1]
            )

        if inputs["attention_mask"] is None and inputs["input_ids"] is not None and input_shape[-1] > 1:
            inputs["attention_mask"] = tf.cast(
                tf.math.not_equal(inputs["input_ids"], self.config.pad_token_id), inputs["input_ids"].dtype
            )
            inputs["attention_mask"] = tf.concat(
                [
                    tf.ones((input_shape[0], past_key_values_length), dtype=inputs["attention_mask"].dtype),
                    inputs["attention_mask"],
                ],
                axis=-1,
            )
        else:
            inputs["attention_mask"] = tf.ones(
                (input_shape[0], input_shape[1] + past_key_values_length), dtype=tf.int32
            )

        if inputs["encoder_hidden_states"] is not None and inputs["encoder_attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            inputs["encoder_attention_mask"] = _expand_mask(inputs["encoder_attention_mask"], tgt_len=input_shape[-1])

        hidden_states = self.layernorm_embedding(hidden_states + positions)
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        present_key_values = ()
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if inputs["output_hidden_states"]:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)

            if inputs["training"] and (dropout_probability < self.layerdrop):
                continue

            past_key_value = inputs["past_key_values"][idx] if inputs["past_key_values"] is not None else None

            hidden_states, layer_self_attn, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=inputs["encoder_hidden_states"],
                encoder_attention_mask=inputs["encoder_attention_mask"],
                past_key_value=past_key_value,
            )

            if inputs["use_cache"]:
                present_key_values += (present_key_value,)

            if inputs["output_attentions"]:
                all_self_attns += (layer_self_attn,)

        if inputs["output_hidden_states"]:
            all_hidden_states += (hidden_states,)
        else:
            all_hidden_states = None

        all_self_attns = list(all_self_attns) if inputs["output_attentions"] else None

        present_key_values = (encoder_hidden_states, present_key_values) if inputs["use_cache"] else None

        if not inputs["return_dict"]:
            return hidden_states, present_key_values, all_hidden_states, all_self_attns
        else:
            return TFBaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=present_key_values,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


@add_start_docstrings(
    "The bare {{cookiecutter.uppercase_modelname}} Model outputting raw hidden-states without any specific head on top.",
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
@keras_serializable
class TF{{cookiecutter.camelcase_modelname}}Model(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, config.pad_token_id, name="model.shared")

        with tf.compat.v1.variable_scope("model.shared") as shared_abs_scope_name:
            pass

        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        embed_tokens.vocab_size = self.shared.vocab_size
        embed_tokens.hidden_size = self.shared.hidden_size

        self.encoder = TF{{cookiecutter.camelcase_modelname}}Encoder(config, embed_tokens, name="encoder")
        self.decoder = TF{{cookiecutter.camelcase_modelname}}Decoder(config, embed_tokens, name="decoder")

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["decoder_input_ids"] is None and inputs["decoder_inputs_embeds"] is None:
            inputs["use_cache"] = False

        inputs["output_hidden_states"] = (
            inputs["output_hidden_states"]
            if inputs["output_hidden_states"] is not None
            else self.config.output_hidden_states
        )

        if inputs["encoder_outputs"] is None:
            inputs["encoder_outputs"] = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                inputs_embeds=inputs["inputs_embeds"],
                output_attentions=inputs["output_attentions"],
                output_hidden_states=inputs["output_hidden_states"],
                return_dict=inputs["return_dict"],
                training=inputs["training"],
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a TFBaseModelOutput when return_dict=True
        elif inputs["return_dict"] and not isinstance(inputs["encoder_outputs"], TFBaseModelOutput):
            inputs["encoder_outputs"] = TFBaseModelOutput(
                last_hidden_state=inputs["encoder_outputs"][0],
                hidden_states=inputs["encoder_outputs"][1] if len(inputs["encoder_outputs"]) > 1 else None,
                attentions=inputs["encoder_outputs"][2] if len(inputs["encoder_outputs"]) > 2 else None,
            )
        # If the user passed a TFBaseModelOutput for encoder_outputs, we wrap it in a tuple when return_dict=False
        elif not inputs["return_dict"] and not isinstance(inputs["encoder_outputs"], tuple):
            inputs["encoder_outputs"] = inputs["encoder_outputs"].to_tuple()

        decoder_outputs = self.decoder(
            inputs["decoder_input_ids"],
            attention_mask=inputs["decoder_attention_mask"],
            encoder_hidden_states=inputs["encoder_outputs"][0],
            encoder_attention_mask=inputs["attention_mask"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        if not inputs["return_dict"]:
            return decoder_outputs + inputs["encoder_outputs"]

        return TFSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=inputs["encoder_outputs"].last_hidden_state,
            encoder_hidden_states=inputs["encoder_outputs"].hidden_states,
            encoder_attentions=inputs["encoder_outputs"].attentions,
        )

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def get_output_embeddings(self):
        return self.shared


@add_start_docstrings(
    "The {{cookiecutter.uppercase_modelname}} Model with a language modeling head. Can be used for summarization.",
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
class TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration(TF{{cookiecutter.camelcase_modelname}}PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TF{{cookiecutter.camelcase_modelname}}Model(config, name="model")
        self.use_cache = config.use_cache
        # final_bias_logits is registered as a buffer in pytorch, so not trainable for the the sake of consistency.
        self.final_logits_bias = self.add_weight(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        return self.model.decoder

    def resize_token_embeddings(self, new_num_tokens):
        super().resize_token_embeddings(new_num_tokens=new_num_tokens)

        # {{cookiecutter.uppercase_modelname}} is a special case where the bias has two dimensions
        # and not named just `bias`
        if new_num_tokens is not None:
            num_tokens_to_copy = min(self.final_logits_bias.shape[0], new_num_tokens)
            init_bias = tf.zeros((new_num_tokens,))
            init_bias[:num_tokens_to_copy] = self.final_logits_bias.value()[:num_tokens_to_copy]
            self.final_logits_bias = self.add_weight(
                shape=(1, new_num_tokens),
                initializer="zeros",
                trainable=False,
                name="final_logits_bias",
            )
            self.final_logits_bias.assign(init_bias)

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        """
        Returns:

        Examples::

            >>> from transformers import {{cookiecutter.camelcase_modelname}}Tokenizer, TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration
            >>> import tensorflow as tf
            >>> mname = '{{cookiecutter.checkpoint_identifier}}'
            >>> tokenizer = {{cookiecutter.camelcase_modelname}}Tokenizer.from_pretrained(mname)
            >>> TXT = "My friends are <mask> but they eat too many carbs."
            >>> model = TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration.from_pretrained(mname)
            >>> batch = tokenizer([TXT], return_tensors='tf')
            >>> logits = model(inputs=batch.input_ids).logits
            >>> probs = tf.nn.softmax(logits[0])
            >>> # probs[5] is associated with the mask token
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["labels"] is not None:
            inputs["use_cache"] = False
            if inputs["decoder_input_ids"] is None:
                inputs["decoder_input_ids"] = shift_tokens_right(
                    inputs["labels"], self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            encoder_outputs=inputs["encoder_outputs"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["inputs_embeds"],
            decoder_inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"]
        )
        lm_logits = self.model.shared(outputs[0], mode="linear")
        lm_logits = lm_logits + self.final_logits_bias
        masked_lm_loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], lm_logits)

        if not inputs["return_dict"]:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            encoder_last_hidden_state=outputs.last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
        )

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs) -> Dict:
        assert past is not None and len(past) in {1, 2}, f"past has to be an iterable of length 1,2 got {past}"
        if len(past) == 1:
            assert isinstance(past[0], tf.Tensor), f"`past[0]` has to be of type `tf.Tensor`, but is {type(past[0])}"
            encoder_outputs = TFBaseModelOutput(last_hidden_state=past[0])
            past_key_values = None
        else:
            assert (
                len(past) == 2
            ), "`past` has to be of length 2 with the encoder_outputs at the first position and past_key_values at the second position."
            encoder_outputs, past_key_values = past
            if isinstance(encoder_outputs, tuple):
                assert isinstance(
                    encoder_outputs[0], tf.Tensor
                ), f"`encoder_outputs[0]` has to be of type `tf.Tensor`, but is {type(encoder_outputs[0])}"
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs[0])
            elif isinstance(encoder_outputs, tf.Tensor):
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs)
            assert (
                past_key_values
            ), f"decoder cached states must be truthy. got {past_key_values} from the 2nd element of past"
            decoder_input_ids = decoder_input_ids[:, -1:]

        assert isinstance(
            encoder_outputs, TFBaseModelOutput
        ), f"encoder_outputs should be a TFBaseModelOutput, Instead got {type(encoder_outputs)}."
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        if len(past) == 1:
            return past

        past_key_values = past[1]

        reordered_past = ()
        for layer_past_key_values in past_key_values:
            reordered_past += (
                tuple(tf.gather(layer_past_key_value, beam_idx) for layer_past_key_value in layer_past_key_values),
            )
        return (past[0], reordered_past)

    def get_output_embeddings(self):
        return self.model.shared

    def get_encoder(self):
        return self.model.encoder

    def compute_loss(self, labels, logits):
        """CrossEntropyLoss that ignores pad tokens"""
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )
        melted_labels = tf.reshape(labels, (-1,))
        active_loss = tf.not_equal(melted_labels, self.config.pad_token_id)
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(melted_labels, active_loss)
        return loss_fn(labels, reduced_logits)
{% endif -%}
