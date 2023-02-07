# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
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
""" TF 2.0 CTRL model."""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFSharedEmbeddings,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "ctrl"
_CONFIG_FOR_DOC = "CTRLConfig"

TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ctrl"
    # See all CTRL models at https://huggingface.co/models?filter=ctrl
]


def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates


def positional_encoding(position, d_model_size):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(d_model_size)[np.newaxis, :], d_model_size)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.convert_to_tensor(np.concatenate([sines, cosines], axis=-1))

    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    # calculate attention
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(shape_list(k)[-1], dtype=matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += tf.cast(mask * -1e4, dtype=scaled_attention_logits.dtype)

    if attention_mask is not None:
        # Apply the attention mask
        attention_mask = tf.cast(attention_mask, dtype=scaled_attention_logits.dtype)
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = stable_softmax(scaled_attention_logits, axis=-1)

    # Mask heads if we want to
    if head_mask is not None:
        attention_weights = attention_weights * head_mask

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class TFMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.output_attentions = output_attentions

        self.depth = int(d_model_size / self.num_heads)

        self.Wq = tf.keras.layers.Dense(d_model_size, name="Wq")
        self.Wk = tf.keras.layers.Dense(d_model_size, name="Wk")
        self.Wv = tf.keras.layers.Dense(d_model_size, name="Wv")

        self.dense = tf.keras.layers.Dense(d_model_size, name="dense")

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        batch_size = shape_list(q)[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)

        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            k = tf.concat((past_key, k), axis=-2)
            v = tf.concat((past_value, v), axis=-2)

        if use_cache:
            present = tf.stack((k, v), axis=0)
        else:
            present = (None,)

        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = tf.transpose(output[0], perm=[0, 2, 1, 3])
        attn = output[1]
        original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model_size))
        output = self.dense(original_size_attention)
        outputs = (output, present)

        if output_attentions:
            outputs = outputs + (attn,)

        return outputs


class TFPointWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_size, dff, **kwargs):
        super().__init__(**kwargs)

        self.dense_0 = tf.keras.layers.Dense(dff, activation="relu", name="0")
        self.dense_2 = tf.keras.layers.Dense(d_model_size, name="2")

    def call(self, inputs, trainable=False):
        dense_0_output = self.dense_0(inputs)
        dense_2_output = self.dense_2(dense_0_output)

        return dense_2_output


class TFEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model_size, num_heads, dff, rate=0.1, layer_norm_epsilon=1e-6, output_attentions=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.output_attentions = output_attentions

        self.multi_head_attention = TFMultiHeadAttention(
            d_model_size, num_heads, output_attentions=self.output_attentions, name="multi_head_attention"
        )
        self.ffn = TFPointWiseFeedForwardLayer(d_model_size, dff, name="ffn")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm2")

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
            training=training,
        )
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        outputs = (out2,) + attn_outputs[1:]
        return outputs


@keras_serializable
class TFCTRLMainLayer(tf.keras.layers.Layer):
    config_class = CTRLConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size)

        self.w = TFSharedEmbeddings(
            config.vocab_size, config.n_embd, initializer_range=config.initializer_range, name="w"
        )

        self.dropout = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [
            TFEncoderLayer(
                config.n_embd,
                config.n_head,
                config.dff,
                config.resid_pdrop,
                config.layer_norm_epsilon,
                self.output_attentions,
                name=f"h_._{i}",
            )
            for i in range(config.n_layer)
        ]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layernorm")

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, value):
        self.w.weight = value
        self.w.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        # If using past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = shape_list(past_key_values[0][0])[-2]
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32), axis=0)
            position_ids = tf.tile(position_ids, [input_shape[0], 1])

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.

            one_cst = tf.constant(1.0)
            ten_thousand_cst = tf.constant(-10000.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(tf.subtract(one_cst, attention_mask), ten_thousand_cst)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_layers

        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.w(token_type_ids, mode="embedding")
            token_type_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, dtype=token_type_embeds.dtype))
        else:
            token_type_embeds = tf.constant(0.0)
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
            # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
            tf.debugging.assert_less(
                input_ids,
                tf.cast(self.w.vocab_size, dtype=input_ids.dtype),
                message=(
                    "input_ids must be smaller than the embedding layer's input dimension (got"
                    f" {tf.math.reduce_max(input_ids)} >= {self.w.vocab_size})"
                ),
            )
            inputs_embeds = self.w(input_ids, mode="embedding")
        seq_len = input_shape[-1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        inputs_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, inputs_embeds.dtype))

        pos_embeds = tf.gather(self.pos_encoding, position_ids)
        pos_embeds = tf.cast(pos_embeds, dtype=token_type_embeds.dtype)
        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, (h, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)
            outputs = h(
                hidden_states,
                mask,
                layer_past,
                attention_mask,
                head_mask[i],
                use_cache,
                output_attentions,
                training=training,
            )
            hidden_states, present = outputs[:2]

            if use_cache:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.layernorm(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CTRLConfig
    base_model_prefix = "transformer"


CTRL_START_DOCSTRING = r"""

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

    Parameters:
        config ([`CTRLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CTRL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past` is `None` else `past[0].shape[-2]` (`sequence_length` of
            input past key value states).

            Indices of input sequence tokens in the vocabulary.

            If `past` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        past (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past` output below). Can be used to speed up sequential decoding. The token ids which have their past
            given to this model should not be passed as input ids as they have already been computed.
        attention_mask (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past` key value states are returned and can be used to speed up decoding (see `past`).
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
    "The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
    CTRL_START_DOCSTRING,
)
class TFCTRLModel(TFCTRLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFCTRLMainLayer(config, name="transformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values) if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPast(
            last_hidden_state=output.last_hidden_state, past_key_values=pkv, hidden_states=hs, attentions=attns
        )


class TFCTRLLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # CTRL has numerical issues in XLA generate
        self.supports_xla_generation = False

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def get_output_embeddings(self):
        return self.input_embeddings

    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@add_start_docstrings(
    """
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    CTRL_START_DOCSTRING,
)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFCTRLMainLayer(config, name="transformer")

        self.lm_head = TFCTRLLMHead(config, self.transformer.w, name="lm_head")
        # CTRL has numerical issues in XLA generate
        self.supports_xla_generation = False

    def get_lm_head(self):
        return self.lm_head

    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = tf.expand_dims(input_ids[:, -1], -1)

        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": use_cache}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFCausalLMOutputWithPast]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        hidden_states = transformer_outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # shift labels to the left and cut last logit token
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.hf_compute_loss(labels, shifted_logits)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values) if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFCausalLMOutputWithPast(logits=output.logits, past_key_values=pkv, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    """
    The CTRL Model transformer with a sequence classification head on top (linear layer).

    [`TFCTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1, GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    CTRL_START_DOCSTRING,
)
class TFCTRLForSequenceClassification(TFCTRLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
            use_bias=False,
        )
        self.transformer = TFCTRLMainLayer(config, name="transformer")

    def get_output_embeddings(self):
        return self.transformer.w

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
        in_logits = None
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    tf.reduce_sum(
                        tf.cast(
                            tf.math.not_equal(input_ids, self.config.pad_token_id),
                            dtype=input_ids.dtype,
                        ),
                        -1,
                        keepdims=False,
                    )
                    - 1
                )
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        loss = None

        if labels is not None:
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))

        pooled_logits = in_logits if in_logits is not None else logits

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification.serving_output
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)
