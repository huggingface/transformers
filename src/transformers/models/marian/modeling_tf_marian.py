# coding=utf-8
# Copyright 2021 The Marian Team Authors and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 Marian model. """


import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
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
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    TFWrappedEmbeddings,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_marian import MarianConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MarianConfig"
_TOKENIZER_FOR_DOC = "MarianTokenizer"


LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart.shift_tokens_right
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


# Copied from transformers.models.bart.modeling_tf_bart._make_causal_mask
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


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None, past_key_values_length: int = 0):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = shape_list(mask)
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = tf.cast(tf.broadcast_to(mask[:, None, None, :], (bsz, 1, tgt_len, src_len)), tf.float32)

    return (1.0 - expanded_mask) * LARGE_NEGATIVE


class TFMarianSinusoidalPositionalEmbedding(tf.keras.layers.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):

        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        super().__init__(
            num_positions,
            embedding_dim,
            **kwargs,
        )

    def build(self, input_shape: tf.TensorShape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        super().build(input_shape)  # Instantiates self.weight so it can be loaded
        weight: np.ndarray = self._init_weight(self.input_dim, self.output_dim)
        self.set_weights([weight])  # overwrite self.weight to correct value

    @staticmethod
    def _init_weight(n_pos: int, dim: int):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # index 0 is all zero
        position_enc[:, 0 : dim // 2] = np.sin(position_enc[:, 0::2])
        position_enc[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
        # convert to tensor
        table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        tf.stop_gradient(table)
        return table

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_shape[:2]

        positions = tf.range(
            past_key_values_length, seq_len + past_key_values_length, delta=1, dtype=tf.int32, name="range"
        )
        return super().call(positions)


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with Bart->Marian
class TFMarianAttention(tf.keras.layers.Layer):
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
        layer_head_mask: Optional[tf.Tensor] = None,
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

        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [self.num_heads],
                message=f"Head mask for a single layer should be of size {(self.num_heads)}, but is {shape_list(layer_head_mask)}",
            )
            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                attn_weights, (bsz, self.num_heads, tgt_len, src_len)
            )
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

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


# Copied from transformers.models.bart.modeling_tf_bart.TFBartEncoderLayer with Bart->Marian
class TFMarianEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: MarianConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFMarianAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, training=False):
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        residual = hidden_states
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
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


# Copied from transformers.models.bart.modeling_tf_bart.TFBartDecoderLayer with Bart->Marian
class TFMarianDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: MarianConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFMarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.encoder_attn = TFMarianAttention(
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
        layer_head_mask: Optional[tf.Tensor] = None,
        encoder_layer_head_mask: Optional[tf.Tensor] = None,
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
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(decoder_attention_heads,)`
            encoder_layer_head_mask (:obj:`tf.Tensor`): mask for encoder attention heads in a given layer of size
                `(encoder_attention_heads,)`
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
            layer_head_mask=layer_head_mask,
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
                layer_head_mask=encoder_layer_head_mask,
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


class TFMarianPreTrainedModel(TFPreTrainedModel):
    config_class = MarianConfig
    base_model_prefix = "model"

    @property
    def dummy_inputs(self):
        pad_token = 1
        input_ids = tf.cast(tf.convert_to_tensor(DUMMY_INPUTS), tf.int32)
        decoder_input_ids = tf.cast(tf.convert_to_tensor(DUMMY_INPUTS), tf.int32)
        dummy_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": tf.math.not_equal(input_ids, pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
                "decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),
                "decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask"),
            }
        ]
    )
    # Copied from transformers.models.bart.modeling_tf_bart.TFBartPretrainedModel.serving
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)


MARIAN_START_DOCSTRING = r"""
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
        config (:class:`~transformers.MarianConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

MARIAN_GENERATION_EXAMPLE = r"""
        TF version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints. Available
        models are listed `here <https://huggingface.co/models?search=Helsinki-NLP>`__.

        Examples::

            >>> from transformers import MarianTokenizer, TFMarianMTModel
            >>> from typing import List
            >>> src = 'fr'  # source language
            >>> trg = 'en'  # target language
            >>> sample_text = "où est l'arrêt de bus ?"
            >>> mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'

            >>> model = MarianMTModel.from_pretrained(mname)
            >>> tok = MarianTokenizer.from_pretrained(mname)
            >>> batch = tok.prepare_seq2seq_batch(src_texts=[sample_text], return_tensors="tf")  # don't need tgt_text for inference
            >>> gen = model.generate(**batch)
            >>> words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)  # returns "Where is the bus stop ?"
"""

MARIAN_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.MarianTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.MarianTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            Marian uses the :obj:`pad_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
        decoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
        head_mask (:obj:`tf.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

        decoder_head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@keras_serializable
class TFMarianEncoder(tf.keras.layers.Layer):
    config_class = MarianConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TFMarianEncoderLayer`.

    Args:
        config: MarianConfig
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens
        self.embed_positions = TFMarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFMarianEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
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

                Indices can be obtained using :class:`~transformers.MarianTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`tf.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value
                in the config will be used instead.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail. This argument can be used only in eager mode, in graph mode the value in the config
                will be used instead.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
                argument can be used in eager mode, in graph mode the value will always be set to True.
            training (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use the model in training mode (some modules like dropout modules have different
                behaviors between training and evaluation).
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
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

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"]) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs["inputs_embeds"] + embed_pos
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # check attention mask and invert
        if inputs["attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(inputs["attention_mask"])
        else:
            attention_mask = None

        encoder_states = () if inputs["output_hidden_states"] else None
        all_attentions = () if inputs["output_attentions"] else None

        # check if head_mask has a correct number of layers specified if desired
        if inputs["head_mask"] is not None:
            tf.debugging.assert_equal(
                shape_list(inputs["head_mask"])[0],
                len(self.layers),
                message=f"The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(inputs['head_mask'])[0]}.",
            )
        # encoder layers
        for idx, encoder_layer in enumerate(self.layers):

            if inputs["output_hidden_states"]:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if inputs["training"] and (dropout_probability < self.layerdrop):  # skip the layer
                continue

            hidden_states, attn = encoder_layer(
                hidden_states,
                attention_mask,
                inputs["head_mask"][idx] if inputs["head_mask"] is not None else None,
            )

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
class TFMarianDecoder(tf.keras.layers.Layer):
    config_class = MarianConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TFMarianDecoderLayer`

    Args:
        config: MarianConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFMarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.layers = [TFMarianDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]

        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
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

                Indices can be obtained using :class:`~transformers.MarianTokenizer`. See
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
            head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`tf.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

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
                returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value
                in the config will be used instead.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail. This argument can be used only in eager mode, in graph mode the value in the config
                will be used instead.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
                argument can be used in eager mode, in graph mode the value will always be set to True.
            training (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use the model in training mode (some modules like dropout modules have different
                behaviors between training and evaluation).
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            encoder_head_mask=encoder_head_mask,
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
            shape_list(inputs["past_key_values"][0][0])[2] if inputs["past_key_values"] is not None else 0
        )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"]) * self.embed_scale

        hidden_states = inputs["inputs_embeds"]

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(
                tf.ones((input_shape[0], input_shape[1] + past_key_values_length)), tgt_len=input_shape[-1]
            )

        if inputs["attention_mask"] is not None:
            combined_attention_mask = combined_attention_mask + _expand_mask(
                inputs["attention_mask"], tgt_len=input_shape[-1]
            )

        if inputs["encoder_hidden_states"] is not None and inputs["encoder_attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            inputs["encoder_attention_mask"] = _expand_mask(inputs["encoder_attention_mask"], tgt_len=input_shape[-1])

        hidden_states = self.dropout(hidden_states + positions, training=inputs["training"])

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        present_key_values = ()

        # check if head_mask has a correct number of layers specified if desired
        if inputs["head_mask"] is not None:
            tf.debugging.assert_equal(
                shape_list(inputs["head_mask"])[0],
                len(self.layers),
                message=f"The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(inputs['head_mask'])[0]}.",
            )
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
                layer_head_mask=inputs["head_mask"][idx] if inputs["head_mask"] is not None else None,
                encoder_layer_head_mask=inputs["encoder_head_mask"][idx]
                if inputs["encoder_head_mask"] is not None
                else None,
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


@keras_serializable
class TFMarianMainLayer(tf.keras.layers.Layer):
    config_class = MarianConfig

    def __init__(self, config: MarianConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, config.pad_token_id, name="model.shared")

        with tf.compat.v1.variable_scope("model.shared") as shared_abs_scope_name:
            pass

        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        embed_tokens.vocab_size = self.shared.vocab_size
        embed_tokens.hidden_size = self.shared.hidden_size

        self.encoder = TFMarianEncoder(config, embed_tokens, name="encoder")
        self.decoder = TFMarianDecoder(config, embed_tokens, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared.weight = new_embeddings
        self.shared.vocab_size = self.shared.weight.shape[0]
        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("model.shared") as shared_abs_scope_name:
            pass
        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        self.encoder.set_embed_tokens(embed_tokens)
        self.decoder.set_embed_tokens(embed_tokens)

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
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
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
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
                head_mask=inputs["head_mask"],
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
            head_mask=inputs["decoder_head_mask"],
            encoder_head_mask=inputs["head_mask"],
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


@add_start_docstrings(
    "The bare MARIAN Model outputting raw hidden-states without any specific head on top.",
    MARIAN_START_DOCSTRING,
)
class TFMarianModel(TFMarianPreTrainedModel):
    def __init__(self, config: MarianConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.model = TFMarianMainLayer(config, name="model")

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="Helsinki-NLP/opus-mt-en-de",
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
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
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
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

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            head_mask=inputs["head_mask"],
            decoder_head_mask=inputs["decoder_head_mask"],
            encoder_outputs=inputs["encoder_outputs"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["inputs_embeds"],
            decoder_inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return outputs

    # Copied from transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output
    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )


@add_start_docstrings(
    "The MARIAN Model with a language modeling head. Can be used for summarization.",
    MARIAN_START_DOCSTRING,
)
class TFMarianMTModel(TFMarianPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFMarianMainLayer(config, name="model")
        self.use_cache = config.use_cache
        # final_bias_logits is registered as a buffer in pytorch, so not trainable for the the sake of consistency.
        self.final_logits_bias = self.add_weight(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        return self.model.decoder

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def get_bias(self):
        return {"final_logits_bias": self.final_logits_bias}

    def set_bias(self, value):
        self.final_logits_bias = value["final_logits_bias"]

    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
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
        r"""
        labels (:obj:`tf.tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
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
            inputs["labels"] = tf.where(
                inputs["labels"] == self.config.pad_token_id,
                tf.fill(shape_list(inputs["labels"]), -100),
                inputs["labels"],
            )
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
            head_mask=inputs["head_mask"],
            decoder_head_mask=inputs["decoder_head_mask"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["inputs_embeds"],
            decoder_inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
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

    # Copied from transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output
    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # Copied from transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past,
        attention_mask,
        head_mask=None,
        use_cache=None,
        **kwargs,
    ) -> Dict:
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
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    # Copied from transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration._reorder_cache
    def _reorder_cache(past, beam_idx):
        if len(past) == 1:
            return past

        past_key_values = past[1]

        reordered_past = ()
        for layer_past_key_values in past_key_values:
            reordered_past += (
                tuple(tf.gather(layer_past_key_value, beam_idx) for layer_past_key_value in layer_past_key_values[:2])
                + layer_past_key_values[2:],
            )
        return (past[0], reordered_past)

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        """Never predict pad_token_id. Predict </s> when max_length is reached."""
        vocab_range = tf.constant(range(self.config.vocab_size))
        logits = tf.where(vocab_range == self.config.pad_token_id, LARGE_NEGATIVE, logits)
        if cur_len == max_length - 1:
            logits = tf.where(vocab_range != self.config.eos_token_id, LARGE_NEGATIVE, logits)
        return logits
