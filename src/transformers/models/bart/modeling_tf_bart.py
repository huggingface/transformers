# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
"""TF BART model, ported from the fairseq repo."""

import math
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
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
    cast_bool_to_primitive,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_bart import BartConfig


_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_START_DOCSTRING = r"""

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

    Args:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""


BART_INPUTS_DOCSTRING = r"""
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
        past_key_values (:obj:`Tuple[Dict[str: tf.Tensor]]` of length :obj:`config.n_layers`)
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
LARGE_NEGATIVE = -1e8


logger = logging.get_logger(__name__)


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = tf.cumsum(mask, axis=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


def causal_attention_mask(nd, ns, dtype):
    """
    1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]), -1,
    ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i < j - ns + nd
    return tf.cast(m, dtype) * LARGE_NEGATIVE


def invert_mask(attention_mask: tf.Tensor):
    """Turns 1->0, 0->1, False->True, True-> False"""
    tf.debugging.assert_rank(attention_mask, 2)
    attention_mask = tf.cast(attention_mask, tf.bool)
    ret = tf.math.logical_not(attention_mask)  # dtype is tf.bool
    return ret


class TFPretrainedBartModel(TFPreTrainedModel):
    config_class = BartConfig
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

    def _shift_right(self, input_ids):
        # Should maybe be decoder_start_token_id. Change for torch and TF in one PR
        position_0_id = self.config.eos_token_id
        pad_token_id = self.config.pad_token_id
        shifted_input_ids = tf.cast(input_ids, tf.int32)
        shifted_input_ids = tf.roll(shifted_input_ids, 1, axis=-1)
        start_tokens = tf.fill((shape_list(shifted_input_ids)[0], 1), position_0_id)
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


# Helper Functions, mostly for making masks


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = tf.math.equal(input_ids, padding_idx)  # bool tensor
    return padding_mask


# Helper Modules

PAST_KV_DEPRECATION_WARNING = (
    "The `past_key_value_states` argument is deprecated and will be removed in a future "
    "version, use `past_key_values` instead."
)


class TFEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(self, x, encoder_padding_mask, training=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, self_attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask)
        assert shape_list(x) == shape_list(
            residual
        ), f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(x)}"
        x = self.dropout(x, training=training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, self_attn_weights


class TFBartEncoder(tf.keras.layers.Layer):
    # config_class = BartConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TFEncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens: TFSharedEmbeddings, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = TFSinusoidalPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                name="embed_positions",
            )
        else:
            self.embed_positions = TFLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                config.extra_pos_embeddings,
                name="embed_positions",
            )
        self.layers = [TFEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
            if config.normalize_embedding
            else tf.keras.layers.Layer()
        )
        self.layer_norm = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
            if config.add_final_layer_norm
            else None
        )
        self.return_dict = config.return_dict

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        training=False,
    ):
        """
        Args:
            input_ids (Tensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (Tensor): indicating which indices are padding tokens

        Returns:
            namedtuple:

                - **x** (Tensor): the last encoder layer's output of shape `(src_len, batch, embed_dim)`

                - **encoder_states** (List[tf.Tensor]): all intermediate hidden states of shape `(src_len, batch,
                  embed_dim)`. Only populated if *output_hidden_states* is True.
                - **all_attentions** (List[tf.Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict

        # check attention mask and invert
        if attention_mask is not None:
            assert (
                attention_mask._rank() == 2
            ), f"expected attention_mask._rank() to be a 2D tensor got {attention_mask._rank()}"
            attention_mask = tf.cast(attention_mask, dtype=tf.float32)
            attention_mask = (1.0 - attention_mask) * LARGE_NEGATIVE
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = self.dropout(x, training=training)

        # B x T x C -> T x B x C
        x = tf.transpose(x, perm=[1, 0, 2])

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # encoder layers
        for encoder_layer in self.layers:

            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

            if output_attentions:
                all_attentions += (attn,)
        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            encoder_states = [tf.transpose(hidden_state, perm=(1, 0, 2)) for hidden_state in encoder_states]
        x = tf.transpose(x, perm=(1, 0, 2))
        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)


class TFDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.encoder_attn = TFAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
            name="encoder_attn",
        )
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(
        self,
        x,
        encoder_hidden_states: tf.Tensor,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        training=False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attn_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding elements are indicated by ``1``.
            need_attn_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:

            Tuple containing, encoded output of shape `(seq_len, batch, embed_dim)`, self_attn_weights, layer_state
        """
        residual = x  # Make a copy of the input tensor to add later.
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # next line mutates layer state and we need a copy of it
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,
            attn_mask=causal_mask,
            key_padding_mask=decoder_padding_mask,
        )
        x = self.dropout(x, training=training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Cross-Attention Block
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = self.dropout(x, training=training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class TFBartDecoder(tf.keras.layers.Layer):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TFDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens, **kwargs):
        super().__init__(**kwargs)
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if config.static_position_embeddings:
            self.embed_positions = TFSinusoidalPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                name="embed_positions",
            )
        else:
            self.embed_positions = TFLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                config.extra_pos_embeddings,
                name="embed_positions",
            )
        self.layers = [TFDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
            if config.normalize_embedding
            else tf.keras.layers.Layer()
        )
        self.layer_norm = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
            if config.add_final_layer_norm
            else None
        )

        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.do_blenderbot_90_layernorm = config.do_blenderbot_90_layernorm

    def call(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        training=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if use_cache:
            assert not training, "Training + use cache are incompatible"
        # check attention mask and invert
        use_cache = cast_bool_to_primitive(use_cache)
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=(use_cache and decoder_cached_states is not None))

        if use_cache and decoder_cached_states is not None:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]

        x = self.embed_tokens(input_ids) * self.embed_scale
        if self.do_blenderbot_90_layernorm:
            x = self.layernorm_embedding(x) + positions
        else:
            x = self.layernorm_embedding(x + positions)
        x = self.dropout(x, training=training)

        # Convert to Bart output format: (BS, seq_len, model_dim) ->  (seq_len, BS, model_dim)
        x = tf.transpose(x, perm=(1, 0, 2))
        assert len(shape_list(encoder_hidden_states)) == 3, "encoder_hidden_states must be a 3D tensor"
        encoder_hidden_states = tf.transpose(encoder_hidden_states, perm=(1, 0, 2))

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if output_attentions:
                all_self_attns += (layer_self_attn,)

        if self.layer_norm is not None:  # same as if config.add_final_layer_norm
            x = self.layer_norm(x)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states += (x,)
            # T x B x C -> B x T x C
            all_hidden_states = tuple(tf.transpose(hs, perm=(1, 0, 2)) for hs in all_hidden_states)
        else:
            all_hidden_states = None
        all_self_attns = list(all_self_attns) if output_attentions else None

        x = tf.transpose(x, perm=(1, 0, 2))
        encoder_hidden_states = tf.transpose(encoder_hidden_states, perm=(1, 0, 2))  # could maybe be avoided.

        next_cache = (encoder_hidden_states, next_decoder_cache) if use_cache else None
        if not return_dict:
            return x, next_cache, all_hidden_states, all_self_attns
        else:
            return TFBaseModelOutputWithPast(
                last_hidden_state=x,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = tf.gather(input_buffer_k, new_order, axis=0)
    return attn_cache


class TFAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor: tf.Tensor, dim_0, bsz) -> tf.Tensor:
        reshaped_T_B_D = tf.reshape(tensor, (dim_0, bsz * self.num_heads, self.head_dim))
        return tf.transpose(reshaped_T_B_D, perm=(1, 0, 2))

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        key_padding_mask: Optional[tf.Tensor] = None,
        layer_state: Optional[Dict[str, tf.Tensor]] = None,
        attn_mask: Optional[tf.Tensor] = None,
        training=False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Input shape: Time(SeqLen) x Batch x Channel

        Args:

            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
        """
        static_kv = self.encoder_decoder_attention  # value=key=encoder_hidden_states,
        tgt_len, bsz, embed_dim = shape_list(query)
        assert (
            embed_dim == self.embed_dim
        ), f"query must be shaped {(tgt_len, bsz, self.embed_dim)} got {shape_list(query)}"
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # get the last k and v for reuse
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            # this branch is hit by encoder
            saved_state = None

        # Project query key values using weights q_proj, k_proj, v_proj
        q = self.q_proj(query) * self.scaling
        if static_kv and key is None:  # cross-attention with cache
            k = v = None
        elif static_kv and key is not None:  # cross-attention no prev_key found in cache
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:  # self-attention
            k = self.k_proj(query)
            v = self.v_proj(query)

        # Reshape
        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
            v = self._shape(v, -1, bsz)

        if saved_state:  # read from cache
            k, v = self._concat_saved_state(k, v, saved_state, static_kv, bsz)

        if layer_state is not None:  # Write to cache every decoder call
            cached_shape = (bsz, self.num_heads, -1, self.head_dim)  # bsz must be first for reorder_cache
            layer_state[self.cache_key] = dict(
                prev_key=tf.reshape(k, cached_shape), prev_value=tf.reshape(v, cached_shape)
            )

        # Compute multi-headed attention
        src_len = shape_list(k)[1]
        attn_weights = tf.matmul(q, k, transpose_b=True)  # shape (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            assert attn_mask.dtype == tf.float32, f"expected dtype tf.float32 got {attn_mask.dtype}"
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attn_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            if key_padding_mask.dtype == tf.bool:
                key_padding_mask = tf.cast(key_padding_mask, attn_weights.dtype) * -1e9
            extended_mask = tf.expand_dims(tf.expand_dims(key_padding_mask, 1), 2)
            attn_weights = attn_weights + extended_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_weights, training=training)

        attn_output = tf.matmul(attn_probs, v)  # shape: (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = tf.transpose(attn_output, perm=(1, 0, 2))
        attn_output = tf.reshape(attn_output, (tgt_len, bsz, embed_dim))
        attn_output = self.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
        return attn_output, attn_weights

    def _concat_saved_state(self, k, v, saved_state, static_kv, bsz) -> Tuple[tf.Tensor]:
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        prev_key = tf.reshape(saved_state["prev_key"], (bsz * self.num_heads, -1, self.head_dim))
        k = prev_key if static_kv else tf.concat([prev_key, k], axis=1)
        prev_value = tf.reshape(saved_state["prev_value"], (bsz * self.num_heads, -1, self.head_dim))
        v = prev_value if static_kv else tf.concat([prev_value, v], axis=1)
        return k, v


class TFLearnedPositionalEmbedding(TFSharedEmbeddings):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset, **kwargs):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None, "padding_idx cannot be None"
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(self, input_ids: tf.Tensor, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = shape_list(input_ids)[:2]

        if use_cache:
            positions = tf.fill((1, 1), seq_len - 1)
        else:
            # starts at 0, ends at 1-seq_len
            positions = tf.range(0, seq_len, delta=1, dtype=tf.int32, name="range")
        return super().call(positions + self.offset)  # super object is not callable for some reason


class TFSinusoidalPositionalEmbedding(tf.keras.layers.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, **kwargs):

        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        super().__init__(
            num_positions,
            embedding_dim,
            **kwargs,
        )

    def build(self, input_shape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        super().build(input_shape)  # Instantiates self.weight so it can be loaded
        weight: np.ndarray = self._init_weight(self.input_dim, self.output_dim)
        self.set_weights([weight])  # overwrite self.weight to correct value

    @staticmethod
    def _init_weight(n_pos, dim):
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

    def call(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = shape_list(input_ids)[:2]
        if use_cache:
            positions = tf.fill((1, 1), seq_len - 1)
        else:
            # starts at 0, ends at 1-seq_len
            positions = tf.range(0, seq_len, delta=1, dtype=tf.int32, name="range")
        return super().call(positions)


# Public API


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
@keras_serializable
class TFBartModel(TFPretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, config.pad_token_id, name="model.shared")

        with tf.compat.v1.variable_scope("model.shared") as shared_abs_scope_name:
            pass

        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        embed_tokens.vocab_size = self.shared.vocab_size
        embed_tokens.hidden_size = self.shared.hidden_size

        self.encoder = TFBartEncoder(config, embed_tokens, name="encoder")
        self.decoder = TFBartDecoder(config, embed_tokens, name="decoder")

    def _prepare_bart_decoder_inputs(
        self,
        inputs,
        decoder_input_ids=None,
        decoder_attn_mask=None,
        mask_dtype=None,
    ):
        """
        Prepare masks that ignore padding tokens decoder and a causal lm mask for the decoder if none are provided.
        This mimics the default behavior in fairseq. To override it pass in masks.
        """
        pad_token_id = self.config.pad_token_id
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(inputs)
        bsz, tgt_len = shape_list(decoder_input_ids)[:2]
        if decoder_attn_mask is None:
            decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
        else:
            decoder_padding_mask = invert_mask(decoder_attn_mask)

        causal_lm_mask = causal_attention_mask(tgt_len, tgt_len, mask_dtype)
        return decoder_input_ids, decoder_padding_mask, causal_lm_mask

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,  # BAD DEFAULT LEFT FOR CONSISTENT SIGNATURE
        decoder_attention_mask=None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values=None,
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
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["decoder_input_ids"] is None:
            inputs["use_cache"] = False

        inputs["output_hidden_states"] = (
            inputs["output_hidden_states"]
            if inputs["output_hidden_states"] is not None
            else self.config.output_hidden_states
        )

        if not use_cache or past_key_values is None:
            inputs["decoder_input_ids"], decoder_padding_mask, causal_mask = self._prepare_bart_decoder_inputs(
                inputs["input_ids"],
                decoder_input_ids=inputs["decoder_input_ids"],
                decoder_attn_mask=inputs["decoder_attention_mask"],
                mask_dtype=self.shared.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None
        if inputs["encoder_outputs"] is None:
            inputs["encoder_outputs"] = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
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
            inputs["encoder_outputs"][0],
            inputs["attention_mask"],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=inputs["past_key_values"],
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
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING,
)
class TFBartForConditionalGeneration(TFPretrainedBartModel):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFBartModel(config, name="model")
        self.use_cache = config.use_cache
        # final_bias_logits is registered as a buffer in pytorch, so not trainable for the the sake of consistency.
        self.final_logits_bias = self.add_weight(
            name="/final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def resize_token_embeddings(self, new_num_tokens):
        super().resize_token_embeddings(new_num_tokens=new_num_tokens)

        # BART is a special case where the bias has two dimensions
        # and not named just `bias`
        if new_num_tokens is not None:
            num_tokens_to_copy = min(self.final_logits_bias.shape[0], new_num_tokens)
            init_bias = tf.zeros((new_num_tokens,))
            init_bias[:num_tokens_to_copy] = self.final_logits_bias.value()[:num_tokens_to_copy]
            name = self.name + "/final_logits_bias"
            self.final_logits_bias = self.add_weight(
                shape=(1, new_num_tokens),
                initializer="zeros",
                trainable=False,
                name=name,
            )
            self.final_logits_bias.assign(init_bias)

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        past_key_values=None,
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

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, TFBartForConditionalGeneration
            import tensorflow as tf
            mname = 'facebook/bart-large'
            tokenizer = BartTokenizer.from_pretrained(mname)
            TXT = "My friends are <mask> but they eat too many carbs."
            model = TFBartForConditionalGeneration.from_pretrained(mname)
            batch = tokenizer([TXT], return_tensors='tf')
            logits = model(inputs=batch.input_ids).logits
            probs = tf.nn.softmax(logits[0])
            # probs[5] is associated with the mask token
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
                inputs["decoder_input_ids"] = self._shift_right(inputs["labels"])

        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            encoder_outputs=inputs["encoder_outputs"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            past_key_values=inputs["past_key_values"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
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

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache=True, **kwargs) -> Dict:
        assert past is not None and len(past) in {1, 2}, f"past has to be an iterable of length 1,2 got {past}"
        if len(past) == 1:
            assert isinstance(past[0], tf.Tensor)
            encoder_outputs = TFBaseModelOutput(last_hidden_state=past[0])
            decoder_cached_states = None
        else:
            assert len(past) == 2
            encoder_outputs, decoder_cached_states = past
            if isinstance(encoder_outputs, tuple):
                assert isinstance(encoder_outputs[0], tf.Tensor)
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs[0])
            elif isinstance(encoder_outputs, tf.Tensor):
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs)
            assert (
                decoder_cached_states
            ), f"decoder cached states must be truthy. got {decoder_cached_states} from the 2nd element of past"

        assert isinstance(
            encoder_outputs, TFBaseModelOutput
        ), f"encoder_outputs should be a TFBaseModelOutput, Instead got {type(encoder_outputs)}."
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        assert len(past) == 2
        (encoder_out, decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        past = (encoder_out, reordered_past)
        return past

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            vocab_range = tf.constant(range(self.config.vocab_size))
            return tf.where(vocab_range != self.config.bos_token_id, LARGE_NEGATIVE, logits)
        elif cur_len == max_length - 1:
            vocab_range = tf.constant(range(self.config.vocab_size))
            return tf.where(vocab_range != self.config.eos_token_id, LARGE_NEGATIVE, logits)
        else:
            return logits

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
