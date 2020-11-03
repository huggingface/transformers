# coding=utf-8
# Copyright 2018 {{cookiecutter.authors}} and The HuggingFace Inc. team.
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
""" TF 2.0 {{cookiecutter.uppercase_modelname}} model. """


{% if cookiecutter.is_encoder_decoder_model == "True" -%}
import warnings
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Dense, LayerNormalization

from .activations_tf import ACT2FN
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Config
from .file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPast, TFSeq2SeqLMOutput, TFSeq2SeqModelOutput

# Public API
from .modeling_tf_utils import (
    DUMMY_INPUTS,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    TFWrappedEmbeddings,
    cast_bool_to_primitive,
    keras_serializable,
    shape_list,
)
from .tokenization_utils_base import BatchEncoding
from .utils import logging

TF_{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "{{cookiecutter.checkpoint_identifier}}",
    # See all {{cookiecutter.modelname}} models at https://huggingface.co/models?filter={{cookiecutter.lowercase_modelname}}
]

_CONFIG_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Config"
_TOKENIZER_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Tokenizer"

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
            Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the model weights.
"""


{{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

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
    """Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = tf.cumsum(mask, axis=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


def causal_attention_mask(nd, ns, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
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


class TFPretrained{{cookiecutter.camelcase_modelname}}Model(TFPreTrainedModel):
    config_class = {{cookiecutter.camelcase_modelname}}Config
    base_model_prefix = "{{cookiecutter.lowercase_modelname}}"

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
    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = TFAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_probs_dropout_prob, name="self_attn"
        )

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout_wt = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.fc1 = Dense(config.encoder_intermediate_dim, name="fc1")
        self.fc2 = Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = LayerNormalization(epsilon=1e-5, name="final_layer_norm")

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
        x, self_attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask)
        assert x.shape == residual.shape, f"Self attn modified the shape of query {residual.shape} to {x.shape}"
        x = self.dropout_wt(x, training=training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout_wt(x, training=training)
        x = residual + x
        x = self.final_layer_norm(x)

        return x, self_attn_weights


class TF{{cookiecutter.camelcase_modelname}}Encoder(tf.keras.layers.Layer):
    # config_class = {{cookiecutter.camelcase_modelname}}Config
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`TFEncoderLayer`.

    Args:
        config: {{cookiecutter.camelcase_modelname}}Config
    """

    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, embed_tokens: TFSharedEmbeddings, **kwargs):
        super().__init__(**kwargs)

        self.dropout = config.hidden_dropout_prob
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_positions = TFLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_tokens.hidden_size,
            self.padding_idx,
            name="embed_positions",
        )
        self.layers = [TFEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
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
            attention_mask (Tensor): indicating which indices are padding tokens.
        Returns:
            namedtuple:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
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
        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)

        # B x T x C -> T x B x C
        x = tf.transpose(x, perm=[1, 0, 2])

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # encoder layers
        for encoder_layer in self.layers:

            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            x, attn = encoder_layer(x, attention_mask)

            if output_attentions:
                all_attentions += (attn,)
        if output_hidden_states:
            encoder_states.append(x)
            encoder_states = [tf.transpose(hidden_state, perm=(1, 0, 2)) for hidden_state in encoder_states]
        x = tf.transpose(x, perm=(1, 0, 2))
        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)


class TFDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = TFAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            name="self_attn",
        )
        self.dropout = config.hidden_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.hidden_dropout_prob

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.encoder_attn = TFAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            encoder_decoder_attention=True,
            name="encoder_attn",
        )
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        self.fc1 = Dense(config.decoder_intermediate_dim, name="fc1")
        self.fc2 = Dense(self.embed_dim, name="fc2")
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
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:

            Tuple containing, encoded output of shape `(seq_len, batch, embed_dim)`, self_attn_weights, layer_state
        """
        if layer_state is None:
            layer_state = {}

        residual = x  # Make a copy of the input tensor to add later.
        # next line mutates layer state and we need a copy of it
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,
            attn_mask=causal_mask,
            key_padding_mask=decoder_padding_mask,
        )
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        # Cross-Attention
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x

        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = tf.nn.dropout(x, rate=self.activation_dropout if training else 0)
        x = self.fc2(x)
        x = tf.nn.dropout(x, rate=self.dropout if training else 0)
        x = residual + x
        x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class TF{{cookiecutter.camelcase_modelname}}Decoder(tf.keras.layers.Layer):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`TFDecoderLayer`.
    Args:
        config: {{cookiecutter.camelcase_modelname}}Config
        embed_tokens: output embedding
    """

    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, embed_tokens, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        self.embed_positions = TFLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            self.padding_idx,
            name="embed_positions",
        )
        self.layers = [TFDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache

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

        embed_pos = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            embed_pos = embed_pos[:, -1:]

        inputs_embeds = self.embed_tokens(input_ids)
        x = (inputs_embeds + embed_pos)
        x = self.dropout(x)

        # Convert to {{cookiecutter.camelcase_modelname}} output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = tf.transpose(x, perm=(1, 0, 2))
        assert len(encoder_hidden_states.shape) == 3, "encoder_hidden_states must be a 3D tensor"
        encoder_hidden_states = tf.transpose(encoder_hidden_states, perm=(1, 0, 2))

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
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
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention

        self.k_proj = Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = Dense(embed_dim, use_bias=bias, name="out_proj")

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
        attn_mask: Optional[Tensor] = None,
        training=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel

        Args:

            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        static_kv = self.encoder_decoder_attention  # value=key=encoder_hidden_states,
        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim, f"query must be shaped {(tgt_len, bsz, self.embed_dim)} got {query.shape}"
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
        src_len = k.shape[1]
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
        attn_probs = tf.nn.dropout(attn_weights, rate=self.dropout if training else 0.0)

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
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, **kwargs):
        # {{cookiecutter.camelcase_modelname}} is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        assert padding_idx is not None, "padding_idx cannot be None"
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(self, input_ids: tf.Tensor, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]

        if use_cache:
            positions = tf.fill((1, 1), seq_len - 1)
        else:
            # starts at 0, ends at 1-seq_len
            positions = tf.range(0, seq_len, delta=1, dtype=tf.int32, name="range")
        return super().call(positions)  # super object is not callable for some reason


# Public API


@add_start_docstrings(
    "The bare {{cookiecutter.modelname}} Model outputting raw hidden-states without any specific head on top.",
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
@keras_serializable
class TF{{cookiecutter.camelcase_modelname}}Model(TFPretrained{{cookiecutter.camelcase_modelname}}Model):
    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = TFSharedEmbeddings(config.vocab_size, config.hidden_size, config.pad_token_id, name="{{cookiecutter.lowercase_modelname}}.shared")

        with tf.compat.v1.variable_scope("{{cookiecutter.lowercase_modelname}}.shared") as shared_abs_scope_name:
            pass

        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        embed_tokens.vocab_size = self.shared.vocab_size
        embed_tokens.hidden_size = self.shared.hidden_size

        self.encoder = TF{{cookiecutter.camelcase_modelname}}Encoder(config, embed_tokens, name="encoder")
        self.decoder = TF{{cookiecutter.camelcase_modelname}}Decoder(config, embed_tokens, name="decoder")

    def _prepare_{{cookiecutter.lowercase_modelname}}_decoder_inputs(
            self,
            inputs,
            decoder_input_ids=None,
            decoder_attn_mask=None,
            mask_dtype=None,
    ):
        """Prepare masks that ignore padding tokens  decoder and a causal lm mask for the decoder if
        none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
        """
        pad_token_id = self.config.pad_token_id
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(inputs)
        bsz, tgt_len = decoder_input_ids.shape[:2]
        if decoder_attn_mask is None:
            decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
        else:
            decoder_padding_mask = invert_mask(decoder_attn_mask)

        causal_lm_mask = causal_attention_mask(tgt_len, tgt_len, mask_dtype)
        return decoder_input_ids, decoder_padding_mask, causal_lm_mask

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
            self,
            inputs,
            attention_mask=None,
            decoder_input_ids=None,  # BAD DEFAULT LEFT FOR CONSISTENT SIGNATURE
            decoder_attention_mask=None,
            encoder_outputs: Optional[TFBaseModelOutput] = None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            training=False,
            **kwargs
    ):
        """
        Returns:
        """
        assert "decoder_cached_states" not in kwargs, "Please use past_key_values to cache intermediate outputs"
        if isinstance(inputs, (tuple, list)):
            assert len(inputs) <= 10, "Too many inputs."
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            decoder_input_ids = inputs[2] if len(inputs) > 2 else decoder_input_ids
            decoder_attention_mask = inputs[3] if len(inputs) > 3 else decoder_attention_mask
            encoder_outputs = inputs[4] if len(inputs) > 4 else encoder_outputs
            past_key_values = inputs[5] if len(inputs) > 5 else past_key_values
            use_cache = inputs[6] if len(inputs) > 6 else use_cache
            output_attentions = inputs[7] if len(inputs) > 7 else output_attentions
            output_hidden_states = inputs[8] if len(inputs) > 8 else output_hidden_states
            return_dict = inputs[9] if len(inputs) > 9 else return_dict
        elif isinstance(inputs, (dict, BatchEncoding)):
            assert len(inputs) <= 10, "Too many inputs."
            if "inputs" in inputs:
                raise ValueError("Using `inputs` as a keyword argument is deprecated. Please use `input_ids` instead.")
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            decoder_input_ids = inputs.get("decoder_input_ids", decoder_input_ids)
            decoder_attention_mask = inputs.get("decoder_attention_mask", decoder_attention_mask)
            encoder_outputs = inputs.get("encoder_outputs", encoder_outputs)
            past_key_values = inputs.get("past_key_values", past_key_values)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
        else:
            input_ids = inputs

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if decoder_input_ids is None:  # Classification
            use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = self._prepare_{{cookiecutter.lowercase_modelname}}_decoder_inputs(
                inputs,
                decoder_input_ids=decoder_input_ids,
                decoder_attn_mask=decoder_attention_mask,
                mask_dtype=self.shared.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None
        assert (
                isinstance(encoder_outputs, TFBaseModelOutput) or encoder_outputs is None
        ), f"got unexpected encoder outputs type {type(encoder_outputs)}"
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs.last_hidden_state,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            # Attention and hidden_states will be [] or None if they aren't needed
            return tuple(x for x in decoder_outputs + encoder_outputs.to_tuple() if x is not None)
        else:
            return TFSeq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def get_output_embeddings(self):
        return self.shared


@add_start_docstrings(
    "The {{cookiecutter.modelname}} Model with a language modeling head. Can be used for summarization.",
    {{cookiecutter.uppercase_modelname}}_START_DOCSTRING,
)
class TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration(TFPretrained{{cookiecutter.camelcase_modelname}}Model):
    base_model_prefix = "{{cookiecutter.lowercase_modelname}}"

    def __init__(self, config: {{cookiecutter.camelcase_modelname}}Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.{{cookiecutter.lowercase_modelname}} = TF{{cookiecutter.camelcase_modelname}}Model(config, name="{{cookiecutter.lowercase_modelname}}")
        self.use_cache = config.use_cache

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
            self,
            inputs,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs: Optional[TFBaseModelOutput] = None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
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
            >>> logits = model(inputs=batch.input_ids, return_dict=True).logits
            >>> probs = tf.nn.softmax(logits[0])
            >>> # probs[5] is associated with the mask token
        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            decoder_input_ids = inputs[2] if len(inputs) > 2 else decoder_input_ids
            decoder_attention_mask = inputs[3] if len(inputs) > 3 else decoder_attention_mask
            encoder_outputs = inputs[4] if len(inputs) > 4 else encoder_outputs
            past_key_values = inputs[5] if len(inputs) > 5 else past_key_values
            labels = inputs[6] if len(inputs) > 6 else labels
            use_cache = inputs[7] if len(inputs) > 7 else use_cache
            output_attentions = inputs[8] if len(inputs) > 8 else output_attentions
            output_hidden_states = inputs[9] if len(inputs) > 9 else output_hidden_states
            return_dict = inputs[10] if len(inputs) > 10 else return_dict
            assert len(inputs) <= 13, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            if "inputs" in inputs:
                warnings.warn("Using `inputs` as a keyword argument is deprecated. Please use `input_ids` instead.")
            if "past_key_value_states" in inputs:
                raise ValueError(PAST_KV_DEPRECATION_WARNING)
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            decoder_input_ids = inputs.get("decoder_input_ids", decoder_input_ids)
            decoder_attention_mask = inputs.get("decoder_attention_mask", decoder_attention_mask)
            encoder_outputs = inputs.get("encoder_outputs", encoder_outputs)
            past_key_values = inputs.get("past_key_values", past_key_values)
            labels = inputs.get("labels", labels)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            assert len(inputs) <= 13, "Too many inputs."

        else:
            input_ids = inputs
        if "past_key_value_states" in kwargs:
            raise ValueError(PAST_KV_DEPRECATION_WARNING)

        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if labels is not None:
            use_cache = False
        outputs: TFSeq2SeqModelOutput = self.{{cookiecutter.lowercase_modelname}}(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            training=training
        )
        logits = self.{{cookiecutter.lowercase_modelname}}.shared(outputs.last_hidden_state, mode="linear")
        loss = None if labels is None else self.compute_loss(labels, logits)

        past = outputs.past_key_values if cast_bool_to_primitive(use_cache, self.config.use_cache) else None

        if return_dict:
            return TFSeq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=past,  # index 1 of d outputs
                decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
                decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
                encoder_last_hidden_state=outputs.last_hidden_state,  # index 0 of encoder outputs
                encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
                encoder_attentions=outputs.encoder_attentions,  # 2 of e out
            )
        else:
            if past is not None:
                decoder_outputs = (past,)
            else:
                decoder_outputs = tuple(
                    [x for x in (outputs.decoder_hidden_states, outputs.decoder_attentions) if x is not None]
                )
            enc_out = (outputs.encoder_last_hidden_state, outputs.encoder_hidden_states, outputs.encoder_attentions)
            encoder_outputs = tuple(x for x in enc_out if x is not None)
            output: Tuple = (logits,) + decoder_outputs + encoder_outputs
            return ((loss,) + output) if loss is not None else output

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
        ), "encoder_outputs should be a TFBaseModelOutput, Instead got "
        return {
            "inputs": None,  # encoder_outputs is defined. input_ids not needed
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
            logits = self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            logits = self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        output_list = []

        # Is there a better way to do scores[:, [x for if x != token_id]] = -float("inf") in TF?
        bs, vocab_size = scores.shape
        for x in range(vocab_size):
            if x != token_id:
                output_list.append(tf.convert_to_tensor([-float("inf")] * bs, dtype=scores.dtype))
            else:
                output_list.append(scores[:, x])
        scores = tf.stack(output_list, axis=1, name="scores")
        assert scores.shape == (bs, vocab_size)
        return scores

    def get_output_embeddings(self):
        return self.{{cookiecutter.lowercase_modelname}}.shared

    def get_encoder(self):
        return self.{{cookiecutter.lowercase_modelname}}.encoder

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


{% else -%}

import tensorflow as tf

from .activations_tf import get_tf_activation
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Config
from .file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from .modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from .modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    TFSequenceSummary,
    get_initializer,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Config"
_TOKENIZER_FOR_DOC = "{{cookiecutter.camelcase_modelname}}Tokenizer"

TF_{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "{{cookiecutter.checkpoint_identifier}}",
    # See all {{cookiecutter.modelname}} models at https://huggingface.co/models?filter={{cookiecutter.lowercase_modelname}}
]


# Copied from transformers.modeling_tf_bert.TFBertEmbeddings with Bert->{{cookiecutter.camelcase_modelname}}
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


# Copied from transformers.modeling_tf_bert.TFBertSelfAttention with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}SelfAttention(tf.keras.layers.Layer):
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

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], attention_scores.dtype)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TF{{cookiecutter.camelcase_modelname}}Model call() function)
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


# Copied from transformers.modeling_tf_bert.TFBertSelfOutput with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}SelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# Copied from transformers.modeling_tf_bert.TFBertAttention with Bert->{{cookiecutter.camelcase_modelname}}
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


# Copied from transformers.modeling_tf_bert.TFBertIntermediate with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Intermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.modeling_tf_bert.TFBertOutput with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Output(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# Copied from transformers.modeling_tf_bert.TFBertLayer with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}Layer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TF{{cookiecutter.camelcase_modelname}}Attention(config, name="attention")
        self.intermediate = TF{{cookiecutter.camelcase_modelname}}Intermediate(config, name="intermediate")
        self.{{cookiecutter.lowercase_modelname}}_output = TF{{cookiecutter.camelcase_modelname}}Output(config, name="output")

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.{{cookiecutter.lowercase_modelname}}_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


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


# Copied from transformers.modeling_tf_bert.TFBertPredictionHead with Bert->{{cookiecutter.camelcase_modelname}}
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


# Copied from transformers.modeling_tf_bert.TFBertLMPredictionHead with Bert->{{cookiecutter.camelcase_modelname}}
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


# Copied from transformers.modeling_tf_bert.TFBertMLMHead with Bert->{{cookiecutter.camelcase_modelname}}
class TF{{cookiecutter.camelcase_modelname}}MLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TF{{cookiecutter.camelcase_modelname}}LMPredictionHead(config, input_embeddings, name="predictions")

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)

        return prediction_scores


class TF{{cookiecutter.camelcase_modelname}}NSPHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.seq_relationship = tf.keras.layers.Dense(
            2, kernel_initializer=get_initializer(config.initializer_range), name="seq_relationship"
        )

    def call(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)

        return seq_relationship_score


@keras_serializable
class TF{{cookiecutter.camelcase_modelname}}MainLayer(tf.keras.layers.Layer):
    config_class = {{cookiecutter.camelcase_modelname}}Config

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

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
        inputs,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            output_attentions = inputs[6] if len(inputs) > 6 else output_attentions
            output_hidden_states = inputs[7] if len(inputs) > 7 else output_hidden_states
            return_dict = inputs[8] if len(inputs) > 8 else return_dict
            assert len(inputs) <= 9, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            assert len(inputs) <= 9, "Too many inputs."
        else:
            input_ids = inputs

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict

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

        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)

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
        extended_attention_mask = tf.cast(extended_attention_mask, embedding_output.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

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

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (
                sequence_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.modeling_tf_bert.TFBertPreTrainedModel with Bert->{{cookiecutter.camelcase_modelname}}
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
    "The bare {{cookiecutter.camelcase_modelname}} Model transformer outputing raw hidden-states without any specific head on top.",
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
    def call(self, inputs, **kwargs):
        outputs = self.{{cookiecutter.lowercase_modelname}}(inputs, **kwargs)

        return outputs


@add_start_docstrings("""{{cookiecutter.camelcase_modelname}} Model with a `language modeling` head on top. """, {{cookiecutter.uppercase_modelname}}_START_DOCSTRING)
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

    @add_start_docstrings_to_model_forward({{cookiecutter.uppercase_modelname}}_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="{{cookiecutter.checkpoint_identifier}}",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
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
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.{{cookiecutter.lowercase_modelname}}.return_dict

        if isinstance(inputs, (tuple, list)):
            labels = inputs[9] if len(inputs) > 9 else labels
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.{{cookiecutter.lowercase_modelname}}(
            inputs,
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
        prediction_scores = self.mlm(sequence_output, training=training)
        loss = None if labels is None else self.compute_loss(labels, prediction_scores)

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
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
    """{{cookiecutter.uppercase_modelname}} Model transformer with a sequence classification/regression head on top 
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
            inputs,
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
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.{{cookiecutter.lowercase_modelname}}.config.return_dict

        if isinstance(inputs, (tuple, list)):
            labels = inputs[9] if len(inputs) > 9 else labels

            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.{{cookiecutter.lowercase_modelname}}(
            inputs,
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
        logits = self.classifier(outputs[0])
        loss = None if labels is None else self.compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """{{cookiecutter.uppercase_modelname}} Model with a multiple choice classification head on top (a linear layer on top of
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
        """Dummy inputs to build the network.

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
        inputs,
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
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            output_attentions = inputs[6] if len(inputs) > 6 else output_attentions
            output_hidden_states = inputs[7] if len(inputs) > 7 else output_hidden_states
            return_dict = inputs[8] if len(inputs) > 8 else return_dict
            labels = inputs[9] if len(inputs) > 9 else labels
            assert len(inputs) <= 10, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            labels = inputs.get("labels", labels)
            assert len(inputs) <= 10, "Too many inputs."
        else:
            input_ids = inputs

        return_dict = return_dict if return_dict is not None else self.{{cookiecutter.lowercase_modelname}}.config.return_dict

        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        outputs = self.{{cookiecutter.lowercase_modelname}}(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        logits = self.sequence_summary(outputs[0])
        logits = self.classifier(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.compute_loss(labels, reshaped_logits)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """{{cookiecutter.camelcase_modelname}} Model with a token classification head on top (a linear layer on top of
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
        inputs=None,
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
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.{{cookiecutter.lowercase_modelname}}.return_dict

        if isinstance(inputs, (tuple, list)):
            labels = inputs[9] if len(inputs) > 9 else labels
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.{{cookiecutter.lowercase_modelname}}(
            inputs,
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
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.compute_loss(labels, logits)

        if not return_dict:
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
        inputs=None,
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
        return_dict = return_dict if return_dict is not None else self.{{cookiecutter.lowercase_modelname}}.return_dict

        if isinstance(inputs, (tuple, list)):
            start_positions = inputs[9] if len(inputs) > 9 else start_positions
            end_positions = inputs[10] if len(inputs) > 10 else end_positions
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            start_positions = inputs.pop("start_positions", start_positions)
            end_positions = inputs.pop("end_positions", start_positions)

        outputs = self.{{cookiecutter.lowercase_modelname}}(
            inputs,
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
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None

        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
{% endif -%}
