# coding=utf-8
# Copyright 2021 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 Speech2Text model. """
import inspect
import math
import random
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ... import BatchEncoding
from ...activations_tf import get_tf_activation
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# Public API
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    booleans_processing,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_speech_to_text import Speech2TextConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "s2t"
_CONFIG_FOR_DOC = "Speech2TextConfig"
_TOKENIZER_FOR_DOC = "Speech2TextTokenizer"


LARGE_NEGATIVE = -1e8

TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/s2t-small-librispeech-asr",
    # See all Speech2Text models at https://huggingface.co/models?filter=speech_to_text
]


def input_features_processing(func, config, input_features, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. :obj:`input_values = keras.Input(shape=(128,),
    dtype='float32', name="input_values")` otherwise the order of the tensors will not be guaranteed during the
    training.

    Args:
        func (:obj:`callable`):
            The callable function of the TensorFlow model.
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    signature.pop("kwargs", None)
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(input_features, (tuple, list)):
        for i, input in enumerate(input_features):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if type(input) == tf.Tensor:
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for {parameter_names[i]}."
                )
    elif isinstance(input_features, (dict, BatchEncoding)):
        if "inputs" in input_features:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_values` instead.",
                FutureWarning,
            )

            output["input_values"] = input_features.pop("inputs")

        if "decoder_cached_states" in input_features:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = input_features.pop("decoder_cached_states")

        for k, v in dict(input_features).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if isinstance(input_features, tf.Tensor) or input_features is None:
            output[parameter_names[0]] = input_features
        else:
            raise ValueError(
                f"Data of type {type(input_features)} is not allowed only {allowed_types} is accepted for {parameter_names[0]}."
            )

    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and type(output["args"]) == tf.Tensor:
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_values`
            output["input_values"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    boolean_dict = {
        k: v
        for k, v in output.items()
        if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
    }

    output.update(booleans_processing(config=config, **boolean_dict))

    return output


# Copied from transformers.models.bart.modeling_tf_bart.shift_tokens_right
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = tf.where(
        shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
    )

    if tf.executing_eagerly():
        # "Verify that `labels` has only positive values and -100"
        assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0))

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
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])

    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None, past_key_values_length: int = 0):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFSpeech2TextGatedLinearUnit(layers.Layer):
    """GLU Activation"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        linear, gated = tf.split(value=inputs, num_or_size_splits=2, axis=-1)
        return linear * tf.nn.sigmoid(gated)


class TFSpeech2TextConv1D(layers.Conv1D):
    def __init__(self, filters, kernel_size, strides, explicit_padding, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding="valid", **kwargs)
        self.explicit_padding = explicit_padding

    def call(self, inputs):
        inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        out = super().call(inputs)
        return out


class TFConv1dSubsampler(layers.Layer):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_layers = config.num_conv_layers
        self.mid_channels = config.conv_channels
        self.out_channels = config.d_model
        self.kernel_sizes = config.conv_kernel_sizes

        self.conv_layers = [
            TFSpeech2TextConv1D(
                self.mid_channels if i < self.num_layers - 1 else self.out_channels * 2,
                kernel_size=k,
                strides=2,
                explicit_padding=k // 2,
                name=f"conv_layers.{i}",
            )
            for i, k in enumerate(self.kernel_sizes)
        ]
        self.glu = TFSpeech2TextGatedLinearUnit()

    def call(self, input_features):
        hidden_states = input_features
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)
            hidden_states = self.glu(hidden_states)
        return hidden_states


class TFSpeech2TextSinusoidalPositionalEmbedding(layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_positions = num_positions

    def build(self, input_shape):

        kernel = self.get_embedding(self.num_positions, self.embedding_dim)

        self.kernel = self.add_weight(name="kernel", shape=[self.num_positions, self.embedding_dim], trainable=False)
        kernel = tf.cast(kernel, dtype=self.kernel.dtype)

        self.kernel.assign(kernel)

        super().build(input_shape)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.range(num_embeddings, dtype=tf.float32)[..., tf.newaxis] * emb[tf.newaxis, ...]
        emb = tf.reshape(tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1), (num_embeddings, -1))
        if embedding_dim % 2 == 1:
            # zero pad
            emb = tf.concat([emb, tf.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            emb = tf.tensor_scatter_nd_update(emb, [[padding_idx]], tf.zeros((1, shape_list(emb)[1])))
        tf.stop_gradient(emb)
        return emb

    def call(self, input_features: tf.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = shape_list(input_features)
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_features(
            input_features, self.padding_idx, past_key_values_length
        )

        # expand embeddings if needed
        # max_pos = self.padding_idx + 1 + seq_len
        # if max_pos > shape_list(self.weight)[0]:
        #    self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        position_ids = tf.reshape(position_ids, [-1])
        embedding = tf.gather([self.kernel], indices=position_ids, axis=1)
        return tf.reshape(embedding, (bsz, seq_len, -1))

    def create_position_ids_from_input_features(
        self, input_features: tf.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: tf.Tensor x:
        Returns: tf.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = tf.where(tf.not_equal(input_features, padding_idx), 1, 0)
        incremental_indices = (tf.cumsum(mask, axis=1) + past_key_values_length) * mask
        return tf.cast(incremental_indices, dtype=tf.int64) + padding_idx


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with TFBart -> TFSpeech2Text
class TFSpeech2TextAttention(layers.Layer):
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

        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(attn_weights),
                [bsz * self.num_heads, tgt_len, src_len],
                message=f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {shape_list(attn_weights)}",
            )

        if attention_mask is not None:
            # The tf.debugging asserts are not compliant with XLA then they
            # have to be disabled in other modes than eager.
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(attention_mask),
                    [bsz, 1, tgt_len, src_len],
                    message=f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {shape_list(attention_mask)}",
                )

            attention_mask = tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            # The tf.debugging asserts are not compliant with XLA then they
            # have to be disabled in other modes than eager.
            if tf.executing_eagerly():
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

        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if tf.executing_eagerly():
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


class TFSpeech2TextEncoderLayer(layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFSpeech2TextAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = layers.Dropout(config.dropout)
        self.activation_fn = layers.Activation(config.activation_function)
        self.activation_dropout = layers.Dropout(config.activation_dropout)
        self.fc1 = layers.Dense(
            config.encoder_ffn_dim, name="fc1", kernel_initializer=get_initializer(config.init_std)
        )
        self.fc2 = layers.Dense(self.embed_dim, name="fc2", kernel_initializer=get_initializer(config.init_std))
        self.dropout_2 = layers.Dropout(config.dropout)
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        output_attentions: bool = False,
        training=False,
    ):
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training,
        )
        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(hidden_states),
                shape_list(residual),
                message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
            )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_2(hidden_states, training=training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFSpeech2TextDecoderLayer(layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFSpeech2TextAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        self.dropout = layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = layers.Dropout(config.activation_dropout)

        self.self_attn_layer_norm = layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.encoder_attn = TFSpeech2TextAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        self.fc1 = layers.Dense(
            config.decoder_ffn_dim, name="fc1", kernel_initializer=get_initializer(config.init_std)
        )
        self.fc2 = layers.Dense(self.embed_dim, name="fc2", kernel_initializer=get_initializer(config.init_std))
        self.dropout_2 = layers.Dropout(config.dropout)
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(
        self,
        hidden_states,
        attention_mask: Optional[tf.Tensor] = None,
        encoder_hidden_states: Optional[tf.Tensor] = None,
        encoder_attention_mask: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        cross_attn_layer_head_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
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
            cross_attn_layer_head_mask (:obj:`tf.Tensor`): mask for heads of the cross-attention module.
                `(decoder_attention_heads,)`
            past_key_value (:obj:`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training,
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                training=training,
            )
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_2(hidden_states, training=training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TFSpeech2TextEncoder(layers.Layer):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TFSpeech2TextEncoderLayer`.

    Args:
        config: Speech2TextConfig
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.dropout = layers.Dropout(config.dropout, name="dropout")
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_target_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv = TFConv1dSubsampler(config, name="conv")

        self.embed_positions = TFSpeech2TextSinusoidalPositionalEmbedding(
            num_positions=self.max_source_positions,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            name="embed_positions",
        )
        self.layer = [TFSpeech2TextEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    def _get_subsampled_encoder_attn_mask(self, attention_mask):
        # generate creates 3D attention mask, because of the shape of input_features
        # convert it to 2D if thats the case
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        attention_mask = tf.cast(attention_mask, tf.int32)
        subsampled_lengths = self._get_subsampled_output_lengths(tf.reduce_sum(attention_mask, -1))
        max_len = tf.reduce_max(subsampled_lengths)
        bsz = shape_list(attention_mask)[0]

        attention_mask = tf.zeros((bsz, max_len), dtype=attention_mask.dtype)

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        indices = tf.stack([tf.range(0, bsz), subsampled_lengths - 1], 1)
        attention_mask = tf.tensor_scatter_nd_update(attention_mask, indices, tf.ones((bsz,), tf.int32))
        attention_mask = tf.reverse(tf.cumsum(tf.reverse(attention_mask, [-1]), -1), [-1])
        return attention_mask

    def _get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def call(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        r"""
        Args:
            input_features (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length, feature_size)`):
                Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a ``.flac`` or ``.wav`` audio file into an array of type :obj:`List[float]` or a
                :obj:`numpy.ndarray`, *e.g.* via the soundfile library (``pip install soundfile``). To prepare the
                array into :obj:`input_features`, the :class:`~transformers.Speech2TextTokenizer` should be used for
                extracting the fbank features, padding and conversion into a tensor of type :obj:`torch.FloatTensor`.
                See :meth:`~transformers.Speech2TextTokenizer.__call__`
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs = input_features_processing(
            func=self.call,
            config=self.config,
            input_features=input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        if inputs["attention_mask"] is not None:
            inputs["attention_mask"] = self._get_subsampled_encoder_attn_mask(inputs["attention_mask"])

        inputs["inputs_embeds"] = self.conv(inputs["input_features"])
        inputs["inputs_embeds"] = self.embed_scale * inputs["inputs_embeds"]

        if inputs["attention_mask"] is None:
            padding_mask = tf.zeros(shape_list(inputs["inputs_embeds"])[:2], dtype=inputs["inputs_embeds"].dtype)
        else:
            padding_mask = tf.cast(tf.not_equal(inputs["attention_mask"], 1), tf.int32)
        embed_pos = self.embed_positions(padding_mask)

        hidden_states = inputs["inputs_embeds"] + embed_pos
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # expand attention_mask
        if inputs["attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            inputs["attention_mask"] = _expand_mask(inputs["attention_mask"])

        encoder_states = () if inputs["output_hidden_states"] else None
        all_attentions = () if inputs["output_attentions"] else None

        # check if head_mask has a correct number of layers specified if desired
        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if inputs["head_mask"] is not None and tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(inputs["head_mask"])[0],
                len(self.layer),
                message=f"The head_mask should be specified for {len(self.layer)} layers, but it is for {shape_list(inputs['head_mask'])[0]}.",
            )

        for idx, encoder_layer in enumerate(self.layer):
            if inputs["output_hidden_states"]:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if inputs["training"] and (dropout_probability < self.layerdrop):  # skip the layer
                continue
            else:
                layer_outputs = encoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=inputs["attention_mask"],
                    layer_head_mask=inputs["head_mask"][idx] if inputs["head_mask"] is not None else None,
                    output_attentions=inputs["output_attentions"],
                    training=inputs["training"],
                )

                hidden_states = layer_outputs[0]

            if inputs["output_attentions"]:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if inputs["output_hidden_states"]:
            encoder_states = encoder_states + (hidden_states,)

        if not inputs["return_dict"]:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class TFSpeech2TextDecoder(layers.Layer):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    :class:`TFSpeech2TextDecoderLayer`

    Args:
        config: Speech2TextConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = layers.Embedding(config.vocab_size, config.d_model, name="embed_tokens")

        self.embed_positions = TFSpeech2TextSinusoidalPositionalEmbedding(
            config.max_target_positions,
            config.d_model,
            self.padding_idx,
            name="embed_positions",
        )
        self.layers = [TFSpeech2TextDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _get_subsampled_encoder_attn_mask(self, attention_mask):
        # generate creates 3D attention mask, because of the shape of input_features
        # convert it to 2D if thats the case
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        attention_mask = tf.cast(attention_mask, tf.int32)
        subsampled_lengths = self._get_subsampled_output_lengths(tf.reduce_sum(attention_mask, -1))
        max_len = tf.reduce_max(subsampled_lengths)
        bsz = shape_list(attention_mask)[0]

        attention_mask = tf.zeros((bsz, max_len), dtype=attention_mask.dtype)

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        indices = tf.stack([tf.range(0, bsz), subsampled_lengths - 1], 1)
        attention_mask = tf.tensor_scatter_nd_update(attention_mask, indices, tf.ones((bsz,), tf.int32))
        attention_mask = tf.reverse(tf.cumsum(tf.reverse(attention_mask, [-1]), -1), [-1])
        return attention_mask

    def _get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
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

                Indices can be obtained using :class:`~transformers.Speech2TextTokenizer`. See
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
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_features. Mask
                values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`Tuple[Tuple[tf.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_features` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_features`
                indices into associated vectors than the model's internal embedding lookup matrix.
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        # retrieve input_features and inputs_embeds
        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
            inputs["input_ids"] = tf.reshape(inputs["input_ids"], (-1, input_shape[-1]))
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = (
            shape_list(inputs["past_key_values"][0][0])[2] if inputs["past_key_values"] is not None else 0
        )

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"]) * self.embed_scale

        inputs["attention_mask"] = self._prepare_decoder_attention_mask(
            inputs["attention_mask"], input_shape, past_key_values_length
        )

        # expand encoder attention mask
        if inputs["encoder_hidden_states"] is not None and inputs["encoder_attention_mask"] is not None:
            inputs["encoder_attention_mask"] = self._get_subsampled_encoder_attn_mask(inputs["encoder_attention_mask"])
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            inputs["encoder_attention_mask"] = _expand_mask(inputs["encoder_attention_mask"], tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(inputs["input_ids"], past_key_values_length=past_key_values_length)

        hidden_states = inputs["inputs_embeds"] + positions
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # decoder layers
        all_hidden_states = () if inputs["output_hidden_states"] else None
        all_self_attns = () if inputs["output_attentions"] else None
        all_cross_attns = () if (inputs["output_attentions"] and inputs["encoder_hidden_states"] is not None) else None
        present_key_values = () if inputs["use_cache"] else None

        # check if head_mask and cross_attn_head_mask have a correct number of layers specified if desired
        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        for attn_mask in ["head_mask", "cross_attn_head_mask"]:
            if inputs[attn_mask] is not None and tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(inputs[attn_mask])[0],
                    len(self.layers),
                    message=f"The {attn_mask} should be specified for {len(self.layers)} layers, but it is for {shape_list(inputs[attn_mask])[0]}.",
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if inputs["output_hidden_states"]:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if inputs["training"] and (dropout_probability < self.layerdrop):
                continue

            past_key_value = inputs["past_key_values"][idx] if inputs["past_key_values"] is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=inputs["attention_mask"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                encoder_attention_mask=inputs["encoder_attention_mask"],
                layer_head_mask=(inputs["head_mask"][idx] if inputs["head_mask"] is not None else None),
                cross_attn_layer_head_mask=(
                    inputs["cross_attn_head_mask"][idx] if inputs["cross_attn_head_mask"] is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=inputs["output_attentions"],
                use_cache=inputs["use_cache"],
                training=inputs["training"],
            )
            hidden_states = layer_outputs[0]

            if inputs["use_cache"]:
                present_key_values += (layer_outputs[3 if inputs["output_attentions"] else 1],)

            if inputs["output_attentions"]:
                all_self_attns += (layer_outputs[1],)

                if inputs["encoder_hidden_states"] is not None:
                    all_cross_attns += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if inputs["output_hidden_states"]:
            all_hidden_states += (hidden_states,)

        next_cache = present_key_values if use_cache else None
        if not inputs["return_dict"]:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attns]
                if v is not None
            )
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
        )


@keras_serializable
class TFSpeech2TextMainLayer(layers.Layer):
    config_class = Speech2TextConfig

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.encoder = TFSpeech2TextEncoder(config, name="encoder")
        self.decoder = TFSpeech2TextDecoder(config, name="decoder")

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def call(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs = input_features_processing(
            func=self.call,
            config=self.config,
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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

        if inputs["encoder_outputs"] is None:
            inputs["encoder_outputs"] = self.encoder(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                head_mask=inputs["head_mask"],
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

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=inputs["decoder_input_ids"],
            attention_mask=inputs["decoder_attention_mask"],
            encoder_hidden_states=inputs["encoder_outputs"][0],
            encoder_attention_mask=inputs["attention_mask"],
            head_mask=inputs["decoder_head_mask"],
            cross_attn_head_mask=inputs["cross_attn_head_mask"],
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
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=inputs["encoder_outputs"].last_hidden_state,
            encoder_hidden_states=inputs["encoder_outputs"].hidden_states,
            encoder_attentions=inputs["encoder_outputs"].attentions,
        )


class TFSpeech2TextPreTrainedModel(TFPreTrainedModel):
    config_class = Speech2TextConfig
    base_model_prefix = "model"

    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"model.encoder.embed_positions.weights",
        r"model.decoder.embed_positions.weights",
        r"model.encoder.embed_positions.weight",
        r"model.decoder.embed_positions.weight",
    ]
    _keys_to_ignore_on_save = [
        r"model.encoder.embed_positions.weights",
        r"model.decoder.embed_positions.weights",
        r"model.encoder.embed_positions.weight",
        r"model.decoder.embed_positions.weight",
    ]

    @property
    def dummy_inputs(self):
        pad_token = 1
        input_features = tf.random.normal((3, 768, self.config.input_feat_per_channel))
        decoder_input_ids = tf.random.uniform(maxval=self.config.vocab_size, shape=(3, 50), dtype=tf.int32)
        attention_mask = tf.cast(tf.not_equal(input_features, pad_token), tf.float32)
        decoder_attention_mask = tf.not_equal(decoder_input_ids, pad_token)
        head_mask = tf.ones((self.config.encoder_layers, self.config.encoder_attention_heads))
        decoder_head_mask = tf.ones((self.config.decoder_layers, self.config.decoder_attention_heads))
        cross_attn_head_mask = tf.ones((self.config.decoder_layers, self.config.decoder_attention_heads))

        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
        }

    @tf.function(
        input_signature=[
            {
                "input_features": tf.TensorSpec((None, None, None), tf.int32, name="input_features"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
                "decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),
                "decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)


SPEECH_TO_TEXT_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`keras.Model.fit` method which currently requires having all the
        tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_features` only and nothing else: :obj:`model(input_features)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_features, attention_mask])` or :obj:`model([input_features, attention_mask,
          token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_features": input_features, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.Speech2TextConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

SPEECH_TO_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_features (:obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.Speech2TextTokenizer`. See
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

            Indices can be obtained using :class:`~transformers.Speech2TextTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            Speech2Text uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the
            :obj:`input_features` to the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
        head_mask (:obj:`tf.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (:obj:`tf.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in ``[0, 1]``:

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


@add_start_docstrings(
    "The bare SPEECH_TO_TEXT Model outputting raw hidden-states without any specific head on top.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
class TFSpeech2TextModel(TFSpeech2TextPreTrainedModel):
    def __init__(self, config: Speech2TextConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.model = TFSpeech2TextMainLayer(config, name="model")

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
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
        inputs = input_features_processing(
            func=self.call,
            config=self.config,
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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
            inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            head_mask=inputs["head_mask"],
            decoder_head_mask=inputs["decoder_head_mask"],
            cross_attn_head_mask=inputs["cross_attn_head_mask"],
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


@add_start_docstrings(
    "The SPEECH_TO_TEXT Model with a language modeling head. Can be used for summarization.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
class TFSpeech2TextForConditionalGeneration(TFSpeech2TextPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFSpeech2TextMainLayer(config, name="model")
        # self.model._set_save_spec(inputs=self.serving.input_signature)
        self.use_cache = config.use_cache
        self.lm_head = layers.Dense(self.config.vocab_size, use_bias=False, name="lm_head")
        self.final_logits_bias = self.add_weight(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        return self.model.decoder

    def get_encoder(self):
        return self.model.encoder

    def resize_token_embeddings(self, new_num_tokens=None):
        return super().resize_token_embeddings(new_num_tokens)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_bias(self):
        return {"final_logits_bias": self.final_logits_bias}

    def set_bias(self, value):
        self.final_logits_bias = value["final_logits_bias"]

    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
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
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_features`` docstring). Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Example::

            >>> import tensorflow as tf
            >>> from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> model = TFSpeech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
            >>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_features = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="tf").input_features  # Batch size 1
            >>> generated_ids = model.generate(input_features=input_features)

            >>> transcription = processor.batch_decode(generated_ids)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs = input_features_processing(
            func=self.call,
            config=self.config,
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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
            inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            encoder_outputs=inputs["encoder_outputs"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            head_mask=inputs["head_mask"],
            decoder_head_mask=inputs["decoder_head_mask"],
            cross_attn_head_mask=inputs["cross_attn_head_mask"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["inputs_embeds"],
            decoder_inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias

        if inputs["labels"] is not None:
            loss = self.compute_loss(inputs["labels"], lm_logits)
        else:
            loss = None

        if not inputs["return_dict"]:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
        )

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past,
        attention_mask,
        head_mask,
        decoder_head_mask,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ) -> Dict:
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
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

    def compute_loss(self, labels, logits):
        """CrossEntropyLoss that ignores pad tokens"""
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )
        melted_labels = tf.reshape(labels, [-1])
        active_loss = tf.not_equal(melted_labels, self.config.pad_token_id)
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(melted_labels, active_loss)
        return loss_fn(labels, reduced_logits)
