# coding=utf-8
# Copyright 2022 Google LLC., LongT5 Authors and HuggingFace Inc. team.
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
""" TF 2.0 LongT5 model."""

import copy
import itertools
import math
import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    ContextManagers,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_longt5 import LongT5Config


logger = logging.get_logger(__name__)

CONFIG_FOR_DOC = "LongT5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "google/long-t5-local-base"

# TODO: Update before the merge
TF_LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/long-t5-local-base",
    "google/long-t5-local-large",
    "google/long-t5-tglobal-base",
    "google/long-t5-tglobal-large",
    "google/long-t5-tglobal-xl",
]


def _pad_to_multiple(x: tf.Tensor, block_len: int, dim: int, pad_value: int = 0) -> tf.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    x_shape = shape_list(x)

    pad_len = -x_shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x_shape):
        new_shape = x_shape
        new_shape[dim] += pad_len
        return tf.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * len(shape_list(x))
    pad[dim] = (0, pad_len)
    return tf.pad(x, pad, constant_values=pad_value)


def _split_into_blocks(x: tf.Tensor, block_len: int, dim: int) -> tf.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    x_shape = shape_list(x)

    if x_shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = shape_list(x)[dim] // block_len
    output_shape = x_shape[:dim] + [num_blocks, block_len] + x_shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return tf.experimental.numpy.empty(output_shape, dtype=x.dtype)
    return tf.reshape(x, output_shape)


def _concatenate_3_blocks(x: tf.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> tf.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = shape_list(x)[block_dim]

    pad = [(0, 0)] * len(shape_list(x))
    pad[block_dim] = (1, 1)
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = tf.pad(x, pad, constant_values=pad_value)

    blocks_list: List[tf.Tensor] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * len(shape_list(x))
        indices[block_dim] = slice(i, i + num_blocks)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 3 * block_len, ...]
    return tf.concat(blocks_list, axis=sequence_dim)


def _make_3block_relative_position_ids(block_len: int) -> tf.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = tf.range(3 * block_len, dtype=tf.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = tf.expand_dims(position_ids, axis=0) - tf.expand_dims(center_position_ids, axis=1)
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: tf.Tensor, block_len: int) -> tf.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    _local_attention_mask_dtype = local_attention_mask.dtype
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = tf.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    local_attention_mask = tf.logical_and(
        tf.cast(local_attention_mask, tf.bool), tf.cast(local_attention_mask, tf.bool)
    )
    return tf.cast(local_attention_mask, _local_attention_mask_dtype)


def _get_local_attention_mask(attention_mask: tf.Tensor, block_len: int) -> tf.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = tf.expand_dims(_blocked_attention_mask, -1)
    _3blocked_attention_mask = tf.expand_dims(_3blocked_attention_mask, -2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = tf.logical_and(
        tf.cast(_blocked_attention_mask, tf.bool), tf.cast(_3blocked_attention_mask, tf.bool)
    )
    local_attention_mask = tf.cast(local_attention_mask, _blocked_attention_mask.dtype)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # [batch_size, 1, num_block, block_len, 3 * block_len]
    return tf.expand_dims(local_attention_mask, 1)


def _make_global_fixed_block_ids(attention_mask: tf.Tensor, global_block_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    batch_size, seq_len = shape_list(attention_mask)[:2]

    def handle_orphan_tokens(block_ids: tf.Tensor) -> tf.Tensor:
        block_ends = tf.cast((tf.range(seq_len) % global_block_size) == global_block_size - 1, tf.bool)
        true_block_ends = tf.cast(tf.logical_and(block_ends, block_ids >= 0), tf.int32)
        full_blocks = tf.cast(tf.expand_dims(tf.math.reduce_sum(true_block_ends, -1), -1), block_ids.dtype) - 1
        block_ids = tf.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids

    fixed_block_mask = tf.ones_like(attention_mask) / global_block_size
    fixed_block_mask = tf.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = tf.cast(tf.where(attention_mask != 0.0, 1.0, -1000.0), attention_mask.dtype)
    global_block_ids = tf.cast(tf.floor(mask + fixed_block_mask - 1.0), attention_mask.dtype)
    _global_block_ids_lower_bound = tf.constant(-1.0, dtype=global_block_ids.dtype)
    global_block_ids = tf.where(
        global_block_ids > _global_block_ids_lower_bound, global_block_ids, _global_block_ids_lower_bound
    )
    # set padding tokens to -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size
    # [batch_size, seq_len // global_block_size]
    if num_globals > 0:
        _sequence_block_ids_max = tf.expand_dims(tf.reduce_max(global_block_ids, axis=-1), axis=1)
        _sequence_block_ids_max = tf.repeat(_sequence_block_ids_max, num_globals, axis=1)
        _sequence_block_ids_max = tf.transpose(_sequence_block_ids_max, [0, 1])
    else:
        _sequence_block_ids_max = tf.zeros((batch_size, 0), dtype=global_block_ids.dtype)
    global_segment_ids = tf.cumsum(tf.ones((batch_size, num_globals)), axis=-1) - 1
    global_segment_ids = tf.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)

    global_block_ids = tf.cast(global_block_ids, tf.int32)
    global_segment_ids = tf.cast(global_segment_ids, tf.int32)
    return global_block_ids, global_segment_ids


def _make_side_relative_position_ids(attention_mask: tf.Tensor, global_block_size: int) -> tf.Tensor:
    """Create the relative position tensor for local -> global attention."""
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    global_seq_len = shape_list(global_segment_ids)[-1]
    global_positions = tf.range(global_seq_len)
    side_relative_position = global_positions - block_ids[..., None]
    return tf.cast(side_relative_position, tf.int64)


def _create_global_aggregates(hidden_states: tf.Tensor, block_ids: tf.Tensor, global_seq_len: int) -> tf.Tensor:
    """Compute individual block aggregates by summing over individual blocks."""
    # (batch..., seq_len, global_seq_len))
    block_ids = tf.where(block_ids >= 0, block_ids, global_seq_len)
    one_hot_block_ids = tf.one_hot(block_ids, global_seq_len + 1)[..., :-1]
    return tf.einsum("...nd,...ng->...gd", hidden_states, tf.cast(one_hot_block_ids, hidden_states.dtype))


# Copied from transformers.models.t5.modeling_tf_t5.TFT5LayerNorm with T5->LongT5
class TFLongT5LayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        """
        Construct a layernorm module in the LongT5 style No bias and no subtraction of mean.
        """
        super().__init__(**kwargs)
        self.variance_epsilon = epsilon

    def build(self, input_shape):
        """Build shared word embedding layer"""
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)

    def call(self, hidden_states):
        variance = tf.math.reduce_mean(tf.math.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# Copied from transformers.models.t5.modeling_tf_t5.TFT5DenseActDense with T5->LongT5
class TFLongT5DenseActDense(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        wi_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model**-0.5)
        )
        wo_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff**-0.5)
        )
        self.wi = tf.keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        self.wo = tf.keras.layers.Dense(
            config.d_model, use_bias=False, name="wo", kernel_initializer=wo_initializer
        )  # Update init weights as in flax
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.act = get_tf_activation(config.dense_act_fn)

    def call(self, hidden_states, training=False):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_tf_t5.TFT5DenseGatedActDense with T5->LongT5
class TFLongT5DenseGatedActDense(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        wi_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model**-0.5)
        )
        wo_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff**-0.5)
        )
        self.wi_0 = tf.keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi_0", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        self.wi_1 = tf.keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi_1", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        self.wo = tf.keras.layers.Dense(
            config.d_model, use_bias=False, name="wo", kernel_initializer=wo_initializer
        )  # Update init weights as in flax
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.act = get_tf_activation(config.dense_act_fn)

    def call(self, hidden_states, training=False):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_tf_t5.TFT5LayerFF with T5->LongT5
class TFLongT5LayerFF(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.is_gated_act:
            self.DenseReluDense = TFLongT5DenseGatedActDense(config, name="DenseReluDense")
        else:
            self.DenseReluDense = TFLongT5DenseActDense(config, name="DenseReluDense")

        self.layer_norm = TFLongT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, hidden_states, training=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        dense_output = self.DenseReluDense(normed_hidden_states, training=training)
        hidden_states = hidden_states + self.dropout(dense_output, training=training)
        return hidden_states


# Copied from transformers.models.t5.modeling_tf_t5.TFT5Attention with T5->LongT5
class TFLongT5Attention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFLongT5Attention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.has_relative_attention_bias = has_relative_attention_bias
        self.output_attentions = config.output_attentions

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        q_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        )
        k_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        v_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        o_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        self.relative_attention_bias_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )

        self.q = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="q", kernel_initializer=q_initializer
        )  # Update init weights as in flax
        self.k = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="k", kernel_initializer=k_initializer
        )  # Update init weights as in flax
        self.v = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="v", kernel_initializer=v_initializer
        )  # Update init weights as in flax
        self.o = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="o", kernel_initializer=o_initializer
        )  # Update init weights as in flax
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

        self.pruned_heads = set()

    def build(self, input_shape):
        if self.has_relative_attention_bias:
            with tf.name_scope("relative_attention_bias"):
                self.relative_attention_bias = self.add_weight(
                    name="embeddings",
                    shape=[self.relative_attention_num_buckets, self.n_heads],
                    initializer=self.relative_attention_bias_initializer,  # Add initializer
                )

        return super().build(input_shape)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        #        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(tf.math.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(tf.cast(relative_position, tf.float32) / tf.cast(max_exact, tf.float32))
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(relative_position_if_large, num_buckets - 1)
        relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = tf.range(query_length)[:, None]
        memory_position = tf.range(key_length)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = tf.gather(
            self.relative_attention_bias, relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = tf.expand_dims(
            tf.transpose(values, [2, 0, 1]), axis=0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        training=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, query_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = shape_list(hidden_states)[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            real_seq_length += shape_list(past_key_value[0])[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else shape_list(key_value_states)[1]

        def shape(hidden_states):
            """projection"""
            return tf.transpose(
                tf.reshape(hidden_states, (batch_size, -1, self.n_heads, self.key_value_proj_dim)), perm=(0, 2, 1, 3)
            )

        def unshape(hidden_states):
            """compute context"""
            return tf.reshape(tf.transpose(hidden_states, perm=(0, 2, 1, 3)), (batch_size, -1, self.inner_dim))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = tf.concat([past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, query_length, dim_per_head)

        # get key/value
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # to cope with keras serialization
        if self.is_decoder and use_cache:
            present_key_value_state = (key_states, value_states)
        else:
            present_key_value_state = None

        scores = tf.einsum(
            "bnqd,bnkd->bnqk", query_states, key_states
        )  # (batch_size, n_heads, query_length, key_length)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = tf.zeros((1, self.n_heads, real_seq_length, key_length))
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated we want only the last query position bias
            if past_key_value is not None:
                if not self.has_relative_attention_bias:
                    position_bias = position_bias[:, :, -seq_length:, :]
                else:
                    # we might have a padded past structure, in which case we want to fetch the position bias slice
                    # right after the most recently filled past index
                    most_recently_filled_past_index = tf.reduce_max(tf.where(past_key_value[0][0, 0, :, 0] != 0.0))
                    position_bias = dynamic_slice(
                        position_bias,
                        (0, 0, most_recently_filled_past_index + 1, 0),
                        (1, self.n_heads, seq_length, real_seq_length),
                    )

            if mask is not None:
                position_bias = tf.cast(position_bias, dtype=mask.dtype)
                position_bias = position_bias + mask  # (batch_size, n_heads, query_length, key_length)

        scores += position_bias
        weights = stable_softmax(scores, axis=-1)  # (batch_size, n_heads, query_length, key_length)
        weights = self.dropout(weights, training=training)  # (batch_size, n_heads, query_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [self.n_heads],
                message=(
                    f"Head mask for a single layer should be of size {(self.n_heads)}, but is"
                    f" {shape_list(layer_head_mask)}"
                ),
            )
            weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * weights

        attn_output = tf.matmul(weights, value_states)  # (batch_size, n_heads, query_length, dim_per_head)

        attn_output = self.o(unshape(attn_output))

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (weights,)

        return outputs


class TFLongT5LocalAttention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.layer_id = next(TFLongT5LocalAttention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.has_relative_attention_bias = has_relative_attention_bias
        self.output_attentions = config.output_attentions

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1

        # Mesh TensorFlow initialization to avoid scaling before softmax
        q_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        )
        k_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        v_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        o_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        self.relative_attention_bias_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )

        self.q = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="q", kernel_initializer=q_initializer
        )  # Update init weights as in flax
        self.k = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="k", kernel_initializer=k_initializer
        )  # Update init weights as in flax
        self.v = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="v", kernel_initializer=v_initializer
        )  # Update init weights as in flax
        self.o = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="o", kernel_initializer=o_initializer
        )  # Update init weights as in flax
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

        self.pruned_heads = set()

    def build(self, input_shape):
        if self.has_relative_attention_bias:
            with tf.name_scope("relative_attention_bias"):
                self.relative_attention_bias = self.add_weight(
                    name="embeddings",
                    shape=[self.relative_attention_num_buckets, self.n_heads],
                    initializer=self.relative_attention_bias_initializer,  # Add initializer
                )

        return super().build(input_shape)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    # Copied from transformers.models.t5.modeling_tf_t5.TFT5Attention._relative_position_bucket
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        #        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(tf.math.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(tf.cast(relative_position, tf.float32) / tf.cast(max_exact, tf.float32))
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(relative_position_if_large, num_buckets - 1)
        relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, block_length: int) -> tf.Tensor:
        """Compute binned relative position bias"""
        memory_position = tf.range(3 * block_length, dtype=tf.int64)
        context_position = memory_position[block_length:-block_length]

        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # (block_length, 3 * block_length, num_heads)
        values = tf.gather(self.relative_attention_bias, relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = tf.expand_dims(tf.expand_dims(tf.transpose(values, [2, 0, 1]), 0), 0)
        return values

    def call(
        self,
        hidden_states: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        position_bias: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        training: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        batch_size, seq_length = shape_list(hidden_states)[:2]

        def shape(states: tf.Tensor) -> tf.Tensor:
            """projection"""
            return tf.reshape(states, [batch_size, -1, self.n_heads, self.key_value_proj_dim])

        def unshape(states: tf.Tensor) -> tf.Tensor:
            """reshape"""
            return tf.reshape(states, [batch_size, -1, self.inner_dim])

        # get query/key/value states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Compute scores; shape: (batch_size, num_block, n_heads, block_len, 3 * block_len)
        scores = tf.einsum("...qhd,...khd->...hqk", query_states, key_states)

        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = tf.zeros((1, 1, self.n_heads, self.block_len, 3 * self.block_len), dtype=scores.dtype)
            else:
                position_bias = self.compute_bias(self.block_len)

            if mask is not None:
                # Replace masked positions with -1e10 (according to the original implementation))
                mask = tf.where(mask > 0, 0.0, -1e10)
                # We need to adjust position bias shape to be sum with mask
                position_bias = tf.cast(position_bias, dtype=mask.dtype)
                position_bias = position_bias + tf.experimental.numpy.swapaxes(mask, 1, 2)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        weights = stable_softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            tf.debugging.assert_equal(
                shape_list(layer_head_mask),
                [self.n_heads],
                message=(
                    f"Head mask for a single layer should be of size {(self.n_heads)}, but is"
                    f" {shape_list(layer_head_mask)}"
                ),
            )
            weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * weights

        attn_output = unshape(tf.einsum("...hqk,...khd->...qhd", weights, value_states))
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (weights,)

        return outputs


class TFLongT5TransientGlobalAttention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.layer_id = next(TFLongT5TransientGlobalAttention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.has_relative_attention_bias = has_relative_attention_bias
        self.output_attentions = config.output_attentions

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1
        self.global_block_size = config.global_block_size

        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

        # Mesh TensorFlow initialization to avoid scaling before softmax
        q_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        )
        k_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        v_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        o_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        self.relative_attention_bias_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )

        self.q = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="q", kernel_initializer=q_initializer
        )  # Update init weights as in flax
        self.k = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="k", kernel_initializer=k_initializer
        )  # Update init weights as in flax
        self.v = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="v", kernel_initializer=v_initializer
        )  # Update init weights as in flax
        self.o = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="o", kernel_initializer=o_initializer
        )  # Update init weights as in flax
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

        self.pruned_heads = set()

        self.global_input_layer_norm = TFLongT5LayerNorm(
            epsilon=config.layer_norm_epsilon, name="global_input_layer_norm"
        )

    def build(self, input_shape):
        if self.has_relative_attention_bias:
            with tf.name_scope("relative_attention_bias"):
                self.relative_attention_bias = self.add_weight(
                    name="embeddings",
                    shape=[self.relative_attention_num_buckets, self.n_heads],
                    initializer=self.relative_attention_bias_initializer,  # Add initializer
                )

            with tf.name_scope("global_relative_attention_bias"):
                self.global_relative_attention_bias = self.add_weight(
                    name="embeddings",
                    shape=[self.relative_attention_num_buckets, self.n_heads],
                    initializer=self.relative_attention_bias_initializer,  # Add initializer
                )

        return super().build(input_shape)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    # Copied from transformers.models.t5.modeling_tf_t5.TFT5Attention._relative_position_bucket
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        #        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(tf.math.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(tf.cast(relative_position, tf.float32) / tf.cast(max_exact, tf.float32))
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(relative_position_if_large, num_buckets - 1)
        relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, block_length: int) -> tf.Tensor:
        """Compute binned relative position bias"""
        memory_position = tf.range(3 * block_length, dtype=tf.int64)
        context_position = memory_position[block_length:-block_length]

        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # (block_length, 3 * block_length, num_heads)
        values = tf.gather(self.relative_attention_bias, relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = tf.expand_dims(tf.expand_dims(tf.transpose(values, [2, 0, 1]), 0), 0)
        return values

    def compute_side_bias(self, mask: tf.Tensor, global_segment_ids: tf.Tensor) -> tf.Tensor:

        # (batch_size, seq_len, global_seq_len)
        side_attention_bias = tf.math.equal(mask[..., None], tf.cast(global_segment_ids[:, None, :], mask.dtype))
        side_attention_bias = tf.cast(side_attention_bias, mask.dtype)
        # (batch_size, 1, seq_len, global_seq_len)
        side_attention_bias = side_attention_bias[:, None, ...]
        attention_side_bias = tf.where(side_attention_bias > 0, 0.0, -1e10)
        # (batch_size, seq_len, global_seq_len)
        side_relative_position = _make_side_relative_position_ids(mask, self.global_block_size)
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (batch_size, seq_len, global_seq_len, num_heads)
        side_bias = tf.gather(self.global_relative_attention_bias, side_relative_position_bucket)
        # (batch_size, num_heads, seq_len, global_seq_len)
        side_bias = tf.transpose(side_bias, [0, 3, 1, 2])
        # (batch_size, num_heads, seq_len, global_seq_len)
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias

    def call(
        self,
        hidden_states: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        position_bias: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        training: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        batch_size, seq_length = shape_list(hidden_states)[:2]

        def shape(states: tf.Tensor) -> tf.Tensor:
            """projection"""
            return tf.reshape(states, [batch_size, -1, self.n_heads, self.key_value_proj_dim])

        def unshape(states: tf.Tensor) -> tf.Tensor:
            """reshape"""
            return tf.reshape(states, [batch_size, -1, self.inner_dim])

        # Prepare components for transient-global attention
        # Obtain block_ids and global_segment_ids
        # global_seq_len := seq_len // self.global_block_size
        # shapes: (batch_size, seq_len) & (batch_size, global_seq_len)
        block_ids, global_segment_ids = _make_global_fixed_block_ids(
            mask if mask is not None else tf.ones(shape_list(hidden_states)[:-1]), self.global_block_size
        )
        # Create global inputs
        _global_seq_len = shape_list(global_segment_ids)[-1]
        global_inputs = _create_global_aggregates(hidden_states, block_ids, _global_seq_len)
        global_inputs = self.global_input_layer_norm(global_inputs)

        # get query states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        # Get global/side key/value states  shape: (batch_size, global_seq_len, n_heads, dim_per_head)
        side_key_states = shape(self.k(global_inputs))
        side_value_states = shape(self.v(global_inputs))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Tile side inputs across local key/value blocks
        # New shape: (batch_size, num_blocks, global_seq_len, n_heads, dim_per_head)
        reps = shape_list(key_states)[1]
        side_key_states = tf.repeat(tf.expand_dims(side_key_states, 1), reps, 1)
        side_value_states = tf.repeat(tf.expand_dims(side_value_states, 1), reps, 1)

        # Concatenate "local" and "side"/"global" key/value states to allow each token to attend global aggregated ones
        # New shape: (batch_size, num_blocks, 3 * block_len + global_seq_len, n_heads, dim_per_head)
        key_states = tf.concat([key_states, side_key_states], axis=2)
        value_states = tf.concat([value_states, side_value_states], axis=2)

        # Compute scores -> (batch_size, num_block, n_heads, block_len, 3 * block_len + global_seq_len)
        scores = tf.einsum("...qhd,...khd->...hqk", query_states, key_states)

        if mask is not None:
            # We need to adjust position bias shape to be sum with mask
            local_attention_mask = _get_local_attention_mask(mask, self.block_len)
            # Replace masked positions with -10_000 (according to the original implementation)
            local_attention_mask = tf.where(local_attention_mask > 0, 0.0, -1e10)
        else:
            local_attention_mask = None

        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = tf.zeros((1, 1, self.n_heads, self.block_len, 3 * self.block_len), dtype=scores.dtype)
            else:
                position_bias = self.compute_bias(self.block_len)

            if local_attention_mask is not None:
                # (batch_size, 1, n_heads, block_len, 3 * block_len)
                position_bias = position_bias + tf.experimental.numpy.swapaxes(local_attention_mask, 1, 2)
            position_bias = tf.cast(position_bias, scores.dtype)

            # Calculate global/side bias - shape: # (batch_size, num_heads, seq_len, global_seq_len)
            if mask is None:
                mask = tf.ones((batch_size, seq_length))
            side_position_bias = self.compute_side_bias(mask, global_segment_ids)
            # (batch_size, num_blocks, num_heads, block_len, global_seq_len)
            side_position_bias = _split_into_blocks(side_position_bias, self.block_len, dim=-2)
            side_position_bias = tf.experimental.numpy.swapaxes(side_position_bias, 1, 2)
            side_position_bias = tf.cast(side_position_bias, scores.dtype)
            # (batch_size, num_blocks, num_heads, block_len, 3 * block_len + global_seq_len)
            position_bias = tf.concat([position_bias, side_position_bias], axis=-1)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len + global_seq_len)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = tf.reshape(layer_head_mask, (1, 1, -1, 1, 1)) * attn_weights
        attn_weights = tf.cast(attn_weights, value_states.dtype)
        attn_output = unshape(tf.einsum("...hqk,...khd->...qhd", attn_weights, value_states))
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


# Copied from transformers.models.t5.modeling_tf_t5.TFT5LayerSelfAttention with T5->LongT5
class TFLongT5LayerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.SelfAttention = TFLongT5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name="SelfAttention",
        )
        self.layer_norm = TFLongT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TFLongT5LayerLocalSelfAttention(tf.keras.layers.Layer):
    """Local self attention used in encoder"""

    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.LocalSelfAttention = TFLongT5LocalAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name="LocalSelfAttention",
        )
        self.layer_norm = TFLongT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_bias: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        training: bool = False,
        **kwargs: Any,  # to accept past_key_value and use_cache kwargs
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.LocalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TFLongT5LayerTransientGlobalSelfAttention(tf.keras.layers.Layer):
    """Transient-Global self attention used in encoder"""

    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.TransientGlobalSelfAttention = TFLongT5TransientGlobalAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name="TransientGlobalSelfAttention",
        )
        self.layer_norm = TFLongT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_bias: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        training: bool = False,
        **kwargs: Any,  # to accept past_key_value and use_cache kwargs
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.TransientGlobalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_tf_t5.TFT5LayerCrossAttention with T5->LongT5
class TFLongT5LayerCrossAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.EncDecAttention = TFLongT5Attention(
            config,
            has_relative_attention_bias=False,
            name="EncDecAttention",
        )
        self.layer_norm = TFLongT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TFLongT5Block(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.is_decoder = config.is_decoder
        if config.is_decoder:
            attention_layer = TFLongT5LayerSelfAttention
        elif config.encoder_attention_type == "local":
            attention_layer = TFLongT5LayerLocalSelfAttention
        elif config.encoder_attention_type == "transient-global":
            attention_layer = TFLongT5LayerTransientGlobalSelfAttention
        else:
            raise ValueError(
                "For encoder attention mechanism, either `local` or `transient-global` attention type is expected, "
                f"but got {config.encoder_attention_type}."
            )
        self.layer = []
        self.layer.append(
            attention_layer(config, has_relative_attention_bias=has_relative_attention_bias, name="layer_._0")
        )
        if self.is_decoder:
            self.layer.append(TFLongT5LayerCrossAttention(config, name="layer_._1"))
        self.layer.append(TFLongT5LayerFF(config, name=f"layer_._{len(self.layer)}"))

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_bias: Optional[tf.Tensor] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_decoder_position_bias: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        cross_attn_layer_head_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention' if expected_num_past_key_values == 4 else ''}."
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = shape_list(present_key_value_state[0])[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, training=training)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


####################################################
# The full model without a specific pretrained or finetuning head is
# provided as a tf.keras.layers.Layer usually called "TFLongT5MainLayer"
####################################################
@keras_serializable
class TFLongT5MainLayer(tf.keras.layers.Layer):
    config_class = LongT5Config

    def __init__(self, config: LongT5Config, embed_tokens=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.config = config
        self.num_hidden_layers = config.num_layers

        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1

        self.block = [
            TFLongT5Block(config, has_relative_attention_bias=bool(i == 0), name=f"block_._{i}")
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = TFLongT5LayerNorm(epsilon=config.layer_norm_epsilon, name="final_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ) -> Tuple:

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            # if `self.embed_tokens.load_weight_prefix` is set, runs the embedding operation with the correct name
            # scope, so that its weights are registered with the desired name for loading/storing. When `tf.name_scope`
            # is used with a name ending in `/`, that name replaces the current name scope.
            # (embeddings with tf.name_scope: self.embed_tokens.load_weight_prefix/self.embed_tokens.name/embeddings:0)
            context = []
            if hasattr(self.embed_tokens, "load_weight_prefix"):
                context.append(tf.name_scope(self.embed_tokens.load_weight_prefix + "/"))
            with ContextManagers(context):
                # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
                # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
                tf.debugging.assert_less(
                    input_ids,
                    tf.cast(self.embed_tokens.input_dim, dtype=input_ids.dtype),
                    message=(
                        "input_ids must be smaller than the embedding layer's input dimension (got"
                        f" {tf.math.reduce_max(input_ids)} >= {self.embed_tokens.input_dim})"
                    ),
                )
                inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            shape_list(past_key_values[0][0])[2] + seq_length if past_key_values is not None else seq_length
        )

        if attention_mask is None:
            attention_mask = tf.fill((batch_size, mask_seq_length), 1)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = shape_list(encoder_hidden_states)[1]
            encoder_attention_mask = tf.fill((batch_size, encoder_seq_length), 1)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        attention_mask = tf.cast(attention_mask, dtype=inputs_embeds.dtype)
        num_dims_attention_mask = len(shape_list(attention_mask))
        if self.is_decoder:
            if num_dims_attention_mask == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif num_dims_attention_mask == 2:
                # Provided a padding mask of dimensions [batch_size, mask_seq_length]
                # - if the model is a decoder, apply a causal mask in addition to the padding mask
                # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
                seq_ids = tf.range(mask_seq_length)
                causal_mask = tf.less_equal(
                    tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                    seq_ids[None, :, None],
                )
                causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                if past_key_values[0] is not None:
                    extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and  -1e9 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # extended_attention_mask = tf.math.equal(extended_attention_mask,
            #                                         tf.transpose(extended_attention_mask, perm=(-1, -2)))

            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        elif self.config.encoder_attention_type == "local":
            extended_attention_mask = _get_local_attention_mask(attention_mask, self.block_len)
        else:  # we need to use both local attention mask and standard extended mask for transient-global attention
            extended_attention_mask = attention_mask

        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=extended_attention_mask.dtype)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        present_key_value_states = () if use_cache and self.is_decoder else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds, training=training)

        for idx, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                training=training,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, past_key_values, (self-attention weights),
            # (self-attention position bias), (cross-attention position bias), (cross-attention weights),
            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            # append next layer key value states
            if present_key_value_state is not None and use_cache and self.is_decoder:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            # need to check if is decoder here as well for special cases when using keras compile
            if use_cache and self.is_decoder:
                outputs = outputs + (present_key_value_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
                if self.is_decoder:
                    outputs + (all_cross_attentions,)
            return outputs  # last-layer hidden state, (past_key_values), (all hidden states), (all attentions), (all_cross_attentions)

        if self.is_decoder:
            return TFBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
        else:
            return TFBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )


####################################################
# TFLongT5PreTrainedModel is a sub-class of tf.keras.Model
# which take care of loading and saving pretrained weights
# and various common utilities.
# Here you just need to specify a few (self-explanatory)
# pointers for your model.
####################################################
class TFLongT5PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LongT5Config
    base_model_prefix = "transformer"
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"decoder\Wblock[\W_0]+layer[\W_1]+EncDecAttention\Wrelative_attention_bias"]

    @property
    def dummy_inputs(self):
        inputs = tf.constant(DUMMY_INPUTS)
        input_mask = tf.constant(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": inputs,
            "decoder_input_ids": inputs,
            "decoder_attention_mask": input_mask,
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
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        if hasattr(self, "decoder"):
            self.decoder.embed_tokens = self.shared

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
            shifted_input_ids,
        )

        # "Verify that `labels` has only positive values and -100"
        assert_gte0 = tf.debugging.assert_greater_equal(
            shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype)
        )

        # Make sure the assertion op is called by wrapping the result in an identity no-op
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

        return shifted_input_ids


LONGT5_START_DOCSTRING = r"""
    The LongT5 model was proposed in [LongT5: Efficient Text-To-Text Transformer for Long
    Sequences](https://arxiv.org/abs/2112.07916) by Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo
    Ni, Yun-Hsuan Sung and Yinfei Yang. It's an encoder-decoder transformer pre-trained in a text-to-text denoising
    generative setting. LongT5 model is an extension of T5 model, and it enables using one of the two different
    efficient attention mechanisms - (1) Local attention, or (2) Transient-Global attention.

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:
    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.
    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`. If you choose this second option, there
    are three possibilities you can use to gather all the input Tensors in the first positional argument :
    - a single Tensor with `input_ids` only and nothing else: `model(inputs_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Parameters:
        config ([`LongT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LONGT5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on the right or the left.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        decoder_input_ids (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for sequence to sequence training. T5 uses the `pad_token_id` as the starting token for
            `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids`
            have to be input (see `past_key_values`). To know more on how to prepare `decoder_input_ids` for
            pretraining take a look at [T5 Training](./t5#training).
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        decoder_attention_mask (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        decoder_head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(tf.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(tf.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`tf.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix. If
            `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of
            `inputs_embeds`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
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

LONGT5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        inputs (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on the right or the left. Indices can be obtained using [`T5Tokenizer`].
            See [`PreTrainedTokenizer.__call__`] and [`PreTrainedTokenizer.encode`] for details. To know more on how to
            prepare `inputs` for pre-training take a look at [T5 Training](./t5#training).
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

_HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = tf.ones((num_layers,
num_heads))`.
"""


@add_start_docstrings(
    "The bare LongT5 Model transformer outputting raw hidden-stateswithout any specific head on top.",
    LONGT5_START_DOCSTRING,
)
class TFLongT5Model(TFLongT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = tf.keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.d_model,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.initializer_factor),
            name="shared",
        )
        # Additional attribute to specify the expected name scope of the layer (for loading/storing weights)
        self.shared.load_weight_prefix = "shared"

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFLongT5MainLayer(encoder_config, self.shared, name="encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TFLongT5MainLayer(decoder_config, self.shared, name="decoder")

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_input_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        cross_attn_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_outputs: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFSeq2SeqModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFLongT5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = TFLongT5Model.from_pretrained("google/long-t5-local-base")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="tf").input_ids  # Batch size 1

        >>> # forward pass
        >>> outputs = model(input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        past = decoder_outputs[1] if use_cache else None

        if not return_dict:
            if past is not None:
                decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
            return decoder_outputs + encoder_outputs

        return TFSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=past,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values[1:]) if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            cross_attentions=cross_attns,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )


@add_start_docstrings("""LongT5 Model with a `language modeling` head on top.""", LONGT5_START_DOCSTRING)
class TFLongT5ForConditionalGeneration(TFLongT5PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model_dim = config.d_model
        self.shared = tf.keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.d_model,
            embeddings_initializer=get_initializer(self.config.initializer_factor),
            name="shared",
        )
        # Additional attribute to specify the expected name scope of the layer (for loading/storing weights)
        self.shared.load_weight_prefix = "shared"

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFLongT5MainLayer(encoder_config, self.shared, name="encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TFLongT5MainLayer(decoder_config, self.shared, name="decoder")

        if not config.tie_word_embeddings:
            lm_head_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor)
            self.lm_head = tf.keras.layers.Dense(
                config.vocab_size, use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # Update init weights as in flax

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.get_input_embeddings()
        else:
            # in a dense layer the kernel has a shape (last_dim, units), for us (dim, num_tokens)
            # value has a shape (num_tokens, dim) then needs to be transposed
            return tf.transpose(self.lm_head.kernel)

    def set_output_embeddings(self, value):
        if self.config.tie_word_embeddings:
            self.set_input_embeddings(value)
        else:
            lm_head_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=self.config.initializer_factor)
            self.lm_head = tf.keras.layers.Dense(
                shape_list(value)[0], use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # Update init weights as in flax
            # in a dense layer the kernel has a shape (last_dim, units), for us (dim, num_tokens)
            # value has a shape (num_tokens, dim) then needs to be transposed
            transposed_value = tf.transpose(value)
            self.lm_head.kernel = transposed_value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_input_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        cross_attn_head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_outputs: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        decoder_inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFSeq2SeqLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFLongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = TFLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")

        >>> # training
        >>> inputs = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="tf").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="tf").input_ids
        >>> outputs = model(inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> inputs = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(inputs)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you
        ```"""
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = decoder_outputs[0]

        # T5v1.1 does not tie output word embeddings and thus does not require downscaling
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
            logits = tf.matmul(sequence_output, self.shared.weights, transpose_b=True)
        else:
            logits = self.lm_head(sequence_output)

        logits = tf.cast(logits, tf.float32)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        past = decoder_outputs[1] if use_cache else None
        if not return_dict:
            if past is not None:
                decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        # If the user passed a tuple for encoder_outputs, we wrap it in a TFBaseModelOutput when return_dict=True
        elif isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
            hidden_states = None
            attentions = None
            idx = 0
            if output_hidden_states:
                idx += 1
                hidden_states = encoder_outputs[idx]
            if output_attentions:
                idx += 1
                attentions = encoder_outputs[idx]

            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=hidden_states,
                attentions=attentions,
            )

        return TFSeq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values[1:]) if self.config.use_cache else None
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
        input_ids,
        past=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": None,  # needs to be passed to make Keras.layer.__call__ happy
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor) -> tf.Tensor:
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    tf.gather(layer_past_state, beam_idx, axis=0),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare LongT5 Model transformer outputting encoder's raw hidden-stateswithout any specific head on top.",
    LONGT5_START_DOCSTRING,
)
class TFLongT5EncoderModel(TFLongT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = tf.keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.d_model,
            embeddings_initializer=get_initializer(self.config.initializer_factor),
            name="shared",
        )
        # Additional attribute to specify the expected name scope of the layer (for loading/storing weights)
        self.shared.load_weight_prefix = "shared"

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFLongT5MainLayer(encoder_config, self.shared, name="encoder")

    @property
    def dummy_inputs(self):
        return {"input_ids": tf.constant(DUMMY_INPUTS)}

    def get_encoder(self):
        return self.encoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFLongT5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = TFLongT5EncoderModel.from_pretrained("google/long-t5-local-base")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids)
        ```"""

        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return encoder_outputs

        return TFBaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)

    # Copied from transformers.models.distilbert.modeling_tf_distilbert.TFDistilBertModel.serving_output
    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutput(last_hidden_state=output.last_hidden_state, hidden_states=hs, attentions=attns)
