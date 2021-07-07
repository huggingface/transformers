# coding=utf-8
# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 LED model. """


import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutputWithPast

# Public API
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSharedEmbeddings,
    TFWrappedEmbeddings,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_led import LEDConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "allenai/led-base-16384"
_CONFIG_FOR_DOC = "LEDConfig"
_TOKENIZER_FOR_DOC = "LEDTokenizer"


LARGE_NEGATIVE = -1e8


def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = tf.where(
        shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
    )

    # "Verify that `labels` has only positive values and -100"
    if tf.executing_eagerly():
        assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0))

        # Make sure the assertion op is called by wrapping the result in an identity no-op
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


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


class TFLEDLearnedPositionalEmbedding(TFSharedEmbeddings):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_shape[:2]

        positions = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name="range")
        return super().call(positions)


# Copied from transformers.models.longformer.modeling_tf_longformer.TFLongformerSelfAttention with TFLongformer->TFLEDEncoder
class TFLEDEncoderSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        # separate projection layers for tokens with global attention
        self.query_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query_global",
        )
        self.key_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key_global",
        )
        self.value_global = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value_global",
        )
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.global_dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]

        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def call(
        self,
        inputs,
        training=False,
    ):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`. Padding to
        `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in :meth:`LongformerModel.forward` from 0, 1, 2 to:

            * -10000: no attention
            * 0: local attention
            * +10000: global attention
        """
        # retrieve input args
        (
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked,
            is_index_global_attn,
            is_global_attn,
        ) = inputs

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        batch_size, seq_len, embed_dim = shape_list(hidden_states)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                embed_dim,
                self.embed_dim,
                message=f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}",
            )

        # normalize query
        query_vectors /= tf.math.sqrt(tf.cast(self.head_dim, dtype=query_vectors.dtype))
        query_vectors = tf.reshape(query_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))
        key_vectors = tf.reshape(key_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))

        # attn_probs = (batch_size, seq_len, num_heads, window*2+1)
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            tf.ones(shape_list(attention_mask)),
            attention_mask,
            self.one_sided_attn_window_size,
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(attn_scores),
                [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1],
                message=f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {shape_list(attn_scores)}",
            )

        # compute global attn indices required through out forward fn
        (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        ) = self._get_global_attn_indices(is_index_global_attn)

        # this function is only relevant for global attention
        attn_scores = tf.cond(
            is_global_attn,
            lambda: self._concat_with_global_key_attn_probs(
                attn_scores=attn_scores,
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            ),
            lambda: attn_scores,
        )
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        # Make sure to create a mask with the proper shape:
        # if is_global_attn==True => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
        # if is_global_attn==False => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1]
        masked_index = tf.cond(
            is_global_attn,
            lambda: tf.tile(
                is_index_masked[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
            ),
            lambda: tf.tile(
                is_index_masked[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
            ),
        )
        attn_probs = tf.where(
            masked_index,
            tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype),
            attn_probs,
        )

        if layer_head_mask is not None:
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(layer_head_mask),
                    [self.num_heads],
                    message=f"Head mask for a single layer should be of size {(self.num_heads)}, but is {shape_list(layer_head_mask)}",
                )

            attn_probs = tf.reshape(layer_head_mask, (1, 1, -1, 1)) * attn_probs

        # apply dropout
        attn_probs = self.dropout(attn_probs, training=training)
        value_vectors = tf.reshape(value_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))

        # if global attention, compute sum of global and local attn
        attn_output = tf.cond(
            is_global_attn,
            lambda: self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            ),
            lambda: self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            ),
        )

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(attn_output),
                [batch_size, seq_len, self.num_heads, self.head_dim],
                message="Unexpected size",
            )

        attn_output = tf.reshape(attn_output, (batch_size, seq_len, embed_dim))

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        attn_output, global_attn_probs = tf.cond(
            is_global_attn,
            lambda: self._compute_global_attn_output_from_hidden(
                attn_output=attn_output,
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
                training=training,
            ),
            lambda: (attn_output, tf.zeros((batch_size, self.num_heads, max_num_global_attn_indices, seq_len))),
        )

        # make sure that local attention probabilities are set to 0 for indices of global attn
        # Make sure to create a mask with the proper shape:
        # if is_global_attn==True => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
        # if is_global_attn==False => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1]
        masked_global_attn_index = tf.cond(
            is_global_attn,
            lambda: tf.tile(
                is_index_global_attn[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
            ),
            lambda: tf.tile(
                is_index_global_attn[:, :, None, None],
                (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
            ),
        )
        attn_probs = tf.where(
            masked_global_attn_index,
            tf.zeros(shape_list(masked_global_attn_index), dtype=attn_probs.dtype),
            attn_probs,
        )

        outputs = (attn_output, attn_probs, global_attn_probs)

        return outputs

    def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = shape_list(query)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                seq_len % (window_overlap * 2),
                0,
                message=f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}",
            )
            tf.debugging.assert_equal(
                shape_list(query),
                shape_list(key),
                message=f"Shape of query and key should be equal, but got query: {shape_list(query)} and key: {shape_list(key)}",
            )

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = tf.reshape(
            tf.transpose(query, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )
        key = tf.reshape(tf.transpose(key, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))
        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        chunked_query = tf.cast(chunked_query, dtype=chunked_key.dtype)
        chunked_attention_scores = tf.einsum("bcxd,bcyd->bcxy", chunked_query, chunked_key)  # multiply

        # convert diagonals into columns
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 1], [0, 0]])
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(chunked_attention_scores, paddings)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        # TODO: This code is most likely not very efficient and should be improved
        diagonal_attn_scores_up_triang = tf.concat(
            [
                diagonal_chunked_attention_scores[:, :, :window_overlap, : window_overlap + 1],
                diagonal_chunked_attention_scores[:, -1:, window_overlap:, : window_overlap + 1],
            ],
            axis=1,
        )

        # - copying the lower triangle
        diagonal_attn_scores_low_triang = tf.concat(
            [
                tf.zeros(
                    (batch_size * num_heads, 1, window_overlap, window_overlap),
                    dtype=diagonal_chunked_attention_scores.dtype,
                ),
                diagonal_chunked_attention_scores[:, :, -(window_overlap + 1) : -1, window_overlap + 1 :],
            ],
            axis=1,
        )
        diagonal_attn_scores_first_chunk = tf.concat(
            [
                tf.roll(
                    diagonal_chunked_attention_scores,
                    shift=[1, window_overlap],
                    axis=[2, 3],
                )[:, :, :window_overlap, :window_overlap],
                tf.zeros(
                    (batch_size * num_heads, 1, window_overlap, window_overlap),
                    dtype=diagonal_chunked_attention_scores.dtype,
                ),
            ],
            axis=1,
        )
        first_chunk_mask = (
            tf.tile(
                tf.range(chunks_count + 1)[None, :, None, None],
                (batch_size * num_heads, 1, window_overlap, window_overlap),
            )
            < 1
        )
        diagonal_attn_scores_low_triang = tf.where(
            first_chunk_mask,
            diagonal_attn_scores_first_chunk,
            diagonal_attn_scores_low_triang,
        )

        # merging upper and lower triangle
        diagonal_attention_scores = tf.concat(
            [diagonal_attn_scores_low_triang, diagonal_attn_scores_up_triang], axis=-1
        )

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = tf.transpose(
            tf.reshape(
                diagonal_attention_scores,
                (batch_size, num_heads, seq_len, 2 * window_overlap + 1),
            ),
            (0, 2, 1, 3),
        )

        diagonal_attention_scores = self._mask_invalid_locations(diagonal_attention_scores, window_overlap)

        return diagonal_attention_scores

    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        # create correct upper triangle bool mask
        mask_2d_upper = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
            axis=[0],
        )

        # pad to full matrix
        padding = tf.convert_to_tensor(
            [[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]]
        )

        # create lower mask
        mask_2d = tf.pad(mask_2d_upper, padding)

        # combine with upper mask
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

        # broadcast to full matrix
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))

        # inf tensor used for masking
        inf_tensor = -float("inf") * tf.ones_like(input_tensor)

        # mask
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

        return input_tensor

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """

        batch_size, seq_len, num_heads, head_dim = shape_list(value)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                seq_len % (window_overlap * 2),
                0,
                message="Seq_len has to be multiple of 2 * window_overlap",
            )
            tf.debugging.assert_equal(
                shape_list(attn_probs)[:3],
                shape_list(value)[:3],
                message="value and attn_probs must have same dims (except head_dim)",
            )
            tf.debugging.assert_equal(
                shape_list(attn_probs)[3],
                2 * window_overlap + 1,
                message="attn_probs last dim has to be 2 * window_overlap + 1",
            )

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap
        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (
                batch_size * num_heads,
                seq_len // window_overlap,
                window_overlap,
                2 * window_overlap + 1,
            ),
        )

        # group batch_size and num_heads dimensions into one
        value = tf.reshape(
            tf.transpose(value, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        frame_size = 3 * window_overlap * head_dim
        frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size) // chunks_count
        chunked_value = tf.signal.frame(
            tf.reshape(padded_value, (batch_size * num_heads, -1)),
            frame_size,
            frame_hop_size,
        )
        chunked_value = tf.reshape(
            chunked_value,
            (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim),
        )

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(chunked_value),
                [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim],
                message="Chunked value has the wrong shape",
            )

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        context = tf.transpose(
            tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
            (0, 2, 1, 3),
        )

        return context

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # padding value is not important because it will be overwritten
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example::

              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        chunked_hidden_states = tf.pad(
            chunked_hidden_states, paddings
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (total_num_heads, num_chunks, -1)
        )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim),
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]

        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

        # define frame size and frame stride (similar to convolution)
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))

        # chunk with overlap
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(chunked_hidden_states),
                [batch_size, num_output_chunks, frame_size],
                message=f"Make sure chunking is correctly applied. `Chunked hidden states should have output  dimension {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}.",
            )

        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
        )

        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)

        # max number of global attn indices in batch
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)

        # indices of global attn
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)

        # helper variable
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(
            num_global_attn_indices, axis=-1
        )

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        attn_scores,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = shape_list(key_vectors)[0]

        # select global key vectors
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)

        # create only global key vectors
        key_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_key_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)

        # (batch_size, max_num_global_attn_indices, seq_len, num_heads)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )
        mask = tf.ones(mask_shape) * -10000.0
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)

        # scatter mask
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans,
            is_local_index_no_global_attn_nonzero,
            mask,
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        # concat to attn_probs
        # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)

        return attn_scores

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = shape_list(attn_probs)[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]

        # select global value vectors
        global_value_vectors = tf.gather_nd(value_vectors, is_index_global_attn_nonzero)

        # create only global value vectors
        value_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_value_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # compute attn output only global
        attn_output_only_global = tf.einsum("blhs,bshd->blhd", attn_probs_only_global, value_vectors_only_global)

        # reshape attn probs
        attn_probs_without_global = attn_probs[:, :, :, max_num_global_attn_indices:]

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )

        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        attn_output,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
        training,
    ):
        batch_size, seq_len = shape_list(hidden_states)[:2]

        # prepare global hidden states
        global_attn_hidden_states = tf.gather_nd(hidden_states, is_index_global_attn_nonzero)
        global_attn_hidden_states = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_attn_hidden_states,
            shape=(batch_size, max_num_global_attn_indices, self.embed_dim),
        )

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= tf.math.sqrt(
            tf.cast(self.head_dim, dtype=global_query_vectors_only_global.dtype)
        )
        global_query_vectors_only_global = self.reshape_and_transpose(global_query_vectors_only_global, batch_size)
        global_key_vectors = self.reshape_and_transpose(global_key_vectors, batch_size)
        global_value_vectors = self.reshape_and_transpose(global_value_vectors, batch_size)

        # compute attn scores
        global_attn_scores = tf.matmul(global_query_vectors_only_global, global_key_vectors, transpose_b=True)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(global_attn_scores),
                [batch_size * self.num_heads, max_num_global_attn_indices, seq_len],
                message=f"global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {shape_list(global_attn_scores)}.",
            )

        global_attn_scores = tf.reshape(
            global_attn_scores,
            (batch_size, self.num_heads, max_num_global_attn_indices, seq_len),
        )
        global_attn_scores_trans = tf.transpose(global_attn_scores, (0, 2, 1, 3))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(global_attn_scores_trans)[-2:]
        )
        global_attn_mask = tf.ones(mask_shape) * -10000.0
        global_attn_mask = tf.cast(global_attn_mask, dtype=global_attn_scores_trans.dtype)

        # scatter mask
        global_attn_scores_trans = tf.tensor_scatter_nd_update(
            global_attn_scores_trans,
            is_local_index_no_global_attn_nonzero,
            global_attn_mask,
        )
        global_attn_scores = tf.transpose(global_attn_scores_trans, (0, 2, 1, 3))

        # mask global attn scores
        attn_mask = tf.tile(is_index_masked[:, None, None, :], (1, shape_list(global_attn_scores)[1], 1, 1))
        global_attn_scores = tf.where(attn_mask, -10000.0, global_attn_scores)
        global_attn_scores = tf.reshape(
            global_attn_scores,
            (batch_size * self.num_heads, max_num_global_attn_indices, seq_len),
        )

        # compute global attn probs
        global_attn_probs_float = tf.nn.softmax(global_attn_scores, axis=-1)

        # apply layer head masking
        if layer_head_mask is not None:
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(layer_head_mask),
                    [self.num_heads],
                    message=f"Head mask for a single layer should be of size {(self.num_heads)}, but is {shape_list(layer_head_mask)}",
                )
            global_attn_probs_float = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                global_attn_probs_float, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            )
            global_attn_probs_float = tf.reshape(
                global_attn_probs_float, (batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
            )

        # dropout
        global_attn_probs = self.global_dropout(global_attn_probs_float, training=training)

        # global attn output
        global_attn_output = tf.matmul(global_attn_probs, global_value_vectors)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(global_attn_output),
                [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim],
                message=f"global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {shape_list(global_attn_output)}.",
            )

        global_attn_output = tf.reshape(
            global_attn_output,
            (batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim),
        )

        # get only non zero global attn output
        nonzero_global_attn_output = tf.gather_nd(
            tf.transpose(global_attn_output, (0, 2, 1, 3)),
            is_local_index_global_attn_nonzero,
        )
        nonzero_global_attn_output = tf.reshape(
            nonzero_global_attn_output,
            (shape_list(is_local_index_global_attn_nonzero)[0], -1),
        )

        # overwrite values with global attention
        attn_output = tf.tensor_scatter_nd_update(
            attn_output, is_index_global_attn_nonzero, nonzero_global_attn_output
        )

        global_attn_probs = tf.reshape(
            global_attn_probs, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        )

        return attn_output, global_attn_probs

    def reshape_and_transpose(self, vector, batch_size):
        return tf.reshape(
            tf.transpose(
                tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, -1, self.head_dim),
        )


class TFLEDEncoderAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        self.longformer_self_attn = TFLEDEncoderSelfAttention(config, layer_id=layer_id, name="longformer_self_attn")
        self.output_dense = tf.keras.layers.Dense(config.d_model, use_bias=True, name="output")

    def call(self, inputs, training=False):
        (
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_index_masked,
            is_index_global_attn,
            is_global_attn,
        ) = inputs

        self_outputs = self.longformer_self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )

        attention_output = self.output_dense(self_outputs[0], training=training)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class TFLEDDecoderAttention(tf.keras.layers.Layer):
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

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(attn_weights),
                [bsz * self.num_heads, tgt_len, src_len],
                message=f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {shape_list(attn_weights)}",
            )

        if attention_mask is not None:
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(attention_mask),
                    [bsz, 1, tgt_len, src_len],
                    message=f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {shape_list(attention_mask)}",
                )

            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + tf.cast(
                attention_mask, dtype=attn_weights.dtype
            )
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
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


class TFLEDEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: LEDConfig, layer_id: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFLEDEncoderAttention(config, layer_id, name="self_attn")
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        is_index_masked: tf.Tensor,
        is_index_global_attn: tf.Tensor,
        is_global_attn: bool,
        training=False,
    ):
        """
        Args:
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`tf.Tensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
        """
        residual = hidden_states
        layer_outputs = self.self_attn(
            [hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,
        )

        hidden_states = layer_outputs[0]

        if tf.executing_eagerly():
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

        return (hidden_states,) + layer_outputs[1:]


class TFLEDDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: LEDConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFLEDDecoderAttention(
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
        self.encoder_attn = TFLEDDecoderAttention(
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
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`tf.Tensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
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


class TFLEDPreTrainedModel(TFPreTrainedModel):
    config_class = LEDConfig
    base_model_prefix = "led"

    @property
    def dummy_inputs(self):
        input_ids = tf.convert_to_tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0]])
        # make sure global layers are initialized
        attention_mask = tf.convert_to_tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0]])
        global_attention_mask = tf.convert_to_tensor([[0, 0, 0, 0, 1], [0, 0, 1, 0, 0]])
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "decoder_input_ids": input_ids,
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


@dataclass
# Copied from transformers.models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutput with TFLongformer->TFLEDEncoder
class TFLEDEncoderBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where ``x`` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first ``x`` values) and to every token in the attention window (remaining
            ``attention_window + 1`` values). Note that the first ``x`` values refer to tokens with fixed positions in
            the text, but the remaining ``attention_window + 1`` values refer to tokens with relative positions: the
            attention weight of a token to itself is located at index ``x + attention_window / 2`` and the
            ``attention_window / 2`` preceding (succeeding) values are the attention weights to the ``attention_window
            / 2`` preceding (succeeding) tokens. If the attention window contains a token with global attention, the
            attention weight at the corresponding index is set to 0; the value should be accessed from the first ``x``
            attention weights. If a token has global attention, the attention weights to all other tokens in
            :obj:`attentions` is set to 0, the values should be accessed from :obj:`global_attentions`.
        global_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x)`,
            where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    global_attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFLEDSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x)`,
            where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    last_hidden_state: tf.Tensor = None
    past_key_values: Optional[List[tf.Tensor]] = None
    decoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[tf.Tensor]] = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_last_hidden_state: Optional[tf.Tensor] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_global_attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFLEDSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, x)`,
            where ``x`` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    past_key_values: Optional[List[tf.Tensor]] = None
    decoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[tf.Tensor]] = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_last_hidden_state: Optional[tf.Tensor] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_global_attentions: Optional[Tuple[tf.Tensor]] = None


LED_START_DOCSTRING = r"""
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
        config (:class:`~transformers.LEDConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

LED_INPUTS_DOCSTRING = r"""
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
        decoder_input_ids (:obj:`tf.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.LedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            LED uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
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
class TFLEDEncoder(tf.keras.layers.Layer):
    config_class = LEDConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TFLEDEncoderLayer`.

    Args:
        config: LEDConfig
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.attention_window = config.attention_window
        self.embed_tokens = embed_tokens
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_encoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFLEDEncoderLayer(config, i, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        global_attention_mask=None,
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

                Indices can be obtained using :class:`~transformers.LEDTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`tf.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

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
            head_mask=head_mask,
            global_attention_mask=global_attention_mask,
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
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(input_shape, 1)

        # merge `global_attention_mask` and `attention_mask`
        if inputs["global_attention_mask"] is not None:
            inputs["attention_mask"] = inputs["global_attention_mask"] + 1

        (
            padding_len,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["inputs_embeds"],
        ) = self._pad_to_window_size(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            pad_token_id=self.padding_idx,
        )

        input_shape = shape_list(inputs["attention_mask"])
        # is index masked or global attention
        is_index_masked = tf.math.less(tf.cast(inputs["attention_mask"], tf.int8), 1)
        is_index_global_attn = tf.math.greater(tf.cast(inputs["attention_mask"], tf.int8), 1)
        is_global_attn = tf.math.reduce_any(is_index_global_attn)

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs["inputs_embeds"] + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # check attention mask and invert
        if inputs["attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            inputs["attention_mask"] = _expand_mask(inputs["attention_mask"])[:, 0, 0, :]
            inputs["attention_mask"] = inputs["attention_mask"][:, :, None, None]

        encoder_states = () if inputs["output_hidden_states"] else None
        all_attentions = all_global_attentions = () if inputs["output_attentions"] else None

        # check if head_mask has a correct number of layers specified if desired
        if inputs["head_mask"] is not None and tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(inputs["head_mask"])[0],
                len(self.layers),
                message=f"The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(inputs['head_mask'])[0]}.",
            )

        # encoder layers
        for idx, encoder_layer in enumerate(self.layers):

            if inputs["output_hidden_states"]:
                hidden_states_to_add = self.compute_hidden_states(hidden_states, padding_len)
                encoder_states = encoder_states + (hidden_states_to_add,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if inputs["training"] and (dropout_probability < self.layerdrop):  # skip the layer
                continue

            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=inputs["attention_mask"],
                layer_head_mask=inputs["head_mask"][idx] if inputs["head_mask"] is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
            )

            hidden_states = layer_outputs[0]

            if inputs["output_attentions"]:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (tf.transpose(layer_outputs[1], (0, 2, 1, 3)),)

                # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                all_global_attentions = all_global_attentions + (tf.transpose(layer_outputs[2], (0, 1, 3, 2)),)

        # undo padding
        # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
        hidden_states = self.compute_hidden_states(hidden_states, padding_len)

        if inputs["output_hidden_states"]:
            encoder_states = encoder_states + (hidden_states,)

        if not inputs["return_dict"]:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFLEDEncoderBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )

    @tf.function
    def compute_hidden_states(self, hidden_states, padding_len):
        return hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states

    def _pad_to_window_size(
        self,
        input_ids,
        attention_mask,
        inputs_embeds,
        pad_token_id,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        # padding
        attention_window = (
            self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"

        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        if padding_len > 0:
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )

        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])

        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)

        if inputs_embeds is not None:

            def pad_embeddings():
                input_ids_padding = tf.fill((batch_size, padding_len), pad_token_id)
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                return tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)

            inputs_embeds = tf.cond(tf.math.greater(padding_len, 0), pad_embeddings, lambda: inputs_embeds)

        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)  # no attention on the padding tokens

        return (
            padding_len,
            input_ids,
            attention_mask,
            inputs_embeds,
        )


@keras_serializable
class TFLEDDecoder(tf.keras.layers.Layer):
    config_class = LEDConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TFLEDDecoderLayer`

    Args:
        config: LEDConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFLEDLearnedPositionalEmbedding(
            config.max_decoder_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFLEDDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        self.dropout = tf.keras.layers.Dropout(config.dropout)

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
                provide it. Indices can be obtained using :class:`~transformers.LEDTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details. `What are input IDs? <../glossary.html#input-ids>`__
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
                - 0 indicates the head is **masked**.

            encoder_head_mask (:obj:`tf.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`Tuple[Tuple[tf.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding. If :obj:`past_key_values` are used, the user can optionally input only the last
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
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"])

        hidden_states = inputs["inputs_embeds"]

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(
                tf.ones((input_shape[0], input_shape[1] + past_key_values_length)), tgt_len=input_shape[-1]
            )

        if inputs["attention_mask"] is not None and input_shape[-1] > 1:
            combined_attention_mask = combined_attention_mask + _expand_mask(
                inputs["attention_mask"], tgt_len=input_shape[-1]
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

        # check if head_mask has a correct number of layers specified if desired
        if inputs["head_mask"] is not None and tf.executing_eagerly():
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
class TFLEDMainLayer(tf.keras.layers.Layer):
    config_class = LEDConfig

    def __init__(self, config: LEDConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, config.pad_token_id, name="led.shared")

        with tf.compat.v1.variable_scope("led.shared") as shared_abs_scope_name:
            pass

        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        embed_tokens.vocab_size = self.shared.vocab_size
        embed_tokens.hidden_size = self.shared.hidden_size

        self.encoder = TFLEDEncoder(config, embed_tokens, name="encoder")
        self.decoder = TFLEDDecoder(config, embed_tokens, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared.weight = new_embeddings
        self.shared.vocab_size = self.shared.weight.shape[0]
        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("led.shared") as shared_abs_scope_name:
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
        encoder_outputs: Optional[Union[Tuple, TFLEDEncoderBaseModelOutput]] = None,
        global_attention_mask=None,
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
            global_attention_mask=global_attention_mask,
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

        if inputs["encoder_outputs"] is None:
            inputs["encoder_outputs"] = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                global_attention_mask=inputs["global_attention_mask"],
                head_mask=inputs["head_mask"],
                inputs_embeds=inputs["inputs_embeds"],
                output_attentions=inputs["output_attentions"],
                output_hidden_states=inputs["output_hidden_states"],
                return_dict=inputs["return_dict"],
                training=inputs["training"],
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a TFLEDEncoderBaseModelOutput when return_dict=True
        elif inputs["return_dict"] and not isinstance(inputs["encoder_outputs"], TFLEDEncoderBaseModelOutput):
            inputs["encoder_outputs"] = TFLEDEncoderBaseModelOutput(
                last_hidden_state=inputs["encoder_outputs"][0],
                hidden_states=inputs["encoder_outputs"][1] if len(inputs["encoder_outputs"]) > 1 else None,
                attentions=inputs["encoder_outputs"][2] if len(inputs["encoder_outputs"]) > 2 else None,
            )
        # If the user passed a TFLEDEncoderBaseModelOutput for encoder_outputs, we wrap it in a tuple when return_dict=False
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

        return TFLEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=inputs["encoder_outputs"].last_hidden_state,
            encoder_hidden_states=inputs["encoder_outputs"].hidden_states,
            encoder_attentions=inputs["encoder_outputs"].attentions,
            encoder_global_attentions=inputs["encoder_outputs"].global_attentions,
        )


@add_start_docstrings(
    "The bare LED Model outputting raw hidden-states without any specific head on top.",
    LED_START_DOCSTRING,
)
class TFLEDModel(TFLEDPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.led = TFLEDMainLayer(config, name="led")

    def get_encoder(self):
        return self.led.encoder

    def get_decoder(self):
        return self.led.decoder

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLEDSeq2SeqModelOutput,
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
        encoder_outputs: Optional[Union[Tuple, TFLEDEncoderBaseModelOutput]] = None,
        global_attention_mask=None,
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
            global_attention_mask=global_attention_mask,
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
        outputs = self.led(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            encoder_outputs=inputs["encoder_outputs"],
            global_attention_mask=inputs["global_attention_mask"],
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

        return outputs

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

        return TFLEDSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            encoder_global_attentions=enc_g_attns,
        )


@add_start_docstrings(
    "The LED Model with a language modeling head. Can be used for summarization.",
    LED_START_DOCSTRING,
)
class TFLEDForConditionalGeneration(TFLEDPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"led.encoder.embed_tokens.weight",
        r"led.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.led = TFLEDMainLayer(config, name="led")
        self.use_cache = config.use_cache
        # final_bias_logits is registered as a buffer in pytorch, so not trainable for the the sake of consistency.
        self.final_logits_bias = self.add_weight(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        return self.led.decoder

    def get_encoder(self):
        return self.led.encoder

    def get_bias(self):
        return {"final_logits_bias": self.final_logits_bias}

    def set_bias(self, value):
        self.final_logits_bias = value["final_logits_bias"]

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLEDSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs: Optional[TFLEDEncoderBaseModelOutput] = None,
        global_attention_mask=None,
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

            >>> from transformers import LEDTokenizer, TFLEDForConditionalGeneration
            >>> import tensorflow as tf
            >>> mname = 'allenai/led-base-16384'
            >>> tokenizer = LEDTokenizer.from_pretrained(mname)
            >>> TXT = "My friends are <mask> but they eat too many carbs."
            >>> model = TFLEDForConditionalGeneration.from_pretrained(mname)
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
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
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

        outputs = self.led(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            encoder_outputs=inputs["encoder_outputs"],
            global_attention_mask=inputs["global_attention_mask"],
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
        lm_logits = self.led.shared(outputs[0], mode="linear")
        lm_logits = lm_logits + self.final_logits_bias
        masked_lm_loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], lm_logits)

        if not inputs["return_dict"]:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return TFLEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            encoder_last_hidden_state=outputs.last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
            encoder_global_attentions=outputs.encoder_global_attentions,
        )

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        enc_g_attns = tf.convert_to_tensor(output.encoder_global_attentions) if self.config.output_attentions else None

        return TFLEDSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            encoder_global_attentions=enc_g_attns,
        )

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
            encoder_outputs = TFLEDEncoderBaseModelOutput(last_hidden_state=past[0])
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
                encoder_outputs = TFLEDEncoderBaseModelOutput(last_hidden_state=encoder_outputs[0])
            elif isinstance(encoder_outputs, tf.Tensor):
                encoder_outputs = TFLEDEncoderBaseModelOutput(last_hidden_state=encoder_outputs)
            assert (
                past_key_values
            ), f"decoder cached states must be truthy. got {past_key_values} from the 2nd element of past"
            decoder_input_ids = decoder_input_ids[:, -1:]

        assert isinstance(
            encoder_outputs,
            TFLEDEncoderBaseModelOutput,
        ), f"encoder_outputs should be a TFLEDEncoderBaseModelOutput, Instead got {type(encoder_outputs)}."
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

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
        melted_labels = tf.reshape(labels, (-1,))
        active_loss = tf.not_equal(melted_labels, self.config.pad_token_id)
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(melted_labels, active_loss)
        return loss_fn(labels, reduced_logits)
