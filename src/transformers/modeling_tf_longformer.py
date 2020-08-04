import logging
import math

import tensorflow as tf

from .configuration_longformer import LongformerConfig
from .modeling_tf_bert import TFBertIntermediate, TFBertOutput, TFBertSelfOutput
from .modeling_tf_roberta import TFRobertaEmbeddings
from .modeling_tf_utils import (
    TFPreTrainedModel,
    cast_bool_to_primitive,
    get_initializer,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "LongformerTokenizer"
TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # See all Longformer models at https://huggingface.co/models?filter=longformer
]


class TFLongformerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # separate projection layers for tokens with global attention
        self.query_global = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="query_global"
        )
        self.key_global = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="key_global"
        )
        self.value_global = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="value_global"
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
        self, inputs, training=False,
    ):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`.
        Padding to `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention

        """
        # retrieve input args
        hidden_states, attention_mask, output_attentions = inputs

        attention_mask = tf.squeeze(tf.squeeze(attention_mask, axis=2), axis=1)
        # is index masked or global attention

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = tf.math.reduce_any(is_index_global_attn)

        hidden_states = tf.transpose(hidden_states, (1, 0, 2))

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = shape_list(hidden_states)
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = tf.transpose(
            tf.reshape(query_vectors, (seq_len, batch_size, self.num_heads, self.head_dim)), (1, 0, 2, 3)
        )
        key_vectors = tf.transpose(
            tf.reshape(key_vectors, (seq_len, batch_size, self.num_heads, self.head_dim)), (1, 0, 2, 3)
        )

        # attn_probs = (batch_size, seq_len, num_heads, window*2+1)
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        float_mask = tf.cast((attention_mask != 0)[:, :, None, None], dtype=tf.float32) * -10000.0

        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            tf.ones(shape_list(float_mask), dtype=tf.float32), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert shape_list(attn_scores) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = tf.concat((global_key_attn_scores, attn_scores), axis=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = tf.nn.softmax(attn_scores, axis=-1)

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        # TODO:(PVP) attn_probs = torch.masked_fill(attn_probs, is_index_masked.unsqueeze(-1).unsqueeze(-1), 0.0)

        # mask probs according to attention_mask
        attn_probs = tf.where(is_index_masked[:, :, None, None], 0.0, attn_probs)

        # apply dropout
        attn_probs = self.dropout(attn_probs, training=training)

        value_vectors = tf.transpose(
            tf.reshape(value_vectors, (seq_len, batch_size, self.num_heads, self.head_dim)), (1, 0, 2, 3)
        )

        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert shape_list(attn_output) == [batch_size, seq_len, self.num_heads, self.head_dim], "Unexpected size"
        attn_output = tf.reshape(tf.transpose(attn_output, (1, 0, 2, 3)), (seq_len, batch_size, embed_dim))

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
                training=training,
            )

            # get only non zero global attn output
            nonzero_global_attn_output_trans = tf.gather_nd(
                tf.transpose(global_attn_output, (0, 2, 1, 3)), is_local_index_global_attn_nonzero
            )
            nonzero_global_attn_output = tf.reshape(
                tf.transpose(nonzero_global_attn_output_trans, (0, 2, 1)),
                (shape_list(is_local_index_global_attn_nonzero)[0], -1),
            )

            # overwrite values with global attention
            attn_output = tf.tensor_scatter_nd_update(
                attn_output, tf.reverse(is_local_index_global_attn_nonzero, axis=[1]), nonzero_global_attn_output
            )

        attn_output = tf.transpose(attn_output, (1, 0, 2))

        if output_attentions:
            if is_global_attn:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_probs are padded with -10000.0 attention scores
                attn_probs = tf.reshape(attn_probs, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len))
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_probs = tf.transpose(attn_probs, (0, 2, 1, 3))

        outputs = (attn_output, attn_probs) if output_attentions else (attn_output,)

        import ipdb

        ipdb.set_trace()
        return outputs

    def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
        """Matrix multiplication of query and key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size window_overlap"""
        batch_size, seq_len, num_heads, head_dim = shape_list(query)
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert shape_list(query) == shape_list(key)

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = tf.reshape(tf.transpose(query, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))
        key = tf.reshape(tf.transpose(key, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))

        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)

        # matrix multipication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x window_overlap
        chunked_attention_scores = tf.einsum("bcxd,bcyd->bcxy", chunked_query, chunked_key)  # multiply

        # convert diagonals into columns
        paddings = tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]], dtype=tf.dtypes.int32)
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(chunked_attention_scores, paddings)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
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
                tf.zeros((batch_size * num_heads, 1, window_overlap, window_overlap)),
                diagonal_chunked_attention_scores[:, :, -(window_overlap + 1) : -1, window_overlap + 1 :],
            ],
            axis=1,
        )
        diagonal_attn_scores_first_chunk = tf.concat(
            [
                tf.roll(diagonal_chunked_attention_scores, shift=[1, window_overlap], axis=[2, 3])[
                    :, :, :window_overlap, :window_overlap
                ],
                tf.zeros((batch_size * num_heads, 1, window_overlap, window_overlap)),
            ],
            axis=1,
        )

        first_chunk_mask = (
            tf.broadcast_to(
                tf.range(chunks_count + 1)[None, :, None, None],
                shape=(batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap),
            )
            < 1
        )

        diagonal_attn_scores_low_triang = tf.where(
            first_chunk_mask, diagonal_attn_scores_first_chunk, diagonal_attn_scores_low_triang
        )

        # merging upper and lower triangle
        diagonal_attention_scores = tf.concat(
            [diagonal_attn_scores_low_triang, diagonal_attn_scores_up_triang], axis=-1
        )

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = tf.transpose(
            tf.reshape(diagonal_attention_scores, (batch_size, num_heads, seq_len, 2 * window_overlap + 1)),
            (0, 2, 1, 3),
        )

        diagonal_attention_scores = self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    @staticmethod
    def _mask_invalid_locations(input_tensor_4d, window_overlap):
        # retrieve correct shape of dims to mask
        mask_2d_shape = (shape_list(input_tensor_4d)[1], shape_list(input_tensor_4d)[-1])

        # create correct subband triangle bool mask to filter
        mask_2d = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=mask_2d_shape), window_overlap - 1, window_overlap), axis=[0]
        )

        # broadcast to full matrix
        mask_4d = tf.broadcast_to(mask_2d[None, :, None, :], shape_list(input_tensor_4d))

        # inf tensor used for masking
        inf_tensor = -float("inf") * tf.ones_like(input_tensor_4d, dtype=tf.dtypes.float32)

        # mask
        input_tensor_4d = tf.where(mask_4d < 1, inf_tensor, input_tensor_4d)

        return input_tensor_4d

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):

        """Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors.
           Returned tensor will be of the same shape as `attn_probs`"""

        batch_size, seq_len, num_heads, head_dim = shape_list(value)
        assert seq_len % (window_overlap * 2) == 0
        assert shape_list(attn_probs)[:3] == shape_list(value)[:3]
        assert shape_list(attn_probs)[3] == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1),
        )

        # group batch_size and num_heads dimensions into one
        value = tf.reshape(tf.transpose(value, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end

        paddings = tf.constant([[0, 0], [window_overlap, window_overlap], [0, 0]], dtype=tf.dtypes.int32)
        padded_value = tf.pad(value, paddings, constant_values=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap

        frame_size = 3 * window_overlap * head_dim
        frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size) // chunks_count

        chunked_value = tf.signal.frame(
            tf.reshape(padded_value, (batch_size * num_heads, -1)), frame_size, frame_hop_size
        )
        chunked_value = tf.reshape(
            chunked_value, (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        )

        assert shape_list(chunked_value) == [
            batch_size * num_heads,
            chunks_count + 1,
            3 * window_overlap,
            head_dim,
        ], "Chunked value has the wrong shape"

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        context = tf.transpose(tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)), (0, 2, 1, 3))
        return context

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # padding value is not important because it will be overwritten
        if tf.rank(hidden_states_padded) > 3:
            batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
            hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))
        else:
            batch_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
            hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, hidden_dim, seq_length))
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """shift every row 1 step right, converting columns into diagonals.
           Example:
                 chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                          -1.8348,  0.7672,  0.2986,  0.0285,
                                          -0.7584,  0.4206, -0.0405,  0.1599,
                                          2.0514, -1.1600,  0.5372,  0.2629 ]
                 window_overlap = num_rows = 4
                (pad & diagonilize) =>
                [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
                  0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
                  0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
                  0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)

        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
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
            chunked_hidden_states, (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

        # define frame size and frame stride (similar to convolution)
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size

        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))

        # chunk with overlap
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)

        assert shape_list(chunked_hidden_states) == [
            batch_size,
            num_output_chunks,
            frame_size,
        ], f"Make sure chunking is correctly applied. `Chunked hidden states should have output  dimension {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}."

        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim)
        )

        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """ compute global attn indices required throughout forward pass """
        # helper variable
        num_global_attn_indices = tf.reduce_sum(tf.cast(is_index_global_attn, dtype=tf.dtypes.int32), axis=1)

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
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # select global key vectors
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)
        # create only global key vectors
        key_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_key_vectors,
            shape=(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim),
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)
        # (batch_size, max_num_global_attn_indices, seq_len, num_heads)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )
        mask = tf.ones(mask_shape) * -10000.0

        # scatter mask
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans, is_local_index_no_global_attn_nonzero, mask
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        return attn_probs_from_global_key

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
            shape=(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim),
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
        hidden_states,
        max_num_global_attn_indices,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
        training,
    ):
        seq_len, batch_size = shape_list(hidden_states)[:2]

        # prepare global hidden states
        global_attn_hidden_states = tf.gather_nd(hidden_states, tf.reverse(is_index_global_attn_nonzero, axis=[1]))
        global_attn_hidden_states = tf.scatter_nd(
            tf.reverse(is_local_index_global_attn_nonzero, axis=[1]),
            global_attn_hidden_states,
            shape=(max_num_global_attn_indices, batch_size, self.embed_dim),
        )

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= tf.math.sqrt(tf.constant(self.head_dim, dtype=tf.dtypes.float32))

        # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_query_vectors_only_global = tf.transpose(
            tf.reshape(
                global_query_vectors_only_global,
                (max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim),
            ),
            (1, 0, 2),
        )

        # (..., batch_size * self.num_heads, seq_len, head_dim)
        global_key_vectors = tf.transpose(
            tf.reshape(global_key_vectors, (-1, batch_size * self.num_heads, self.head_dim)), (1, 0, 2)
        )

        # (..., batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = tf.transpose(
            tf.reshape(global_value_vectors, (-1, batch_size * self.num_heads, self.head_dim)), (1, 0, 2)
        )

        # compute attn scores
        global_attn_scores = tf.matmul(global_query_vectors_only_global, tf.transpose(global_key_vectors, (0, 2, 1)))

        assert shape_list(global_attn_scores) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ], f"global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}."

        global_attn_scores = tf.reshape(
            global_attn_scores, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        )

        global_attn_scores_trans = tf.transpose(global_attn_scores, (0, 2, 1, 3))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(global_attn_scores_trans)[-2:]
        )
        global_attn_mask = tf.ones(mask_shape) * -10000.0

        # scatter mask
        global_attn_scores_trans = tf.tensor_scatter_nd_update(
            global_attn_scores_trans, is_local_index_no_global_attn_nonzero, global_attn_mask
        )
        global_attn_scores = tf.transpose(global_attn_scores_trans, (0, 2, 1, 3))

        # mask global attn scores
        attn_mask = tf.broadcast_to(is_index_masked[:, None, None, :], shape_list(global_attn_scores))
        global_attn_scores = tf.where(attn_mask, -10000.0, global_attn_scores)

        global_attn_scores = tf.reshape(
            global_attn_scores, (batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        )

        # compute global attn probs
        global_attn_probs_float = tf.nn.softmax(global_attn_scores, axis=-1)

        # dropout
        global_attn_probs = self.global_dropout(global_attn_probs_float, training=training)

        # global attn output
        global_attn_output = tf.matmul(global_attn_probs, global_value_vectors)

        assert shape_list(global_attn_output) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], f"global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}."

        global_attn_output = tf.reshape(
            global_attn_output, (batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim)
        )
        return global_attn_output


class TFLongformerAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFLongformerSelfAttention(config, layer_id, name="self")
        self.dense_output = TFBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        input_tensor, attention_mask, output_attentions = inputs

        self_outputs = self.self_attention([input_tensor, attention_mask, output_attentions], training=training)
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TFLongformerLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFLongformerAttention(config, layer_id, name="attention")
        self.intermediate = TFBertIntermediate(config, name="intermediate")
        self.longformer_output = TFBertOutput(config, name="output")

    def call(self, inputs, training=False):
        hidden_states, attention_mask, output_attentions = inputs

        attention_outputs = self.attention([hidden_states, attention_mask, output_attentions], training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.longformer_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class TFLongformerEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layer = [
            TFLongformerLayer(config, i, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, output_attentions, output_hidden_states = inputs

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if cast_bool_to_primitive(output_hidden_states) is True:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module([hidden_states, attention_mask, output_attentions], training=training)
            hidden_states = layer_outputs[0]

            if cast_bool_to_primitive(output_attentions) is True:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if cast_bool_to_primitive(output_hidden_states) is True:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if cast_bool_to_primitive(output_hidden_states) is True:
            outputs = outputs + (all_hidden_states,)
        if cast_bool_to_primitive(output_attentions) is True:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


@keras_serializable
class TFLongformerMainLayer(tf.keras.layers.Layer):
    config_class = LongformerConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.pad_token_id = config.pad_token_id
        self.attention_window = config.attention_window

        self.embeddings = TFRobertaEmbeddings(config, name="embeddings")
        self.encoder = TFLongformerEncoder(config, name="encoder")

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        inputs,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            global_attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            output_attentions = inputs[6] if len(inputs) > 6 else output_attentions
            output_hidden_states = inputs[7] if len(inputs) > 7 else output_hidden_states
            assert len(inputs) <= 8, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            global_attention_mask = inputs.get("global_attention_mask", global_attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            assert len(inputs) <= 8, "Too many inputs."
        else:
            input_ids = inputs

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

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

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.pad_token_id,
        )

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

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        encoder_outputs = self.encoder(
            [embedding_output, extended_attention_mask, output_attentions, output_hidden_states], training=training,
        )

        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def _pad_to_window_size(
        self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, pad_token_id,
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
                "Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}".format(
                    seq_len, seq_len + padding_len, attention_window
                )
            )
            paddings = tf.constant([[0, 0], [0, padding_len]])
            if input_ids is not None:
                input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = tf.pad(position_ids, paddings, constant_values=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = tf.fill((batch_size, padding_len), self.config.pad_token_id)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)

            attention_mask = tf.pad(
                attention_mask, paddings, constant_values=False
            )  # no attention on the padding tokens
            token_type_ids = tf.pad(token_type_ids, paddings, constant_values=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    @staticmethod
    def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask


class TFLongformerPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = LongformerConfig
    base_model_prefix = "longformer"

    @property
    def dummy_inputs(self):
        input_ids = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        # make sure global layers are initialized
        attention_mask = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        global_attention_mask = tf.constant([[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
        }


class TFLongformerModel(TFLongformerPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.longformer = TFLongformerMainLayer(config, name="longformer")

    def call(self, inputs, **kwargs):
        outputs = self.longformer(inputs, **kwargs)
        return outputs
