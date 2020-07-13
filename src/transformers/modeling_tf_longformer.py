import logging

import tensorflow as tf

from .modeling_tf_utils import get_initializer, shape_list


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "LongformerTokenizer"
LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
    # See all Longformer models at https://huggingface.co/models?filter=longformer
]


class TFLongformerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id):
        super().__init__()
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

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

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
