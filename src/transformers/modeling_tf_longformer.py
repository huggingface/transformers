import logging

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

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def forward(
        self, hidden_states, attention_mask=None, output_attentions=False,
    ):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`.
        Padding to `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention

        """
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)

        # is index masked or global attention
        is_index_masked = attention_mask < 0
        #        is_index_global_attn = attention_mask > 0
        #        is_global_attn = any(is_index_global_attn.flatten())

        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # attn_probs = (batch_size, seq_len, num_heads, window*2+1)
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0).unsqueeze(dim=-1).unsqueeze(dim=-1)

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, -10000.0
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

        # compute local attention probs from global attention keys and contact over window dim
        #        if is_global_attn:
        # compute global attn indices required through out forward fn
        #            (
        #                max_num_global_attn_indices,
        #                is_index_global_attn_nonzero,
        #                is_local_index_global_attn_nonzero,
        #                is_local_index_no_global_attn_nonzero,
        #            ) = self._get_global_attn_indices(is_index_global_attn)
        # calculate global attn probs from global key
        #            global_key_attn_scores = self._concat_with_global_key_attn_probs(
        #                query_vectors=query_vectors,
        #                key_vectors=key_vectors,
        #                max_num_global_attn_indices=max_num_global_attn_indices,
        #                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
        #                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
        #                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
        #            )
        # concat to attn_probs
        # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        #            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
        #
        # free memory
        #            del global_key_attn_scores

        attn_probs_fp32 = F.softmax(attn_scores, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        attn_probs = attn_probs_fp32.type_as(attn_scores)

        # free memory
        del attn_probs_fp32

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked.unsqueeze(-1).unsqueeze(-1), 0.0)

        # apply dropout
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        #        if is_global_attn:
        # compute sum of global and local attn
        #            attn_output = self._compute_attn_output_with_global_indices(
        #                value_vectors=value_vectors,
        #                attn_probs=attn_probs,
        #                max_num_global_attn_indices=max_num_global_attn_indices,
        #                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
        #                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
        #            )
        #        else:
        if True:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        #        if is_global_attn:
        #            global_attn_output = self._compute_global_attn_output_from_hidden(
        #                hidden_states=hidden_states,
        #                max_num_global_attn_indices=max_num_global_attn_indices,
        #                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
        #                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
        #                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
        #                is_index_masked=is_index_masked,
        #            )
        #
        # get only non zero global attn output
        #            nonzero_global_attn_output = global_attn_output[
        #                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
        #            ]
        # overwrite values with global attention
        #            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
        #                len(is_local_index_global_attn_nonzero[0]), -1
        #            )

        attn_output = attn_output.transpose(0, 1)

        if output_attentions:
            #            if is_global_attn:
            # With global attention, return global attention probabilities only
            # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
            # which is the attention weights from tokens with global attention to all tokens
            # It doesn't not return local attention
            # In case of variable number of global attantion in the rows of a batch,
            # attn_probs are padded with -10000.0 attention scores
            #                attn_probs = attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            #            else:
            if True:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_probs = attn_probs.permute(0, 2, 1, 3)

        outputs = (attn_output, attn_probs) if output_attentions else (attn_output,)
        return outputs

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """Matrix multiplication of query and key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size window_overlap"""
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)

        # matrix multipication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x window_overlap
        chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunked_query, chunked_key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors.
           Returned tensor will be of the same shape as `attn_probs`"""
        batch_size, seq_len, num_heads, head_dim = value.size()
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = F.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

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

    def _mask_invalid_locations(self, input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8


class TFLongformerAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFLongformerSelfAttention(config, layer_id, name="self")
        self.dense_output = TFBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        input_tensor, attention_mask, head_mask, output_attentions = inputs

        self_outputs = self.self_attention(
            [input_tensor, attention_mask, head_mask, output_attentions], training=training
        )
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TFLongformerLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFLongformerAttention(config, layer_id, name="attention")
        self.intermediate = TFBertIntermediate(config, name="intermediate")
        self.longformer_output = TFBertOutput(config, name="output")

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask, output_attentions = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask, head_mask, output_attentions], training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.longformer_output([intermediate_output, attention_output], training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class TFLongformerEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layer = [
            TFLongformerLayer(config, i, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states = inputs

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if cast_bool_to_primitive(output_hidden_states) is True:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                [hidden_states, attention_mask, head_mask[i], output_attentions], training=training
            )
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
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

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
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
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
            assert len(inputs) <= 8, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
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

        embedding_output = self.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
        encoder_outputs = self.encoder(
            [embedding_output, extended_attention_mask, head_mask, output_attentions, output_hidden_states],
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class TFLongformerPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = LongformerConfig
    base_model_prefix = "longformer"


class TFLongformerModel(TFLongformerPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.longformer = TFLongformerMainLayer(config, name="bert")

    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        outputs = self.longformer(inputs, **kwargs)
        return outputs
