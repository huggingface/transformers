# coding=utf-8
# Copyright 2020-present Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" TF 2.0 Funnel model. """

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from .activations_tf import get_tf_activation
from .configuration_funnel import FunnelConfig
from .file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_tf_outputs import (
    TFBaseModelOutput,
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
    get_initializer,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FunnelConfig"
_TOKENIZER_FOR_DOC = "FunnelTokenizer"

TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

INF = 1e6


class TFFunnelEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)

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
            return self._embedding(input_ids, inputs_embeds, training=training)
        elif mode == "linear":
            return self._linear(input_ids)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids, inputs_embeds, training=False):
        """Applies embedding based on inputs tensor."""
        assert not (input_ids is None and inputs_embeds is None)
        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)

        embeddings = self.layer_norm(inputs_embeds)
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


class TFFunnelAttentionStructure:
    """
    Contains helpers for `TFFunnelRelMultiheadAttention `.
    """

    cls_token_type_id: int = 2

    def __init__(self, config):
        self.d_model = config.d_model
        self.attention_type = config.attention_type
        self.num_blocks = config.num_blocks
        self.separate_cls = config.separate_cls
        self.truncate_seq = config.truncate_seq
        self.pool_q_only = config.pool_q_only
        self.pooling_type = config.pooling_type

        self.sin_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.cos_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # Track where we are at in terms of pooling from the original input, e.g., by how much the sequence length was
        # dividide.
        self.pooling_mult = None

    def init_attention_inputs(self, input_embeds, attention_mask=None, token_type_ids=None, training=False):
        """ Returns the attention inputs associated to the inputs of the model. """
        # input_embeds has shape batch_size x seq_len x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        self.pooling_mult = 1
        self.seq_len = seq_len = input_embeds.shape[1]
        position_embeds = self.get_position_embeds(seq_len, dtype=input_embeds.dtype, training=training)
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        cls_mask = (
            tf.pad(tf.ones([seq_len - 1, seq_len - 1], dtype=input_embeds.dtype), [[1, 0], [1, 0]])
            if self.separate_cls
            else None
        )
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        token_type_mat = tf.equal(tf.expand_dims(token_type_ids, -1), tf.expand_dims(token_type_ids, -2))
        # Treat <cls> as in the same segment as both A & B
        cls_ids = tf.equal(token_type_ids, tf.constant([self.cls_token_type_id], dtype=token_type_ids.dtype))
        cls_mat = tf.logical_or(tf.expand_dims(cls_ids, -1), tf.expand_dims(cls_ids, -2))
        return tf.logical_or(cls_mat, token_type_mat)

    def get_position_embeds(self, seq_len, dtype=tf.float32, training=False):
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shif attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        if self.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula.
            # We need to create and return the matrics phi, psi, pi and omega.
            pos_seq = tf.range(0, seq_len, 1.0, dtype=dtype)
            freq_seq = tf.range(0, self.d_model // 2, 1.0, dtype=dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (self.d_model // 2)))
            sinusoid = tf.einsum("i,d->id", pos_seq, inv_freq)

            sin_embed = tf.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed, training=training)
            cos_embed = tf.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed, training=training)
            # This is different from the formula on the paper...
            phi = tf.concat([sin_embed_d, sin_embed_d], axis=-1)
            psi = tf.concat([cos_embed, sin_embed], axis=-1)
            pi = tf.concat([cos_embed_d, cos_embed_d], axis=-1)
            omega = tf.concat([-sin_embed, cos_embed], axis=-1)
            return (phi, pi, psi, omega)
        else:
            # Notations from the paper, appending A.2.1, final formula.
            # We need to create and return all the possible vectors R for all blocks and shifts.
            freq_seq = tf.range(0, self.d_model // 2, 1.0, dtype=dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (self.d_model // 2)))
            # Maximum relative positions for the first input
            rel_pos_id = tf.range(-seq_len * 2, seq_len * 2, 1.0, dtype=dtype)
            zero_offset = seq_len * 2
            sinusoid = tf.einsum("i,d->id", rel_pos_id, inv_freq)
            sin_embed = self.sin_dropout(tf.sin(sinusoid), training=training)
            cos_embed = self.cos_dropout(tf.cos(sinusoid), training=training)
            pos_embed = tf.concat([sin_embed, cos_embed], axis=-1)

            pos = tf.range(0, seq_len, dtype=dtype)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.num_blocks):
                # For each block with block_index > 0, we need two types position embeddings:
                #   - Attention(pooled-q, unpooled-kv)
                #   - Attention(pooled-q, pooled-kv)
                # For block_index = 0 we only need the second one and leave the first one as None.

                # First type
                if block_index == 0:
                    position_embeds_pooling = None
                else:
                    pooled_pos = self.stride_pool_pos(pos, block_index)

                    # construct rel_pos_id
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                    # rel_pos = tf.expand_dims(rel_pos,1) + zero_offset
                    # rel_pos = tf.broadcast_to(rel_pos, (rel_pos.shape[0], self.d_model))
                    rel_pos = rel_pos + zero_offset
                    position_embeds_pooling = tf.gather(pos_embed, rel_pos, axis=0)

                # Second type
                pos = pooled_pos
                stride = 2 ** block_index
                rel_pos = self.relative_pos(pos, stride)

                # rel_pos = tf.expand_dims(rel_pos,1) + zero_offset
                # rel_pos = tf.broadcast_to(rel_pos, (rel_pos.shape[0], self.d_model))
                rel_pos = rel_pos + zero_offset
                position_embeds_no_pooling = tf.gather(pos_embed, rel_pos, axis=0)

                position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id, block_index):
        """
        Pool `pos_id` while keeping the cls token separate (if `self.separate_cls=True`).
        """
        if self.separate_cls:
            # Under separate <cls>, we treat the <cls> as the first token in
            # the previous block of the 1st real block. Since the 1st real
            # block always has position 1, the position of the previous block
            # will be at `1 - 2 ** block_index`.
            cls_pos = tf.constant([-(2 ** block_index) + 1], dtype=pos_id.dtype)
            pooled_pos_id = pos_id[1:-1] if self.truncate_seq else pos_id[1:]
            return tf.concat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos

        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        return tf.range(max_dist, min_dist - 1, -stride, dtype=tf.int64)

    def stride_pool(self, tensor, axis):
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None

        # Do the stride pool recursively if axis is a list or a tuple of ints.
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # Do the stride pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # Deal with negative axis
        axis %= tensor.shape.ndims

        axis_slice = slice(None, -1, 2) if self.separate_cls and self.truncate_seq else slice(None, None, 2)
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = tf.concat([tensor[cls_slice], tensor], axis)
        return tensor[enc_slice]

    def pool_tensor(self, tensor, mode="mean", stride=2):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # Do the pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        if self.separate_cls:
            suffix = tensor[:, :-1] if self.truncate_seq else tensor
            tensor = tf.concat([tensor[:, :1], suffix], axis=1)

        ndim = tensor.shape.ndims
        if ndim == 2:
            tensor = tensor[:, :, None]

        if mode == "mean":
            tensor = tf.nn.avg_pool1d(tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        elif mode == "max":
            tensor = tf.nn.max_pool1d(tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        elif mode == "min":
            tensor = -tf.nn.max_pool1d(-tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        return tf.squeeze(tensor, 2) if ndim == 2 else tensor

    def pre_attention_pooling(self, output, attention_inputs):
        """ Pool `output` and the proper parts of `attention_inputs` before the attention layer. """
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.pool_q_only:
            if self.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.pooling_type)
        else:
            self.pooling_mult *= 2
            if self.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            output = self.pool_tensor(output, mode=self.pooling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs):
        """ Pool the proper parts of `attention_inputs` after the attention layer. """
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.pool_q_only:
            self.pooling_mult *= 2
            if self.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs


def _relative_shift_gather(positional_attn, context_len, shift):
    batch_size, n_head, seq_len, max_rel_len = shape_list(positional_attn)
    # max_rel_len = 2 * context_len + shift -1 is the numbers of possible relative positions i-j

    # What's next is the same as doing the following gather in PyTorch, which might be clearer code but less efficient.
    # idxs = context_len + torch.arange(0, context_len).unsqueeze(0) - torch.arange(0, seq_len).unsqueeze(1)
    # # matrix of context_len + i-j
    # return positional_attn.gather(3, idxs.expand([batch_size, n_head, context_len, context_len]))

    positional_attn = tf.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    positional_attn = positional_attn[:, :, shift:, :]
    positional_attn = tf.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    positional_attn = positional_attn[..., :context_len]
    return positional_attn


class TFFunnelRelMultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        self.attention_type = config.attention_type
        self.n_head = n_head = config.n_head
        self.d_head = d_head = config.d_head
        self.d_model = d_model = config.d_model
        self.initializer_range = config.initializer_range
        self.block_index = block_index

        self.hidden_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout)

        initializer = get_initializer(config.initializer_range)

        self.q_head = tf.keras.layers.Dense(
            n_head * d_head, use_bias=False, kernel_initializer=initializer, name="q_head"
        )
        self.k_head = tf.keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name="k_head")
        self.v_head = tf.keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name="v_head")

        self.post_proj = tf.keras.layers.Dense(d_model, kernel_initializer=initializer, name="post_proj")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.scale = 1.0 / (d_head ** 0.5)

    def build(self, input_shape):
        n_head, d_head, d_model = self.n_head, self.d_head, self.d_model
        initializer = get_initializer(self.initializer_range)

        self.r_w_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_w_bias"
        )
        self.r_r_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_r_bias"
        )
        self.r_kernel = self.add_weight(
            shape=(d_model, n_head, d_head), initializer=initializer, trainable=True, name="r_kernel"
        )
        self.r_s_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_s_bias"
        )
        self.seg_embed = self.add_weight(
            shape=(2, n_head, d_head), initializer=initializer, trainable=True, name="seg_embed"
        )
        super().build(input_shape)

    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """ Relative attention score for the positional encodings """
        # q_head has shape batch_size x sea_len x n_head x d_head
        if self.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
            # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
            phi, pi, psi, omega = position_embeds
            # Shape n_head x d_head
            u = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape batch_size x sea_len x n_head x d_model
            q_r_attention = tf.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]

            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = tf.einsum("bind,jd->bnij", q_r_attention_1, psi) + tf.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        else:
            shift = 2 if q_head.shape[1] != context_len else 1
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the proper positional encoding, shape max_rel_len x d_model
            r = position_embeds[self.block_index][shift - 1]
            # Shape n_head x d_head
            v = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape max_rel_len x n_head x d_model
            r_head = tf.einsum("td,dnh->tnh", r, w_r)
            # Shape batch_size x n_head x seq_len x max_rel_len
            positional_attn = tf.einsum("binh,tnh->bnit", q_head + v, r_head)
            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """ Relative attention score for the token_type_ids """
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = shape_list(token_type_mat)
        # q_head has shape batch_size x seq_len x n_head x d_head
        # Shape n_head x d_head
        r_s_bias = self.r_s_bias * self.scale

        # Shape batch_size x n_head x seq_len x 2
        token_type_bias = tf.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        # Shape batch_size x n_head x seq_len x context_len
        new_shape = [batch_size, q_head.shape[2], seq_len, context_len]
        token_type_mat = tf.broadcast_to(token_type_mat[:, None], new_shape)
        # Shapes batch_size x n_head x seq_len
        diff_token_type, same_token_type = tf.split(token_type_bias, 2, axis=-1)
        # Shape batch_size x n_head x seq_len x context_len
        token_type_attn = tf.where(
            token_type_mat, tf.broadcast_to(same_token_type, new_shape), tf.broadcast_to(diff_token_type, new_shape)
        )

        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = shape_list(query)
        context_len = key.shape[1]
        n_head, d_head = self.n_head, self.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = tf.reshape(self.q_head(query), [batch_size, seq_len, n_head, d_head])
        # Shapes batch_size x context_len x n_head x d_head
        k_head = tf.reshape(self.k_head(key), [batch_size, context_len, n_head, d_head])
        v_head = tf.reshape(self.v_head(value), [batch_size, context_len, n_head, d_head])

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        content_score = tf.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        if dtype != tf.float32:
            attn_score = tf.cast(attn_score, tf.float32)
        # perform masking
        if attention_mask is not None:
            attn_score = attn_score - INF * (1 - tf.cast(attention_mask[:, None, None], tf.float32))
        # attention probability
        attn_prob = tf.nn.softmax(attn_score, axis=-1)
        if dtype != tf.float32:
            attn_prob = tf.cast(attn_prob, dtype)
        attn_prob = self.attention_dropout(attn_prob, training=training)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = tf.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(tf.reshape(attn_vec, [batch_size, seq_len, n_head * d_head]))
        attn_out = self.hidden_dropout(attn_out, training=training)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class TFFunnelPositionwiseFFN(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        initializer = get_initializer(config.initializer_range)
        self.linear_1 = tf.keras.layers.Dense(config.d_inner, kernel_initializer=initializer, name="linear_1")
        self.activation_function = get_tf_activation(config.hidden_act)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.linear_2 = tf.keras.layers.Dense(config.d_model, kernel_initializer=initializer, name="linear_2")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")

    def call(self, hidden, training=False):
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h, training=training)
        h = self.linear_2(h)
        h = self.dropout(h, training=training)
        return self.layer_norm(hidden + h)


class TFFunnelLayer(tf.keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFFunnelRelMultiheadAttention(config, block_index, name="attention")
        self.ffn = TFFunnelPositionwiseFFN(config, name="ffn")

    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        attn = self.attention(
            query, key, value, attention_inputs, output_attentions=output_attentions, training=training
        )
        output = self.ffn(attn[0], training=training)
        return (output, attn[1]) if output_attentions else (output,)


class TFFunnelEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.separate_cls = config.separate_cls
        self.pool_q_only = config.pool_q_only
        self.block_repeats = config.block_repeats
        self.attention_structure = TFFunnelAttentionStructure(config)
        self.blocks = [
            [TFFunnelLayer(config, block_index, name=f"blocks_._{block_index}_._{i}") for i in range(block_size)]
            for block_index, block_size in enumerate(config.block_sizes)
        ]

    def call(
        self,
        inputs_embeds,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        training=False,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        # attention_mask = tf.cast(attention_mask, inputs_embeds.dtype)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
        )
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = shape_list(hidden)[1] > (2 if self.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(
                        query, key, value, attention_inputs, output_attentions=output_attentions, training=training
                    )
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return TFBaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


def upsample(x, stride, target_len, separate_cls=True, truncate_seq=False):
    """Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length
    dimension."""
    if stride == 1:
        return x
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    output = tf.repeat(x, repeats=stride, axis=1)
    if separate_cls:
        if truncate_seq:
            output = tf.pad(output, [[0, 0], [0, stride - 1], [0, 0]])
        output = output[:, : target_len - 1]
        output = tf.concat([cls, output], axis=1)
    else:
        output = output[:, :target_len]
    return output


class TFFunnelDecoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.separate_cls = config.separate_cls
        self.truncate_seq = config.truncate_seq
        self.stride = 2 ** (len(config.block_sizes) - 1)
        self.attention_structure = TFFunnelAttentionStructure(config)
        self.layers = [TFFunnelLayer(config, 0, name=f"layers_._{i}") for i in range(config.num_decoder_layers)]

    def call(
        self,
        final_hidden,
        first_block_hidden,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        training=False,
    ):
        upsampled_hidden = upsample(
            final_hidden,
            stride=self.stride,
            target_len=first_block_hidden.shape[1],
            separate_cls=self.separate_cls,
            truncate_seq=self.truncate_seq,
        )

        hidden = upsampled_hidden + first_block_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
        )

        for layer in self.layers:
            layer_output = layer(
                hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions, training=training
            )
            hidden = layer_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return TFBaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


@keras_serializable
class TFFunnelBaseLayer(tf.keras.layers.Layer):
    """ Base model without decoder """

    config_class = FunnelConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        self.embeddings = TFFunnelEmbeddings(config, name="embeddings")
        self.encoder = TFFunnelEncoder(config, name="encoder")

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
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
            inputs_embeds = inputs[3] if len(inputs) > 3 else inputs_embeds
            output_attentions = inputs[4] if len(inputs) > 4 else output_attentions
            output_hidden_states = inputs[5] if len(inputs) > 5 else output_hidden_states
            return_dict = inputs[6] if len(inputs) > 6 else return_dict
            assert len(inputs) <= 7, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            assert len(inputs) <= 7, "Too many inputs."
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

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, training=training)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return encoder_outputs


@keras_serializable
class TFFunnelMainLayer(tf.keras.layers.Layer):
    """ Base model with decoder """

    config_class = FunnelConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.block_sizes = config.block_sizes
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        self.embeddings = TFFunnelEmbeddings(config, name="embeddings")
        self.encoder = TFFunnelEncoder(config, name="encoder")
        self.decoder = TFFunnelDecoder(config, name="decoder")

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
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
            inputs_embeds = inputs[3] if len(inputs) > 3 else inputs_embeds
            output_attentions = inputs[4] if len(inputs) > 4 else output_attentions
            output_hidden_states = inputs[5] if len(inputs) > 5 else output_hidden_states
            return_dict = inputs[6] if len(inputs) > 6 else return_dict
            assert len(inputs) <= 7, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            assert len(inputs) <= 7, "Too many inputs."
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

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, training=training)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            training=training,
        )

        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[1][self.block_sizes[0]],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        return TFBaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


class TFFunnelDiscriminatorPredictions(tf.keras.layers.Layer):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        initializer = get_initializer(config.initializer_range)
        self.dense = tf.keras.layers.Dense(config.d_model, kernel_initializer=initializer, name="dense")
        self.activation_function = get_tf_activation(config.hidden_act)
        self.dense_prediction = tf.keras.layers.Dense(1, kernel_initializer=initializer, name="dense_prediction")

    def call(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation_function(hidden_states)
        logits = tf.squeeze(self.dense_prediction(hidden_states))
        return logits


class TFFunnelMaskedLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, hidden_states, training=False):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


class TFFunnelClassificationHead(tf.keras.layers.Layer):
    def __init__(self, config, n_labels, **kwargs):
        super().__init__(**kwargs)
        initializer = get_initializer(config.initializer_range)
        self.linear_hidden = tf.keras.layers.Dense(
            config.d_model, kernel_initializer=initializer, name="linear_hidden"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.linear_out = tf.keras.layers.Dense(n_labels, kernel_initializer=initializer, name="linear_out")

    def call(self, hidden, training=False):
        hidden = self.linear_hidden(hidden)
        hidden = tf.keras.activations.tanh(hidden)
        hidden = self.dropout(hidden, training=training)
        return self.linear_out(hidden)


class TFFunnelPreTrainedModel(TFPreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = FunnelConfig
    base_model_prefix = "funnel"


@dataclass
class TFFunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.FunnelForPreTrainingModel`.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(tf.ensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


FUNNEL_START_DOCSTRING = r"""

    The Funnel Transformer model was proposed in
    `Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
    <https://arxiv.org/abs/2006.03236>`__ by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

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

    Parameters:
        config (:class:`~transformers.XxxConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

FUNNEL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.FunnelTokenizer`.
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
    """ The base Funnel Transformer Model transformer outputting raw hidden-states without upsampling head (also called
    decoder) or any task-specific head on top.""",
    FUNNEL_START_DOCSTRING,
)
class TFFunnelBaseModel(TFFunnelPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.funnel = TFFunnelBaseLayer(config, name="funnel")

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small-base",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(self, inputs, **kwargs):
        return self.funnel(inputs, **kwargs)


@add_start_docstrings(
    "The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.",
    FUNNEL_START_DOCSTRING,
)
class TFFunnelModel(TFFunnelPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.funnel = TFFunnelMainLayer(config, name="funnel")

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(self, inputs, **kwargs):
        return self.funnel(inputs, **kwargs)


@add_start_docstrings(
    """Funnel model with a binary classification head on top as used during pre-training for identifying generated
    tokens.""",
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForPreTraining(TFFunnelPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.funnel = TFFunnelMainLayer(config, name="funnel")
        self.discriminator_predictions = TFFunnelDiscriminatorPredictions(config, name="discriminator_predictions")

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFFunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import FunnelTokenizer, TFFunnelForPreTraining
            >>> import torch

            >>> tokenizer = TFFunnelTokenizer.from_pretrained('funnel-transformer/small')
            >>> model = TFFunnelForPreTraining.from_pretrained('funnel-transformer/small')

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors= "tf")
            >>> logits = model(inputs).logits
        """
        return_dict = return_dict if return_dict is not None else self.funnel.return_dict

        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)

        if not return_dict:
            return (logits,) + discriminator_hidden_states[1:]

        return TFFunnelForPreTrainingOutput(
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


@add_start_docstrings("""Funnel Model with a `language modeling` head on top. """, FUNNEL_START_DOCSTRING)
class TFFunnelForMaskedLM(TFFunnelPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.funnel = TFFunnelMainLayer(config, name="funnel")
        self.lm_head = TFFunnelMaskedLMHead(config, self.funnel.embeddings, name="lm_head")

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
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
        return_dict = return_dict if return_dict is not None else self.funnel.return_dict
        if isinstance(inputs, (tuple, list)):
            labels = inputs[7] if len(inputs) > 7 else labels
            if len(inputs) > 7:
                inputs = inputs[:7]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.funnel(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, training=training)

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


@add_start_docstrings(
    """Funnel Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForSequenceClassification(TFFunnelPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.funnel = TFFunnelBaseLayer(config, name="funnel")
        self.classifier = TFFunnelClassificationHead(config, config.num_labels, name="classifier")

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small-base",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
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
        return_dict = return_dict if return_dict is not None else self.funnel.return_dict
        if isinstance(inputs, (tuple, list)):
            labels = inputs[7] if len(inputs) > 7 else labels
            if len(inputs) > 7:
                inputs = inputs[:7]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.funnel(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output, training=training)

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
    """Funnel Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForMultipleChoice(TFFunnelPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.funnel = TFFunnelBaseLayer(config, name="funnel")
        self.classifier = TFFunnelClassificationHead(config, 1, name="classifier")

    @property
    def dummy_inputs(self):
        """Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {"input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)}

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small-base",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
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
            inputs_embeds = inputs[3] if len(inputs) > 3 else inputs_embeds
            output_attentions = inputs[4] if len(inputs) > 4 else output_attentions
            output_hidden_states = inputs[5] if len(inputs) > 5 else output_hidden_states
            return_dict = inputs[6] if len(inputs) > 6 else return_dict
            labels = inputs[7] if len(inputs) > 7 else labels
            assert len(inputs) <= 8, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            labels = inputs.get("labels", labels)
            assert len(inputs) <= 8, "Too many inputs."
        else:
            input_ids = inputs
        return_dict = return_dict if return_dict is not None else self.funnel.return_dict

        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        outputs = self.funnel(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output, training=training)
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
    """Funnel Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForTokenClassification(TFFunnelPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.funnel = TFFunnelMainLayer(config, name="funnel")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
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
        return_dict = return_dict if return_dict is not None else self.funnel.return_dict
        if isinstance(inputs, (tuple, list)):
            labels = inputs[7] if len(inputs) > 7 else labels
            if len(inputs) > 7:
                inputs = inputs[:7]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.funnel(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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
    """Funnel Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForQuestionAnswering(TFFunnelPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.funnel = TFFunnelMainLayer(config, name="funnel")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    @add_start_docstrings_to_callable(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
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
        return_dict = return_dict if return_dict is not None else self.funnel.return_dict
        if isinstance(inputs, (tuple, list)):
            start_positions = inputs[7] if len(inputs) > 7 else start_positions
            end_positions = inputs[8] if len(inputs) > 8 else end_positions
            if len(inputs) > 7:
                inputs = inputs[:7]
        elif isinstance(inputs, (dict, BatchEncoding)):
            start_positions = inputs.pop("start_positions", start_positions)
            end_positions = inputs.pop("end_positions", start_positions)

        outputs = self.funnel(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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
            labels = {"start_position": start_positions, "end_position": end_positions}
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
