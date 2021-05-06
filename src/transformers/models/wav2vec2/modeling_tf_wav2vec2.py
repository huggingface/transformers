# coding=utf-8
# Copyright 2021 Will Rice (Spokestack) and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 Wav2Vec2 model. """

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import TFPreTrainedModel, get_initializer, input_processing, keras_serializable, shape_list
from ...utils import logging
from .configuration_wav2vec2 import Wav2Vec2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-base-960h"
_CONFIG_FOR_DOC = "Wav2Vec2Config"
_TOKENIZER_FOR_DOC = "Wav2Vec2Tokenizer"

TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]

LARGE_NEGATIVE = -1e8


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[tf.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    Adapted from `fairseq's data_utils.py
    <https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376>`__.
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    padding_mask = tf.math.not_equal(attention_mask, 1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - tf.math.reduce_sum(padding_mask[i])
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


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


class GroupNorm(tf.keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: tf.keras.initializers.Initializer = "zeros",
        gamma_initializer: tf.keras.initializers.Initializer = "ones",
        beta_regularizer: tf.keras.regularizers.Regularizer = None,
        gamma_regularizer: tf.keras.regularizers.Regularizer = None,
        beta_constraint: tf.keras.constraints.Constraint = None,
        gamma_constraint: tf.keras.constraints.Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape


class WeightNormConv1D(tf.keras.layers.Conv1D):
    def __init__(self, config, **kwargs):
        super().__init__(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
            padding="valid",
            use_bias=True,
            bias_initializer="he_normal",
            **kwargs,
        )
        self._padding = config.num_conv_pos_embeddings // 2
        self.filter_axis = 0
        self.initialized = False

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _compute_weights(self):
        """Generate weights with normalization."""

        # `self.kernel_norm_axes` is determined by `self.filter_axis` and the rank
        # of the layer kernel, and is thus statically known.
        self.kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * self.weight_g

    def build(self, input_shape):
        batch_size, seq_length, channels = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=(batch_size, seq_length, channels))

        if not self.built:
            super().build(input_shape)
            kernel_norm_axes = list(range(self.kernel.shape.rank))
            kernel_norm_axes.pop(self.filter_axis)
            # Convert `kernel_norm_axes` from a list to a constant Tensor to allow
            # TF checkpoint saving.
            self.kernel_norm_axes = tf.constant(kernel_norm_axes)
            self.kernel = tf.Variable(self.kernel, name="weight_v", trainable=True)
            self.weight_v = self.kernel

            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        if not self.initialized:
            self._init_norm()
            self.initialized = True

        self._compute_weights()
        output = tf.pad(inputs, ((0, 0), (self._padding, self._padding), (0, 0)))
        output = super().call(output)
        return output


class TFWav2Vec2NoLayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, distribution="uniform"),
            name="conv",
        )
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2LayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=config.conv_dim[layer_id],
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, distribution="uniform"),
            name="conv",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm", epsilon=config.layer_norm_eps)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2GroupNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, distribution="uniform"),
            name="conv",
        )
        self.activation = get_tf_activation(config.feat_extract_activation)
        self.layer_norm = GroupNorm(groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name="layer_norm")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2PositionalConvEmbedding(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conv = WeightNormConv1D(config, name="conv")
        self.padding = TFWav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2SamePadLayer(tf.keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def call(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        return hidden_states


class TFWav2Vec2FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if config.feat_extract_norm == "group":
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0, name=f"conv_layers.{0}")] + [
                TFWav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=f"conv_layers.{i+1}")
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                TFWav2Vec2LayerNormConvLayer(config, layer_id=i, name=f"conv_layers.{i}")
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers

    def call(self, input_values):
        hidden_states = tf.expand_dims(input_values, -1)
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class TFWav2Vec2FeatureProjection(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.projection = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.feat_proj_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with TFBart->TFWav2Vec2
class TFWav2Vec2Attention(tf.keras.layers.Layer):
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


class TFWav2Vec2FeedForward(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        self.intermediate_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        self.intermediate_dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        self.intermediate_act_fn = tf.keras.layers.Activation(config.hidden_act)

        self.output_dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="output_dense",
        )
        self.output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, training=training)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states


class TFWav2Vec2Output(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        return hidden_states


class TFWav2Vec2EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFWav2Vec2EncoderLayerStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFWav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [TFWav2Vec2EncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TFWav2Vec2EncoderStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [
            TFWav2Vec2EncoderLayerStableLayerNorm(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@keras_serializable
class TFWav2Vec2MainLayer(tf.keras.layers.Layer):
    config_class = Wav2Vec2Config

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.feature_extractor = TFWav2Vec2FeatureExtractor(config, name="feature_extractor")
        self.feature_projection = TFWav2Vec2FeatureProjection(config, name="feature_projection")

        if config.do_stable_layer_norm:
            self.encoder = TFWav2Vec2EncoderStableLayerNorm(config, name="encoder")
        else:
            self.encoder = TFWav2Vec2Encoder(config, name="encoder")

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs: Any,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        hidden_states = self.feature_extractor(tf.cast(inputs["input_ids"], tf.float32), training=inputs["training"])
        if inputs["attention_mask"] is not None:
            # compute real output lengths according to convolution formula
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(inputs["attention_mask"]))
            batch_size, seq_length, _ = tf.shape(hidden_states)
            attention_mask = tf.ones((batch_size, seq_length), dtype=tf.int32)
            attention_mask = _expand_mask(attention_mask, output_lengths)

        hidden_states = self.feature_projection(hidden_states, training=inputs["training"])

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        hidden_states = encoder_outputs[0]

        if not inputs["return_dict"]:
            return (hidden_states,) + encoder_outputs[1:]

        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2Config
    base_model_prefix = "wav2vec2"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        pad_token = 1.0
        input_ids = tf.convert_to_tensor(np.random.rand(1, 16000), tf.float32)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": tf.cast(tf.not_equal(input_ids, pad_token), tf.float32),
        }
        return dummy_inputs


WAV_2_VEC_2_START_DOCSTRING = r"""

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
        config (:class:`~transformers.Wav2Vec2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`np.ndarray`, :obj:`tf.Tensor`, :obj:`List[tf.Tensor]` :obj:`Dict[str, tf.Tensor]` or :obj:`Dict[str, np.ndarray]` and each example must have the shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
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
    "The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs: Any,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        """

        Returns:

        Example::

            >>> from transformers import Wav2Vec2Processor, TFWav2Vec2Model
            >>> from datasets import load_dataset
            >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", from_pt=True)

            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
            >>> hidden_states = model(input_values).last_hidden_state
        """

        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        inputs["output_hidden_states"] = (
            inputs["output_hidden_states"] if inputs["output_hidden_states"] else self.config.output_hidden_states
        )
        inputs["output_attentions"] = (
            inputs["output_attentions"] if inputs["output_attentions"] else self.config.output_attentions
        )
        inputs["return_dict"] = inputs["return_dict"] if inputs["return_dict"] else self.config.return_dict

        outputs = self.wav2vec2(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return outputs

    @tf.function
    def serving_output(self, input_ids: tf.Tensor) -> tf.Tensor:
        hidden_states = self.wav2vec2(input_ids, training=False)[0]
        return hidden_states


@add_start_docstrings(
    """TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). """,
    WAV_2_VEC_2_START_DOCSTRING,
)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
        self.dropout = tf.keras.layers.Dropout(config.final_dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Example:

            >>> import tensorflow as tf >>> from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC >>> from
            datasets import load_dataset >>> import soundfile as sf

            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h") >>> model =
            TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

            >>> def map_to_array(batch): >>> speech, _ = sf.read(batch["file"]) >>> batch["speech"] = speech >>> return
            batch

            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation") >>> ds =
            ds.map(map_to_array)

            >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values # Batch size 1 >>> logits =
            model(input_values).logits >>> predicted_ids = tf.argmax(logits, axis=-1)

            >>> transcription = processor.decode(predicted_ids[0])

            >>> # compute loss >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

            >>> # wrap processor as target processor to encode labels >>> with processor.as_target_processor(): >>>
            labels = processor(transcription, return_tensors="tf").input_ids

            >>> loss = model(input_values, labels=labels).loss
        """

        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        outputs = self.wav2vec2(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        logits = self.lm_head(hidden_states)

        if labels is not None:
            attention_mask = (
                attention_mask if attention_mask is not None else tf.ones_like(inputs["input_ids"], dtype=tf.float32)
            )
            input_lengths = self._get_feat_extract_output_lengths(tf.sum(attention_mask, axis=-1))

            labels_mask = labels >= 0
            target_lengths = tf.reduce_sum(labels_mask, axis=-1)
            flattened_targets = tf.sequence_mask(target_lengths, maxlen=target_lengths)

            log_probs = tf.nn.log_softmax(logits, axis=-1)

            loss = tf.nn.ctc_loss(
                logits=log_probs,
                labels=flattened_targets,
                logit_length=input_lengths,
                label_length=target_lengths,
                blank_index=self.config.pad_token_id,
            )
        else:
            loss = None

        if not inputs["return_dict"]:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @tf.function
    def serving_output(self, input_ids: tf.Tensor) -> tf.Tensor:
        hidden_states = self.wav2vec2(input_ids, training=False)[0]
        logits = self.lm_head(hidden_states)
        return logits
