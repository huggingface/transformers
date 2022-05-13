# coding=utf-8
# Copyright 2022 Meta Platforms and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 Data2Vec Vision model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from transformers.tf_utils import shape_list, stable_softmax

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_data2vec_vision import Data2VecVisionConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Data2VecVisionConfig"
_FEAT_EXTRACTOR_FOR_DOC = "BeitFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/data2vec-vision-base"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/data2vec-vision-base-ft1k"
_IMAGE_CLASS_EXPECTED_OUTPUT = "remote control, remote"

TF_DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-vision-base-ft1k",
    # See all Data2VecVision models at https://huggingface.co/models?filter=data2vec-vision
]


@dataclass
class TFData2VecVisionModelOutputWithPooling(TFBaseModelOutputWithPooling):
    """
    Class for outputs of [`TFData2VecVisionModel`].

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: tf.Tensor = None
    pooler_output: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


class TFDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class TFData2VecVisionEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.patch_embeddings = TFPatchEmbeddings(
            config=config, image_size=config.image_size, patch_size=config.patch_size, name="patch_embeddings"
        )
        self.num_patches = self.patch_embeddings.num_patches
        self.config = config

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )
        if self.config.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="mask_token",
            )
        else:
            self.mask_token = None

        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.add_weight(
                shape=(1, self.num_patches + 1, self.config.hidden_size),
                initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
                trainable=True,
                name="position_embeddings",
            )
        else:
            self.position_embeddings = None

        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: Optional[tf.Tensor] = None) -> tf.Tensor:

        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, projection_dim = shape_list(embeddings)

        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))

        if bool_masked_pos is not None:
            mask_tokens = tf.broadcast_to(self.mask_token, (batch_size, seq_len, projection_dim))
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos[..., None]
            w = tf.cast(w, mask_tokens.dtype)
            # since TF doesn't support eager tensor assignment
            embeddings = embeddings * (1 - w) + mask_tokens * w

        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class TFPatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config: Data2VecVisionConfig, image_size: int = 224, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size

        self.projection = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # following torch.nn.Linear
            bias_initializer="zeros",
            name="projection",
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if getattr(height, "numpy", None) and getattr(width, "numpy", None):
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])

        return tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))


class TFData2VecVisionSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
            use_bias=False,
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        if window_size:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                config, window_size=window_size, name="relative_position_bias"
            )
        else:
            self.relative_position_bias = None

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.sqrt_att_head_size

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            # Passing `0.0` to the `relative_position_bias()` layer because otherwise Keras
            # might complain about `Layer.call()` not being invoked properly. In this case this input
            # i.e., 0.0 is not going to be used in any calculations so we're safe.
            attention_scores = attention_scores + self.relative_position_bias(0.0)[None, ...]

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs


class TFData2VecVisionSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFData2VecVisionLayer instead of here (as is the case with other models), due
    to the layernorm applied before each block.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, gamma=None, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states


class TFData2VecVisionAttention(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFData2VecVisionSelfAttention(config, window_size=window_size, name="attention")
        self.dense_output = TFData2VecVisionSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


# Copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate with ViT->Data2VecVision
class TFData2VecVisionIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TFData2VecVisionOutput(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states


class TFData2VecVisionLayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config

        self.attention = TFData2VecVisionAttention(config, window_size=window_size, name="attention")
        self.intermediate = TFData2VecVisionIntermediate(config, name="intermediate")
        self.data2vec_output = TFData2VecVisionOutput(config, name="output")

        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )
        # Using `layers.Activation` instead of `tf.identity` to better control `training`
        # behaviour.
        self.drop_path = (
            TFDropPath(drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )
        self.init_values = config.layer_scale_init_value

    def build(self, input_shape: tf.TensorShape):
        if self.init_values > 0:
            self.lambda_1 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_1",
            )
            self.lambda_2 = self.add_weight(
                shape=(self.config.hidden_size),
                initializer="ones",
                trainable=True,
                name="lambda_2",
            )
            self.lambda_1.assign(self.init_values * tf.ones((self.config.hidden_size)))
            self.lambda_2.assign(self.init_values * tf.ones((self.config.hidden_size)))
        else:
            self.lambda_1, self.lambda_2 = None, None

        super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: Optional["TFData2VecVisionRelativePositionBias"] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_attention_outputs = self.attention(
            # in Data2VecVision, layernorm is applied before self-attention
            input_tensor=self.layernorm_before(inputs=hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            training=training,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Data2VecVision, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.data2vec_output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Taken and modified from here:
# https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/beit.py#L28
class TFData2VecVisionRelativePositionBias(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        self.window_size = window_size
        # +3 for cls_token_pos_len
        # window_size can be something like (14, 14)
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

        self.relative_position_index = self.get_position_index()

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(self.num_relative_distance, self.config.num_attention_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )  # [2*Wh-1 * 2*Ww-1, nH]
        # cls to token & token 2 cls & cls to cls

        super().build(input_shape)

    def get_position_index(self):
        # get pair-wise relative position index for each token inside the window
        xx, yy = tf.meshgrid(range(self.window_size[0]), range(self.window_size[1]))
        coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
        coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])  # [Wh*Ww, Wh*Ww, 2]

        xx = (relative_coords[:, :, 0] + self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        yy = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords = tf.stack([xx, yy], axis=-1)

        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]

        top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 3
        )
        left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (
            self.num_relative_distance - 2
        )
        corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (self.num_relative_distance - 1)

        left_corner = tf.concat([corner, left], axis=0)
        relative_position_index = tf.concat([top, relative_position_index], axis=0)
        relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)  # [Wh*Ww + 1, Wh*Ww + 1]
        return relative_position_index

    def call(self, inputs=None) -> tf.Tensor:
        relative_position_bias = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=0)
        return tf.transpose(relative_position_bias, [2, 0, 1])


class TFData2VecVisionEncoder(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(
                config, window_size=window_size, name="relative_position_bias"
            )
        else:
            self.relative_position_bias = None

        # stochastic depth decay rule
        dpr = [x for x in tf.linspace(0.0, config.drop_path_rate, config.num_hidden_layers)]
        self.layer = [
            TFData2VecVisionLayer(
                config,
                window_size=window_size if config.use_relative_position_bias else None,
                drop_path_rate=dpr[i],
                name=f"layer_._{i}",
            )
            for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # Passing `0.0` to the `relative_position_bias()` layer because otherwise Keras
            # might complain about `Layer.call()` not being invoked properly. In this case this input
            # i.e., 0.0 is not going to be used in any calculations so we're safe.
            relative_position_bias = (
                self.relative_position_bias(0.0) if self.relative_position_bias is not None else None
            )
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

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
class TFData2VecVisionMainLayer(tf.keras.layers.Layer):
    config_class = Data2VecVisionConfig

    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.add_pooling_layer = add_pooling_layer

        self.embeddings = TFData2VecVisionEmbeddings(config, name="embeddings")
        self.encoder = TFData2VecVisionEncoder(
            config, window_size=self.embeddings.patch_embeddings.patch_shape, name="encoder"
        )
        self.layernorm = (
            tf.identity
            if config.use_mean_pooling
            else tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        )

        # We are setting the `data_format` like so because from here on we will revert to the
        # NCHW output format
        self.pooler = TFData2VecVisionPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        bool_masked_pos: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(pixel_values, bool_masked_pos, training=training)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return TFData2VecVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFData2VecVisionPooler(tf.keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.layernorm = (
            tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
            if config.use_mean_pooling
            else None
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(tf.reduce_mean(patch_tokens, axis=1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output


class TFData2VecVisionPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Data2VecVisionConfig
    base_model_prefix = "data2vec_vision"
    main_input_name = "pixel_values"
    _keys_to_ignore_on_load_unexpected = [r"relative_position_index"]

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network. Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(3, self.config.num_channels, self.config.image_size, self.config.image_size),
            dtype=tf.float32,
        )
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
            }
        ]
    )
    def serving(self, inputs):
        """
        Method used for serving the model.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """

        return self.call(inputs)


DATA2VEC_VISION_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.).

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    </Tip>

    Args:
        config ([`Data2VecVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

DATA2VEC_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`BeitFeatureExtractor`]. See
            [`BeitFeatureExtractor.__call__`] for details.

        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
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
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This argument can be used
            in eager mode, in graph mode the value will always be set to True.

        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Data2VecVision Model transformer outputting raw hidden-states without any specific head on top.",
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionModel(TFData2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        self.data2vec_vision = TFData2VecVisionMainLayer(
            config, add_pooling_layer=add_pooling_layer, name="data2vec_vision"
        )

    def get_input_embeddings(self):
        return self.data2vec_vision.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFData2VecVisionModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        bool_masked_pos: Optional[tf.Tensor] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:

        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs


@add_start_docstrings(
    """
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
    the final hidden states of the patch tokens) e.g. for ImageNet.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForImageClassification(TFData2VecVisionPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=True, name="data2vec_vision")

        # Classifier head
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, tuple]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.data2vec_vision(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
