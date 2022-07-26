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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from transformers.tf_utils import shape_list, stable_softmax

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFSemanticSegmenterOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
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


class TFData2VecVisionDropPath(tf.keras.layers.Layer):
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


class TFData2VecVisionEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.patch_embeddings = TFData2VecVisionPatchEmbeddings(config, name="patch_embeddings")
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


class TFData2VecVisionPatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels

        self.projection = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # following torch.nn.Linear
            bias_initializer="zeros",
            name="projection",
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the"
                    " configuration."
                )
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
            TFData2VecVisionDropPath(drop_path_rate, name="drop_path")
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
        output = self.call(inputs)
        return self.serving_output(output)


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

    def serving_output(self, output: TFData2VecVisionModelOutputWithPooling) -> TFData2VecVisionModelOutputWithPooling:
        hidden_states = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attentions = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFData2VecVisionModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hidden_states,
            attentions=attentions,
        )


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

    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hidden_states = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attentions = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hidden_states, attentions=attentions)


class TFData2VecVisionConvModule(tf.keras.layers.Layer):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: str = "valid",
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=bias,
            dilation_rate=dilation,
            name="conv",
        )
        self.bn = tf.keras.layers.BatchNormalization(name="bn")
        self.activation = tf.nn.relu

    def call(self, input: tf.Tensor) -> tf.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)
        return output


# Copied from:
# https://gist.github.com/Rocketknight1/43abbe6e73f1008e6e459486e01e0ceb
class TFAdaptiveAvgPool1D(tf.keras.layers.Layer):
    def __init__(self, output_dim, mode="dense", **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.mode = mode
        self.map = None

    def build(self, input_shape):
        super().build(input_shape)
        """We pre-compute the sparse matrix for the build() step once. The below code comes
        from https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993."""

        def get_kernels(ind, outd) -> List:
            """Returns a List [(kernel_offset_start,kernel_length)] defining all the pooling kernels for a 1-D adaptive
            pooling layer that takes an input of dimension `ind` and yields an output of dimension `outd`"""

            def start_index(a, b, c):
                return math.floor((float(a) * float(c)) / b)

            def end_index(a, b, c):
                return math.ceil((float(a + 1) * float(c)) / b)

            results = []
            for ow in range(outd):
                start = start_index(ow, outd, ind)
                end = end_index(ow, outd, ind)
                sz = end - start
                results.append((start, sz))
            return results

        in_dim = int(input_shape[-1])
        kernels = get_kernels(in_dim, self.output_dim)
        sparse_map = np.zeros((in_dim, self.output_dim), dtype=np.float32)
        for i, kernel in enumerate(kernels):
            sparse_map[kernel[0] : kernel[0] + kernel[1], i] = 1 / kernel[1]
        if self.mode == "dense":
            self.map = tf.constant(sparse_map)
        else:
            self.map = tf.sparse.from_dense(sparse_map)

    def call(self, inputs):
        if self.mode == "dense":
            return inputs @ self.map
        else:
            input_dims = inputs.shape
            input_matrix = tf.reshape(inputs, (-1, input_dims[-1]))
            out = tf.sparse.sparse_dense_matmul(input_matrix, self.map)
            return tf.reshape(out, input_dims[:-1].as_list() + [-1])

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim, "mode": self.mode})
        return config


class TFAdaptiveAvgPool2D(tf.keras.layers.Layer):
    def __init__(self, output_shape, mode="dense", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.h_pool = TFAdaptiveAvgPool1D(output_shape[0], mode=mode, name="h_pool")
        self.w_pool = TFAdaptiveAvgPool1D(output_shape[1], mode=mode, name="w_pool")

    def call(self, inputs):
        # Rearrange from NHWC -> NCHW
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        # Perform W-pooling
        inputs = self.w_pool(inputs)
        # Rearrange NCHW -> NCWH
        inputs = tf.transpose(inputs, perm=[0, 1, 3, 2])
        # Perform H-pooling
        inputs = self.h_pool(inputs)
        # Rearrange from NCWH -> NHWC
        inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config


class TFData2VecVisionPyramidPoolingModule(tf.keras.layers.Layer):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        channels (int): Channels after modules, before conv_seg.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: Tuple[int, ...], channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pool_scales = pool_scales
        self.channels = channels

        self.layer_list = []
        for idx, pool_scale in enumerate(pool_scales):
            pool_scale = pool_scale if isinstance(pool_scale, collections.abc.Iterable) else (pool_scale, pool_scale)
            self.layer_list.append(
                [
                    TFAdaptiveAvgPool2D(output_shape=pool_scale),
                    TFData2VecVisionConvModule(out_channels=self.channels, kernel_size=1, name=f"{idx}.1"),
                ]
            )

    def call(self, x: tf.Tensor) -> List[tf.Tensor]:
        ppm_outs = []
        inputs = x

        for ppm in self.layer_list:
            for layer_module in ppm:
                ppm_out = layer_module(x)
                x = ppm_out

            upsampled_ppm_out = tf.image.resize(ppm_out, size=shape_list(inputs)[1:-1], method="bilinear")
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class TFData2VecVisionUperHead(tf.keras.layers.Layer):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None:
        super().__init__(**kwargs)

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.classifier = tf.keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")

        # PSP Module
        self.psp_modules = TFData2VecVisionPyramidPoolingModule(self.pool_scales, self.channels, name="psp_modules")
        self.bottleneck = TFData2VecVisionConvModule(self.channels, kernel_size=3, padding="same", name="bottleneck")
        # FPN Module
        self.lateral_convs = []
        self.fpn_convs = []
        for idx, _ in enumerate(self.in_channels[:-1]):  # skip the top layer
            l_conv = TFData2VecVisionConvModule(out_channels=self.channels, kernel_size=1, name=f"lateral_convs.{idx}")
            fpn_conv = TFData2VecVisionConvModule(
                out_channels=self.channels, kernel_size=3, padding="same", name=f"fpn_convs.{idx}"
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = TFData2VecVisionConvModule(
            out_channels=self.channels, kernel_size=3, padding="same", name="fpn_bottleneck"
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = tf.concat(psp_outs, axis=-1)
        output = self.bottleneck(psp_outs)

        return output

    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = shape_list(laterals[i - 1])[1:-1]
            laterals[i - 1] = laterals[i - 1] + tf.image.resize(laterals[i], size=prev_shape, method="bilinear")

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = tf.image.resize(fpn_outs[i], size=shape_list(fpn_outs[0])[1:-1], method="bilinear")
        fpn_outs = tf.concat(fpn_outs, axis=-1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class TFData2VecVisionFCNHead(tf.keras.layers.Layer):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented from
    [FCNNet](https://arxiv.org/abs/1411.4038).

    Args:
        config (Data2VecVisionConfig): Configuration.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        convs = []
        convs.append(
            TFData2VecVisionConvModule(
                out_channels=self.channels,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
                name="convs.0",
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                TFData2VecVisionConvModule(
                    out_channels=self.channels,
                    kernel_size=kernel_size,
                    padding="same",
                    dilation=dilation,
                    name=f"conv_module_{i+2}",
                )
            )
        if self.num_convs == 0:
            self.convs = [tf.identity]
        else:
            self.convs = convs
        if self.concat_input:
            self.conv_cat = TFData2VecVisionConvModule(
                out_channels=self.channels, kernel_size=kernel_size, padding="same", name="conv_cat"
            )

        self.classifier = tf.keras.layers.Conv2D(config.num_labels, kernel_size=1, name="classifier")

    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = hidden_states
        for layer_module in self.convs:
            output = layer_module(output)
        if self.concat_input:
            output = self.conv_cat(tf.concat([hidden_states, output], axis=-1))
        output = self.classifier(output)
        return output


@add_start_docstrings(
    """
    Data2VecVision Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForSemanticSegmentation(TFData2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=False, name="data2vec_vision")

        # FPNs
        self.fpn1 = [
            tf.keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.0"),
            tf.keras.layers.BatchNormalization(name="fpn1.1"),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn1.3"),
        ]
        self.fpn2 = [tf.keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name="fpn2.0")]

        self.fpn3 = tf.identity
        self.fpn4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        # Semantic segmentation head(s)
        self.decode_head = TFData2VecVisionUperHead(config, name="decode_head")
        self.auxiliary_head = (
            TFData2VecVisionFCNHead(config, name="auxiliary_head") if config.use_auxiliary_head else None
        )

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        if len(shape_list(labels)) > 3:
            label_interp_shape = shape_list(labels)[1:-1]
        else:
            label_interp_shape = shape_list(labels)[-2:]

        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method="bilinear")
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = tf.image.resize(auxiliary_logits, size=label_interp_shape, method="bilinear")
        # compute weighted loss
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

        # Copied from https://www.tensorflow.org/text/tutorials/transformer#loss_and_metrics.
        # Utility to mask the index to ignore during computing the loss.
        def masked_loss(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, self.config.semantic_loss_ignore_index))
            loss_ = loss_fct(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        main_loss = masked_loss(labels, upsampled_logits)
        auxiliary_loss = masked_loss(labels, upsampled_auxiliary_logits)
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TFSemanticSegmenterOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, TFData2VecVisionForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/data2vec-vision-base")
        >>> model = TFData2VecVisionForSemanticSegmentation.from_pretrained("facebook/data2vec-vision-base")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.data2vec_vision(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        batch_size = shape_list(pixel_values)[0]
        patch_resolution = self.config.image_size // self.config.patch_size

        def reshape_features(x):
            x = tf.reshape(x, (batch_size, patch_resolution, patch_resolution, -1))
            return x

        features = [reshape_features(x[:, 1:, :]) for x in features]

        # apply FPNs
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for module in ops[0]:
            features[0] = module(features[0])
        features[1] = ops[1][0](features[1])
        for i in range(len(features[2:])):
            features[i + 2] = ops[i + 2](features[i + 2])

        logits = self.decode_head(features)
        # Tranpose the logits to maintain consistency in the output formats.
        transposed_logits = tf.transpose(logits, perm=[0, 3, 1, 2])

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.compute_loss(logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSemanticSegmenterOutput(
            loss=loss,
            logits=transposed_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFSemanticSegmenterOutput) -> TFSemanticSegmenterOutput:
        hidden_states = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attentions = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSemanticSegmenterOutput(logits=output.logits, hidden_states=hidden_states, attentions=attentions)
