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
""" TF 2.0 Data2VecVision model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from transformers.tf_utils import shape_list, stable_softmax

from ...activations_tf import get_tf_activation
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
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

DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-vision-base-ft1k",
    # See all Data2VecVision models at https://huggingface.co/models?filter=data2vec-vision
]


@dataclass
class Data2VecVisionModelOutputWithPooling(TFBaseModelOutputWithPooling):
    """
    Class for outputs of [`Data2VecVisionModel`].

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

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


# copied from transformers.models.vit.modeling_tf_vit.to_2tuple
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


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

        self.patch_embeddings = TFPatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
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
        batch_size, seq_len, _ = shape_list(embeddings)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos[..., None]
            w = tf.cast(w, mask_tokens.dtype)
            # since TF doesn't support eager tensor assignment
            embeddings_modified = embeddings * (1 - w) + mask_tokens * w
        else:
            embeddings_modified = embeddings

        embeddings = tf.concat([cls_tokens, embeddings_modified], axis=1)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copied from transformers.models.vit.modeling_tf_vit_mae.TFPatchEmbeddings
class TFPatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.

    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        image_size = to_2tuple(config.image_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size
        self.config = config

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
                    f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                )

        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        x = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return x


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
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        if window_size:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(config, window_size=window_size)
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
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias()[None, ...]

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
    The residual connection is defined in TFData2VecVisionLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
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

        self.self_attention = TFData2VecVisionSelfAttention(config, window_size=window_size, name="attention")
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
        self_outputs = self.self_attention(
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
        self.vit_output = TFData2VecVisionOutput(config, name="output")

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
        layer_output = self.output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs
