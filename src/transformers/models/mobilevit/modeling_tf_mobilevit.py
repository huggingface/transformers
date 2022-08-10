# coding=utf-8
# Copyright 2022 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
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
#
# Original license: https://github.com/apple/ml-cvnets/blob/main/LICENSE
""" TensorFlow 2.0 MobileViT model."""

import math
from typing import Dict, Optional, Tuple, Union

import tensorflow as tf


from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "MobileViTConfig"
_FEAT_EXTRACTOR_FOR_DOC = "MobileViTFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/mobilevit-small"
_EXPECTED_OUTPUT_SHAPE = [1, 640, 8, 8]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevit-small"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevit-small",
    "apple/mobilevit-x-small",
    "apple/mobilevit-xx-small",
    "apple/deeplabv3-mobilevit-small",
    "apple/deeplabv3-mobilevit-x-small",
    "apple/deeplabv3-mobilevit-xx-small",
    # See all MobileViT models at https://huggingface.co/models?filter=mobilevit
]


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class TFMobileViTConvLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        padding = int((kernel_size - 1) / 2) * dilation
        self.padding = tf.keras.layers.ZeroPadding2D(padding)

        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        self.convolution = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            dilation=dilation,
            groups=groups,
            use_bias=bias,
            name="convolution"
        )

        if use_normalization:
            self.normalization = tf.keras.layers.BatchNormalization(
                epsilon=1e-5, momentum=0.1,
                name="normalization"
            )
        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                self.activation = get_tf_activation(use_activation)
            elif isinstance(config.hidden_act, str):
                self.activation = get_tf_activation(config.hidden_act)
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    def call(self, features: tf.Tensor) -> tf.Tensor:
        features = self.convolution(self.padding(features))
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features


class TFMobileViTInvertedResidual(tf.keras.layers.Layer):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int = 1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = TFMobileViTConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1, name="expand_1x1"
        )

        self.conv_3x3 = TFMobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            strides=stride,
            groups=expanded_channels,
            dilation=dilation,
            name="conv_3x3"
        )

        self.reduce_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
            name="reduce_1x1"
        )

    def call(self, features: tf.Tensor) -> tf.Tensor:
        residual = features

        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        return residual + features if self.use_residual else features


class MobileViTMobileNetLayer(tf.keras.layers.Layer):
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1, **kwargs
    ) -> None:
        super().__init__( **kwargs)

        self.layers = []
        for i in range(num_stages):
            layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
                name=f"layer.{i}"
            )
            self.layers.append(layer)
            in_channels = out_channels

    def call(self, features: tf.Tensor) -> tf.Tensor:
        for layer_module in self.layers:
            features = layer_module(features)
        return features


class TFMobileViTSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)

        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        scale = tf.cast(self.attention_head_size, dtype=tf.float32)
        self.scale = tf.math.sqrt(scale)

        self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="query")
        self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="key")
        self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name="value")

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        new_x_shape = shape_list(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = tf.reshape(x, shape=new_x_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        batch_size = shape_list(hidden_states)[0]

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, shape=(batch_size, -1, self.all_head_size))
        return context_layer


class TFMobileViTSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TFMobileViTAttention(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = TFMobileViTSelfAttention(config, hidden_size, name="attention")
        self.output = TFMobileViTSelfOutput(config, hidden_size, name="output")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs)
        return attention_output


class TFMobileViTIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(intermediate_size, name="dense")
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TFMobileViTOutput(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Layer(hidden_size, name="dense")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class TFMobileViTTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = TFMobileViTAttention(config, hidden_size, name="attention")
        self.intermediate = TFMobileViTIntermediate(config, hidden_size, intermediate_size, name="intermediate")
        self.output = TFMobileViTOutput(config, hidden_size, intermediate_size, name="output")
        self.layernorm_before = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        self.layernorm_after =  tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


class TFMobileViTTransformer(tf.keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.layers = []
        for i in range(num_stages):
            transformer_layer = TFMobileViTTransformerLayer(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
                name=f"layer.{i}"
            )
            self.layers.append(transformer_layer)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class TFMobileViTLayer(tf.keras.layers.Layer):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        if stride == 2:
            self.downsampling_layer = TFMobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
                name="downsampling_layer"
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        self.conv_kxk = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            name="conv_kxk"
        )

        self.conv_1x1 = TFMobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            name="conv_1x1"
        )

        self.transformer = TFMobileViTTransformer(
            config,
            hidden_size=hidden_size,
            num_stages=num_stages,
            name="transformer"
        )

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

        self.conv_projection = TFMobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1, name="conv_projection"
        )

        self.fusion = TFMobileViTConvLayer(
            config, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size, name="fusion")

    def unfolding(self, features: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = tf.cast(patch_width * patch_height, "int32")

        batch_size, orig_height, orig_width, channels = shape_list(features)

        new_height = tf.cast(tf.math.ceil(orig_height / patch_height) * patch_height, "int32")
        new_width = tf.cast(tf.math.ceil(orig_width / patch_width) * patch_width, "int32")

        interpolate = False
        if new_width != orig_width or new_height != orig_height:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            features = tf.image.resize(
                features, size=(new_height, new_width), method="bilinear"
            )
            interpolate = True

        # number of patches along width and height
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # convert from shape (batch_size, orig_height, orig_width, channels)
        # to the shape (batch_size * patch_area, num_patches, channels)
        features = tf.transpose(features, [0, 3, 1, 2])
        patches = tf.reshape(features, (
            batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width
        ))
        patches = tf.transpose(patches, [0, 2, 1, 3])
        patches = tf.reshape(patches, (batch_size, channels, num_patches, patch_area))
        patches = tf.transpose(patches, [0, 3, 2, 1])
        patches = tf.reshape(patches, (batch_size * patch_area, num_patches, channels))

        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        return patches, info_dict

    def folding(self, patches: tf.Tensor, info_dict: Dict) -> tf.Tensor:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # convert from shape (batch_size * patch_area, num_patches, channels)
        # back to shape (batch_size, channels, orig_height, orig_width)
        features = tf.reshape(patches, (batch_size, patch_area, num_patches, -1))
        features = tf.transpose(features, perm=(0, 3, 2, 1))
        features = tf.reshape(features, (
            batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width
        ))
        features = tf.transpose(features, perm=(0, 2, 1, 3)) 
        features = tf.reshape(features, (
            batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width
        ))
        features = tf.transpose(features, perm=(0, 2, 3, 1))

        if info_dict["interpolate"]:
            features = tf.image.resize(
                features, size=info_dict["orig_size"], method="bilinear"
            )

        return features

    def call(self, features: tf.Tensor) -> tf.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        residual = features

        # local representation
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)

        # convert feature map to patches
        patches, info_dict = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        features = self.folding(patches, info_dict)

        features = self.conv_projection(features)
        features = self.fusion(tf.concat([residual, features], axis=-1))
        return features