# coding=utf-8
# Copyright 2023 Snapchat Research and The HuggingFace Inc. team. All rights reserved.
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
""" TensorFlow EfficientFormer model."""

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_efficientformer import EfficientFormerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "snap-research/efficientformer-l1-300"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 448]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1-300"
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_281"


TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "snap-research/efficientformer-l1-300",
    # See all EfficientFormer models at https://huggingface.co/models?filter=efficientformer
]


class TFEfficientFormerPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """

    def __init__(
        self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_channels = num_channels

        self.padding = tf.keras.layers.ZeroPadding2D(padding=config.downsample_pad)
        self.projection = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=config.downsample_patch_size,
            strides=config.downsample_stride,
            padding="valid",
            name="projection",
        )
        # Use same default momentum and epsilon as PyTorch equivalent for BatchNormalization
        self.norm = (
            tf.keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="norm")
            if apply_norm
            else tf.identity
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        tf.debugging.assert_shapes(
            [(pixel_values, (..., None, None, self.num_channels))],
            message="Make sure that the channel dimension of the pixel values match with the one set in the configuration.",
        )
        embeddings = self.projection(self.padding(pixel_values))
        embeddings = self.norm(embeddings, training=training)
        return embeddings


class TFEfficientFormerSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attention_ratio: int,
        resolution: int,
        config: EfficientFormerConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.scale = key_dim**-0.5
        self.total_key_dim = key_dim * num_heads
        self.expanded_key_dim = int(attention_ratio * key_dim)
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2

        self.qkv = tf.keras.layers.Dense(
            units=hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="qkv"
        )
        self.projection = tf.keras.layers.Dense(
            units=dim, kernel_initializer=get_initializer(config.initializer_range), name="projection"
        )
        self.resolution = resolution

    def build(self, input_shape: tf.TensorShape) -> None:
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        num_points = len(points)
        attention_offsets = {}

        idxs = []

        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = self.add_weight(
            shape=(self.num_heads, len(attention_offsets)),
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            name="attention_biases",
        )
        self.attention_bias_idxs = self.add_weight(
            shape=(num_points, num_points),
            trainable=False,
            dtype=tf.int32,
            name="attention_bias_idxs",
        )

        self.attention_bias_idxs.assign(tf.reshape(tf.cast(idxs, dtype=tf.int32), (num_points, num_points)))

        super().build(input_shape)

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        batch_size, sequence_length, *_ = shape_list(hidden_states)
        qkv = self.qkv(inputs=hidden_states)

        query_layer, key_layer, value_layer = tf.split(
            tf.reshape(tensor=qkv, shape=(batch_size, sequence_length, self.num_heads, -1)),
            num_or_size_splits=[self.key_dim, self.key_dim, self.expanded_key_dim],
            axis=3,
        )

        query_layer = tf.transpose(query_layer, perm=[0, 2, 1, 3])
        key_layer = tf.transpose(key_layer, perm=[0, 2, 1, 3])
        value_layer = tf.transpose(value_layer, perm=[0, 2, 1, 3])

        attention_probs = tf.matmul(query_layer, tf.transpose(key_layer, perm=[0, 1, 3, 2]))
        scale = tf.cast(self.scale, dtype=attention_probs.dtype)
        attention_probs = tf.multiply(attention_probs, scale)

        attention_biases = tf.gather(params=self.attention_biases, indices=self.attention_bias_idxs, axis=1)
        attention_probs = attention_probs + attention_biases
        attention_probs = stable_softmax(logits=attention_probs, axis=-1)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        context_layer = tf.reshape(
            tensor=context_layer, shape=(batch_size, sequence_length, self.total_expanded_key_dim)
        )
        context_layer = self.projection(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TFEfficientFormerConvStem(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, out_channels: int, **kwargs):
        super().__init__(**kwargs)

        self.padding = tf.keras.layers.ZeroPadding2D(padding=1)
        self.convolution1 = tf.keras.layers.Conv2D(
            filters=out_channels // 2, kernel_size=3, strides=2, padding="valid", name="convolution1"
        )
        # Use same default momentum and epsilon as PyTorch equivalent for BatchNormalization
        self.batchnorm_before = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_before"
        )

        self.convolution2 = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            strides=2,
            padding="valid",
            name="convolution2",
        )
        # Use same default momentum and epsilon as PyTorch equivalent for BatchNormalization
        self.batchnorm_after = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_after"
        )

        self.activation = tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="activation")

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        features = self.batchnorm_before(self.convolution1(self.padding(pixel_values)), training=training)
        features = self.activation(features)
        features = self.batchnorm_after(self.convolution2(self.padding(features)), training=training)
        features = self.activation(features)
        return features


class TFEfficientFormerPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size: int, **kwargs):
        super().__init__(**kwargs)
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1, padding="same")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        output = self.pool(hidden_states)
        output = output - hidden_states
        return output


class TFEfficientFormerDenseMlp(tf.keras.layers.Layer):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear_in = tf.keras.layers.Dense(
            units=hidden_features, kernel_initializer=get_initializer(config.initializer_range), name="linear_in"
        )
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

        self.linear_out = tf.keras.layers.Dense(
            units=out_features, kernel_initializer=get_initializer(config.initializer_range), name="linear_out"
        )

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.linear_in(inputs=hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.linear_out(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states


class TFEfficientFormerConvMlp(tf.keras.layers.Layer):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.convolution1 = tf.keras.layers.Conv2D(
            filters=hidden_features,
            kernel_size=1,
            name="convolution1",
            padding="valid",
        )

        self.activation = ACT2FN[config.hidden_act]

        self.convolution2 = tf.keras.layers.Conv2D(
            filters=out_features,
            kernel_size=1,
            name="convolution2",
            padding="valid",
        )

        self.dropout = tf.keras.layers.Dropout(rate=drop)

        # Use same default momentum and epsilon as PyTorch equivalent for BatchNormalization
        self.batchnorm_before = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_before"
        )
        # Use same default momentum and epsilon as PyTorch equivalent for BatchNormalization
        self.batchnorm_after = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name="batchnorm_after"
        )

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.batchnorm_before(hidden_state, training=training)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        hidden_state = self.convolution2(hidden_state)
        hidden_state = self.batchnorm_after(hidden_state, training=training)
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state


# Copied from transformers.models.convnext.modeling_tf_convnext.TFConvNextDropPath with ConvNext->EfficientFormer
class TFEfficientFormerDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class TFEfficientFormerFlat(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor]:
        batch_size, _, _, in_channels = shape_list(hidden_states)
        hidden_states = tf.reshape(hidden_states, shape=[batch_size, -1, in_channels])
        return hidden_states


class TFEfficientFormerMeta3D(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self.token_mixer = TFEfficientFormerSelfAttention(
            dim=config.dim,
            key_dim=config.key_dim,
            num_heads=config.num_attention_heads,
            attention_ratio=config.attention_ratio,
            resolution=config.resolution,
            name="token_mixer",
            config=config,
        )
        self.dim = dim
        self.config = config

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm2")
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = TFEfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim, name="mlp")

        # Using `layers.Activation` instead of `tf.identity` to better control `training' behavior.
        self.drop_path = (
            TFEfficientFormerDropPath(drop_path)
            if drop_path > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )
        self.config = config

    def build(self, input_shape: tf.TensorShape):
        self.layer_scale_1 = None
        self.layer_scale_2 = None

        if self.config.use_layer_scale:
            self.layer_scale_1 = self.add_weight(
                shape=(self.dim,),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_1",
            )
            self.layer_scale_2 = self.add_weight(
                shape=(self.dim,),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_2",
            )
        super().build(input_shape)

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        self_attention_outputs = self.token_mixer(
            hidden_states=self.layernorm1(hidden_states, training=training),
            output_attentions=output_attentions,
            training=training,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.config.use_layer_scale:
            layer_output = hidden_states + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * attention_output,
                training=training,
            )
            layer_output = layer_output + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0)
                * self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training),
                training=training,
            )
        else:
            layer_output = hidden_states + self.drop_path(attention_output, training=training)
            layer_output = layer_output + self.drop_path(
                self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training),
                training=training,
            )

        outputs = (layer_output,) + outputs

        return outputs


class TFEfficientFormerMeta3DLayers(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:-1]))
            for block_idx in range(config.num_meta3d_blocks)
        ]
        self.blocks = [
            TFEfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path, name=f"blocks.{i}")
            for i, drop_path in enumerate(drop_paths)
        ]

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        all_attention_outputs = () if output_attentions else None

        for i, layer_module in enumerate(self.blocks):
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            hidden_states = layer_module(
                hidden_states=hidden_states, output_attentions=output_attentions, training=training
            )
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)

        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs

        return hidden_states


class TFEfficientFormerMeta4D(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        pool_size = config.pool_size if config.pool_size is not None else 3
        self.token_mixer = TFEfficientFormerPooling(pool_size=pool_size, name="token_mixer")
        self.dim = dim
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = TFEfficientFormerConvMlp(
            config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob, name="mlp"
        )

        self.drop_path = (
            TFEfficientFormerDropPath(drop_path, name="drop_path")
            if drop_path > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )
        self.config = config

    def build(self, input_shape: tf.TensorShape):
        self.layer_scale_1 = None
        self.layer_scale_2 = None

        if self.config.use_layer_scale:
            self.layer_scale_1 = self.add_weight(
                shape=(self.dim),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_1",
            )
            self.layer_scale_2 = self.add_weight(
                shape=(self.dim),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_2",
            )
        super().build(input_shape)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        outputs = self.token_mixer(hidden_states)

        if self.config.use_layer_scale:
            layer_output = hidden_states + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * outputs,
                training=training,
            )

            layer_output = layer_output + self.drop_path(
                tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0)
                * self.mlp(hidden_state=layer_output, training=training),
                training=training,
            )

        else:
            layer_output = hidden_states + self.drop_path(outputs, training=training)
            layer_output = layer_output + self.drop_path(
                self.mlp(hidden_state=layer_output, training=training), training=training
            )

        return layer_output


class TFEfficientFormerMeta4DLayers(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, stage_idx: int, **kwargs):
        super().__init__(**kwargs)
        num_layers = (
            config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        )
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)
        ]

        self.blocks = [
            TFEfficientFormerMeta4D(
                config=config, dim=config.hidden_sizes[stage_idx], drop_path=drop_paths[i], name=f"blocks.{i}"
            )
            for i in range(len(drop_paths))
        ]

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states=hidden_states, training=training)
        return hidden_states


class TFEfficientFormerIntermediateStage(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, index: int, **kwargs):
        super().__init__(**kwargs)
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=index, name="meta4D_layers")

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        return hidden_states


class TFEfficientFormerLastStage(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=-1, name="meta4D_layers")
        self.flat = TFEfficientFormerFlat(name="flat")
        self.meta3D_layers = TFEfficientFormerMeta3DLayers(config, name="meta3D_layers")

    def call(
        self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
    ) -> Tuple[tf.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        hidden_states = self.flat(hidden_states=hidden_states)
        hidden_states = self.meta3D_layers(
            hidden_states=hidden_states, output_attentions=output_attentions, training=training
        )

        return hidden_states


class TFEfficientFormerEncoder(tf.keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        num_intermediate_stages = len(config.depths) - 1
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]

        intermediate_stages = []
        layer_count = -1
        for i in range(num_intermediate_stages):
            layer_count += 1
            intermediate_stages.append(
                TFEfficientFormerIntermediateStage(config, i, name=f"intermediate_stages.{layer_count}")
            )
            if downsamples[i]:
                layer_count += 1
                intermediate_stages.append(
                    TFEfficientFormerPatchEmbeddings(
                        config,
                        config.hidden_sizes[i],
                        config.hidden_sizes[i + 1],
                        name=f"intermediate_stages.{layer_count}",
                    )
                )
        self.intermediate_stages = intermediate_stages
        self.last_stage = TFEfficientFormerLastStage(config, name="last_stage")

    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        training: bool = False,
    ) -> TFBaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states, training=training)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions, training=training)

        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        if not return_dict:
            return tuple(v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@keras_serializable
class TFEfficientFormerMainLayer(tf.keras.layers.Layer):
    config_class = EfficientFormerConfig

    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        self.patch_embed = TFEfficientFormerConvStem(config, config.hidden_sizes[0], name="patch_embed")
        self.encoder = TFEfficientFormerEncoder(config, name="encoder")
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # When running on CPU, tf.keras.layers.Conv2D and tf.keras.layers.AveragePool2D do not
        # support channels first NCHW format. A number of blocks contain both.
        # So change the input format from (batch_size, num_channels, height, width) to
        # (batch_size, height, width, num_channels) here.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        embedding_output = self.patch_embed(pixel_values, training=training)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)

        # Change the hidden states from (batch_size, height, width, num_channels) to
        # (batch_size, num_channels, height, width).
        # The hidden states are in (batch_size, height, width, num_channels)
        # shape after all stages except the MB3D blocks.
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1][:-1]]) + (
                encoder_outputs[1][-1],
            )

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFEfficientFormerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientFormerConfig
    base_model_prefix = "efficientformer"
    main_input_name = "pixel_values"


EFFICIENTFORMER_START_DOCSTRING = r"""
    This model is a TensorFlow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular
    TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.


    Parameters:
        config ([`EfficientFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values ((`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`EfficientFormerImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerModel(TFEfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs


@add_start_docstrings(
    """
    EfficientFormer Model transformer with an image classification head on top of pooled last hidden state, e.g. for
    ImageNet.
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerForImageClassification(TFEfficientFormerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

        # Classifier head
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier")
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tf.Tensor, TFImageClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]

        logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@dataclass
class TFEfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Args:
    Output type of [`EfficientFormerForImageClassificationWithTeacher`].
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: tf.Tensor = None
    cls_logits: tf.Tensor = None
    distillation_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@add_start_docstrings(
    """
    EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden
    state and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

    .. warning::
            This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
            supported.
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerForImageClassificationWithTeacher(TFEfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = TFEfficientFormerMainLayer(config, name="efficientformer")

        # Classifier heads
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier")
        )
        self.distillation_classifier = (
            tf.keras.layers.Dense(config.num_labels, name="distillation_classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="distillation_classifier")
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFEfficientFormerForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFEfficientFormerForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if training:
            raise Exception(
                "This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported."
            )

        outputs = self.efficientformer(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]

        cls_logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))
        distillation_logits = self.distillation_classifier(tf.reduce_mean(sequence_output, axis=-2))
        logits = (cls_logits + distillation_logits) / 2

        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        return TFEfficientFormerForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
