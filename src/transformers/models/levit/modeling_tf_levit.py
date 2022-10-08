# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" TensorFlow LeViT model."""

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from numpy import indices

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras import backend as K

from ...modeling_outputs import ModelOutput
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPooling,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import TFPreTrainedModel, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "LevitConfig"
_FEAT_EXTRACTOR_FOR_DOC = "LevitFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/levit-128S"
_EXPECTED_OUTPUT_SHAPE = [1, 16, 384]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/levit-128S"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/levit-128S",
    # See all LeViT models at https://huggingface.co/models?filter=levit
]


@dataclass
class TFLevitForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`LevitForImageClassificationWithTeacher`].

    Args:
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the `cls_logits` and `distillation_logits`.
        cls_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """

    logits: tf.Tensor = None
    cls_logits: tf.Tensor = None
    distillation_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None


class TFLevitConvEmbeddings(tf.keras.layers.Layer):
    """
    LeViT Conv Embeddings with Batch Norm, used in the initial patch embedding layer.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bn_weight_init=1, **kwargs,
    ):
        super().__init__(**kwargs)
        self.convolution = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=(padding, padding), # TODO @ariG23498: Make sure the padding is a tuple
            dilation_rate=dilation,
            groups=groups,
            use_bias=False,
            data_format="channels_first",
            name="convolution",
        )
        # The epsilon and momentum used here are the defaults in torch batch norm layer.
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="batch_norm")

    def call(self, embeddings, training=None):
        embeddings = self.convolution(embeddings, training=training)
        embeddings = self.batch_norm(embeddings, training=training)
        return embeddings


# Defining hard swish with keras backend.
def hard_swish(x):
    return x * (K.relu(x + 3.0, max_value=6.0) / 6.0)


class TFLevitPatchEmbeddings(tf.keras.layers.Layer):
    """
    LeViT patch embeddings, for final embeddings to be passed to transformer blocks. It consists of multiple
    `TFLevitConvEmbeddings`.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer_1 = TFLevitConvEmbeddings(
            in_channels=config.num_channels,
            out_channels=config.hidden_sizes[0] // 8,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            name="embedding_layer_1",
        )
        self.activation_layer_1 = hard_swish

        self.embedding_layer_2 = TFLevitConvEmbeddings(
            in_channels=config.hidden_sizes[0] // 8,
            out_channels=config.hidden_sizes[0] // 4,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            name="embedding_layer_2",
        )
        self.activation_layer_2 = hard_swish

        self.embedding_layer_3 = TFLevitConvEmbeddings(
            in_channels=config.hidden_sizes[0] // 4,
            out_channels=config.hidden_sizes[0] // 2,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            name="embedding_layer_3",
        )
        self.activation_layer_3 = hard_swish

        self.embedding_layer_4 = TFLevitConvEmbeddings(
            in_channels=config.hidden_sizes[0] // 2,
            out_channels=config.hidden_sizes[0],
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            name="embedding_layer_4",
        )
        self.num_channels = config.num_channels

    def call(self, pixel_values, training=None):
        batch_size = tf.shape(pixel_values)[0]
        num_channels = tf.shape(pixel_values)[1]
        
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        embeddings = self.embedding_layer_1(pixel_values, training=training)
        embeddings = self.activation_layer_1(embeddings)
        embeddings = self.embedding_layer_2(embeddings, training=training)
        embeddings = self.activation_layer_2(embeddings)
        embeddings = self.embedding_layer_3(embeddings, training=training)
        embeddings = self.activation_layer_3(embeddings)
        embeddings = self.embedding_layer_4(embeddings, training=training)
        # Flatten the embeddings
        flattended_embeddings = tf.reshape(embeddings, shape=(batch_size, num_channels, -1))
        # Transpose the channel and spatial axis of the flattened embeddings
        transpose_embeddings = tf.transpose(flattended_embeddings, perm=(0, 2, 1))
        return transpose_embeddings


class TFMLPLayerWithBN(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, bn_weight_init=1, **kwargs):
        super().__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(units=output_dim, use_bias=False, name="linear")
        # The epsilon and momentum used here are the defaults in torch batch norm layer.
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="batch_norm")

    def call(self, hidden_state, training=None):
        num_channels = tf.shape(hidden_state)[2]
        hidden_state = self.linear(hidden_state, training=training)
        
        # Before sending the hidden state to the batch normalization layer, we would have to
        # flatten the hidden states in the batch and seq len dimension
        flattened_hidden_state = tf.reshape(hidden_state, shape=(-1, num_channels))
        batch_norm_hidden_state = self.batch_norm(flattened_hidden_state, training=training)
        
        # Reshape the output of batch norm to have the same shape as the original hidden state
        hidden_state = tf.reshape(batch_norm_hidden_state, shape=tf.shape(hidden_state))
        return hidden_state


class TFLevitSubsample(tf.keras.layers.Layer):
    """
    Layer to subsample the activatioin maps
    """
    def __init__(self, stride, resolution, **kwargs):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def call(self, hidden_state):
        batch_size = tf.shape(hidden_state)[0]
        channels = tf.shape(hidden_state)[2]
        
        reshaped_hidden_state = tf.reshape(
            hidden_state, shape=(batch_size, self.resolution, self.resolution, channels)
        )
        strided_hidden_state = reshaped_hidden_state[:, :: self.stride, :: self.stride]
        hidden_state = tf.reshape(strided_hidden_state, shape=(batch_size, -1, channels))
        
        return hidden_state


class TFLevitAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads * 2
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads

        self.queries_keys_values = TFMLPLayerWithBN(hidden_sizes, self.out_dim_keys_values, name="queries_keys_values")
        self.activation = hard_swish
        self.projection = TFMLPLayerWithBN(self.out_dim_projection, hidden_sizes, bn_weight_init=0, name="projection")

        # Build tuples of points in the entire resolution range of the pixel values
        points = list(itertools.product(range(resolution), range(resolution)))
        self.len_points = len(points)

        # Initialize the attention offsets and indices
        attention_offsets, indices = {}, []

        # Iterate over the points generator and calculate the offset between the initial
        # point (0, 0) and the rest of the points [(0, 1), (0, 2)...]
        for p1 in points: # this iterates only once
            for p2 in points: # iterate over all the points other than (0, 0)
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])
        
        # Store the attention offsets, indices and attention bias cache
        self.attention_offsets = attention_offsets
        self.indices = indices
        self.attention_bias_cache = {}

    def build(self, input_shape):
        self.attention_biases = self.add_weight(
            shape=(self.num_attention_heads, len(self.attention_offsets)),
            initializer="zeros",
            trainable=True,
            name="attention_biases",
        )
        self.attention_bias_idxs = tf.Variable(
            initial_value=tf.reshape(self.indices, (self.len_points, self.len_points)),
            trainable=False, # this is a registered buffer and not a parameter
            dtype=tf.float32,
            name="attention_bias_idxs",
        )
        super().build(input_shape)

    # # TODO @ariG23498
    # @torch.no_grad()
    # def train(self, mode=True):
    #     super().train(mode)
    #     if mode and self.attention_bias_cache:
    #         self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device, training=None):
        if training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def call(self, hidden_state, training=None):
        batch_size = tf.shape(hidden_state)[0]
        seq_length = tf.shape(hidden_state)[1]
        queries_keys_values = self.queries_keys_values(hidden_state)

        # Reshape queries_keys_values
        reshaped_queries_keys_values = tf.reshape(
            queries_keys_values, shape=(batch_size, seq_length, self.num_attention_heads, -1)
        )
        query, key, value = tf.split(
            value=reshaped_queries_keys_values,
            num_or_size_splits=[self.key_dim, self.key_dim, self.attention_ratio * self.key_dim],
            axis=3,
        )
        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 1, 3))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        attention = tf.matmul(query, key, transpose_b=True) * self.scale + self.get_attention_biases(
            hidden_state.device, training=training
        )
        attention = stable_softmax(attention, axis=-1)
        hidden_state = tf.matmul(attention, value)
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, seq_length, self.out_dim_projection))
        hidden_state = self.projection(self.activation(hidden_state))
        return hidden_state


class TFLevitAttentionSubsample(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        key_dim,
        num_attention_heads,
        attention_ratio,
        stride,
        resolution_in,
        resolution_out,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_keys_values = attention_ratio * key_dim * num_attention_heads + key_dim * num_attention_heads
        self.out_dim_projection = attention_ratio * key_dim * num_attention_heads
        self.resolution_out = resolution_out
        # resolution_in is the intial resolution, resoloution_out is final resolution after downsampling
        self.keys_values = TFMLPLayerWithBN(input_dim, self.out_dim_keys_values, name="keys_values")
        self.queries_subsample = TFLevitSubsample(stride, resolution_in, name="queries_subsample")
        self.queries = TFMLPLayerWithBN(input_dim, key_dim * num_attention_heads, name="queries")
        self.activation = hard_swish
        self.projection = TFMLPLayerWithBN(self.out_dim_projection, output_dim, name="projection")

        self.attention_bias_cache = {}

        points = list(itertools.product(range(resolution_in), range(resolution_in)))
        points_ = list(itertools.product(range(resolution_out), range(resolution_out)))
        self.len_points, self.len_points_ = len(points), len(points_)
        attention_offsets, indices = {}, []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                indices.append(attention_offsets[offset])

        self.attention_offsets = attention_offsets
        self.indices = indices

    def build(self, input_shape):
        self.attention_biases = self.add_weight(
            shape=(self.num_attention_heads, len(self.attention_offsets)),
            initializer="zeros",
            trainable=True,
            name="attention_biases",
        )

        self.attention_bias_idxs = tf.Variable(
            initial_value=tf.reshape(self.indices, (self.len_points_, self.len_points)),
            trainable=False,
            dtype=tf.float32,
            name="attention_bias_idxs",
        )
        super().build(input_shape)

    # # TODO @ariG23498
    # @torch.no_grad()
    # def train(self, mode=True):
    #     super().train(mode)
    #     if mode and self.attention_bias_cache:
    #         self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device, training=None):
        if training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def call(self, hidden_state, training=None):
        batch_size = tf.shape(hidden_state)[0]
        seq_length = tf.shape(hidden_state)[1]

        # Process the hidden states and reshape it
        reshaped_hidden_state = tf.reshape(
            self.keys_values(hidden_state), shape=(batch_size, seq_length, self.num_attention_heads, -1)
        )
        # Split the reshaped hidden state into key and value
        key, value = tf.split(
            reshaped_hidden_state,
            num_or_size_splits=[self.key_dim, self.attention_ratio * self.key_dim],
            axis=3,
        )
        key = tf.transpose(key, perm=(0, 2, 1, 3))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        query = self.queries(self.queries_subsample(hidden_state))
        query = tf.reshape(query, shape=(batch_size, self.resolution_out**2, self.num_attention_heads, self.key_dim))
        query = tf.transpose(query, perm=(0, 2, 1, 3))

        attention = tf.matmul(query, key, transpose_b=True) * self.scale + self.get_attention_biases(
            hidden_state.device, training=training
        )
        attention = stable_softmax(attention, axis=-1)
        hidden_state = tf.matmul(attention, value)
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
        hidden_state = tf.reshape(hidden_state, (batch_size, -1, self.out_dim_projection))
        hidden_state = self.projection(self.activation(hidden_state), training=training)
        return hidden_state


class TFLevitMLPLayer(tf.keras.layers.Layer):
    """
    MLP Layer with `2X` expansion in contrast to ViT with `4X`.
    """

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.linear_up = TFMLPLayerWithBN(input_dim, hidden_dim, name="linear_up")
        self.activation = hard_swish
        self.linear_down = TFMLPLayerWithBN(hidden_dim, input_dim, name="linear_down")

    def call(self, hidden_state, training=None):
        hidden_state = self.linear_up(hidden_state, training=training)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.linear_down(hidden_state, training=training)
        return hidden_state


class TFLevitResidualLayer(tf.keras.layers.Layer):
    """
    Residual Block for TFLeViT
    """

    def __init__(self, module, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.drop_rate = drop_rate

    def call(self, hidden_state, training=None):
        if training and self.drop_rate > 0:
            rnd = tf.random.normal(shape=(tf.shape(hidden_state)[0], 1, 1), minval=0, maxval=1)
            rnd = tf.math.greater(rnd, self.drop_rate)
            rnd = tf.math.divide(rnd, (1 - self.drop_rate))
            hidden_state = hidden_state + self.module(hidden_state) * rnd
            return hidden_state
        else:
            hidden_state = hidden_state + self.module(hidden_state)
            return hidden_state


class TFLevitStage(tf.keras.layers.Layer):
    """
    LeViT Stage consisting of `TFLevitMLPLayer` and `TFLevitAttention` layers.
    """

    def __init__(
        self,
        config,
        idx,
        hidden_sizes,
        key_dim,
        depths,
        num_attention_heads,
        attention_ratio,
        mlp_ratio,
        down_ops,
        resolution_in,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = []
        self.config = config
        self.resolution_in = resolution_in
        # resolution_in is the intial resolution, resolution_out is final resolution after downsampling

        for idx in range(depths):
            self.layers.append(
                TFLevitResidualLayer(
                    TFLevitAttention(hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution_in),
                    self.config.drop_path_rate,
                    name=f"layers.{idx}",
                )
            )
            if mlp_ratio > 0:
                hidden_dim = hidden_sizes * mlp_ratio
                self.layers.append(
                    TFLevitResidualLayer(
                        TFLevitMLPLayer(hidden_sizes, hidden_dim),
                        self.config.drop_path_rate,
                        name=f"layers.{idx}",
                    )
                )

        if down_ops[0] == "Subsample":

            print("info", self.config.hidden_sizes)
            print("info", idx)
            self.resolution_out = (self.resolution_in - 1) // down_ops[5] + 1
            self.layers.append(
                TFLevitAttentionSubsample(
                    input_dim=self.config.hidden_sizes[idx],
                    output_dim=self.config.hidden_sizes[idx + 1],
                    key_dim=down_ops[1],
                    num_attention_heads=down_ops[2],
                    attention_ratio=down_ops[3],
                    stride=down_ops[5],
                    resolution_in=resolution_in,
                    resolution_out=self.resolution_out,
                    name=f"layers.{idx}",
                )
            )
            self.resolution_in = self.resolution_out
            if down_ops[4] > 0:
                hidden_dim = self.config.hidden_sizes[idx + 1] * down_ops[4]
                self.layers.append(
                    TFLevitResidualLayer(
                        TFLevitMLPLayer(self.config.hidden_sizes[idx + 1], hidden_dim),
                        self.config.drop_path_rate,
                        name=f"layers.{idx}",
                    )
                )

    def get_resolution(self):
        return self.resolution_in

    def call(self, hidden_state):
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class TFLevitEncoder(tf.keras.layers.Layer):
    """
    LeViT Encoder consisting of multiple `TFLevitStage` stages.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        resolution = self.config.image_size // self.config.patch_size
        self.stages = []
        self.config.down_ops.append([""])

        # TODO ariG23498: add the index values to the layer names
        for stage_idx in range(len(config.depths)):
            stage = TFLevitStage(
                config,
                stage_idx,
                config.hidden_sizes[stage_idx],
                config.key_dim[stage_idx],
                config.depths[stage_idx],
                config.num_attention_heads[stage_idx],
                config.attention_ratio[stage_idx],
                config.mlp_ratio[stage_idx],
                config.down_ops[stage_idx],
                resolution,
                name=f"stages.{stage_idx}",
            )
            resolution = stage.get_resolution()
            self.stages.append(stage)

    def call(self, hidden_state, output_hidden_states=False, return_dict=True, training=None):
        all_hidden_states = () if output_hidden_states else None

        for stage in self.stages:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            hidden_state = stage(hidden_state)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)


class TFLevitClassificationLayer(tf.keras.layers.Layer):
    """
    LeViT Classification Layer
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # The epsilon and momentum used here are the defaults in torch batch norm layer.
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="batch_norm")
        self.linear = tf.keras.layers.Dense(units=output_dim, use_bias=False, name="linear")

    def call(self, hidden_state, training=None):
        hidden_state = self.batch_norm(hidden_state, training=training)
        logits = self.linear(hidden_state, training=training)
        return logits


@keras_serializable
class TFLevitMainLayer(tf.keras.layers.Layer):
    config_class = LevitConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.patch_embeddings = TFLevitPatchEmbeddings(config, name="patch_embeddings")
        self.encoder = TFLevitEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Apply patch embeddings to the pixel values
        embeddings = self.patch_embeddings(pixel_values, training=training)

        # Apply encoder to the encoded pixel values
        encoder_outputs = self.encoder(
            embeddings,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # Obtain the `last_hidden_state`
        last_hidden_state = encoder_outputs[0]  # encoder_outputs.last_hidden_state

        # global average pooling, (batch_size, seq_length, hidden_sizes) -> (batch_size, hidden_sizes)
        pooled_output = tf.math.reduce_mean(last_hidden_state, axis=1)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,  # only if the `output_hidden_states` is set to True
        )


class TFLevitPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LevitConfig
    base_model_prefix = "levit"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(3, self.config.num_channels, self.config.image_size, self.config.image_size), dtype=tf.float32
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


LEVIT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`LevitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LEVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Levit model outputting raw features without any specific head on top.",
    LEVIT_START_DOCSTRING,
)
class TFLevitModel(TFLevitPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.levit = TFLevitMainLayer(config=config, name="levit")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ):
        outputs = self.levit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    # TODO @ariG23498: Check the output type for serving.
    def serving_output(self, output: TFBaseModelOutputWithPoolingAndNoAttention) -> TFBaseModelOutputWithPooling:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hs,
            attentions=attns,
        )


@add_start_docstrings(
    """
    Levit Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    LEVIT_START_DOCSTRING,
)
class TFLevitForImageClassification(TFLevitPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.levit = TFLevitMainLayer(config, name="levit")

        # Classifier head
        self.classifier = (
            TFLevitClassificationLayer(config.hidden_sizes[-1], config.num_labels, name="classifier")
            if config.num_labels > 0
            else tf.identity
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor = None,
        labels: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ):
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # Get the outputs from the levit main layer
        outputs = self.levit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # Get the `last_hidden_state` and average it along the number of sequences
        sequence_output = outputs[0]  # outputs.last_hidden_state
        sequence_output = tf.math.reduce_mean(sequence_output, axis=1)

        # Apply the classifier head and obtain the logits
        logits = self.classifier(sequence_output, training=training)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                # TODO @ariG23498: Check with the dtypes (long and int in torch)
                elif self.num_labels > 1 and (labels.dtype == tf.float64 or labels.dtype == tf.int64):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MeanSquaredError()
                if self.num_labels == 1:
                    loss = loss_fct(tf.squeeze(logits), tf.squeeze(labels))
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CategoricalCrossentropy()
                loss = loss_fct(tf.reshape(logits, shape=(-1, self.num_labels)), tf.flatten(labels))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BinaryCrossentropy()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,  # only if `output_hidden_states` flag is set to True
        )


@add_start_docstrings(
    """
    LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and
    a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning::
           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    """,
    LEVIT_START_DOCSTRING,
)
class TFLevitForImageClassificationWithTeacher(TFLevitPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.levit = TFLevitMainLayer(config, name="levit")

        # Classifier head
        self.classifier = (
            TFLevitClassificationLayer(config.hidden_sizes[-1], config.num_labels, name="classifier")
            if config.num_labels > 0
            else tf.identity
        )
        self.classifier_distill = (
            TFLevitClassificationLayer(config.hidden_sizes[-1], config.num_labels, name="classifier_distill")
            if config.num_labels > 0
            else tf.identity
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFLevitForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ):
        # Get the output from the levit main layer
        outputs = self.levit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # Get the `last_hidden_state` and average it along the number of sequences
        sequence_output = outputs[0]  # outputs.last_hidden_state
        sequence_output = tf.math.reduce_mean(sequence_output, axis=1)

        # Apply the classifier heads and obtain the `cls_logits` and `distill_logits`
        cls_logits, distill_logits = self.classifier(sequence_output, training=training), self.classifier_distill(
            sequence_output, training=training
        )

        # According to the paper, the cls and distill logits are averaged
        logits = (cls_logits + distill_logits) / 2

        if not return_dict:
            output = (logits, cls_logits, distill_logits) + outputs[2:]
            return output

        return TFLevitForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distill_logits,
            hidden_states=outputs.hidden_states,  # only if `output_hidden_states` flag is set to True
        )
