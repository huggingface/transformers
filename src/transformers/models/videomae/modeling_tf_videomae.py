# coding=utf-8
# Copyright 2022 Multimedia Computing Group, Nanjing University and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 VideoMAE (masked autoencoder) model."""


import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFImageClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import stable_softmax
from ...utils import logging
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_videomae import VideoMAEConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VideoMAEConfig"
_CHECKPOINT_FOR_DOC = "MCG-NJU/videomae-base"

TF_VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MCG-NJU/videomae-base",
    # See all VideoMAE models at https://huggingface.co/models?filter=videomae
]


@dataclass
class TFVideoMAEDecoderOutput(ModelOutput):
    """
    Class for TFVideoMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFVideoMAEForPreTrainingOutput(ModelOutput):
    """
    Class for TFVideoMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`tf.Tensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`tf.Tensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return tf.convert_to_tensor(sinusoid_table)[None, ...]


class TFVideoMAEEmbeddings(tf.keras.layers.Layer):
    """
    Construct the patch and position embeddings.

    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.patch_embeddings = TFVideoMAEPatchEmbeddings(config, name="patch_embeddings")
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = tf.cast(
            get_sinusoid_encoding_table(self.num_patches, config.hidden_size), "float32"
        )
        self.config = config

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor) -> tf.Tensor:
        # create patch embeddings
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings
        embeddings = embeddings + self.position_embeddings

        # only keep visible patches
        # ~bool_masked_pos means visible
        if bool_masked_pos is not None:
            batch_size, _, num_channels = tf.shape(embeddings)
            embeddings = embeddings[~bool_masked_pos]
            embeddings = tf.reshape(embeddings, (batch_size, -1, num_channels))

        return embeddings


class TFVideoMAEPatchEmbeddings(tf.keras.layers.Layer):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        )
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Conv3D(
            filters=hidden_size,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            strides=(self.tubelet_size, patch_size[0], patch_size[1]),
            name="projection",
        )

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        batch_size, num_channels, height, width = (
            tf.shape(pixel_values)[0],
            tf.shape(pixel_values)[2],
            tf.shape(pixel_values)[3],
            tf.shape(pixel_values)[4],
        )
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # TensorFlow's Conv3D layer (in the channels' last mode) has the following shape:
        # (batch_size, num_frames, height, width, num_channels). Also, in CPU mode,
        # the Conv3D layer won't support the channels' first format.
        # So, permute to (batch_size, num_frames, height, width, num_channels).
        # Conv3D in TensorFlow: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D
        pixel_values = tf.transpose(pixel_values, [0, 1, 3, 4, 2])
        embeddings = self.projection(pixel_values)
        embeddings = tf.reshape(embeddings, [batch_size, -1, tf.shape(embeddings)[-1]])
        return embeddings


class TFVideoMAESelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: VideoMAEConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(units=self.all_head_size, use_bias=False, name="query")
        self.key = tf.keras.layers.Dense(units=self.all_head_size, use_bias=False, name="key")
        self.value = tf.keras.layers.Dense(units=self.all_head_size, use_bias=False, name="value")

        self.qkv_bias = config.qkv_bias

        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.qkv_output_cache = None

    def build(self, input_shape: tf.TensorShape):
        if self.qkv_bias:
            self.q_bias = self.add_weight(shape=(self.all_head_size,), initializer="zeros", name="q_bias")
            self.v_bias = self.add_weight(shape=(self.all_head_size,), initializer="zeros", name="v_bias")
        else:
            self.q_bias = None
            self.v_bias = None

        super().build(input_shape)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def linear_transformation(self, inputs, weight, bias=None):
        # weight = tf.transpose(weight) # To match the PT weights (particularly for cross-loading).
        w_x = tf.matmul(inputs, weight, transpose_b=True)
        return w_x + bias if bias is not None else w_x

    def call(
        self, hidden_states: tf.Tensor, head_mask: Optional[tf.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]:
        batch_size = tf.shape(hidden_states)[0]
        k_bias = tf.zeros_like(self.v_bias) if self.q_bias is not None else None

        # So that we have the weights for the q, k, v layers when we're running
        # it for the first time.
        if self.qkv_output_cache is None:
            _ = self.query(inputs=hidden_states)
            _ = self.key(inputs=hidden_states)
            _ = self.value(inputs=hidden_states)
            self.qkv_output_cache = tf.constant(1)

        keys = self.linear_transformation(inputs=hidden_states, weight=self.key.kernel, bias=k_bias)
        values = self.linear_transformation(inputs=hidden_states, weight=self.value.kernel, bias=self.v_bias)
        queries = self.linear_transformation(inputs=hidden_states, weight=self.query.kernel, bias=self.q_bias)

        key_layer = self.transpose_for_scores(keys, batch_size)
        value_layer = self.transpose_for_scores(values, batch_size)
        query_layer = self.transpose_for_scores(queries, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs


# Copied from transformers.models.vit.modeling_tf_vit.TFViTSelfOutput with ViT->VideoMAE
class TFVideoMAESelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFVideoMAELayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """

    def __init__(self, config: VideoMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states


# Copied from transformers.models.vit.modeling_tf_vit.TFViTAttention with ViT->VideoMAE
class TFViTAttention(tf.keras.layers.Layer):
    def __init__(self, config: VideoMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFVideoMAESelfAttention(config, name="attention")
        self.dense_output = TFVideoMAESelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


# Copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate ViT->VideoMAE
class TFVideoMAEIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: VideoMAEConfig, **kwargs):
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


# Copied from transformers.models.vit.modeling_tf_vit.TFViTOutput ViT->VideoMAE
class TFVideoMAEOutput(tf.keras.layers.Layer):
    def __init__(self, config: VideoMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_tf_vit.TFViTLayer with ViT->VideoMAE
class TFVideoMAELayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: VideoMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFViTAttention(config, name="attention")
        self.intermediate = TFVideoMAEIntermediate(config, name="intermediate")
        self.vit_output = TFVideoMAEOutput(config, name="output")

        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(
            # in ViT, layernorm is applied before self-attention
            input_tensor=self.layernorm_before(inputs=hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(inputs=hidden_states)

        intermediate_output = self.intermediate(hidden_states=layer_output)

        # second residual connection is done here
        layer_output = self.vit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training
        )
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


# Copied from transformers.models.vit.modeling_tf_vit.TFViTEncoder with ViT->VideoMAE
class TFVideoMAEEncoder(tf.keras.layers.Layer):
    def __init__(self, config: VideoMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.layer = [TFVideoMAELayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


@keras_serializable
class TFVideoMAEMainLayer(tf.keras.layers.Layer):
    config_class = VideoMAEConfig

    def __init__(self, config: VideoMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFVideoMAEEmbeddings(config, name="embeddings")
        self.encoder = TFVideoMAEEncoder(config, name="encoder")

        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

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
        pixel_values: Optional[TFModelInputType] = None,
        bool_masked_pos: tf.Tensor = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values, bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFVideoMAEPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VideoMAEConfig
    base_model_prefix = "videomae"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(
                3,
                self.config.num_frames,
                self.config.num_channels,
                self.config.image_size,
                self.config.image_size,
            ),
            dtype=tf.float32,
        )
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None, None), tf.float32, name="pixel_values"),
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


VIDEOMAE_START_DOCSTRING = r"""

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
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIDEOMAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
            [`ViTFeatureExtractor.__call__`] for details.

        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare VideoMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIDEOMAE_START_DOCSTRING,
)
class TFVideoMAEModel(TFVideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.videomae = TFVideoMAEMainLayer(config, name="videomae")

    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        bool_masked_pos: tf.Tensor = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from decord import VideoReader, cpu
        >>> import numpy as np

        >>> from transformers import TFVideoMAEFeatureExtractor, TFVideoMAEModel
        >>> from huggingface_hub import hf_hub_download


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

        >>> # sample 16 frames
        >>> videoreader.seek(0)
        >>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(videoreader))
        >>> video = videoreader.get_batch(indices).asnumpy()

        >>> feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = TFVideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        >>> # prepare video for the model
        >>> inputs = feature_extractor(list(video), return_tensors="tf")

        >>> # forward pass
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1568, 768]
        ```"""
        outputs = self.videomae(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output: TFBaseModelOutput) -> TFBaseModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=hs,
            attentions=attns,
        )


class TFVideoMAEDecoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        decoder_num_labels = config.num_channels * config.tubelet_size * config.patch_size**2

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = [
            [
                TFVideoMAELayer(decoder_config, name=f"decoder_layers.{j}")
                for j in range(config.decoder_num_hidden_layers)
            ]
        ]

        self.norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm")
        self.head = tf.keras.layers.Dense(decoder_num_labels, name="head") if decoder_num_labels > 0 else tf.identity

    def call(
        self,
        hidden_states,
        return_token_num,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_token_num > 0:
            hidden_states = hidden_states[:, -return_token_num:]

        # predictor projection
        hidden_states = self.norm(hidden_states)
        logits = self.head(hidden_states)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return TFVideoMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)


@add_start_docstrings(
    "The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.",
    VIDEOMAE_START_DOCSTRING,
)
class TFVideoMAEForPreTraining(TFVideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.videomae = TFVideoMAEMainLayer(config, name="videomae")

        self.encoder_to_decoder = tf.keras.layers.Dense(
            config.decoder_hidden_size, use_bias=False, name="encoder_to_decoder"
        )
        self.position_embeddings = get_sinusoid_encoding_table(
            self.videomae.embeddings.num_patches, config.decoder_hidden_size
        )

        self.decoder = TFVideoMAEDecoder(config, name="decoder")

        self.mean = tf.convert_to_tensor(IMAGENET_DEFAULT_MEAN)[None, None, :, None, None]
        self.std = tf.convert_to_tensor(IMAGENET_DEFAULT_STD)[None, None, :, None, None]

    def build(self, input_shape):
        self.mask_token = self.add_weight(
            shape=(1, 1, self.config.decoder_hidden_size), initializer="zeros", name="mask_token"
        )
        super().build(input_shape)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFVideoMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        bool_masked_pos: tf.Tensor = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        r"""
        Returns:

        Examples:
        ```python
        >>> from transformers import VideoMAEFeatureExtractor, TFVideoMAEForPreTraining
        >>> import numpy as np

        >>> num_frames = 16
        >>> video = list(np.random.randn(16, 3, 224, 224))

        >>> feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

        >>> pixel_values = feature_extractor(video, return_tensors="tf").pixel_values

        >>> num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
        >>> seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
        >>> bool_masked_pos = tf.experimental.numpy.random.randint(0, 2, (1, seq_length))
        >>> bool_masked_pos = tf.cast(bool_masked_pos, "bool")

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.videomae(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]
        sequence_output = self.encoder_to_decoder(
            sequence_output
        )  # [batch_size, num_visible_patches, decoder_hidden_size]
        batch_size, seq_len, num_channels = tf.shape(sequence_output)

        # we don't unshuffle the correct visible token order, but shuffle the position embeddings accordingly.
        if bool_masked_pos is None:
            raise ValueError("One must provide a boolean mask ")
        expanded_position_embeddings = tf.tile(self.position_embeddings, (batch_size, 1, 1))
        expanded_position_embeddings = tf.cast(expanded_position_embeddings, pixel_values.dtype)
        pos_emb_visible = expanded_position_embeddings[~bool_masked_pos]
        pos_emb_visible = tf.reshape(pos_emb_visible, (batch_size, -1, num_channels))
        pos_emb_mask = tf.reshape(expanded_position_embeddings[bool_masked_pos], (batch_size, -1, num_channels))

        # [batch_size, num_patches, decoder_hidden_size]
        x_full = tf.concat([sequence_output + pos_emb_visible, self.mask_token + pos_emb_mask], axis=1)

        # [batch_size, num_masked_patches, num_channels * patch_size * patch_size]
        decoder_outputs = self.decoder(x_full, tf.shape(pos_emb_mask)[1])
        logits = decoder_outputs.logits

        loss = None
        # calculate the labels to be predicted
        # first, unnormalize the frames
        frames = pixel_values * self.std + self.mean  # in [0, 1]

        batch_size, time, num_channels, height, width = tf.shape(frames)
        tubelet_size, patch_size = self.config.tubelet_size, self.config.patch_size
        if self.config.norm_pix_loss:
            # step 1: split up dimensions (time by tubelet_size, height by patch_size, width by patch_size)
            frames = tf.reshape(
                frames,
                (
                    batch_size,
                    time // tubelet_size,
                    tubelet_size,
                    num_channels,
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                ),
            )
            # step 2: move dimensions to concatenate:
            frames = tf.transpose(frames, perm=(0, 1, 4, 6, 2, 5, 7, 3))
            # step 3: concatenate:
            frames = tf.transpose(
                frames,
                perm=(
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size,
                    num_channels,
                ),
            )
            # step 4: normalize. The authors find that the mean is about 0.48 and standard deviation is about 0.08.
            frames_norm = (frames - tf.reduce_mean(frames, axis=-2, keepdims=True)) / (
                tf.math.reduce_std(frames, axis=-2, keepdims=True) + 1e-6
            )
            # step 5: reshape to (batch_size, T//ts * H//ps * W//ps, ts * ps * ps * C)
            videos_patch = tf.reshape(
                frames_norm,
                (
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size * num_channels,
                ),
            )
        else:
            # step 1: split up dimensions (time by tubelet_size, height by patch_size, width by patch_size)
            frames = tf.reshape(
                frames,
                (
                    batch_size,
                    time // tubelet_size,
                    tubelet_size,
                    num_channels,
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                ),
            )
            # step 2: move dimensions to concatenate: (batch_size, T//ts, H//ps, W//ps, ts, ps, ps, C)
            frames = tf.transpose(frames, perm=(0, 1, 4, 6, 2, 5, 7, 3))
            # step 3: concatenate
            videos_patch = tf.reshape(
                frames,
                (
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size * num_channels,
                ),
            )

            batch_size, _, num_channels = tf.shape(videos_patch)
            labels = tf.reshape(videos_patch[bool_masked_pos], (batch_size, -1, num_channels))

        loss = tf.keras.losses.mean_squared_error(labels, logits)
        loss = tf.reshape(loss, (1,))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFVideoMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """VideoMAE Model transformer with a video classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.""",
    VIDEOMAE_START_DOCSTRING,
)
class TFVideoMAEForVideoClassification(TFVideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.videomae = TFVideoMAEMainLayer(config, name="videomae")

        # Classifier head
        self.fc_norm = (
            tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="fc_norm")
            if config.use_mean_pooling
            else None
        )
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier") if config.num_labels > 0 else tf.identity
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFImageClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from decord import VideoReader, cpu
        >>> import torch
        >>> import numpy as np

        >>> from transformers import VideoMAEFeatureExtractor, TFVideoMAEForVideoClassification
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

        >>> # sample 16 frames
        >>> videoreader.seek(0)
        >>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(videoreader))
        >>> video = videoreader.get_batch(indices).asnumpy()

        >>> feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        >>> model = TFVideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        >>> inputs = feature_extractor(list(video), return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits

        >>> # model predicts one of the 400 Kinetics-400 classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        eating spaghetti
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.videomae(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(tf.reduce_mean(sequence_output, axis=1))
        else:
            sequence_output = sequence_output[:, 0]

        logits = self.classifier(sequence_output)

        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFImageClassifierOutput) -> TFImageClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFImageClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)
