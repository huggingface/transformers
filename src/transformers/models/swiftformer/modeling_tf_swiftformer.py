# coding=utf-8
# Copyright 2023 MBZUAI and The HuggingFace Inc. team. All rights reserved.
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
""" TensorFlow SwiftFormer model."""


import collections.abc
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_swiftformer import SwiftFormerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SwiftFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "MBZUAI/swiftformer-xs"
_EXPECTED_OUTPUT_SHAPE = [1, 220, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "MBZUAI/swiftformer-xs"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


TF_SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MBZUAI/swiftformer-xs",
    # See all SwiftFormer models at https://huggingface.co/models?filter=swiftformer
]


class TFSwiftFormerPatchEmbeddingSequential(tf.keras.layers.Layer):
    """
    The sequential component of the patch embedding layer.

    Input: tensor of shape `[batch_size, in_channels, height, width]`

    Output: tensor of shape `[batch_size, out_channels, height/4, width/4]`
    """

    def __init__(self, config: SwiftFormerConfig, **kwargs):
        super().__init__(**kwargs)
        out_chs = config.embed_dims[0]

        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.conv1 = tf.keras.layers.Conv2D(out_chs // 2, kernel_size=3, strides=2, name="0")
        self.batch_norm1 = tf.keras.layers.BatchNormalization(
            epsilon=config.batch_norm_eps, momentum=0.9, name="1"
        )  # FIXME: is this the equivalent momentum?
        self.conv2 = tf.keras.layers.Conv2D(out_chs, kernel_size=3, strides=2, name="3")
        self.batch_norm2 = tf.keras.layers.BatchNormalization(
            epsilon=config.batch_norm_eps, momentum=0.9, name="4"
        )  # FIXME: is this the equivalent momentum?

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.zero_padding(x)
        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = get_tf_activation("relu")(x)
        x = self.zero_padding(x)
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = get_tf_activation("relu")(x)
        return x


class TFSwiftFormerPatchEmbedding(tf.keras.layers.Layer):
    """
    Patch Embedding Layer constructed of two 2D convolutional layers.

    Input: tensor of shape `[batch_size, in_channels, height, width]`

    Output: tensor of shape `[batch_size, out_channels, height/4, width/4]`
    """

    def __init__(self, config: SwiftFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = TFSwiftFormerPatchEmbeddingSequential(config, name="patch_embedding")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.patch_embedding(x, training=training)


# TODO: I think this is available in the KerasCV package, should we use that?
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input  # FIXME: shouldn't this be x?
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + tf.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class TFSwiftFormerDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        return drop_path(hidden_states, self.drop_prob, training)


class TFSwiftFormerEmbeddings(tf.keras.layers.Layer):
    """
    Embeddings layer consisting of a single 2D convolutional and batch normalization layer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height/stride, width/stride]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int, **kwargs):
        super().__init__(**kwargs)

        patch_size = config.down_patch_size
        stride = config.down_stride
        padding = config.down_pad
        embed_dims = config.embed_dims

        embed_dim = embed_dims[index + 1]

        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)

        self.pad = tf.keras.layers.ZeroPadding2D(padding=padding)
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=stride, name="proj")
        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=config.batch_norm_eps, momentum=0.9, name="norm"
        )  # FIXME: is this the correct momentum?

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.pad(x)
        x = self.proj(x)
        x = self.norm(x, training=training)
        return x


class TFSwiftFormerConvEncoder(tf.keras.layers.Layer):
    """
    `SwiftFormerConvEncoder` with 3*3 and 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = int(config.mlp_ratio * dim)

        self.dim = dim
        self.pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.depth_wise_conv = tf.keras.layers.Conv2D(dim, kernel_size=3, groups=dim, name="depth_wise_conv")
        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=config.batch_norm_eps, momentum=0.9, name="norm"
        )  # FIXME
        self.point_wise_conv1 = tf.keras.layers.Conv2D(hidden_dim, kernel_size=1, name="point_wise_conv1")
        self.act = get_tf_activation("gelu")
        self.point_wise_conv2 = tf.keras.layers.Conv2D(dim, kernel_size=1, name="point_wise_conv2")
        self.drop_path = tf.keras.layers.Identity(name="drop_path")  # FIXME: is this supposed to be like this?

    def build(self, input_shape: tf.TensorShape):
        self.layer_scale = self.add_weight(
            name="layer_scale",
            shape=(self.dim),  # TODO: check this
            initializer="ones",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        input = x
        x = self.pad(x)
        x = self.depth_wise_conv(x)
        x = self.norm(x, training=training)
        x = self.point_wise_conv1(x)
        x = self.act(x)
        x = self.point_wise_conv2(x)
        x = input + self.drop_path(self.layer_scale * x)
        return x


class TFSwiftFormerMlp(tf.keras.layers.Layer):
    """
    MLP layer with 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, in_features: int, **kwargs):
        super().__init__(**kwargs)

        hidden_features = int(in_features * config.mlp_ratio)
        self.norm1 = tf.keras.layers.BatchNormalization(
            epsilon=config.batch_norm_eps, momentum=0.9, name="norm1"
        )  # FIXME: is this the correct momentum?
        self.fc1 = tf.keras.layers.Conv2D(hidden_features, 1, name="fc1")
        act_layer = get_tf_activation(config.hidden_act)
        self.act = act_layer
        self.fc2 = tf.keras.layers.Conv2D(in_features, 1, name="fc2")
        self.drop = tf.keras.layers.Dropout(rate=0.0)  # FIXME: is this supposed to be 0?

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.norm1(x, training=training)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class TFSwiftFormerEfficientAdditiveAttention(tf.keras.layers.Layer):
    """
    Efficient Additive Attention module for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int = 512, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim

        self.to_query = tf.keras.layers.Dense(dim, name="to_query")
        self.to_key = tf.keras.layers.Dense(dim, name="to_key")

        self.scale_factor = dim**-0.5
        self.proj = tf.keras.layers.Dense(dim, name="proj")
        self.final = tf.keras.layers.Dense(dim, name="final")

    def build(self, input_shape: tf.TensorShape):
        self.w_g = self.add_weight(
            name="w_g",
            shape=(self.dim, 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        query = self.to_query(x)
        key = self.to_key(x)

        query = tf.math.l2_normalize(query, dim=-1)
        key = tf.math.l2_normalize(key, dim=-1)

        query_weight = query @ self.w_g
        scaled_query_weight = query_weight * self.scale_factor
        scaled_query_weight = tf.nn.softmax(scaled_query_weight, axis=-1)

        global_queries = tf.math.reduce_sum(scaled_query_weight * query, axis=1)
        global_queries = tf.tile(tf.expand_dims(global_queries, 1), (1, key.shape[1], 1))

        out = self.proj(global_queries * key) + query
        out = self.final(out)

        return out


class TFSwiftFormerLocalRepresentation(tf.keras.layers.Layer):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim

        self.pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.depth_wise_conv = tf.keras.layers.Conv2D(dim, kernel_size=3, groups=dim, name="depth_wise_conv")
        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=config.batch_norm_eps, momentum=0.9, name="norm"
        )  # FIXME: momentum
        self.point_wise_conv1 = tf.keras.layers.Conv2D(dim, kernel_size=1, name="point_wise_conv1")
        self.act = get_tf_activation("gelu")
        self.point_wise_conv2 = tf.keras.layers.Conv2D(dim, kernel_size=1, name="point_wise_conv2")
        self.drop_path = tf.keras.layers.Identity(name="drop_path")  # FIXME: is this correct?

    def build(self, input_shape):
        self.layer_scale = self.add_weight(
            name="layer_scale",
            shape=(self.dim),  # FIXME: check this
            initializer="ones",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        input = x
        x = self.pad(x)
        x = self.depth_wise_conv(x)
        x = self.norm(x, training=training)
        x = self.point_wise_conv1(x)
        x = self.act(x)
        x = self.point_wise_conv2(x)
        x = input + self.drop_path(self.layer_scale * x, training=training)
        return x


class TFSwiftFormerEncoderBlock(tf.keras.layers.Layer):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2)
    SwiftFormerEfficientAdditiveAttention, and (3) MLP block.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels,height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        layer_scale_init_value = config.layer_scale_init_value
        use_layer_scale = config.use_layer_scale

        self.local_representation = TFSwiftFormerLocalRepresentation(config, dim=dim, name="local_representation")
        self.attn = TFSwiftFormerEfficientAdditiveAttention(config, dim=dim, name="attn")
        self.linear = TFSwiftFormerMlp(config, in_features=dim, name="linear")
        self.drop_path = TFSwiftFormerDropPath(drop_path) if drop_path > 0.0 else tf.keras.layers.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.dim = dim
            self.layer_scale_init_value = layer_scale_init_value

    def build(self, input_shape: tf.TensorShape):
        self.layer_scale_1 = self.add_weight(
            name="layer_scale_1",
            shape=(self.dim),  # FIXME
            initializer=tf.keras.initializers.constant(self.layer_scale_init_value),
            trainable=True,
        )
        self.layer_scale_2 = self.add_weight(
            name="layer_scale_2",
            shape=(self.dim),  # FIXME
            initializer=tf.keras.initializers.constant(self.layer_scale_init_value),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False):
        x = self.local_representation(x, training=training)
        batch_size, height, width, channels = x.shape

        res = tf.reshape(x, [-1, height * width, channels])
        res = self.attn(res)
        res = tf.reshape(res, [-1, height, width, channels])
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * res, training=training)
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x), training=training)
        else:
            x = x + self.drop_path(res, training=training)
            x = x + self.drop_path(self.linear(x), training=training)
        return x


class TFSwiftFormerStage(tf.keras.layers.Layer):
    """
    A Swiftformer stage consisting of a series of `SwiftFormerConvEncoder` blocks and a final
    `SwiftFormerEncoderBlock`.

    Input: tensor in shape `[batch_size, channels, height, width]`

    Output: tensor in shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int, **kwargs) -> None:
        super().__init__(**kwargs)

        layer_depths = config.depths
        dim = config.embed_dims[index]
        depth = layer_depths[index]

        self.blocks = []
        for block_idx in range(depth):
            block_dpr = config.drop_path_rate * (block_idx + sum(layer_depths[:index])) / (sum(layer_depths) - 1)

            if depth - block_idx <= 1:
                self.blocks.append(
                    TFSwiftFormerEncoderBlock(config, dim=dim, drop_path=block_dpr, name=f"blocks_._{block_idx}")
                )
            else:
                self.blocks.append(TFSwiftFormerConvEncoder(config, dim=dim, name=f"blocks_._{block_idx}"))

    def call(self, input: tf.Tensor, training: bool = False) -> tf.Tensor:
        for i, block in enumerate(self.blocks):
            input = block(input, training=training)
        return input


class TFSwiftFormerEncoder(tf.keras.layers.Layer):
    def __init__(self, config: SwiftFormerConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        embed_dims = config.embed_dims
        downsamples = config.downsamples
        layer_depths = config.depths

        # Transformer model
        self.network = []
        name_i = 0
        for i in range(len(layer_depths)):
            stage = TFSwiftFormerStage(config, index=i, name=f"network_._{name_i}")
            self.network.append(stage)
            name_i += 1
            if i >= len(layer_depths) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                self.network.append(TFSwiftFormerEmbeddings(config, index=i, name=f"network_._{name_i}"))
                name_i += 1

        self.gradient_checkpointing = False

    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TFBaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for i, block in enumerate(self.network):
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = tf.transpose(hidden_states, perm=[0, 3, 1, 2])
        if all_hidden_states:  # TODO: needed this to pass test_compile_tf_model
            all_hidden_states = tuple(tf.transpose(s, perm=[0, 3, 1, 2]) for s in all_hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class TFSwiftFormerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SwiftFormerConfig
    base_model_prefix = "swiftformer"
    main_input_name = "pixel_values"


TFSWIFTFORMER_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:
    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.
    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.
    If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
    first positional argument :
    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
      `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
      `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Args:
        config ([`SwiftFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TFSWIFTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to run the model in training mode.
"""


@keras_serializable
class TFSwiftFormerMainLayer(tf.keras.layers.Layer):
    config_class = SwiftFormerConfig  # FIXME: why is this used (copied from modeling_tf_bert)

    def __init__(self, config: SwiftFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.patch_embed = TFSwiftFormerPatchEmbedding(config, name="patch_embed")
        self.encoder = TFSwiftFormerEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutputWithNoAttention]:
        r""" """

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TF 2.0 image layers can't use NCHW format when running on CPU.
        # We transpose to NHWC format and then transpose back after the full forward pass.
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.patch_embed(pixel_values, training=training)
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return tuple(v for v in encoder_outputs if v is not None)

        return TFBaseModelOutputWithNoAttention(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    "The bare TFSwiftFormer Model transformer outputting raw hidden-states without any specific head on top.",
    TFSWIFTFORMER_START_DOCSTRING,
)
class TFSwiftFormerModel(TFSwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.swiftformer = TFSwiftFormerMainLayer(config, name="swiftformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFSWIFTFORMER_INPUTS_DOCSTRING)
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithNoAttention, Tuple[tf.Tensor]]:
        outputs = self.swiftformer(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs


@add_start_docstrings(
    """
    TFSwiftFormer Model transformer with an image classification head on top (e.g. for ImageNet).
    """,
    TFSWIFTFORMER_START_DOCSTRING,
)
class TFSwiftFormerForImageClassification(TFSwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.num_labels = config.num_labels
        self.swiftformer = TFSwiftFormerMainLayer(config, name="swiftformer")

        # Classifier head
        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=config.batch_norm_eps, momentum=0.9, name="norm"
        )  # FIXME
        self.head = (
            tf.keras.layers.Dense(self.num_labels, name="head")
            if self.num_labels > 0
            else tf.keras.layers.Identity(name="head")
        )
        self.dist_head = (
            tf.keras.layers.Dense(self.num_labels, name="dist_head")
            if self.num_labels > 0
            else tf.keras.layers.Identity(name="dist_head")
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFSWIFTFORMER_INPUTS_DOCSTRING)
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tuple, TFImageClassifierOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # run base model
        outputs = self.swiftformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]
        sequence_output = tf.transpose(sequence_output, perm=[0, 2, 3, 1])

        # run classification head
        sequence_output = self.norm(sequence_output, training=training)
        # FIXME: review this conversion
        sequence_output = tf.transpose(sequence_output, perm=[0, 3, 1, 2])
        _, c, h, w = sequence_output.shape
        sequence_output = tf.reshape(sequence_output, [-1, c, h * w])
        sequence_output = tf.reduce_mean(sequence_output, axis=-1)
        cls_out = self.head(sequence_output)
        distillation_out = self.dist_head(sequence_output)
        logits = (cls_out + distillation_out) / 2

        # calculate loss
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == tf.int64 or labels.dtype == tf.int32
                ):  # FIXME: is this it?
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = tf.keras.losses.MSE
                if self.num_labels == 1:
                    loss = loss_fct(labels.squeeze(), logits.squeeze())
                else:
                    loss = loss_fct(labels, logits)
            # FIXME: multilabel or multiclass?
            elif self.config.problem_type == "single_label_classification":
                # FIXME: from_logits? Initially I had False from somewhere
                loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
                )
                loss = loss_fct(labels, logits)
            elif self.config.problem_type == "multi_label_classification":
                # FIXME: from_logits? Initially I had False from somewhere
                loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE,
                )
                loss = loss_fct(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
