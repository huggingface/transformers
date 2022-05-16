# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" Tensorflow ResNet model."""

from typing import Iterable, Union, Callable, Optional, Dict

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]


class TFResNetConvLayer(tf.keras.layers.Layer):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu", **kwargs
    ):
        super().__init__(**kwargs)
        padding = 'same' if kernel_size // 2 else 'valid'
        self.convolution = tf.keras.layers.Conv2D(
            out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False, name="convolution"
        )
        self.normalization = tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-5, name="normalization")
        self.activation = ACT2FN[activation] if activation is not None else IdentityLayer()

    def call(self, x, training: bool = False):
        hidden_state = x
        # B, C, H, W -> B, H, W, C
        hidden_state = tf.transpose(hidden_state, (0, 2, 3, 1))
        hidden_state = self.convolution(hidden_state)
        # B, H, W, C -> B, C, H, W
        hidden_state = tf.transpose(hidden_state, (0, 3, 1, 2))
        hidden_state = self.normalization(hidden_state, training=training)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetEmbeddings(tf.keras.layers.Layer):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.embedder = TFResNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act, name="embedder"
        )
        self.pooler = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="pooler")

    def call(self, x, training: bool = False):
        hidden_state = x
        hidden_state = self.embedder(hidden_state, training=training)
        # B, C, H, W -> B, H, W, C
        hidden_state = tf.transpose(hidden_state, (0, 2, 3, 1))
        hidden_state = self.pooler(hidden_state)
        # B, H, W, C -> B, C, H, W
        hidden_state = tf.transpose(hidden_state, (0, 3, 1, 2))
        return hidden_state


class TFResNetShortCut(tf.keras.layers.Layer):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.convolution = tf.keras.layers.Conv2D(out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution")
        self.normalization = tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-5, name="normalization")

    def call(self, x, training: bool = False):
        hidden_state = x
        # B, C, H, W -> B, H, W, C
        hidden_state = tf.transpose(hidden_state, (0, 2, 3, 1))
        hidden_state = self.convolution(hidden_state)
        # B, H, W, C -> B, C, H, W
        hidden_state = tf.transpose(hidden_state, (0, 3, 1, 2))
        hidden_state = self.normalization(hidden_state, training=training)
        return hidden_state


class TFResNetBasicLayer(tf.keras.layers.Layer):
    """
    A classic ResNet's residual layer composed by a two `3x3` convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.conv1 = TFResNetConvLayer(in_channels, out_channels, stride=stride)
        self.conv2 = TFResNetConvLayer(out_channels, out_channels, activation=None)
        self.shortcut = (
            TFResNetShortCut(in_channels, out_channels, stride=stride, name="shortcut") if should_apply_shortcut else IdentityLayer(name="shortcut")
        )
        self.activation = ACT2FN[activation]

    def call(self, hidden_state, training: bool = False):
        residual = hidden_state
        hidden_state = self.conv1(hidden_state, training=training)
        hidden_state = self.conv2(hidden_state, training=training)
        residual = self.shortcut(residual, training=training)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class IdentityLayer(tf.keras.layers.Layer):
    """ Helper class to give identity a layer API. """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return x


class TFResNetBottleNeckLayer(tf.keras.layers.Layer):
    """
    A classic ResNet's bottleneck layer composed by a three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remap the reduced features to `out_channels`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", reduction: int = 4, **kwargs
    ):
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.conv0 = TFResNetConvLayer(in_channels, reduces_channels, kernel_size=1, name="layer.0")
        self.conv1 = TFResNetConvLayer(reduces_channels, reduces_channels, stride=stride, name="layer.1")
        self.conv2 = TFResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None, name="layer.2")
        self.shortcut = (
            TFResNetShortCut(in_channels, out_channels, stride=stride, name="shortcut") if should_apply_shortcut else IdentityLayer(name="shortcut")
        )
        self.activation = ACT2FN[activation]

    def call(self, hidden_state, training: bool = False):
        residual = hidden_state

        hidden_state = self.conv0(hidden_state, training=training)
        hidden_state = self.conv1(hidden_state, training=training)
        hidden_state = self.conv2(hidden_state, training=training)

        residual = self.shortcut(residual, training=training)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetStage(tf.keras.layers.Layer):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: ResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)

        layer = TFResNetBottleNeckLayer if config.layer_type == "bottleneck" else TFResNetBasicLayer

        layers = [layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, name="layers.0")]
        layers += [
            layer(out_channels, out_channels, activation=config.hidden_act, name=f"layers.{i + 1}")
            for i in range(depth - 1)
        ]
        self.stage_layers = layers

    def call(self, hidden_state, training: bool = False):
        for layer in self.stage_layers:
            hidden_state = layer(hidden_state, training=training)
        return hidden_state



class TFResNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config: ResNetConfig, **kwargs):
        super().__init__(**kwargs)
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages = [
            TFResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                name=f"stages.0"
            )
        ]
        for i, (in_channels, out_channels, depth) in enumerate(zip(config.hidden_sizes, config.hidden_sizes[1:], config.depths[1:])):
            self.stages.append(TFResNetStage(config, in_channels, out_channels, depth=depth, name=f"stages.{i + 1}"))

    def call(
        self, hidden_state: tf.Tensor, output_hidden_states: bool = False, return_dict: bool = True, training: bool = False
    ) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state, training=training)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class TFResNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ResNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network. Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(3, self.config.num_channels, 224, 224), # FIXME - do I need the image size here? #self.config.image_size, self.config.image_size),
            dtype=tf.float32,
        )
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}


RESNET_START_DOCSTRING = r"""
    This model is a Tensorflow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def normalize_data_format(value: str) -> str:
    """
    From tensorflow addons
    https://github.com/tensorflow/addons/blob/8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/tensorflow_addons/utils/keras_utils.py#L71
    """
    if value is None:
        value = tf.keras.backend.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            'The `data_format` argument must be one of "channels_first", "channels_last". Received: ' + str(value)
        )
    return data_format


class AdaptivePooling2D(tf.keras.layers.Layer):
    """Parent class for 2D pooling layers with adaptive kernel size.
    This class only exists for code reuse. It will never be an exposed API.
    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 2 integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
    """
    def __init__(
        self,
        reduce_function: Callable,
        output_size: Union[int, Iterable[int]],
        data_format=None,
        **kwargs,
    ):
        self.data_format = normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = (output_size, output_size,) if isinstance(output_size, int) else tuple(output_size)
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            out_vect = self.reduce_function(split_rows, axis=[2, 4])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            out_vect = self.reduce_function(split_rows, axis=[3, 5])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    input_shape[3],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_size[0],
                    self.output_size[1],
                ]
            )

        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}



class AdaptiveAveragePooling2D(AdaptivePooling2D):
    """Average Pooling with adaptive kernel size.
    Args:
      output_size: Tuple of integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.
    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, height, width, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, height, width)`.
    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
    """
    def __init__(
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_mean, output_size, data_format, **kwargs)


@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class TFResNetModel(TFResNetPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.embedder = TFResNetEmbeddings(config, name="embedder")
        self.encoder = TFResNetEncoder(config, name="encoder")

    def pooler(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # B, C, H, W -> B, H, W, C
        hidden_state = tf.transpose(hidden_state, (0, 2, 3, 1))
        hidden_state = AdaptiveAveragePooling2D((1, 1))(hidden_state)
        # B, H, W, C -> B, C, H, W
        hidden_state = tf.transpose(hidden_state, (0, 3, 1, 2))
        return hidden_state

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    @unpack_inputs
    def call(
        self, pixel_values: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, training: bool = False
    ) -> TFBaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values, training=training)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
class TFResNetForImageClassification(TFResNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.resnet = TFResNetModel(config, name="resnet")
        # classification head
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier.1") if config.num_labels > 0 else IdentityLayer(name="classifier.1")

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor = None,
        labels: tf.Tensor = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        training: bool = False
    ) -> TFImageClassifierOutputWithNoAttention:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(tf.keras.layers.Flatten()(pooled_output), training=training) #FIXME - tidyup

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
