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
""" TF 2.0 ResNet model."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import ModelOutput
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras_serializable,
    unpack_inputs,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_resnet import ResNetConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 7, 7, 2048]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]


@dataclass
class TFBaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None


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


class TFAdaptiveAvgPool2D(tf.keras.layers.Layer):
    def __init__(self, output_shape, mode="dense", transpose_tf=True, **kwargs):
        super().__init__(**kwargs)
        self.transpose_tf = transpose_tf
        self.h_pool = TFAdaptiveAvgPool1D(output_shape[0], mode=mode)
        self.w_pool = TFAdaptiveAvgPool1D(output_shape[1], mode=mode)

    def call(self, inputs):
        # Rearrange from NHWC -> NCHW
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        # Perform W-pooling
        inputs = self.w_pool(inputs)
        # Rearrange NCHW -> NCWH
        inputs = tf.transpose(inputs, perm=[0, 1, 3, 2])
        # Perform H-pooling
        inputs = self.h_pool(inputs)
        if self.transpose_tf:
            # Rearrange from NCWH -> NHWC
            inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
        else:
            # Rearrange from NCWH -> NCHW
            inputs = tf.transpose(inputs, perm=[0, 1, 3, 2])
        return inputs


class TFResNetConvLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        padding_amnt = kernel_size // 2
        self.padding = tf.keras.layers.ZeroPadding2D(padding=padding_amnt)
        self.convolution = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="VALID",
            use_bias=False,
            name="convolution",
        )
        self.normalization = tf.keras.layers.BatchNormalization(name="normalization")
        self.activation = get_tf_activation(activation) if activation is not None else tf.identity

    def call(self, input: tf.Tensor) -> tf.Tensor:
        input = self.padding(input)
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetEmbeddings(tf.keras.layers.Layer):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.embedder = TFResNetConvLayer(
            config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act, name="embedder"
        )
        self.pooler = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="SAME", name="pooler")

    def call(self, input: tf.Tensor) -> tf.Tensor:
        embedding = self.embedder(input)
        embedding = self.pooler(embedding)
        return embedding


class TFResNetShortCut(tf.keras.layers.Layer):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `strides=2`.
    """

    def __init__(self, out_channels: int, stride: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.convolution = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )
        self.normalization = tf.keras.layers.BatchNormalization(name="normalization")

    def call(self, input: tf.Tensor) -> tf.Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class TFResNetBasicLayer(tf.keras.layers.Layer):
    """
    A classic ResNet's residual layer composed by a two `3x3` convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut = (
            TFResNetShortCut(out_channels, strides=stride, name="shortcut") if should_apply_shortcut else tf.identity
        )
        # self.layer = nn.Sequential(
        #     ResNetConvLayer(in_channels, out_channels, stride=stride),
        #     ResNetConvLayer(out_channels, out_channels, activation=None),
        # )
        self.layers = [
            TFResNetConvLayer(out_channels, stride=stride, name="layer.0"),
            TFResNetConvLayer(out_channels, activation=None, name="layer.1"),
        ]
        self.activation = get_tf_activation(activation)

    def call(self, hidden_state):
        residual = hidden_state
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetBottleNeckLayer(tf.keras.layers.Layer):
    """
    A classic ResNet's bottleneck layer composed by a three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remap the reduced features to `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        reduction: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.shortcut = (
            TFResNetShortCut(out_channels, stride=stride, name="shortcut") if should_apply_shortcut else tf.identity
        )
        self.layers = [
            TFResNetConvLayer(reduces_channels, kernel_size=1, name="layer.0"),
            TFResNetConvLayer(reduces_channels, stride=stride, name="layer.1"),
            TFResNetConvLayer(out_channels, kernel_size=1, activation=None, name="layer.2"),
        ]
        self.activation = get_tf_activation(activation)

    def call(self, hidden_state):
        residual = hidden_state
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetStage(tf.keras.layers.Layer):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(
        self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ):
        super().__init__(**kwargs)

        layer = TFResNetBottleNeckLayer if config.layer_type == "bottleneck" else TFResNetBasicLayer

        self.layers = [
            # downsampling is done in the first layer with stride of 2
            layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, name="layers.0"),
            *[
                layer(out_channels, out_channels, activation=config.hidden_act, name=f"layers.{i+1}")
                for i in range(depth - 1)
            ],
        ]

    def call(self, input: tf.Tensor) -> tf.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class TFResNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config: ResNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.stages = []
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages.append(
            TFResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                name="stages.0",
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, config.depths[1:])):
            self.stages.append(TFResNetStage(config, in_channels, out_channels, depth=depth, name=f"stages.{i+1}"))

    def call(
        self, hidden_state: tf.Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


@keras_serializable
class TFResNetMainLayer(tf.keras.layers.Layer):
    config_class = ResNetConfig

    def __init__(self, config: ResNetConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedder = TFResNetEmbeddings(config, name="embedder")
        self.encoder = TFResNetEncoder(config, name="encoder")
        self.pooler = TFAdaptiveAvgPool2D(output_shape=(1, 1), transpose_tf=False, name="pooler")

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> TFBaseModelOutputWithPooling:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        embedding_output = self.embedder(pixel_values, training=training)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        last_hidden_state = encoder_outputs[0]
        # Change to NCHW output format have uniformity in the modules
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))

        pooled_output = self.pooler(last_hidden_state)
        # Change the other hidden state outputs to NCHW as well
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
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
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(
                3,
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


RESNET_START_DOCSTRING = r"""
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

    </Tip>

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ConvNextFeatureExtractor`]. See
            [`ConvNextFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
"""


@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class TFResNetModel(TFResNetPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.resnet = TFResNetMainLayer(config, name="resnet")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, TFResNetModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        >>> model = TFConvNextModel.from_pretrained("microsoft/resnet-50")

        >>> inputs = feature_extractor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.resnet(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return (outputs[0],) + outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )


@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
class TFResNetForImageClassification(TFResNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ResNetConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.resnet = TFResNetMainLayer(config, name="resnet")

        # Classifier head
        self.classifier = [
            tf.keras.layers.Flatten(name="classifier.0"),
            tf.keras.layers.Dense(
                units=config.num_labels,
                name="classifier.1",
            )
            if config.num_labels > 0
            else partial(tf.identity, name="classifier.1"),
        ]

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, TFResNetForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = ConvNextFeatureExtractor.from_pretrained("microsoft/resnet-50")
        >>> model = TFConvNextForImageClassification.from_pretrained("microsoft/resnet-50")

        >>> inputs = feature_extractor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.resnet(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        flattened_pooled_output = self.classifier[0](pooled_output)
        logits = self.classifier[1](flattened_pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
