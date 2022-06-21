# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" MobileNetV1 model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Matthijs/mobilenet_v1-small": "https://huggingface.co/Matthijs/mobilenet_v1-small/resolve/main/config.json",
    "Matthijs/mobilenet_v1-x-small": "https://huggingface.co/Matthijs/mobilenet_v1-x-small/resolve/main/config.json",
    "Matthijs/mobilenet_v1-xx-small": "https://huggingface.co/Matthijs/mobilenet_v1-xx-small/resolve/main/config.json",
    "Matthijs/deeplabv3-mobilenet_v1-small": (
        "https://huggingface.co/Matthijs/deeplabv3-mobilenet_v1-small/resolve/main/config.json"
    ),
    "Matthijs/deeplabv3-mobilenet_v1-x-small": (
        "https://huggingface.co/Matthijs/deeplabv3-mobilenet_v1-x-small/resolve/main/config.json"
    ),
    "Matthijs/deeplabv3-mobilenet_v1-xx-small": (
        "https://huggingface.co/Matthijs/deeplabv3-mobilenet_v1-xx-small/resolve/main/config.json"
    ),
    # See all MobileNetV1 models at https://huggingface.co/models?filter=mobilenet_v1
}


class MobileNetV1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileNetV1Model`]. It is used to instantiate a
    MobileNetV1 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileNetV1
    [Matthijs/MIHTODO](https://huggingface.co/Matthijs/MIHTODO) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.
        hidden_sizes (`List[int]`, *optional*, defaults to `[144, 192, 240]`):
            Dimensionality (hidden size) of the Transformer encoders at each stage.
        neck_hidden_sizes (`List[int]`, *optional*, defaults to `[16, 32, 64, 96, 128, 160, 640]`):
            The number of channels for the feature maps of the backbone.
        expand_ratio (`float`, *optional*, defaults to 4.0):
            Expansion factor for the MobileNetV2 layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The size of the convolutional kernel in the MobileNetV1 layer.
        output_stride (`int`, `optional`, defaults to 32):
            The ratio of the spatial resolution of the output to the resolution of the input image.
        tf_padding (`bool`, `optional`, defaults to `True`):
            Whether to use TensorFlow padding rules on the convolution layers.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the Transformer encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for attached classifiers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        aspp_out_channels (`int`, `optional`, defaults to 256):
            Number of output channels used in the ASPP layer for semantic segmentation.
        atrous_rates (`List[int]`, *optional*, defaults to `[6, 12, 18]`):
            Dilation (atrous) factors used in the ASPP layer for semantic segmentation.
        aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the ASPP layer for semantic segmentation.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example:

    ```python
    >>> from transformers import MobileNetV1Config, MobileNetV1Model

    >>> # Initializing a mobilenet_v1-smallMIHTODO style configuration
    >>> configuration = MobileNetV1Config()

    >>> # Initializing a model from the mobilenet_v1-smallMIHTODO style configuration
    >>> model = MobileNetV1Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mobilenet_v1"

    def __init__(
        self,
        num_channels=3,
        image_size=224,
        depth_multiplier=1,
        # hidden_sizes=[144, 192, 240],
        # neck_hidden_sizes=[16, 32, 64, 96, 128, 160, 640],
        # expand_ratio=4.0,
        hidden_act="relu6",
        conv_kernel_size=3,
        output_stride=32,
        tf_padding=True,
        # hidden_dropout_prob=0.1,
        # attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        # qkv_bias=True,
        # aspp_out_channels=256,
        # atrous_rates=[6, 12, 18],
        # aspp_dropout_prob=0.1,
        # semantic_loss_ignore_index=255,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.depth_multiplier = depth_multiplier
        # self.hidden_sizes = hidden_sizes
        # self.neck_hidden_sizes = neck_hidden_sizes
        # self.expand_ratio = expand_ratio
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.tf_padding = tf_padding
        # self.hidden_dropout_prob = hidden_dropout_prob
        # self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # decode head attributes for semantic segmentation
        # self.aspp_out_channels = aspp_out_channels
        # self.atrous_rates = atrous_rates
        # self.aspp_dropout_prob = aspp_dropout_prob
        # self.semantic_loss_ignore_index = semantic_loss_ignore_index


class MobileNetV1OnnxConfig(OnnxConfig):

    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch"})])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    @property
    def atol_for_validation(self) -> float:
        return 1e-4
