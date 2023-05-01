# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" MobileViTv2 model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "apple/mobilevitv2-1.0": "https://huggingface.co/apple/mobilevitv2-1.0/resolve/main/config.json",
}


class MobileViTv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileViTv2Model`]. It is used to instantiate a
    MobileViTv2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileViTv2
    [apple/mobilevitv2-1.0](https://huggingface.co/apple/mobilevitv2-1.0) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 256):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 2):
            The size (resolution) of each patch.
        hidden_sizes (`List[int]`, *optional*, defaults to `[144, 192, 240]`):
            Dimensionality (hidden size) of the Transformer encoders at each stage.
        mlp_ratio (`float`, *optional*, defaults to 2.0):
            The ratio of the number of channels in the output of the MLP to the number of channels in the input.
        expand_ratio (`float`, *optional*, defaults to 4.0):
            Expansion factor for the MobileNetv2 layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The size of the convolutional kernel in the MobileViTv2 layer.
        output_stride (`int`, `optional`, defaults to 32):
            The ratio of the spatial resolution of the output to the resolution of the input image.
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
    >>> from transformers import MobileViTv2Config, MobileViTv2Model

    >>> # Initializing a mobilevitv2-small style configuration
    >>> configuration = MobileViTv2Config()

    >>> # Initializing a model from the mobilevitv2-small style configuration
    >>> model = MobileViTv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mobilevitv2"

    def __init__(
        self,
        num_channels=3,
        image_size=256,
        patch_size=2,
        mlp_ratio=2.0,
        expand_ratio=2.0,
        hidden_act="swish",
        conv_kernel_size=3,
        output_stride=32,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        aspp_out_channels=256,
        atrous_rates=[6, 12, 18],
        aspp_dropout_prob=0.1,
        semantic_loss_ignore_index=255,
        classifier_dropout_prob=0.1,
        ####
        width_multiplier=1.0,    # Width multiplier. Defaults to 1.0
        ffn_dropout=0.0,         # "Dropout between FFN layers. Defaults to 0.0",
        attn_dropout=0.0,        # Dropout in attention layer. Defaults to 0.0
        
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.expand_ratio = expand_ratio
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # decode head attributes for semantic segmentation
        self.aspp_out_channels = aspp_out_channels
        self.atrous_rates = atrous_rates
        self.aspp_dropout_prob = aspp_dropout_prob
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        
        ####
        self.width_multiplier = width_multiplier
        self.ffn_dropout = ffn_dropout
        self.attn_dropout = attn_dropout
        self.classifier_dropout_prob=classifier_dropout_prob
        


class MobileViTv2OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"})])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    @property
    def atol_for_validation(self) -> float:
        return 1e-4
