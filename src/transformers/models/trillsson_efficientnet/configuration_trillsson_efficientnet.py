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
""" TrillssonEfficientNet model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TRILLSSON_EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "vumichien/nonsemantic-speech-trillsson3": (
        "https://huggingface.co/vumichien/nonsemantic-speech-trillsson3/resolve/main/config.json"
    ),
}


class TrillssonEfficientNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TrillssonEfficientNet Model`]. It is used to
    instantiate an TrillssonEfficientNet model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    TrillssonEfficientNet
    [vumichien/nonsemantic-speech-trillsson3](https://huggingface.co/vumichien/nonsemantic-speech-trillsson3)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        output_size (`int`, *optional*, defaults to 1024):
            The number of output dimensions.
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            The depth multiplier for controling the size of the network.
        depth_divisible_by (`int`, *optional*, defaults to 8):
            The value that the network depth should be divisible by.
        min_depth (`int`, *optional*, defaults to 8):
            The minimum depth of the network.
        hidden_act (`str`, *optional*, defaults to "swish"):
            The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu",
            "relu", "silu", "swish", "gelu_new" and "tanh" are supported.
        drop_connect_rate (`float`, *optional*, defaults to 0.2):
            The dropout ratio for drop connect.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 0.001):
            The epsilon used by the layer normalization layers.
        norm_momentum (`float`, *optional*, defaults to 0.1):
            The momentum used by the layer normalization layers.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.8):
            The dropout probability for the classifier.
        block_config (`list`, *optional*):
            The block configuration for the model.

    Example:
    ```python
    >>> from transformers import TrillssonEfficientNetConfig, TrillssonEfficientNetModel

    >>> # Initializing a "nonsemantic-speech-trillsson3" style configuration
    >>> configuration = TrillssonEfficientNetConfig()
    >>> # Initializing a model from the "nonsemantic-speech-trillsson3" style configuration
    >>> model = TrillssonEfficientNetModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "trillsson_efficientnet"

    def __init__(
        self,
        output_size=1024,
        depth_multiplier=1.0,
        depth_divisible_by=8,
        min_depth=8,
        hidden_act="swish",
        drop_connect_rate=0.2,
        initializer_range=0.02,
        norm_eps=0.001,
        norm_momentum=0.1,
        classifier_dropout_prob=0.8,
        block_configs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if block_configs is None:
            block_configs = [
                [1, 24, 2, 1, 0, 1],
                [4, 48, 4, 2, 0, 1],
                [4, 64, 4, 2, 0, 1],
                [4, 128, 6, 2, 1, 0],
                [6, 160, 9, 1, 1, 0],
                [6, 256, 15, 2, 1, 0],
            ]
        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")

        self.output_size = output_size
        self.depth_multiplier = depth_multiplier
        self.depth_divisible_by = depth_divisible_by
        self.min_depth = min_depth
        self.hidden_act = hidden_act
        self.drop_connect_rate = drop_connect_rate
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.classifier_dropout_prob = classifier_dropout_prob
        self.block_configs = block_configs
