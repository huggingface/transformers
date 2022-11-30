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
""" BiT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

RESNETV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/resnetv2-50": "https://huggingface.co/google/resnetv2-50/resolve/main/config.json",
}


class BitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BitModel`]. It is used to instantiate an BiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BiT
    [google/resnetnv2-50](https://huggingface.co/google/resnetnv2-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"preactivation"`):
            The layer to use, it can be either `"preactivation"` or `"bottleneck"`.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        num_groups (`int`, *optional*, defaults to `32`):
            Number of groups used for the `BitGroupNormActivation` layers
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop path rate for the stochastic depth.
        output_stride (`int`, *optional*, defaults to 32):
            The output stride of the model.
        width_factor (`int`, *optional*, defaults to 1):
            The width factor for the model.

    Example:
    ```python
    >>> from transformers import BitConfig, BitModel

    >>> # Initializing a BiT resnetv2-50 style configuration
    >>> configuration = BitConfig()

    >>> # Initializing a model (with random weights) from the resnetv2-50 style configuration
    >>> model = BitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "bit"
    layer_types = ["preactivation", "bottleneck"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        stem_type="",
        layer_type="preactivation",
        hidden_act="relu",
        drop_path_rate=0.0,
        output_stride=32,
        width_factor=1,
        conv_layer="std_conv",
        num_groups=32,
        out_features=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.stem_type = stem_type
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.num_groups = num_groups
        self.drop_path_rate = drop_path_rate
        self.output_stride = output_stride
        self.width_factor = width_factor
        self.conv_layer = conv_layer
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        if out_features is not None:
            if not isinstance(out_features, list):
                raise ValueError("out_features should be a list")
            for feature in out_features:
                if feature not in self.stage_names:
                    raise ValueError(
                        f"Feature {feature} is not a valid feature name. Valid names are {self.stage_names}"
                    )
        self.out_features = out_features
