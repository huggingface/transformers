# coding=utf-8
# Copyright 2022 Amazon, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" ResNetD model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

RESNETD_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "resnetd-50": "https://huggingface.co/zuppif/resnetd-50/blob/main/config.json",
}



class ResNetDConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ResNetDModel`]. It is used to instantiate an
    ResNetD model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [resnetd-50](https://huggingface.co/microsoft/resnetd-50) architecture.

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
        layer_type (`str`, *optional*, defaults to `"bottleneck"`):
            The layer to use, it can be either `"basic"` (used for smaller models, like resnetd-18 or resnetd-34) or
            `"bottleneck"` (used for larger models like resnetd-50 and above).
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.

    Example:
    ```python
    >>> from transformers import ResNetDConfig, ResNetDModel

    >>> # Initializing a ResNetD resnetd-50 style configuration
    >>> configuration = ResNetDConfig()
    >>> # Initializing a model from the resnetd-50 style configuration
    >>> model = ResNetDModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "resnetd"
    layer_types = ["basic", "bottleneck"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="bottleneck",
        hidden_act="relu",
        downsample_in_first_stage=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.downsample_in_first_stage = downsample_in_first_stage
