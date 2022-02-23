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
""" ResNet model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "resnet50-224-1k": "https://huggingface.co/Francesco/resnet50-224-1k/blob/main/config.json",
}


class ResNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ResNetModel`]. It is used to instantiate an
    ResNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the resnet50 architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_size (`int`, optional, defaults to 4):
            Patch size to use in the patch embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to [64, 256, 512, 1024, 2048]):
            Dimensionality (hidden size) embeddings + at each stage .
        depths (`List[int]`, *optional*, defaults to [3, 4, 6, 3]):
            Depth (number of blocks) for each stage.
        embeddings_type (`str`, *optional*, defaults to `"classic"`):
            The embedding layer to use, either `"classic"` or `"3x3"`. If `"classic"`, the original resnet embedding, a
            single agressive `7x7` convolution, is applied. If `"3x3"`, three `3x3` are applied instead.
        layer_type (`str`, *optional*, defaults to `"bottleneck"`):
            The layer to use, it can be either `"basic"` (`ResNetBasicLayer`) or `"bottleneck"`
            (`ResNetBottleNeckLayer`).
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.

    Example:
    ```python
    >>> from transformers import ResNetModel, ResNetConfig

    >>> # Initializing a ResNet resnet50-224 style configuration
    >>> configuration = ResNetConfig()
    >>> # Initializing a model from the resnet50-224 style configuration
    >>> model = ResNetModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "resnet"

    def __init__(
        self,
        num_channels=3,
        hidden_sizes=None,
        depths=None,
        embeddings_type="classic",
        layer_type="bottleneck",
        hidden_act="relu",
        downsample_in_first_stage=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.hidden_sizes = [64, 256, 512, 1024, 2048] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 4, 6, 3] if depths is None else depths
        self.layer_type = layer_type
        self.embeddings_type = embeddings_type
        self.hidden_act = hidden_act
        self.downsample_in_first_stage = downsample_in_first_stage
