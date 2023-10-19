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
""" VitMatte model configuration"""

import copy
from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hustvl/vitmatte-small-composition-1k": "https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json",
}


class VitMatteConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`VitMatteForImageMatting`]. It is used to
    instantiate a ViTMatte model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ViTMatte
    [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `VitDetConfig()`):
            The configuration of the backbone model.
        hidden_size (`int`, *optional*, defaults to 384):
            The number of input channels of the decoder.
        batch_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the batch norm layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        convstream_hidden_sizes (`List[int]`, *optional*, defaults to `[48, 96, 192]`):
            The output channels of the ConvStream module.
        fusion_hidden_sizes (`List[int]`, *optional*, defaults to `[256, 128, 64, 32]`):
            The output channels of the Fusion blocks.

    Example:

    ```python
    >>> from transformers import VitMatteConfig, VitMatteForImageMatting

    >>> # Initializing a ViTMatte hustvl/vitmatte-small-composition-1k style configuration
    >>> configuration = VitMatteConfig()

    >>> # Initializing a model (with random weights) from the hustvl/vitmatte-small-composition-1k style configuration
    >>> model = VitMatteForImageMatting(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vitmatte"

    def __init__(
        self,
        backbone_config: PretrainedConfig = None,
        hidden_size: int = 384,
        batch_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        convstream_hidden_sizes: List[int] = [48, 96, 192],
        fusion_hidden_sizes: List[int] = [256, 128, 64, 32],
        **kwargs,
    ):
        super().__init__(**kwargs)

        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `VitDet` backbone.")
            backbone_config = CONFIG_MAPPING["vitdet"](out_features=["stage4"])
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        self.backbone_config = backbone_config
        self.batch_norm_eps = batch_norm_eps
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.convstream_hidden_sizes = convstream_hidden_sizes
        self.fusion_hidden_sizes = fusion_hidden_sizes

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["backbone_config"] = self.backbone_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
