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


from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hustvl/vitmatte-small-composition-1k": "https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json",
}


class VitMatteConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitMatteForImageMatting`]. It is used to
    instantiate an VitMatte model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the VitMatte
    [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `VitDetConfig()`):
            The configuration of the backbone model.
        hidden_size (`int`, *optional*, defaults to 384):
            The number of input channels of the decoder.

    Example:

    ```python
    >>> from transformers import VitMatteConfig, VitMatteForImageMatting

    >>> # Initializing a VitMatte vitdet-base-patch16-224 style configuration
    >>> configuration = VitMatteConfig()

    >>> # Initializing a model (with random weights) from the vitdet-base-patch16-224 style configuration
    >>> model = VitMatteForImageMatting(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vitmatte"

    def __init__(
        self,
        backbone_config=None,
        hidden_size=384,
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
        self.hidden_size = hidden_size
