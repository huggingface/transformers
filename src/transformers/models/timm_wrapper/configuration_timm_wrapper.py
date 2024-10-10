# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for Backbone models"""

import os
from typing import Any, Dict, Tuple, Union

from ...configuration_utils import PretrainedConfig
from ...utils import is_timm_hub_checkpoint, logging


logger = logging.get_logger(__name__)


class TimmWrapperConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmWrapper`].

    It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
    ```python
    >>> from transformers import TimmWrapperConfig, TimmWrapper

    >>> # Initializing a timm backbone
    >>> configuration = TimmWrapperConfig("resnet50")

    >>> # Initializing a model from the configuration
    >>> model = TimmWrapper(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "timm_wrapper"

    def __init__(self, initializer_range: float = 0.02, **kwargs):
        self.model_name = kwargs.pop("model_name", None)
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        name_or_path = kwargs.get("name_or_path", kwargs.get("pretrained_model_name_or_path", None))

        if "model_name" not in config_dict and is_timm_hub_checkpoint(name_or_path):
            # We are loading from an official timm checkpoint, we need to store the model_name in order to be able to
            # load the model using timm.create_model
            config_dict["model_name"] = name_or_path

        kwargs["num_labels"] = kwargs.get("num_labels", config_dict.get("num_classes", 2))
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_name" not in config_dict and is_timm_hub_checkpoint(pretrained_model_name_or_path):
            # We are loading from an official timm checkpoint, we need to store the model_name in order to be able to
            # load the model using timm.create_model
            config_dict["model_name"] = pretrained_model_name_or_path

        # We first try and use timm's num_classes, otherwise we default to the default value of 2
        kwargs["num_labels"] = kwargs.get("num_labels", config_dict.get("num_classes", 2))
        kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
