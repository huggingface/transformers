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
from ...utils import logging, is_timm_hub_checkpoint


logger = logging.get_logger(__name__)


class TimmWrapperConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmWrapper`].

    It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

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

    def __init__(self, **kwargs):
        self.model_name = kwargs.pop("model_name", None)
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_name" not in config_dict and is_timm_hub_checkpoint(pretrained_model_name_or_path):
            # We are loading from an official timm checkpoint, we need to store the model_name in order to be able to
            # load the model using timm.create_model
            config_dict["model_name"] = pretrained_model_name_or_path

        kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
