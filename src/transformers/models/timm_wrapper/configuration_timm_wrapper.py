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

from ...configuration_utils import PretrainedConfig
from ...utils import logging


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
        self.initializer_range = initializer_range

        # timm models have `num_classes` instead of `num_labels`
        kwargs["num_labels"] = kwargs.get("num_labels", kwargs.get("num_classes", 2))
        super().__init__(**kwargs)
