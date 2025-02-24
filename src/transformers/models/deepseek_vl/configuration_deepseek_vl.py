# coding=utf-8
# Copyright 2025 The DeepseekAI and HuggingFace Team. All rights reserved.
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
"""DeepseekVL model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..sam import SamVisionConfig
from ..siglip import SiglipVisionConfig
from ..llama import LlamaConfig


logger = logging.get_logger(__name__)


class DeepseekVLVisionConfig(PretrainedConfig):
    r"""
    This class handles the configuration for the vision component of the DeepseekVL model.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the model of the DeepseekVL
    [geetu040/deepseek-vl-7b-chat](https://huggingface.co/geetu040/deepseek-vl-7b-chat) architecture.
    This class wraps the 2 vision model configurations: `SiglipVisionConfig` and `SamVisionConfig`.

    Args:

    """

    model_type = "deepseek_vl_vision_model"
    base_config_key = "vision_config"
    sub_configs = {"low_res_config": SiglipVisionConfig, "high_res_config": SamVisionConfig}

    def __init__(self, use_high_res=False, low_res_config=None, high_res_config=None, **kwargs):
        super().__init__(**kwargs)

        if low_res_config is None:
            low_res_config = {}
            logger.info("`low_res_config` is `None`. Initializing the `SiglipVisionConfig` with default values.")

        if high_res_config is None:
            high_res_config = {}
            logger.info("`high_res_config` is `None`. Initializing the `SamVisionConfig` with default values.")

        self.use_high_res = use_high_res
        self.low_res_config = SiglipVisionConfig(**low_res_config)
        self.high_res_config = SamVisionConfig(**high_res_config)


class DeepseekVLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekVLModel`]. It is used to instantiate a
    DeepseekVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DeepseekVL
    [geetu040/deepseek-vl-7b-chat](https://huggingface.co/geetu040/deepseek-vl-7b-chat) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:

    Example:

    ```python
    >>> from transformers import DeepseekVLConfig, DeepseekVLModel

    >>> # Initializing a DeepseekVL geetu040/deepseek-vl-7b-chat style configuration
    >>> configuration = DeepseekVLConfig()

    >>> # Initializing a model (with random weights) from the geetu040/deepseek-vl-7b-chat style configuration
    >>> model = DeepseekVLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl"
    sub_configs = {
        "text_config": LlamaConfig,
        "vision_config": DeepseekVLVisionConfig,
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `LlamaConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `DeepseekVLVisionConfig` with default values.")

        self.text_config = LlamaConfig(**text_config)
        self.vision_config = DeepseekVLVisionConfig(**vision_config)

    @classmethod
    def from_text_vision_configs(cls, text_config: LlamaConfig, vision_config: DeepseekVLVisionConfig, **kwargs):
        r"""
        Instantiate a [`DeepseekVLConfig`] from llama text model configuration and DeepseekVLVisionModel
        model configuration.

        Returns:
            [`DeepseekVLConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


__all__ = ["DeepseekVLVisionConfig", "DeepseekVLConfig"]
