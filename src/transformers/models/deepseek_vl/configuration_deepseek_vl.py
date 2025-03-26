# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
from ..llama import LlamaConfig
from ..sam import SamVisionConfig
from ..siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)


class DeepseekVLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekVLModel`]. It is used to instantiate a
    DeepseekVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DeepseekVL
    [deepseek-ai/deepseek-vl-7b-chat-hf](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[LlamaConfig, dict]`  *optional*, defaults to , LlamaConfig`):
            Configuration for the `Llama` based language model.
        low_res_vision_config (`Union[SiglipVisionConfig, dict]`  *optional*, defaults to , SiglipVisionConfig`):
            Configuration for the `SiglipVisionModel` based low resolution image model.
        high_res_vision_config (`Union[SamVisionConfig, dict]`  *optional*, defaults to , SamVisionConfig`):
            Configuration for the `SamVisionModel` based high resolution image model.
        use_high_res_vision (`bool`, *optional*, defaults to `True`):
            Whether to use high resolution image model (SamVisionModel)
            along with low resolution image model (SiglipVisionModel).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import DeepseekVLConfig, DeepseekVLModel

    >>> # Initializing a DeepseekVL deepseek-ai/deepseek-vl-7b-chat-hf style configuration
    >>> configuration = DeepseekVLConfig()

    >>> # Initializing a model (with random weights) from the deepseek-ai/deepseek-vl-7b-chat-hf style configuration
    >>> model = DeepseekVLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl"
    sub_configs = {
        "text_config": LlamaConfig,
        "low_res_vision_config": SiglipVisionConfig,
        "high_res_vision_config": SamVisionConfig,
    }

    def __init__(
        self,
        text_config: LlamaConfig = None,
        low_res_vision_config: SiglipVisionConfig = None,
        high_res_vision_config: SamVisionConfig = None,
        use_high_res_vision: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `LlamaConfig` with default values.")

        if low_res_vision_config is None:
            low_res_vision_config = {}
            logger.info(
                "`low_res_vision_config` is `None`. Initializing the `SiglipVisionConfig` with default values."
            )

        if high_res_vision_config is None:
            high_res_vision_config = {}
            logger.info("`high_res_vision_config` is `None`. Initializing the `SamVisionConfig` with default values.")

        self.use_high_res_vision = use_high_res_vision
        self.text_config = LlamaConfig(**text_config)
        self.low_res_vision_config = SiglipVisionConfig(**low_res_vision_config)
        self.high_res_vision_config = SamVisionConfig(**high_res_vision_config)
        self.initializer_range = initializer_range
        self.image_token_index = kwargs.get("image_token_index", 100015)


__all__ = ["DeepseekVLConfig"]
