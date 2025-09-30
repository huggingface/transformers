# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LFM2-VL model."""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class Lfm2VlConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Lfm2VlForConditionalGeneration`]. It is used to instantiate an
    Lfm2Vl model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Lfm2-VL-1.6B.

    e.g. [LiquidAI/LFM2-VL-1.6B](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`AutoConfig | dict`,  *optional*, defaults to `Siglip2ImageConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`AutoConfig | dict`, *optional*, defaults to `Lfm2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 396):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        projector_hidden_size (`int`, *optional*, defaults to 2560):
            The hidden size of the multimodal projector.
        projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.
        downsample_factor (`int`, *optional*, defaults to 2):
            The downsample_factor factor of the vision backbone.
    """

    model_type = "lfm2-vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=396,
        projector_hidden_act="gelu",
        projector_hidden_size=2560,
        projector_bias=True,
        downsample_factor=2,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.projector_bias = projector_bias
        self.downsample_factor = downsample_factor

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip2_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["siglip2_vision_model"]()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "lfm2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["lfm2"]()

        self.vision_config = vision_config
        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["Lfm2VlConfig"]
