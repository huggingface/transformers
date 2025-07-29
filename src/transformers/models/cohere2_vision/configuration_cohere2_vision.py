# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Cohere2 Vision model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class Cohere2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Cohere2VisionForConditionalGeneration`]. It is used to instantiate an
    Cohere2 Vision model according to the specified arguments, defining the model architecture.

    [CohereLabs/command-a-vision-07-2025](https://huggingface.co/CohereLabs/command-a-vision-07-2025)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        downsample_factor (`int`, *optional*, defaults to 2): The factor by which to downsample the input image.
        image_token (`int`, *optional*, defaults to 255036): The token ID to use for the image input.
        alignment_intermediate_size (`int`, *optional*, defaults to 36864): The size of the intermediate layer for alignment.
    """

    model_type = "cohere2_vision"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        downsample_factor=2,
        image_token=255036,
        alignment_intermediate_size=36864,
        **kwargs,
    ):
        self.image_token = image_token
        self.downsample_factor = downsample_factor
        self.alignment_intermediate_size = alignment_intermediate_size

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                image_size=512,
                intermediate_size=3072,
                num_attention_heads=12,
                num_hidden_layers=27,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["cohere2"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=True, **kwargs)


__all__ = ["Cohere2VisionConfig"]
