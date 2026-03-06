# Copyright 2025 the Cohere Inc. team. All rights reserved.
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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="CohereLabs/command-a-vision-07-2025")
class Cohere2VisionConfig(PreTrainedConfig):
    r"""
    downsample_factor (`int`, *optional*, defaults to 2):
        The factor by which to downsample the input image.
    alignment_intermediate_size (`int`, *optional*, defaults to 36864):
        The size of the intermediate layer for alignment.
    """

    model_type = "cohere2_vision"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        downsample_factor=2,
        image_token_id=255036,
        alignment_intermediate_size=36864,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.downsample_factor = downsample_factor
        self.image_token_id = image_token_id
        self.alignment_intermediate_size = alignment_intermediate_size

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=3072,
                image_size=512,
                num_hidden_layers=27,
                num_attention_heads=12,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "cohere2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["cohere2"](tie_word_embeddings=tie_word_embeddings)

        self.text_config = text_config
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Cohere2VisionConfig"]
