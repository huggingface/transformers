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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="CohereLabs/command-a-vision-07-2025")
@strict
class Cohere2VisionConfig(PreTrainedConfig):
    r"""
    downsample_factor (`int`, *optional*, defaults to 2):
        The factor by which to downsample the input image.
    alignment_intermediate_size (`int`, *optional*, defaults to 36864):
        The size of the intermediate layer for alignment.
    """

    model_type = "cohere2_vision"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    downsample_factor: int = 2
    image_token_id: int = 255036
    alignment_intermediate_size: int = 36864
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=3072,
                image_size=512,
                num_hidden_layers=27,
                num_attention_heads=12,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "cohere2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["cohere2"](tie_word_embeddings=self.tie_word_embeddings)

        super().__post_init__(**kwargs)


__all__ = ["Cohere2VisionConfig"]
