# Copyright 2025 Cohere team. All rights reserved.
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
"""AyaVision model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="CohereForAI/aya-vision-8b")
@strict
class AyaVisionConfig(PreTrainedConfig):
    r"""
    downsample_factor (`int`, *optional*, defaults to 2):
        The downsample factor to apply to the vision features.
    adapter_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon value used for layer normalization in the adapter.
    """

    model_type = "aya_vision"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int | list[int] = -1
    downsample_factor: int = 2
    adapter_layer_norm_eps: float = 1e-6
    image_token_index: int = 255036
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=4304,
                patch_size=14,
                image_size=384,
                num_hidden_layers=26,
                num_attention_heads=14,
                vision_use_head=False,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "cohere2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["cohere2"]()

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {self.vision_feature_select_strategy}"
            )


__all__ = ["AyaVisionConfig"]
