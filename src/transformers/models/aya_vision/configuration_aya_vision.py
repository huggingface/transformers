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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="CohereForAI/aya-vision-8b")
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

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        downsample_factor=2,
        adapter_layer_norm_eps=1e-6,
        image_token_index=255036,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.downsample_factor = downsample_factor
        self.adapter_layer_norm_eps = adapter_layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=4304,
                patch_size=14,
                image_size=384,
                num_hidden_layers=26,
                num_attention_heads=14,
                vision_use_head=False,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "cohere2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["cohere2"]()

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["AyaVisionConfig"]
