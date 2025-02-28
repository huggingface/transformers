# coding=utf-8
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

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class AyaVisionConfig(PretrainedConfig):
    pass


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
"""AyaVision model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class AyaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AyaVisionModel`]. It is used to instantiate an
    AyaVision model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to select the vision feature.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
    """

    model_type = "aya_vision"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        tie_word_embeddings=True,
        downsample_factor=2,
        adapter_layer_norm_eps=1e-6,
        image_token=255036,
        **kwargs,
    ):
        self.image_token = image_token
        self.projector_hidden_act = projector_hidden_act
        self.downsample_factor = downsample_factor
        self.adapter_layer_norm_eps = adapter_layer_norm_eps
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
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
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["cohere2"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
