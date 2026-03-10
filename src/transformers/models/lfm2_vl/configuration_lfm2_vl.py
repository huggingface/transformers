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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="LiquidAI/LFM2-VL-1.6B")
class Lfm2VlConfig(PreTrainedConfig):
    r"""
    downsample_factor (`int`, *optional*, defaults to 2):
        The downsample_factor factor of the vision backbone.
    projector_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in the multimodal projector.
    projector_use_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to use layernorm in the multimodal projector.
    """

    model_type = "lfm2_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=396,
        projector_hidden_act="gelu",
        projector_hidden_size=2560,
        projector_bias=True,
        projector_use_layernorm=True,
        downsample_factor=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.projector_bias = projector_bias
        self.projector_use_layernorm = projector_use_layernorm
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
        self.tie_word_embeddings = getattr(text_config, "tie_embedding", tie_word_embeddings)

        super().__init__(**kwargs)


__all__ = ["Lfm2VlConfig"]
