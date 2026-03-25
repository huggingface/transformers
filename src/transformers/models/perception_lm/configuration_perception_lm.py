# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""PerceptionLM model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig


@auto_docstring(checkpoint="facebook/Perception-LM-1B")
@strict
class PerceptionLMConfig(PreTrainedConfig):
    r"""
    vision_use_cls_token (`bool`, *optional*, defaults to `True`):
        Whether CLS token is used in the vision backbone. If used, we remove CLS token embedding from vision output.
    projector_pooling_ratio (`int`, *optional*, defaults to 1):
        The pooling ratio used in the multimodal projector.
    """

    model_type = "perception_lm"
    sub_configs = {"text_config": AutoConfig, "vision_config": TimmWrapperConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    vision_use_cls_token: bool = True
    projector_pooling_ratio: int = 1
    image_token_id: int = 128002
    video_token_id: int = 128003

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = TimmWrapperConfig(**self.vision_config)
        elif isinstance(self.vision_config, TimmWrapperConfig):
            pass
        elif self.vision_config is None:
            self.vision_config = TimmWrapperConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        super().__post_init__(**kwargs)


__all__ = ["PerceptionLMConfig"]
