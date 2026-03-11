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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/Perception-LM-1B")
class PerceptionLMConfig(PreTrainedConfig):
    r"""
    vision_use_cls_token (`bool`, *optional*, defaults to `True`):
        Whether CLS token is used in the vision backbone. If used, we remove CLS token embedding from vision output.
    projector_pooling_ratio (`int`, *optional*, defaults to 1):
        The pooling ratio used in the multimodal projector.
    """

    model_type = "perception_lm"
    sub_configs = {"text_config": AutoConfig, "vision_config": TimmWrapperConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        vision_use_cls_token=True,
        projector_pooling_ratio=1,
        image_token_id=128002,
        video_token_id=128003,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        if isinstance(vision_config, dict):
            vision_config = TimmWrapperConfig(**vision_config)
        elif isinstance(vision_config, TimmWrapperConfig):
            pass
        elif vision_config is None:
            vision_config = TimmWrapperConfig()
        self.vision_config = vision_config
        self.vision_use_cls_token = vision_use_cls_token

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        self.projector_pooling_ratio = projector_pooling_ratio
        super().__init__(**kwargs)


__all__ = ["PerceptionLMConfig"]
