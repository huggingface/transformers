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

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


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
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="llama"),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="timm_wrapper"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    vision_use_cls_token: bool = True
    projector_pooling_ratio: int = 1
    image_token_id: int = 128002
    video_token_id: int = 128003
    tie_word_embeddings: bool | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.tie_word_embeddings is None:
            self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)


__all__ = ["PerceptionLMConfig"]
