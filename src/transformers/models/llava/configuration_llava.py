# Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""Llava model configuration"""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="llava-hf/llava-1.5-7b-hf")
@strict
class LlavaConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava llava-1.5-7b style configuration
    >>> configuration = LlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-1.5-7b style configuration
    >>> model = LlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 32000
    image_seq_length: int = 576
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: Literal["default", "full"] = "default"
    vision_feature_layer: int | list[int] = -2
    multimodal_projector_bias: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "clip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        # The default value is `False` but this config is used with many model types
        # Attr `tie_word_embeddings` was saved in text config for those models, so we
        # need an ugly workaround and forward-pass the attr from text config
        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings

        super().__post_init__(**kwargs)


__all__ = ["LlavaConfig"]
