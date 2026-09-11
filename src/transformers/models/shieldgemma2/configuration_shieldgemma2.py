# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="google/shieldgemma-2-4b-it")
@strict
class ShieldGemma2Config(PreTrainedConfig):
    r"""
    tie_word_embeddings (`bool`, *optional*):
        Whether to tie the word embeddings. Defaults to the value of `text_config.tie_word_embeddings` if not set.
    mm_tokens_per_image (`int`, *optional*, defaults to 256):
        The number of tokens per image embedding.
    boi_token_index (`int`, *optional*, defaults to 255999):
        The begin-of-image token index to wrap the image prompt.
    eoi_token_index (`int`, *optional*, defaults to 256000):
        The end-of-image token index to wrap the image prompt.

    Example:

    ```python
    >>> from transformers import ShieldGemma2ForConditionalGeneration, ShieldGemma2Config, SiglipVisionConfig, ShieldGemma2TextConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a ShieldGemma2 Text config
    >>> text_config = ShieldGemma2TextConfig()

    >>> # Initializing a ShieldGemma2 gemma-3-4b style configuration
    >>> configuration = ShieldGemma2Config(vision_config, text_config)

    >>> # Initializing a model from the gemma-3-4b style configuration
    >>> model = ShieldGemma2TextConfig(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "shieldgemma2"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="gemma3_text"),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="siglip_vision_model"),
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    mm_tokens_per_image: int = 256
    boi_token_index: int = 255_999
    eoi_token_index: int = 256_000
    image_token_index: int = 262_144
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if not self.tie_word_embeddings and getattr(self.text_config, "tie_word_embeddings", False):
            self.tie_word_embeddings = True


__all__ = ["ShieldGemma2Config"]
