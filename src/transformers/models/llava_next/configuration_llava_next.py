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
"""Llava-NeXT model configuration"""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="llava-hf/llava-v1.6-mistral-7b-hf")
@strict
class LlavaNextConfig(PreTrainedConfig):
    r"""
    image_grid_pinpoints (`List`, *optional*, defaults to `[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]`):
        A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
        of the form `(height, width)`.

    Example:

    ```python
    >>> from transformers import LlavaNextForConditionalGeneration, LlavaNextConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava-Next llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> configuration = LlavaNextConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> model = LlavaNextForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_next"
    attribute_map = {"image_token_id": "image_token_index"}
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="llama"),
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="clip_vision_model",
            init_kwargs={
                "intermediate_size": 4096,
                "hidden_size": 1024,
                "patch_size": 14,
                "image_size": 336,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "vocab_size": 32000,
                "projection_dim": 768,
            },
        ),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 32000
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: Literal["default", "full"] = "default"
    vision_feature_layer: int | list[int] = -2
    multimodal_projector_bias: bool = True
    tie_word_embeddings: bool = False
    image_grid_pinpoints: list | None = None
    image_seq_length: int = 576

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.image_grid_pinpoints = (
            self.image_grid_pinpoints
            if self.image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )


__all__ = ["LlavaNextConfig"]
