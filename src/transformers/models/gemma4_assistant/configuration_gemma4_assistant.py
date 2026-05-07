# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..gemma4.configuration_gemma4 import Gemma4TextConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4AssistantConfig(PreTrainedConfig):
    r"""
    backbone_hidden_size (`int`, defaults to 1536):
        Hidden size of the target model this assistant was trained with.
    use_ordered_embeddings (`bool`, defaults to False):
        If True, uses an embedding table ordered for optimal assistant model performance that needs to be re-ordered to
        align with the main model.
    num_centroids (`int`, defaults to 2048):
        The total numer of centroids.
    centroid_intermediate_top_k (`int`, defaults to 32):
        The number of active centroids.

    Example:

    ```python
    >>> from transformers import (
    >>>     Gemma4AssistantConfig,
    >>>     Gemma4AssistantForCausalLM,
    >>>     Gemma4TextConfig,
    >>> )

    >>> # Initializing a Gemma 4 Text config similar to google/gemma-4-e2b-it-assistant.
    >>> text_config = Gemma4TextConfig(...)

    >>> # Initializing a Gemma 4 Assistant config similar to google/gemma-4-e2b-it-assistant.
    >>> configuration = Gemma4AssistantConfig(text_config)

    >>> # Initializing a model from the google/gemma-4-e2b-it-assistant configuration.
    >>> model = Gemma4AssistantForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma4_assistant"
    sub_configs = {
        "text_config": Gemma4TextConfig,
    }

    text_config: Gemma4TextConfig | dict[str, Any] | None = None

    backbone_hidden_size: int = 1536
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = Gemma4TextConfig(**self.text_config)

        # Assistant reuses the shared kvs across all layers to skip their calculation
        # I.e. it acts as cache shared across the layers
        if self.text_config is not None and not self.text_config.num_kv_shared_layers:
            self.text_config.num_kv_shared_layers = self.text_config.num_hidden_layers

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        text_config: Gemma4TextConfig | None = self.text_config

        if text_config is not None:
            if text_config.hidden_size_per_layer_input != 0:
                raise ValueError(
                    f"hidden_size_per_layer_input must be 0, got {text_config.hidden_size_per_layer_input}"
                )
            if text_config.enable_moe_block is not False:
                raise ValueError(f"enable_moe_block must be False, got {text_config.enable_moe_block}")
            if text_config.use_double_wide_mlp is not False:
                raise ValueError(f"use_double_wide_mlp must be False, got {text_config.use_double_wide_mlp}")
            if text_config.vocab_size_per_layer_input != 0:
                raise ValueError(f"vocab_size_per_layer_input must be 0, got {text_config.vocab_size_per_layer_input}")
            if text_config.num_kv_shared_layers != text_config.num_hidden_layers:
                raise ValueError(
                    "All layers in a Gemma 4 Assistant models must be shared."
                    f" Expected {text_config.num_hidden_layers}. Got {text_config.num_kv_shared_layers}"
                )
        return super().validate_architecture()


__all__ = ["Gemma4AssistantConfig"]
