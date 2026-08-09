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
from ...utils import auto_docstring
from ..gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig


@auto_docstring(checkpoint="google/gemma-4-12b-it")
@strict
class Gemma4UnifiedAssistantConfig(PreTrainedConfig):
    r"""
    backbone_hidden_size (`int`, defaults to 3840):
        Hidden size of the target unified model this assistant was trained with.
    use_ordered_embeddings (`bool`, defaults to `False`):
        If `True`, uses an optimally ordered embedding table that needs re-ordering.
    num_centroids (`int`, defaults to 2048):
        Total number of centroids for the centroid-based masked embedder.
    centroid_intermediate_top_k (`int`, defaults to 32):
        Number of active centroids selected during inference.

    Example:

    ```python
    >>> from transformers import (
    ...     Gemma4UnifiedAssistantConfig,
    ...     Gemma4UnifiedAssistantForCausalLM,
    ...     Gemma4UnifiedTextConfig,
    ... )

    >>> # Initializing a Gemma 4 Unified Text config for the assistant.
    >>> text_config = Gemma4UnifiedTextConfig(...)

    >>> # Initializing a Gemma 4 Unified Assistant config.
    >>> configuration = Gemma4UnifiedAssistantConfig(text_config=text_config)

    >>> # Initializing a model from the configuration.
    >>> model = Gemma4UnifiedAssistantForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma4_unified_assistant"
    sub_configs = {
        "text_config": Gemma4UnifiedTextConfig,
    }

    text_config: Gemma4UnifiedTextConfig | dict[str, Any] | None = None

    backbone_hidden_size: int = 3840
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)

        # Assistant reuses the shared kvs across all layers to skip their calculation
        # I.e. it acts as cache shared across the layers
        if self.text_config is not None and not self.text_config.num_kv_shared_layers:
            self.text_config.num_kv_shared_layers = self.text_config.num_hidden_layers

        super().__post_init__(**kwargs)


__all__ = ["Gemma4UnifiedAssistantConfig"]
