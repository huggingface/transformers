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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="TODO")
@strict
class OnyxAssistantConfig(PreTrainedConfig):
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
    >>>     OnyxAssistantConfig,
    >>>     OnyxAssistantForCausalLM,
    >>>     OnyxTextConfig,
    >>> )

    >>> # Initializing a Onyx Text config similar to TODO.
    >>> text_config = OnyxTextConfig(...)

    >>> # Initializing a Onyx Assistant config similar to TODO.
    >>> configuration = OnyxAssistantConfig(text_config)

    >>> # Initializing a model from the TODO configuration.
    >>> model = OnyxAssistantForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "onyx_assistant"
    default_theta = 500000.0

    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_parameters: dict | None = None
    max_position_embeddings: int = 131072
    sliding_window: int = 2048
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0
    hidden_act: str = "silu"
    bos_token_id: int | None = 200000
    eos_token_id: int | None = 200001
    pad_token_id: int | None = 200018

    block_size: int = 16
    mask_token_id: int = 201818
    target_layer_ids: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        if self.target_layer_ids is None:
            self.target_layer_ids = [2, 14, 26, 38, 50]
        super().__post_init__(**kwargs)


__all__ = ["OnyxAssistantConfig"]
