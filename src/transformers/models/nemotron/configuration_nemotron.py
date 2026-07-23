# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Nemotron model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="thhaus/nemotron3-8b")
@strict
class NemotronConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import NemotronModel, NemotronConfig

    >>> # Initializing a Nemotron nemotron-15b style configuration
    >>> configuration = NemotronConfig()

    >>> # Initializing a model from the nemotron-15b style configuration
    >>> model = NemotronModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nemotron"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 256000
    hidden_size: int = 6144
    intermediate_size: int = 24576
    num_hidden_layers: int = 32
    num_attention_heads: int = 48
    head_dim: int | None = None
    num_key_value_heads: int | None = None
    hidden_act: str = "relu2"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.0134
    norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 3
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False

    def __post_init__(self, **kwargs):
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["NemotronConfig"]
