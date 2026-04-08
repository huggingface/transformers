# Copyright 2023 Adept AI and the HuggingFace Inc. team. All rights reserved.
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
"""Persimmon model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="adept/persimmon-8b-base")
@strict
class PersimmonConfig(PreTrainedConfig):
    r"""
    qk_layernorm (`bool`, *optional*, default to `True`):
        Whether or not to normalize the Queries and Keys after projecting the hidden states

    Example:

    ```python
    >>> from transformers import PersimmonModel, PersimmonConfig

    >>> # Initializing a Persimmon persimmon-7b style configuration
    >>> configuration = PersimmonConfig()
    ```"""

    model_type = "persimmon"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 262144
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 36
    num_attention_heads: int = 64
    hidden_act: str = "relu2"
    max_position_embeddings: int = 16384
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    qk_layernorm: bool = True
    hidden_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["PersimmonConfig"]
