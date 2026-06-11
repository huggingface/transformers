# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""ALBERT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="albert/albert-xxlarge-v2")
@strict
class AlbertConfig(PreTrainedConfig):
    r"""
    num_hidden_groups (`int`, *optional*, defaults to 1):
        Number of groups for the hidden layers, parameters in the same group are shared.
    inner_group_num (`int`, *optional*, defaults to 1):
        The number of inner repetition of attention and ffn.

    Examples:

    ```python
    >>> from transformers import AlbertConfig, AlbertModel

    >>> # Initializing an ALBERT-xxlarge style configuration
    >>> albert_xxlarge_configuration = AlbertConfig()

    >>> # Initializing an ALBERT-base style configuration
    >>> albert_base_configuration = AlbertConfig(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     intermediate_size=3072,
    ... )

    >>> # Initializing a model (with random weights) from the ALBERT-base style configuration
    >>> model = AlbertModel(albert_xxlarge_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "albert"

    vocab_size: int = 30000
    embedding_size: int = 128
    hidden_size: int = 4096
    num_hidden_layers: int = 12
    num_hidden_groups: int = 1
    num_attention_heads: int = 64
    intermediate_size: int = 16384
    inner_group_num: int = 1
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: int | float = 0.0
    attention_probs_dropout_prob: int | float = 0.0
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    classifier_dropout_prob: int | float = 0.1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 3
    tie_word_embeddings: bool = True


__all__ = ["AlbertConfig"]
