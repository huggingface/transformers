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
"""BERT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google-bert/bert-base-uncased")
@strict
class BertConfig(PreTrainedConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True


__all__ = ["BertConfig"]
