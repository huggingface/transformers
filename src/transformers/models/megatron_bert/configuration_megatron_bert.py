# Copyright 2021- NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""MEGATRON_BERT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/megatron-bert-uncased-345m")
@strict
class MegatronBertConfig(PreTrainedConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import MegatronBertConfig, MegatronBertModel

    >>> # Initializing a MEGATRON_BERT google-bert/bert-base-uncased style configuration
    >>> configuration = MegatronBertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = MegatronBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "megatron-bert"

    vocab_size: int = 29056
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    use_cache: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["MegatronBertConfig"]
