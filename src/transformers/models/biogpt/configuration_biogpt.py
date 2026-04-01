# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science All rights reserved.
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
"""BioGPT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/biogpt")
@strict
class BioGptConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import BioGptModel, BioGptConfig

    >>> # Initializing a BioGPT microsoft/biogpt style configuration
    >>> configuration = BioGptConfig()

    >>> # Initializing a model from the microsoft/biogpt style configuration
    >>> model = BioGptModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "biogpt"

    vocab_size: int = 42384
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    scale_embedding: bool = True
    use_cache: bool = True
    layerdrop: float | int = 0.0
    activation_dropout: float | int = 0.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True


__all__ = ["BioGptConfig"]
