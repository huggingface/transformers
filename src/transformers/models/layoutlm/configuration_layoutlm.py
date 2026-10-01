# Copyright 2010, The Microsoft Research Asia LayoutLM Team authors
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
"""LayoutLM model configuration"""

from huggingface_hub.dataclasses import strict

from ... import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/layoutlm-base-uncased")
@strict
class LayoutLMConfig(PreTrainedConfig):
    r"""
    max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the 2D position embedding might ever used. Typically set this to something large
        just in case (e.g., 1024).

    Examples:

    ```python
    >>> from transformers import LayoutLMConfig, LayoutLMModel

    >>> # Initializing a LayoutLM configuration
    >>> configuration = LayoutLMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = LayoutLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "layoutlm"

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
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    use_cache: bool = True
    max_2d_position_embeddings: int = 1024
    tie_word_embeddings: bool = True


__all__ = ["LayoutLMConfig"]
