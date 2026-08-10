# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""RemBERT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/rembert")
@strict
class RemBertConfig(PreTrainedConfig):
    r"""
    input_embedding_size (`int`, *optional*, defaults to 256):
        Dimensionality of the input embeddings.
    output_embedding_size (`int`, *optional*, defaults to 1664):
        Dimensionality of the output embeddings.

    Example:

    ```python
    >>> from transformers import RemBertModel, RemBertConfig

    >>> # Initializing a RemBERT rembert style configuration
    >>> configuration = RemBertConfig()

    >>> # Initializing a model from the rembert style configuration
    >>> model = RemBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rembert"

    vocab_size: int = 250300
    hidden_size: int = 1152
    num_hidden_layers: int = 32
    num_attention_heads: int = 18
    input_embedding_size: int = 256
    output_embedding_size: int = 1664
    intermediate_size: int = 4608
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    classifier_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 312
    eos_token_id: int | list[int] | None = 313
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = False


__all__ = ["RemBertConfig"]
