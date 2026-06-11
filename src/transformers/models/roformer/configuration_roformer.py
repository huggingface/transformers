# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""RoFormer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="junnyu/roformer_chinese_base")
@strict
class RoFormerConfig(PreTrainedConfig):
    r"""
    rotary_value (`bool`, *optional*, defaults to `False`):
        Whether or not apply rotary position embeddings on value layer.

    Example:

    ```python
    >>> from transformers import RoFormerModel, RoFormerConfig

    >>> # Initializing a RoFormer junnyu/roformer_chinese_base style configuration
    >>> configuration = RoFormerConfig()

    >>> # Initializing a model (with random weights) from the junnyu/roformer_chinese_base style configuration
    >>> model = RoFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "roformer"

    vocab_size: int = 50000
    embedding_size: int | None = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 1536
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    rotary_value: bool = False
    use_cache: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.embedding_size = self.hidden_size if self.embedding_size is None else self.embedding_size
        super().__post_init__(**kwargs)


__all__ = ["RoFormerConfig"]
