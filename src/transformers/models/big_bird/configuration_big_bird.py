# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
"""BigBird model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/bigbird-roberta-base")
@strict
class BigBirdConfig(PreTrainedConfig):
    r"""
    attention_type (`str`, *optional*, defaults to `"block_sparse"`):
        Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
        layer (with n^2 complexity). Possible values are `"original_full"` and `"block_sparse"`.
    use_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in query, key, value.
    rescale_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to rescale embeddings with (hidden_size ** 0.5).
    block_size (`int`, *optional*, defaults to 64):
        Size of each block. Useful only when `attention_type == "block_sparse"`.
    num_random_blocks (`int`, *optional*, defaults to 3):
        Each query is going to attend these many number of random blocks. Useful only when `attention_type ==
        "block_sparse"`.

    Example:

    ```python
    >>> from transformers import BigBirdConfig, BigBirdModel

    >>> # Initializing a BigBird google/bigbird-roberta-base style configuration
    >>> configuration = BigBirdConfig()

    >>> # Initializing a model (with random weights) from the google/bigbird-roberta-base style configuration
    >>> model = BigBirdModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "big_bird"

    vocab_size: int = 50358
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 4096
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    sep_token_id: int | None = 66
    attention_type: str = "block_sparse"
    use_bias: bool = True
    rescale_embeddings: bool = False
    block_size: int = 64
    num_random_blocks: int = 3
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["BigBirdConfig"]
