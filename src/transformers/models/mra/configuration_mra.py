# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""MRA model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="uw-madison/mra-base-512-4")
@strict
class MraConfig(PreTrainedConfig):
    r"""
    block_per_row (`int`, *optional*, defaults to 4):
        Used to set the budget for the high resolution scale.
    approx_mode (`str`, *optional*, defaults to `"full"`):
        Controls whether both low and high resolution approximations are used. Set to `"full"` for both low and
        high resolution and `"sparse"` for only low resolution.
    initial_prior_first_n_blocks (`int`, *optional*, defaults to 0):
        The initial number of blocks for which high resolution is used.
    initial_prior_diagonal_n_blocks (`int`, *optional*, defaults to 0):
        The number of diagonal blocks for which high resolution is used.

    Example:

    ```python
    >>> from transformers import MraConfig, MraModel

    >>> # Initializing a Mra uw-madison/mra-base-512-4 style configuration
    >>> configuration = MraConfig()

    >>> # Initializing a model (with random weights) from the uw-madison/mra-base-512-4 style configuration
    >>> model = MraModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mra"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    block_per_row: int = 4
    approx_mode: str = "full"
    initial_prior_first_n_blocks: int = 0
    initial_prior_diagonal_n_blocks: int = 0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["MraConfig"]
