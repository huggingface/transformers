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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="uw-madison/mra-base-512-4")
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

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        block_per_row=4,
        approx_mode="full",
        initial_prior_first_n_blocks=0,
        initial_prior_diagonal_n_blocks=0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        add_cross_attention=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.block_per_row = block_per_row
        self.approx_mode = approx_mode
        self.initial_prior_first_n_blocks = initial_prior_first_n_blocks
        self.initial_prior_diagonal_n_blocks = initial_prior_diagonal_n_blocks


__all__ = ["MraConfig"]
