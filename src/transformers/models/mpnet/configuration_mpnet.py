# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
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
"""MPNet model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/mpnet-base")
class MPNetConfig(PreTrainedConfig):
    r"""
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.

    Examples:

    ```python
    >>> from transformers import MPNetModel, MPNetConfig

    >>> # Initializing a MPNet mpnet-base style configuration
    >>> configuration = MPNetConfig()

    >>> # Initializing a model from the mpnet-base style configuration
    >>> model = MPNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mpnet"

    def __init__(
        self,
        vocab_size=30527,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        relative_attention_num_buckets=32,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.relative_attention_num_buckets = relative_attention_num_buckets


__all__ = ["MPNetConfig"]
