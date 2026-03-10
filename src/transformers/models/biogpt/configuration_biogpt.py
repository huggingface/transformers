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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/biogpt")
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

    def __init__(
        self,
        vocab_size=42384,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        scale_embedding=True,
        use_cache=True,
        layerdrop=0.0,
        activation_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
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
        self.layer_norm_eps = layer_norm_eps
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache
        self.layerdrop = layerdrop
        self.activation_dropout = activation_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["BioGptConfig"]
