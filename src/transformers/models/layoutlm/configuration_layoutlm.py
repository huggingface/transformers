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

from ... import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/layoutlm-base-uncased")
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

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        eos_token_id=None,
        bos_token_id=None,
        use_cache=True,
        max_2d_position_embeddings=1024,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["LayoutLMConfig"]
