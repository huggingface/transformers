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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/rembert")
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

    def __init__(
        self,
        vocab_size=250300,
        hidden_size=1152,
        num_hidden_layers=32,
        num_attention_heads=18,
        input_embedding_size=256,
        output_embedding_size=1664,
        intermediate_size=4608,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=312,
        eos_token_id=313,
        is_decoder=False,
        add_cross_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.output_embedding_size = output_embedding_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = False


__all__ = ["RemBertConfig"]
