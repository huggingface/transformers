# Copyright 2022 The HuggingFace Inc. team.
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
"""XLM_ROBERTa_XL configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="FacebookAI/xlm-roberta-xl")
class XLMRobertaXLConfig(PreTrainedConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import XLMRobertaXLConfig, XLMRobertaXLModel

    >>> # Initializing a XLM_ROBERTA_XL google-bert/bert-base-uncased style configuration
    >>> configuration = XLMRobertaXLConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = XLMRobertaXLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlm-roberta-xl"

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        intermediate_size=10240,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        use_cache=True,
        classifier_dropout=None,
        is_decoder=False,
        add_cross_attention=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        self.classifier_dropout = classifier_dropout


__all__ = ["XLMRobertaXLConfig"]
