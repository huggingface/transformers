# Copyright 2021, The Microsoft Research Asia MarkupLM Team authors
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
"""MarkupLM model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/markuplm-base")
class MarkupLMConfig(PreTrainedConfig):
    r"""
    max_tree_id_unit_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the tree id unit embedding might ever use. Typically set this to something large
        just in case (e.g., 1024).
    max_xpath_tag_unit_embeddings (`int`, *optional*, defaults to 256):
        The maximum value that the xpath tag unit embedding might ever use. Typically set this to something large
        just in case (e.g., 256).
    max_xpath_subs_unit_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
        large just in case (e.g., 1024).
    tag_pad_id (`int`, *optional*, defaults to 216):
        The id of the padding token in the xpath tags.
    subs_pad_id (`int`, *optional*, defaults to 1001):
        The id of the padding token in the xpath subscripts.
    xpath_tag_unit_hidden_size (`int`, *optional*, defaults to 32):
        The hidden size of each tree id unit. One complete tree index will have
        (50*xpath_tag_unit_hidden_size)-dim.
    max_depth (`int`, *optional*, defaults to 50):
        The maximum depth in xpath.
    xpath_unit_hidden_size (`int`, *optional*, defaults to 32):
        The hidden size of each unit in xpath.

    Examples:

    ```python
    >>> from transformers import MarkupLMModel, MarkupLMConfig

    >>> # Initializing a MarkupLM microsoft/markuplm-base style configuration
    >>> configuration = MarkupLMConfig()

    >>> # Initializing a model from the microsoft/markuplm-base style configuration
    >>> model = MarkupLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "markuplm"

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
        bos_token_id=0,
        eos_token_id=2,
        max_xpath_tag_unit_embeddings=256,
        max_xpath_subs_unit_embeddings=1024,
        tag_pad_id=216,
        subs_pad_id=1001,
        xpath_unit_hidden_size=32,
        max_depth=50,
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        # additional properties
        self.max_depth = max_depth
        self.max_xpath_tag_unit_embeddings = max_xpath_tag_unit_embeddings
        self.max_xpath_subs_unit_embeddings = max_xpath_subs_unit_embeddings
        self.tag_pad_id = tag_pad_id
        self.subs_pad_id = subs_pad_id
        self.xpath_unit_hidden_size = xpath_unit_hidden_size


__all__ = ["MarkupLMConfig"]
