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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/markuplm-base")
@strict
class MarkupLMConfig(PreTrainedConfig):
    r"""
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
    xpath_unit_hidden_size (`int`, *optional*, defaults to 32):
        The hidden size of each unit in xpath.
    max_depth (`int`, *optional*, defaults to 50):
        The maximum depth in xpath.

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

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    max_xpath_tag_unit_embeddings: int = 256
    max_xpath_subs_unit_embeddings: int = 1024
    tag_pad_id: int = 216
    subs_pad_id: int = 1001
    xpath_unit_hidden_size: int = 32
    max_depth: int = 50
    use_cache: bool = True
    classifier_dropout: float | int | None = None


__all__ = ["MarkupLMConfig"]
