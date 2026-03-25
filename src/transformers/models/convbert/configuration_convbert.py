# Copyright The HuggingFace team. All rights reserved.
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
"""ConvBERT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="YituTech/conv-bert-base")
@strict
class ConvBertConfig(PreTrainedConfig):
    r"""
    head_ratio (`int`, *optional*, defaults to 2):
        Ratio gamma to reduce the number of attention heads.
    num_groups (`int`, *optional*, defaults to 1):
        The number of groups for grouped linear layers for ConvBert model

    Example:

    ```python
    >>> from transformers import ConvBertConfig, ConvBertModel

    >>> # Initializing a ConvBERT convbert-base-uncased style configuration
    >>> configuration = ConvBertConfig()

    >>> # Initializing a model (with random weights) from the convbert-base-uncased style configuration
    >>> model = ConvBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "convbert"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    embedding_size: int = 768
    head_ratio: int = 2
    conv_kernel_size: int = 9
    num_groups: int = 1
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["ConvBertConfig"]
