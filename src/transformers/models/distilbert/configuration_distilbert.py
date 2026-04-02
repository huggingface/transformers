# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
"""DistilBERT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/distilbert-base-uncased")
@strict
class DistilBertConfig(PreTrainedConfig):
    r"""
    sinusoidal_pos_embds (`boolean`, *optional*, defaults to `False`):
        Whether to use sinusoidal positional embeddings.
    dim (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    qa_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probabilities used in the question answering model [`DistilBertForQuestionAnswering`].
    seq_classif_dropout (`float`, *optional*, defaults to 0.2):
        The dropout probabilities used in the sequence classification and the multiple choice model
        [`DistilBertForSequenceClassification`].

    Examples:

    ```python
    >>> from transformers import DistilBertConfig, DistilBertModel

    >>> # Initializing a DistilBERT configuration
    >>> configuration = DistilBertConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DistilBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "distilbert"
    attribute_map = {
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    vocab_size: int = 30522
    max_position_embeddings: int = 512
    sinusoidal_pos_embds: bool = False
    n_layers: int = 6
    n_heads: int = 12
    dim: int = 768
    hidden_dim: int = 4 * 768
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation: str = "gelu"
    initializer_range: float = 0.02
    qa_dropout: float | int = 0.1
    seq_classif_dropout: float | int = 0.2
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    tie_word_embeddings: bool = True


__all__ = ["DistilBertConfig"]
