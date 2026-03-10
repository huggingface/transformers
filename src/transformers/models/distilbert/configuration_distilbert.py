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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/distilbert-base-uncased")
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

    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=4 * 768,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        initializer_range=0.02,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        eos_token_id=None,
        bos_token_id=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["DistilBertConfig"]
