# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""MobileBERT model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/mobilebert-uncased")
class MobileBertConfig(PreTrainedConfig):
    r"""
    embedding_size (`int`, *optional*, defaults to 128):
        The dimension of the word embedding vectors.
    trigram_input (`bool`, *optional*, defaults to `True`):
        Use a convolution of trigram as input.
    use_bottleneck (`bool`, *optional*, defaults to `True`):
        Whether to use bottleneck in BERT.
    intra_bottleneck_size (`int`, *optional*, defaults to 128):
        Size of bottleneck layer output.
    use_bottleneck_attention (`bool`, *optional*, defaults to `False`):
        Whether to use attention inputs from the bottleneck transformation.
    key_query_shared_bottleneck (`bool`, *optional*, defaults to `True`):
        Whether to use the same linear transformation for query&key in the bottleneck.
    num_feedforward_networks (`int`, *optional*, defaults to 4):
        Number of FFNs in a block.
    normalization_type (`str`, *optional*, defaults to `"no_norm"`):
        The normalization type in MobileBERT.

    Examples:

    ```python
    >>> from transformers import MobileBertConfig, MobileBertModel

    >>> # Initializing a MobileBERT configuration
    >>> configuration = MobileBertConfig()

    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = MobileBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mobilebert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=24,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        embedding_size=128,
        trigram_input=True,
        use_bottleneck=True,
        intra_bottleneck_size=128,
        use_bottleneck_attention=False,
        key_query_shared_bottleneck=True,
        num_feedforward_networks=4,
        normalization_type="no_norm",
        classifier_activation=True,
        classifier_dropout=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
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
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.trigram_input = trigram_input
        self.use_bottleneck = use_bottleneck
        self.intra_bottleneck_size = intra_bottleneck_size
        self.use_bottleneck_attention = use_bottleneck_attention
        self.key_query_shared_bottleneck = key_query_shared_bottleneck
        self.num_feedforward_networks = num_feedforward_networks
        self.normalization_type = normalization_type
        self.classifier_activation = classifier_activation

        if self.use_bottleneck:
            self.true_hidden_size = intra_bottleneck_size
        else:
            self.true_hidden_size = hidden_size

        self.classifier_dropout = classifier_dropout


__all__ = ["MobileBertConfig"]
