# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""ELECTRA model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/electra-small-discriminator")
class ElectraConfig(PreTrainedConfig):
    r"""
    summary_type (`str`, *optional*, defaults to `"first"`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Has to be one of the following options:
            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Pass `"gelu"` for a gelu activation to the output, any other value will result in no activation.
    summary_last_dropout (`float`, *optional*, defaults to 0.0):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        The dropout ratio to be used after the projection and activation.

    Examples:

    ```python
    >>> from transformers import ElectraConfig, ElectraModel

    >>> # Initializing a ELECTRA electra-base-uncased style configuration
    >>> configuration = ElectraConfig()

    >>> # Initializing a model (with random weights) from the electra-base-uncased style configuration
    >>> model = ElectraModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "electra"

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        summary_type="first",
        summary_use_proj=True,
        summary_activation="gelu",
        summary_last_dropout=0.1,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
        is_decoder=False,
        add_cross_attention=False,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


__all__ = ["ElectraConfig"]
