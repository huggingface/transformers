# Copyright 2022 WeChatAI and The HuggingFace Inc. team. All rights reserved.
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
"""RoCBert model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="weiweishi/roc-bert-base-zh")
class RoCBertConfig(PreTrainedConfig):
    r"""
    enable_pronunciation (`bool`, *optional*, defaults to `True`):
        Whether or not the model use pronunciation embed when training.
    enable_shape (`bool`, *optional*, defaults to `True`):
        Whether or not the model use shape embed when training.
    pronunciation_embed_dim (`int`, *optional*, defaults to 768):
        Dimension of the pronunciation_embed.
    pronunciation_vocab_size (`int`, *optional*, defaults to 910):
        Pronunciation Vocabulary size of the RoCBert model. Defines the number of different tokens that can be
        represented by the `input_pronunciation_ids` passed when calling [`RoCBertModel`].
    shape_embed_dim (`int`, *optional*, defaults to 512):
        Dimension of the shape_embed.
    shape_vocab_size (`int`, *optional*, defaults to 24858):
        Shape Vocabulary size of the RoCBert model. Defines the number of different tokens that can be represented
        by the `input_shape_ids` passed when calling [`RoCBertModel`].
    concat_input (`bool`, *optional*, defaults to `True`):
        Defines the way of merging the shape_embed, pronunciation_embed and word_embed, if the value is true,
        output_embed = torch.cat((word_embed, shape_embed, pronunciation_embed), -1), else output_embed =
        (word_embed + shape_embed + pronunciation_embed) / 3

    Example:

    ```python
    >>> from transformers import RoCBertModel, RoCBertConfig

    >>> # Initializing a RoCBert weiweishi/roc-bert-base-zh style configuration
    >>> configuration = RoCBertConfig()

    >>> # Initializing a model from the weiweishi/roc-bert-base-zh style configuration
    >>> model = RoCBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "roc_bert"

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
        use_cache=True,
        pad_token_id=0,
        classifier_dropout=None,
        enable_pronunciation=True,
        enable_shape=True,
        pronunciation_embed_dim=768,
        pronunciation_vocab_size=910,
        shape_embed_dim=512,
        shape_vocab_size=24858,
        concat_input=True,
        is_decoder=False,
        add_cross_attention=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.enable_pronunciation = enable_pronunciation
        self.enable_shape = enable_shape
        self.pronunciation_embed_dim = pronunciation_embed_dim
        self.pronunciation_vocab_size = pronunciation_vocab_size
        self.shape_embed_dim = shape_embed_dim
        self.shape_vocab_size = shape_vocab_size
        self.concat_input = concat_input
        self.classifier_dropout = classifier_dropout
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id


__all__ = ["RoCBertConfig"]
