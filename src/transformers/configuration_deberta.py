# coding=utf-8
# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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
""" DeBERTa model configuration """

from .configuration_utils import PretrainedConfig
from .utils import logging


logger = logging.get_logger(__name__)

DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-base/config.json",
    "microsoft/deberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-large/config.json",
}


class DebertaConfig(PretrainedConfig):
    r"""
    :class:`~transformers.DebertaConfig` is the configuration class to store the configuration of a
    :class:`~transformers.DebertaModel`.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"swish"`, :obj:`"gelu"`, :obj:`"tanh"`, :obj:`"gelu_fast"`,
            :obj:`"mish"`, :obj:`"linear"`, :obj:`"sigmoid"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        relative_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether use relative position encoding.
        max_relative_positions (:obj:`int`, `optional`, defaults to 1):
            The range of relative positions :obj:`[-max_position_embeddings, max_position_embeddings]`.
            Use the same value as :obj:`max_position_embeddings`.
        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (:obj:`List[str]`, `optional`):
            The type of relative position attention, it can be a combination of :obj:`["p2c", "c2p", "p2p"]`,
            e.g. :obj:`["p2c"]`, :obj:`["p2c", "c2p"]`, :obj:`["p2c", "c2p", 'p2p"]`.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """
    model_type = "deberta"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)

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
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
