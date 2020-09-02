# coding=utf-8
# Copyright 2020, Microsoft
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

import logging

from .configuration_utils import PretrainedConfig


__all__ = ["DeBERTaConfig", "DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"]

logger = logging.getLogger(__name__)

DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-base/config.json",
    "microsoft/deberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-large/config.json",
}


class DeBERTaConfig(PretrainedConfig):
    r"""
    :class:`~transformers.DeBERTaConfig` is the configuration class to store the configuration of a
    `DeBERTaModel`.

    Arguments:
        hidden_size (int): Size of the encoder layers and the pooler layer, default: `768`.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder, default: `12`.
        num_attention_heads (int): Number of attention heads for each attention layer in
            the Transformer encoder, default: `12`.
        intermediate_size (int): The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder, default: `3072`.
        hidden_act (str): The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported, default: `gelu`.
        hidden_dropout_prob (float): The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler, default: `0.1`.
        attention_probs_dropout_prob (float): The dropout ratio for the attention
            probabilities, default: `0.1`.
        max_position_embeddings (int): The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048), default: `512`.
        type_vocab_size (int): The vocabulary size of the `token_type_ids` passed into
            `DeBERTa` model, default: `0`.
        initializer_range (int): The sttdev of the _normal_initializer for
            initializing all weight matrices, default: `0.02`.
        relative_attention (:obj:`bool`): Whether use relative position encoding, default: `False`.
        max_relative_positions (int): The range of relative positions [`-max_position_embeddings`, `max_position_embeddings`], default: -1, use the same value as `max_position_embeddings`.
        padding_idx (int): The value used to pad input_ids, default: `0`.
        position_biased_input (:obj:`bool`): Whether add absolute position embedding to content embedding, default: `True`.
        pos_att_type (:obj:`str`): The type of relative position attention, it can be a combination of [`p2c`, `c2p`, `p2p`], e.g. "p2c", "p2c|c2p", "p2c|c2p|p2p", default: "None".
        vocab_size (int): The size of the vocabulary, default: `-1`.
    """
    model_type = "deberta"

    def __init__(
        self,
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
        relative_attention=False,
        max_relative_positions=-1,
        padding_idx=0,
        position_biased_input=True,
        pos_att_type="None",
        vocab_size=-1,
        layer_norm_eps=1e-7,
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
        self.padding_idx = padding_idx
        self.position_biased_input = position_biased_input
        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
