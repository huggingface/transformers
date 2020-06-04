# coding=utf-8
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
""" ELECTRA model configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/config.json",
    "google/electra-base-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-generator/config.json",
    "google/electra-large-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-generator/config.json",
    "google/electra-small-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-discriminator/config.json",
    "google/electra-base-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/config.json",
    "google/electra-large-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-discriminator/config.json",
}


class ElectraConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a :class:`~transformers.ElectraModel`.
        It is used to instantiate an ELECTRA model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the ELECTRA `google/electra-small-discriminator <https://huggingface.co/google/electra-small-discriminator>`__
        architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 30522):
                Vocabulary size of the ELECTRA model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.ElectraModel`.
            embedding_size (:obj:`int`, optional, defaults to 128):
                Dimensionality of the encoder layers and the pooler layer.
            hidden_size (:obj:`int`, optional, defaults to 256):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 4):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 1024):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.ElectraModel`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.

        Example::

            from transformers import ElectraModel, ElectraConfig

            # Initializing a ELECTRA electra-base-uncased style configuration
            configuration = ElectraConfig()

            # Initializing a model from the electra-base-uncased style configuration
            model = ElectraModel(configuration)

            # Accessing the model configuration
            configuration = model.config
    """
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
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

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
