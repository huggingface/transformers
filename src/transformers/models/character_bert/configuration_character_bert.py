# coding=utf-8
# Copyright 2023 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" CHARACTER_BERT model configuration"""
from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CHARACTER_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "helboukkouri/character-bert-base-uncased": "https://huggingface.co/helboukkouri/character-bert-base-uncased/resolve/main/config.json",
}


class CharacterBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CharacterBertModel`] or a [`TFCharacterBertModel`]. It is used to
    instantiate a CHARACTER_BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CHARACTER_BERT
    [helboukkouri/character-bert-base-uncased](https://huggingface.co/helboukkouri/character-bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        character_embeddings_dim (`int`, *optional*, defaults to 16):
            The size of the character embeddings.
        cnn_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function to apply to the cnn representations.
        cnn_filters (`list(list(int))`, *optional*):
            The list of CNN filters to use in the CharacterCNN module. Defaults to `[[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]`):
        num_highway_layers (`int`, *optional*, defaults to 2):
            The number of Highway layers to apply to the CNNs output.
        max_word_length (`int`, *optional*, defaults to 50):
            The maximum token length in characters (actually, in bytes as any non-ascii characters will be converted to
            a sequence of utf-8 bytes).
        mlm_vocab_size (`int`, *optional*, defaults to 100000):
            Size of the output vocabulary for MLM.
        vocab_size (`int`, *optional*):
            Vocabulary size of the CHARACTER_BERT model. This is not relevant here since CharacterBERT does not
            rely on WordPieces and can support any input token.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`CharacterBertModel`] or [`TFCharacterBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The padding token had a character_id vector filled with this value. This is not to be confused with
            the id use to pad character sequences to the same token length (PAD_CHARACTER_ID + 1).
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer. Has to always be False for CharacterBERT.

    Examples:

    ```python
    >>> from transformers import CharacterBertConfig, CharacterBertModel

    >>> # Initializing a CHARACTER_BERT helboukkouri/character-bert-base-uncased style configuration
    >>> configuration = CharacterBertConfig()

    >>> # Initializing a model (with random weights) from the helboukkouri/character-bert-base-uncased style configuration
    >>> model = CharacterBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "character_bert"

    def __init__(
        self,
        character_embeddings_dim=16,
        cnn_activation="relu",
        cnn_filters=None,
        num_highway_layers=2,
        max_word_length=50,
        mlm_vocab_size=100000,
        vocab_size=None,
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
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        if tie_word_embeddings:
            raise ValueError(
                "Cannot tie word embeddings in CharacterBERT. " "Please set `config.tie_word_embeddings=False`."
            )
        self.tie_word_embeddings = tie_word_embeddings

        self.character_embeddings_dim = character_embeddings_dim
        self.cnn_activation = cnn_activation
        self.cnn_filters = cnn_filters or [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
        self.num_highway_layers = num_highway_layers
        self.max_word_length = max_word_length
        self.mlm_vocab_size = mlm_vocab_size
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
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


