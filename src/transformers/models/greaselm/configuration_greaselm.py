# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2022 The HuggingFace Inc. team.
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
""" GreaseLM model configuration"""

from ... import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

GREASELM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Xikun/greaselm-csqa": "https://huggingface.co/Xikun/greaselm-csqa/resolve/main/config.json",
    "Xikun/greaselm-obqa": "https://huggingface.co/Xikun/greaselm-obqa/resolve/main/config.json",
}


class GreaseLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GreaseLMModel`]. It is used to instantiate a
    GreaseLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GreaseLM
    [Xikun/greaselm-csqa](https://huggingface.co/Xikun/greaselm-csqa) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The [`GreaseLMConfig`] class is identical to [`BertConfig`] with a few additional attributes.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GreaseLM model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`GreaseLMModel`]
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
            The vocabulary size of the `token_type_ids` passed when calling [`GreaseLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        num_gnn_layers (`int`, *optional*, defaults to 5):
            Number of Graph Neural Network (GNN) layers.
        num_node_types (`int`, *optional*, defaults to 4):
            Number of node types in the graph.
        num_edge_types (`int`, *optional*, defaults to 38):
            Number of edge types in the graph.
        concept_dim (`int`, *optional*, defaults to 200):
            Dimension of the concept embeddings.
        gnn_hidden_size (`int`, *optional*, defaults to 200):
            Hidden size of the Graph Neural Network (GNN).

    Examples:

    ```python
    >>> from transformers import GreaseLMConfig, GreaseLMModel

    >>> # Initializing a greaselm configuration
    >>> configuration = GreaseLMConfig()

    >>> # Initializing a model from the configuration
    >>> model = GreaseLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "greaselm"

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
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        num_gnn_layers=5,
        num_node_types=4,
        num_edge_types=38,
        concept_dim=200,
        gnn_hidden_size=200,
        **kwargs
    ):
        """Constructs GreaseLMConfig."""
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # LM parameters
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

        # GNN parameters
        self.num_gnn_layers = num_gnn_layers
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.concept_dim = concept_dim
        self.gnn_hidden_size = gnn_hidden_size

        # These GNN layer configs rarely change, but they are parameters in
        # the original config file. Keep them here for now.
        default_dropout = 0.2
        self.num_lm_gnn_attention_heads = kwargs.pop("num_lm_gnn_attention_heads", 2)
        self.fc_dim = kwargs.pop("fc_dim", 200)
        self.n_fc_layer = kwargs.pop("n_fc_layer", 0)
        self.p_emb = kwargs.pop("p_emb", default_dropout)
        self.p_gnn = kwargs.pop("p_gnn", default_dropout)
        self.p_fc = kwargs.pop("p_fc", default_dropout)
        self.ie_dim = kwargs.pop("ie_dim", 200)
        self.info_exchange = kwargs.pop("info_exchange", True)
        self.ie_layer_num = kwargs.pop("ie_layer_num", 1)
        self.sep_ie_layers = kwargs.pop("sep_ie_layers", False)
        self.layer_id = kwargs.pop("layer_id", -1)
