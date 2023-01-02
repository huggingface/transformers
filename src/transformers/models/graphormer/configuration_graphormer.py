# coding=utf-8
# Copyright 2022 Microsoft, clefourrier and The HuggingFace Inc. team. All rights reserved.
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
""" Graphormer model configuration """

from typing import Callable

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "graphormer-base-pcqm4mv1": "https://huggingface.co/graphormer-base-pcqm4mv1/resolve/main/config.json",
    # See all Graphormer models at https://huggingface.co/models?filter=graphormer
}


class GraphormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GraphormerModel`].
    It is used to instantiate an Graphormer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the Graphormer [graphormer-base-pcqm4mv1](https://huggingface.co/graphormer-base-pcqm4mv1) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        num_classes (`int`, *optional*, defaults to 2):
            Number of target classes or labels.
        num_atoms (`int`, *optional*, defaults to 512*9):
            Number of nodes.
        num_classes (`int`, *optional*, defaults to 512*3):
            Number of edges.
        num_in_degree (`int`, *optional*, defaults to 512):
            Number of in degrees.
        num_out_degree (`int`, *optional*, defaults to 512):
            Number of out degrees.
        num_edge_dis (`int`, *optional*, defaults to 128):
            Edge distance between nodes.
        multi_hop_max_dist (`int`, *optional*, defaults to 20):
            Maximum distance between two nodes (in hops).
        spatial_pos_max (`int`, *optional*, defaults to 1024):
            TODO!!!
        edge_type (`str`, *optional*, defaults to multihop):
            Type of edge relation.
        init_fn (`Callable`, *optional*, defaults to None):
            Function to use for initialisation of layer TODO.
        max_nodes (`int`, *optional*, defaults to 512):
            Maximum number of nodes which can be parsed for the input graphs.
        share_input_output_embed (`bool`, *optional*, defaults to False):
            Shares the embedding layer between encoder and decoder - careful, True is not implemented.
        num_layers (`int`, *optional*, defaults to 12):
            Number of layers.
        embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the embedding layer in encoder.
        ffn_embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the encoder.
        self_attention (`bool`, *optional*, defaults to True):
            Model is self attentive (False not implemented).
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see
            https://arxiv.org/abs/1909.11556) for more details.
        bias (`bool`, *optional*, defaults to True):
            Uses bias in the attention module - unsupported at the moment
        embed_scale(`float`, *optional*, defaults to None):
            TODO
        num_trans_layers_to_freeze (`int`, *optional*, defaults to 0):
            Number of transformer layers to freeze
        encoder_normalize_before (`bool`, *optional*, defaults to False):
            Normalize features before encoding the graph.
        pre_layernorm (`bool`, *optional*, defaults to False):
            Normalize layer before. # todo
        apply_graphormer_init (`bool`, *optional*, defaults to False):
            Apply a graphormer initialisation to the model before training.
        freeze_embeddings (`bool`, *optional*, defaults to False):
            Freeze the embedding layer, or train it along the model.
        encoder_normalize_before (`bool`, *optional*, defaults to False):
            Normalize features before encoding the graph.
        q_noise (`float`, *optional*, defaults to 0.0):
            Noise.
        qn_block_size (`int`, *optional*, defaults to 8):
            Block size.
        kdim (`int`, *optional*, defaults to None):
            Dimension of the key in the attention, if different from the other values
        vdim (`int`, *optional*, defaults to None):
            Dimension of the value in the attention, if different from the other values
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        export (`bool`, *optional*, defaults to False):
            TODO
        traceable (`bool`, *optional*, defaults to False):
            TODO
        Example:

    ```python
    >>> from transformers import GraphormerModel, GraphormerConfig

    >>> # Initializing a Graphormer graphormer-base-pcqm4mv1 style configuration
    >>> configuration = GraphormerConfig()

    >>> # Initializing a model from the graphormer-base-pcqm4mv1 style configuration
    >>> model = GraphormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "graphormer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_classes: int = 2,
        num_atoms: int = 512 * 9,
        num_edges: int = 512 * 3,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        multi_hop_max_dist: int = 20,
        spatial_pos_max: int = 1024,
        edge_type: str = "multi_hop",
        init_fn: Callable = None,
        max_nodes: int = 512,
        share_input_output_embed: bool = False,
        num_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        num_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        self_attention: bool = True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.max_nodes = max_nodes
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.num_trans_layers_to_freeze = num_trans_layers_to_freeze
        self.export = export
        self.share_input_output_embed = share_input_output_embed
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.init_fn = init_fn

        # These parameters are here for future extensions
        # atm, the model only supports self attention
        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class GraphormerConfigSmall(GraphormerConfig):
    def __init__(self, **kwargs):
        super().__init__(
            share_encoder_input_output_embed=False,
            num_layers=2,
            embedding_dim=120,
            ffn_embedding_dim=120,
            num_attention_heads=3,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            encoder_normalize_before=True,
            pre_layernorm=False,
            apply_graphormer_init=True,
            activation_fn="gelu",
            no_token_positional_embeddings=False,
            **kwargs,
        )


class GraphormerConfigBase(GraphormerConfig):
    def __init__(self, **kwargs):
        super().__init__(
            share_input_output_embed=False,
            num_layers=12,
            embedding_dim=768,
            ffn_embedding_dim=768,
            num_attention_heads=32,
            dropout=0.0,
            attention_dropout=0.1,
            activation_dropout=0.1,
            encoder_normalize_before=True,
            pre_layernorm=False,
            apply_graphormer_init=True,
            activation_fn="gelu",
            no_token_positional_embeddings=False,
            **kwargs,
        )


class GraphormerConfigSlim(GraphormerConfig):
    def __init__(self, **kwargs):
        super().__init__(
            share_input_output_embed=False,
            num_layers=12,
            embedding_dim=80,
            ffn_embedding_dim=80,
            num_attention_heads=8,
            dropout=0.0,
            attention_dropout=0.1,
            activation_dropout=0.1,
            encoder_normalize_before=True,
            pre_layernorm=False,
            apply_graphormer_init=True,
            activation_fn="gelu",
            no_token_positional_embeddings=False,
            **kwargs,
        )


class GraphormerConfigLarge(GraphormerConfig):
    def __init__(self, **kwargs):
        super().__init__(
            share_input_output_embed=False,
            num_layers=24,
            embedding_dim=1024,
            ffn_embedding_dim=1024,
            num_attention_heads=32,
            dropout=0.0,
            attention_dropout=0.1,
            activation_dropout=0.1,
            encoder_normalize_before=True,
            pre_layernorm=False,
            apply_graphormer_init=True,
            activation_fn="gelu",
            no_token_positional_embeddings=False,
            **kwargs,
        )
