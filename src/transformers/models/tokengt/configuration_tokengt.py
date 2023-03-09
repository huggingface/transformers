# coding=utf_8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE_2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Nystromformer model configuration"""

from typing import Callable

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

TOKENGT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # pcqm4mv1 now deprecated
    "tokengt-lap16": "https://huggingface.co/raman-ai/tokengt-base-lap-pcqm4mv2/blob/main/config.json",
}


class TokenGTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~TokenGTModel`]. It is used to instantiate an
    Graphormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Graphormer
    [tokengt-base-lap-pcqm4mv2](https://huggingface.co/tokengt-base-lap-pcqm4mv2) architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        num_classes (`int`, *optional*, defaults to 2):
            Number of target classes or labels, set to 1 if the task is a regression task.
        num_atoms (`int`, *optional*, defaults to 512*9):
            Number of node types in the graphs.
        num_edges (`int`, *optional*, defaults to 512*3):
            Number of edges types in the graph.
        num_in_degree (`int`, *optional*, defaults to 512):
            Number of in degrees types in the input graphs.
        num_out_degree (`int`, *optional*, defaults to 512):
            Number of out degrees types in the input graphs.
        num_edge_dis (`int`, *optional*, defaults to 128):
            Number of edge dis in the input graphs.
        multi_hop_max_dist (`int`, *optional*, defaults to 20):
            Maximum distance of multi hop edges between two nodes.
        spatial_pos_max (`int`, *optional*, defaults to 1024):
            Maximum distance between nodes in the graph attention bias matrices, used during preprocessing and
            collation.
        edge_type (`str`, *optional*, defaults to multihop):
            Type of edge relation chosen.
        max_nodes (`int`, *optional*, defaults to 512):
            Maximum number of nodes which can be parsed for the input graphs.
        share_input_output_embed (`bool`, *optional*, defaults to `False`):
            Shares the embedding layer between encoder and decoder - careful, True is not implemented.
        num_layers (`int`, *optional*, defaults to 12):
            Number of layers.
        embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the embedding layer in encoder.
        ffn_embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the encoder.
        self_attention (`bool`, *optional*, defaults to `True`):
            Model is self attentive (False not implemented).
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention weights.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability after activation in the FFN.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        bias (`bool`, *optional*, defaults to `True`):
            Uses bias in the attention module - unsupported at the moment.
        embed_scale(`float`, *optional*, defaults to None):
            Scaling factor for the node embeddings.
        num_trans_layers_to_freeze (`int`, *optional*, defaults to 0):
            Number of transformer layers to freeze.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Normalize features before encoding the graph.
        pre_layernorm (`bool`, *optional*, defaults to `False`):
            Apply layernorm before self attention and the feed forward network. Without this, post layernorm will be
            used.
        apply_graphormer_init (`bool`, *optional*, defaults to `False`):
            Apply a custom graphormer initialisation to the model before training.
        freeze_embeddings (`bool`, *optional*, defaults to `False`):
            Freeze the embedding layer, or train it along the model.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Apply the layer norm before each encoder block.
        q_noise (`float`, *optional*, defaults to 0.0):
            Amount of quantization noise (see "Training with Quantization Noise for Extreme Model Compression"). (For
            more detail, see fairseq's documentation on quant_noise).
        qn_block_size (`int`, *optional*, defaults to 8):
            Size of the blocks for subsequent quantization with iPQ (see q_noise).
        kdim (`int`, *optional*, defaults to None):
            Dimension of the key in the attention, if different from the other values.
        vdim (`int`, *optional*, defaults to None):
            Dimension of the value in the attention, if different from the other values.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        traceable (`bool`, *optional*, defaults to `False`):
            Changes return value of the encoder's inner_state to stacked tensors.
        Example:
            ```python
            >>> from transformers import GraphormerForGraphClassification, GraphormerConfig

            >>> # Initializing a Graphormer graphormer-base-pcqm4mv2 style configuration
            >>> configuration = GraphormerConfig()
            >>> # Initializing a model from the graphormer-base-pcqm4mv1 style configuration
            >>> model = GraphormerForGraphClassification(configuration)
            >>> # Accessing the model configuration
            >>> configuration = model.config
            ```
    """
    model_type = "tokengt"

    def __init__(
        self,
        num_classes: int = 1,
        num_atoms: int = 512 * 9,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_edges: int = 512 * 4,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        edge_type: str = "multi_hop",
        multi_hop_max_dist: int = 5,
        max_nodes: int = 128,
        spatial_pos_max: int = 1024,
        # for tokenization
        rand_node_id: bool = False,
        rand_node_id_dim: int = 64,
        orf_node_id: bool = False,
        orf_node_id_dim: int = 64,
        lap_node_id: bool = False,
        lap_node_id_k: int = 8,
        lap_node_id_sign_flip: bool = False,
        lap_node_id_eig_dropout: float = 0.0,
        type_id: bool = False,
        share_encoder_input_output_embed: bool = False,
        prenorm: bool = False,
        postnorm: bool = False,
        stochastic_depth: bool = False,
        performer: bool = False,
        performer_finetune: bool = False,
        performer_nb_features: int = None,
        performer_feature_redraw_interval: int = 1000,
        performer_generalized_attention: bool = False,
        performer_auto_check_redraw: bool = True,
        num_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,  # 3072 in TokenGTGraphEncoderLayer
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        layernorm_style: str = "postnorm",
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        self_attention: bool = True,
        uses_fixed_gaussian_features: bool = False,
        return_attention: bool = False,
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
        self.rand_node_id = rand_node_id
        self.rand_node_id_dim = rand_node_id_dim
        self.orf_node_id = orf_node_id
        self.orf_node_id_dim = orf_node_id_dim
        self.lap_node_id = lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.lap_node_id_sign_flip = lap_node_id_sign_flip
        self.lap_node_id_eig_dropout = lap_node_id_eig_dropout
        self.type_id = type_id
        self.share_encoder_input_output_embed = share_encoder_input_output_embed
        self.stochastic_depth = stochastic_depth
        self.performer = performer
        self.performer_finetune = performer_finetune
        self.performer_nb_features = performer_nb_features
        self.performer_feature_redraw_interval = performer_feature_redraw_interval
        self.performer_generalized_attention = performer_generalized_attention
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.max_nodes = max_nodes
        self.spatial_pos_max = spatial_pos_max
        self.num_classes = num_classes

        self.encoder_normalize_before = encoder_normalize_before
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.n_trans_layers_to_freeze = n_trans_layers_to_freeze
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.return_attention = return_attention
        self.init_fn = init_fn

        # These parameters are here for future extensions
        # atm, the model only supports self attention
        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias

        # For pretraining, removes node/edge feature embedding layers
        self.uses_fixed_gaussian_features = uses_fixed_gaussian_features

        assert not (prenorm and postnorm)
        if prenorm:
            self.layernorm_style = "prenorm"
        else:  # Default choice is postnorm
            self.layernorm_style = "postnorm"

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
