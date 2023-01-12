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

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Callable


logger = logging.get_logger(__name__)

class TokenGTConfig(PretrainedConfig):
    model_type = "tokengt"

    def __init__(
        self,
        num_labels: float = None, # Either num of classes for multiclass, or number of labels in the class in single class. One for regression
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
        ffn_embedding_dim: int = 768, # 3072 in TokenGTGraphEncoderLayer
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
        self.num_labels = num_labels if num_labels is not None else 2
 
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
        elif postnorm:
            self.layernorm_style = "postnorm"
        else:
            self.layernorm_style = "postnorm" # Default choice is postnorm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
