# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Salesforce CTRL configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP = {"ctrl": "https://storage.googleapis.com/sf-ctrl/pytorch/ctrl-config.json"}

class CTRLConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CTRLModel`.

    Args:
        vocab_size: Vocabulary size of `inputs_ids` in `CTRLModel` or a configuration json file.
        n_positions: Number of positional embeddings.
        n_ctx: Size of the causal mask (usually same as n_positions).
        dff: Size of the inner dimension of the FFN.
        n_embd: Dimensionality of the embeddings and hidden states.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        layer_norm_epsilon: epsilon to use in the layer norm layers
        resid_pdrop: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attn_pdrop: The dropout ratio for the attention
            probabilities.
        embd_pdrop: The dropout ratio for the embeddings.
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
    """
    pretrained_config_archive_map = CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        vocab_size=246534,
        n_positions=256,
        n_ctx=256,
        n_embd=1280,
        dff=8192,
        n_layer=48,
        n_head=16,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        summary_type='cls_index',
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        **kwargs
    ):
        """Constructs CTRLConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `CTRLModel` or a configuration json file.
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            dff: Size of the inner dimension of the FFN.
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        super(CTRLConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dff = dff
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
