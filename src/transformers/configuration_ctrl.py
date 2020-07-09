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


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP = {"ctrl": "https://s3.amazonaws.com/models.huggingface.co/bert/ctrl-config.json"}


class CTRLConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.CTRLModel`.
        It is used to instantiate an CTRL model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the `ctrl <https://huggingface.co/ctrl>`__ architecture from SalesForce.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        Args:
            vocab_size (:obj:`int`, optional, defaults to 246534):
                Vocabulary size of the CTRL model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.CTRLModel`.
            n_positions (:obj:`int`, optional, defaults to 256):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            n_ctx (:obj:`int`, optional, defaults to 256):
                Dimensionality of the causal mask (usually same as n_positions).
            n_embd (:obj:`int`, optional, defaults to 1280):
                Dimensionality of the embeddings and hidden states.
            dff (:obj:`int`, optional, defaults to 8192):
                Dimensionality of the inner dimension of the FFN.
            n_layer (:obj:`int`, optional, defaults to 48):
                Number of hidden layers in the Transformer encoder.
            n_head (:obj:`int`, optional, defaults to 16):
                Number of attention heads for each attention layer in the Transformer encoder.
            resid_pdrop (:obj:`float`, optional, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            embd_pdrop (:obj:`int`, optional, defaults to 0.1):
                The dropout ratio for the embeddings.
            attn_pdrop (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention.
            layer_norm_epsilon (:obj:`float`, optional, defaults to 1e-6):
                The epsilon to use in the layer normalization layers
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

        Example::

            >>> from transformers import CTRLModel, CTRLConfig

            >>> # Initializing a CTRL configuration
            >>> configuration = CTRLConfig()

            >>> # Initializing a model from the configuration
            >>> model = CTRLModel(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
    """

    model_type = "ctrl"

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
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
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
