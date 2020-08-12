# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Transformer XL configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "transfo-xl-wt103": "https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json",
}


class TransfoXLConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.TransfoXLModel`.
        It is used to instantiate a Transformer XL model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the `Transformer XL <https://huggingface.co/transfo-xl-wt103>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        Args:
            vocab_size (:obj:`int`, optional, defaults to 267735):
                Vocabulary size of the Transformer XL model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.TransfoXLModel`.
            cutoffs (:obj:`List[int]`, optional, defaults to :obj:`[20000, 40000, 200000]`):
                Cutoffs for the adaptive softmax
            d_model (:obj:`int`, optional, defaults to 1024):
                Dimensionality of the model's hidden states.
            d_embed (:obj:`int`, optional, defaults to 1024):
                Dimensionality of the embeddings
            n_head (:obj:`int`, optional, defaults to 16):
                Number of attention heads for each attention layer in the Transformer encoder.
            d_head (:obj:`int`, optional, defaults to 64):
                Dimensionality of the model's heads.
            d_inner (:obj:`int`, optional, defaults to 4096):
                Inner dimension in FF
            div_val (:obj:`int`, optional, defaults to 4):
                Divident value for adapative input and softmax
            pre_lnorm (:obj:`boolean`, optional, defaults to :obj:`False`):
                Apply LayerNorm to the input instead of the output
            n_layer (:obj:`int`, optional, defaults to 18):
                Number of hidden layers in the Transformer encoder.
            tgt_len (:obj:`int`, optional, defaults to 128):
                Number of tokens to predict
            ext_len (:obj:`int`, optional, defaults to 0):
                Length of the extended context
            mem_len (:obj:`int`, optional, defaults to 1600):
                Length of the retained previous heads
            clamp_len (:obj:`int`, optional, defaults to 1000):
                use the same pos embeddings after clamp_len
            same_length (:obj:`boolean`, optional, defaults to :obj:`True`):
                Use the same attn length for all tokens
            proj_share_all_but_first (:obj:`boolean`, optional, defaults to :obj:`True`):
                True to share all but first projs, False not to share.
            attn_type (:obj:`int`, optional, defaults to 0):
                Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
            sample_softmax (:obj:`int`, optional, defaults to -1):
                number of samples in sampled softmax
            adaptive (:obj:`boolean`, optional, defaults to :obj:`True`):
                use adaptive softmax
            tie_weight (:obj:`boolean`, optional, defaults to :obj:`True`):
                tie the word embedding and softmax weights
            dropout (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            dropatt (:obj:`float`, optional, defaults to 0):
                The dropout ratio for the attention probabilities.
            untie_r (:obj:`boolean`, optional, defaults to :obj:`True`):
                Untie relative position biases
            init (:obj:`string`, optional, defaults to `normal`):
                Parameter initializer to use
            init_range (:obj:`float`, optional, defaults to 0.01):
                Parameters initialized by U(-init_range, init_range).
            proj_init_std (:obj:`float`, optional, defaults to 0.01):
                Parameters initialized by N(0, init_std)
            init_std (:obj:`float`, optional, defaults to 0.02):
                Parameters initialized by N(0, init_std)
            layer_norm_epsilon (:obj:`float`, optional, defaults to 1e-5):
                The epsilon to use in the layer normalization layers

        Example::

            >>> from transformers import TransfoXLConfig, TransfoXLModel

            >>> # Initializing a Transformer XL configuration
            >>> configuration = TransfoXLConfig()

            >>> # Initializing a model from the configuration
            >>> model = TransfoXLModel(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
    """

    model_type = "transfo-xl"

    def __init__(
        self,
        vocab_size=267735,
        cutoffs=[20000, 40000, 200000],
        d_model=1024,
        d_embed=1024,
        n_head=16,
        d_head=64,
        d_inner=4096,
        div_val=4,
        pre_lnorm=False,
        n_layer=18,
        tgt_len=128,
        ext_len=0,
        mem_len=1600,
        clamp_len=1000,
        same_length=True,
        proj_share_all_but_first=True,
        attn_type=0,
        sample_softmax=-1,
        adaptive=True,
        tie_weight=True,
        dropout=0.1,
        dropatt=0.0,
        untie_r=True,
        init="normal",
        init_range=0.01,
        proj_init_std=0.01,
        init_std=0.02,
        layer_norm_epsilon=1e-5,
        eos_token_id=0,
        **kwargs
    ):
        super().__init__(eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.cutoffs = []
        self.cutoffs.extend(cutoffs)
        self.tie_weight = tie_weight
        if proj_share_all_but_first:
            self.tie_projs = [False] + [True] * len(self.cutoffs)
        else:
            self.tie_projs = [False] + [False] * len(self.cutoffs)
        self.d_model = d_model
        self.d_embed = d_embed
        self.d_head = d_head
        self.d_inner = d_inner
        self.div_val = div_val
        self.pre_lnorm = pre_lnorm
        self.n_layer = n_layer
        self.n_head = n_head
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len
        self.same_length = same_length
        self.attn_type = attn_type
        self.clamp_len = clamp_len
        self.sample_softmax = sample_softmax
        self.adaptive = adaptive
        self.dropout = dropout
        self.dropatt = dropatt
        self.untie_r = untie_r
        self.init = init
        self.init_range = init_range
        self.proj_init_std = proj_init_std
        self.init_std = init_std
        self.layer_norm_epsilon = layer_norm_epsilon

    @property
    def max_position_embeddings(self):
        return self.tgt_len + self.ext_len + self.mem_len

    @property
    def n_token(self):  # Backward compatibility
        return self.vocab_size

    @n_token.setter
    def n_token(self, value):  # Backward compatibility
        self.vocab_size = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
