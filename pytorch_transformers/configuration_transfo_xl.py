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

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'transfo-xl-wt103': "https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json",
}

class TransfoXLConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `TransfoXLModel`.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `TransfoXLModel` or a configuration json file.
            cutoffs: cutoffs for the adaptive softmax
            d_model: Dimensionality of the model's hidden states.
            d_embed: Dimensionality of the embeddings
            d_head: Dimensionality of the model's heads.
            div_val: divident value for adapative input and softmax
            pre_lnorm: apply LayerNorm to the input instead of the output
            d_inner: Inner dimension in FF
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            tgt_len: number of tokens to predict
            ext_len: length of the extended context
            mem_len: length of the retained previous heads
            same_length: use the same attn length for all tokens
            proj_share_all_but_first: True to share all but first projs, False not to share.
            attn_type: attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
            clamp_len: use the same pos embeddings after clamp_len
            sample_softmax: number of samples in sampled softmax
            adaptive: use adaptive softmax
            tie_weight: tie the word embedding and softmax weights
            dropout: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            dropatt: The dropout ratio for the attention probabilities.
            untie_r: untie relative position biases
            embd_pdrop: The dropout ratio for the embeddings.
            init: parameter initializer to use
            init_range: parameters initialized by U(-init_range, init_range).
            proj_init_std: parameters initialized by N(0, init_std)
            init_std: parameters initialized by N(0, init_std)
    """
    pretrained_config_archive_map = TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=267735,
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
                 **kwargs):
        """Constructs TransfoXLConfig.
        """
        super(TransfoXLConfig, self).__init__(**kwargs)

        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_token = vocab_size_or_config_json_file
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
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")

    @property
    def max_position_embeddings(self):
        return self.tgt_len + self.ext_len + self.mem_len

    @property
    def vocab_size(self):
        return self.n_token

    @vocab_size.setter
    def vocab_size(self, value):
        self.n_token = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
