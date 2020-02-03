# coding=utf-8
# Copyright 2018 The Fairseq Authors and The HuggingFace Inc. team.
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
""" BART configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

BART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/transformer-base-config.json",

}

class BARTConfig(PretrainedConfig):
    model_type = 'bart'
    def __init__(self,
                 activation_dropout=0.,
                 vocab_size=50265,  # bert's is 30522, why the difference
                 pad_token_id=1,  # TODO(SS): feels like wrong place?
                 d_model=1024,
                 encoder_ffn_dim=4096, encoder_layers=12, encoder_attention_heads=16,
                 decoder_ffn_dim=4096, decoder_layers=12, decoder_attention_heads=16,
                 encoder_layerdrop=0., decoder_layerdrop=0.,


                 attention_dropout=0.0,
                 dropout=0.1,
                 max_position_embeddings=1024,
                 activation_fn='gelu',
                 **common_kwargs,
                 ):
        super().__init__(**common_kwargs)

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim

        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_embed_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.activation_fn = activation_fn

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout
