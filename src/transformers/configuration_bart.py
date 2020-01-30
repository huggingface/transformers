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


_FAIRSEQ_DEFAULTS = dict(
    encoder_embed_dim=1024,
    encoder_ffn_embed_dim=4096,
    encoder_layers=12,
    encoder_attention_heads=16,
    encoder_normalize_before=False,
    encoder_learned_pos=True,
    decoder_embed_path=None,
    decoder_embed_dim=1024,
    decoder_ffn_embed_dim=4096,
    decoder_layers=12,
    decoder_attention_heads=16,
    decoder_normalize_before=False,
    decoder_learned_pos=True,
    attention_dropout=0.0,
    relu_dropout=0.0,
    dropout=0.1,
    max_target_positions=1024,
    max_source_positions=1024,
    adaptive_softmax_cutoff=None,
    adaptive_softmax_dropout=0,
    share_decoder_input_output_embed=True,
    share_all_embeddings=True,
    decoder_output_dim=1024,
    decoder_input_dim=1024,
    no_scale_embedding=True,
    layernorm_embedding=True,
    activation_fn='gelu',
    pooler_activation_fn='tanh',
    pooler_dropout=0.0,
    encoder_layerdrop=0.,
    decoder_layerdrop=0.,
#vocab_size = 30522,
#             pad_token_id = 1
)

class BARTConfig(PretrainedConfig):
    model_type = 'bart'
    def __init__(self,
                 vocab_size=50265,
                 pad_token_id=1, # TODO(SS): feels like wrong place?
                 encoder_embed_dim=1024,
                 encoder_ffn_embed_dim=4096,
                 encoder_layers=12,
                 encoder_attention_heads=16,
                 encoder_normalize_before=False,
                 encoder_layerdrop=0., decoder_layerdrop=0.,
                 encoder_learned_pos=True,
                 decoder_embed_path=None,
                 decoder_embed_dim=1024,
                 decoder_ffn_embed_dim=4096,
                 decoder_layers=12,
                 decoder_attention_heads=16,
                 decoder_normalize_before=False,
                 decoder_learned_pos=True,
                 attention_dropout=0.0,
                 relu_dropout=0.0,
                 dropout=0.1,
                 max_target_positions=1024,  # dont need two args if weight tying
                 max_source_positions=1024,
                 adaptive_softmax_cutoff=None,
                 adaptive_softmax_dropout=0,
                 share_decoder_input_output_embed=True,
                 share_all_embeddings=True,
                 decoder_output_dim=1024,
                 decoder_input_dim=1024,
                 no_scale_embedding=True,
                 no_cross_attention=False,
                 layernorm_embedding=True,
                 activation_fn='gelu',
                 pooler_activation_fn='tanh',
                 pooler_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        for k in _FAIRSEQ_DEFAULTS:
            setattr(self, k, locals()[k])   # hack to avoid 1 million LOC
