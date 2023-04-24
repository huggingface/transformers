# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.from transformers import PretrainedConfig
from transformers import PretrainedConfig


class EncoderConfig(PretrainedConfig):
    def __init__(self, embed_dim=768,
                 attention_heads=12,
                 ffn_embed_dim=3072,
                 layers=12,
                 normalize_before=True,
                 activation_fn="gelu",
                 dropout=0.0,
                 drop_path_rate=0.0,
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 deepnorm=False,
                 subln=True,
                 bert_init=False,
                 multiway=True,
                 layernorm_eps=1e-5,
                 vocab_size=64010,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_heads = attention_heads
        self.ffn_embed_dim = ffn_embed_dim
        self.layers = layers
        self.normalize_before = normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.deepnorm = deepnorm
        self.subln = subln
        self.bert_init = bert_init
        self.multiway = multiway
        self.layernorm_eps = layernorm_eps
        # Text
        self.vocab_size = vocab_size
        # Vision
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        # Fairscale

        if self.deepnorm:
            self.normalize_before = False
            self.subln = False
        if self.subln:
            self.normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0
