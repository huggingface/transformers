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

BEIT3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/beit-base-patch16-224-pt22k": (
        "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k/resolve/main/config.json"
    ),
    # See all BEiT models at https://huggingface.co/models?filter=beit
}

class Beit3Config(PretrainedConfig):
    def __init__(
        self,
        embed_dim=768,
        num_attention_heads=12,
        hidden_size=3072,
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
        max_source_positions=1024,
        layernorm_eps=1e-5,
        vocab_size=64010,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_labels=2,
        initializer_range=0.02,
        label_smoothing=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
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
        self.max_source_positions = max_source_positions
        self.layernorm_eps = layernorm_eps
        self.initializer_range = initializer_range
        # Text
        self.vocab_size = vocab_size
        # Vision
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        if self.subln:
            self.normalize_before = True
            self.deepnorm = False
