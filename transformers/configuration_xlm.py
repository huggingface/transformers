# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
""" XLM configuration """
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

XLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-config.json",
    'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-config.json",
    'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-config.json",
    'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-config.json",
    'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-config.json",
    'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-config.json",
    'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-config.json",
    'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-config.json",
    'xlm-mlm-17-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-config.json",
    'xlm-mlm-100-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-config.json",
}


class XLMConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `XLMModel`.

    Args:
        vocab_size: Vocabulary size of `inputs_ids` in `XLMModel`.
        d_model: Size of the encoder layers and the pooler layer.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        d_inner: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        ff_activation: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        untie_r: untie relative position biases
        attn_type: 'bi' for XLM, 'uni' for Transformer-XL

        dropout: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.

        dropout: float, dropout rate.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
    """
    pretrained_config_archive_map = XLM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size=30145,
                 emb_dim=2048,
                 n_layers=12,
                 n_heads=16,
                 dropout=0.1,
                 attention_dropout=0.1,
                 gelu_activation=True,
                 sinusoidal_embeddings=False,
                 causal=False,
                 asm=False,
                 n_langs=1,
                 use_lang_emb=True,
                 max_position_embeddings=512,
                 embed_init_std=2048 ** -0.5,
                 layer_norm_eps=1e-12,
                 init_std=0.02,
                 bos_index=0,
                 eos_index=1,
                 pad_index=2,
                 unk_index=3,
                 mask_index=5,
                 is_encoder=True,
                 summary_type='first',
                 summary_use_proj=True,
                 summary_activation=None,
                 summary_proj_to_labels=True,
                 summary_first_dropout=0.1,
                 start_n_top=5,
                 end_n_top=5,
                 **kwargs):
        """Constructs XLMConfig.
        """
        super(XLMConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.use_lang_emb = use_lang_emb
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
        self.init_std = init_std
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top

        if "n_words" in kwargs:
            self.n_words = kwargs["n_words"]

    @property
    def n_words(self):  # For backward compatibility
        return self.vocab_size

    @n_words.setter
    def n_words(self, value):  # For backward compatibility
        self.vocab_size = value

    @property
    def hidden_size(self):
        return self.emb_dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers
