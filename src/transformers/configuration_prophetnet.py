# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
""" ProphetNet model configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/prophetnet-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/prophetnet-large-uncased/config.json",
}


class ProphetNetConfig(PretrainedConfig):
    r"""
    Configuration class for ProphetNet. Parameters are renamed from the fairseq implementation
    """
    model_type = "prophetnet"

    def __init__(
        self,
        activation_dropout=0.1,
        activation_function="gelu",
        vocab_size=30522,
        hidden_size=1024,
        encoder_ffn_dim=4096,
        num_encoder_layers=12,
        num_encoder_attention_heads=16,
        decoder_ffn_dim=4096,
        num_decoder_layers=12,
        num_decoder_attention_heads=16,
        encoder_layerdrop=0.1,
        decoder_layerdrop=0.1,
        attention_dropout=0.1,
        dropout=0.1,
        max_position_embeddings=512,
        init_std=0.02,
        is_encoder_decoder=True,
        pad_token_id=0,
        bos_token_id=102,
        eos_token_id=102,
        ngram=2,
        num_buckets=32,
        relative_max_distance=128,
        disable_ngram_loss=False,
        eps=0.0,
        **common_kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # parameters for prophetnet
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.disable_ngram_loss = disable_ngram_loss
        self.eps = eps

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

    @property
    def num_attention_heads(self) -> int:
        return self.num_encoder_attention_heads
