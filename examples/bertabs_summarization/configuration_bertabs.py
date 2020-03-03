# coding=utf-8
# Copyright 2019 The HuggingFace Inc. team.
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
""" BertAbs configuration """
import logging

from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


BERTABS_FINETUNED_CONFIG_MAP = {
    "bertabs-finetuned-cnndm": "https://s3.amazonaws.com/models.huggingface.co/bert/remi/bertabs-finetuned-cnndm-extractive-abstractive-summarization-config.json",
}


class BertAbsConfig(PretrainedConfig):
    r""" Class to store the configuration of the BertAbs model.

    Arguments:
        vocab_size: int
            Number of tokens in the vocabulary.
        max_pos: int
            The maximum sequence length that this model will be used with.
        enc_layer: int
            The numner of hidden layers in the Transformer encoder.
        enc_hidden_size: int
            The size of the encoder's layers.
        enc_heads: int
            The number of attention heads for each attention layer in the encoder.
        enc_ff_size: int
            The size of the encoder's feed-forward layers.
        enc_dropout: int
            The dropout probabilitiy for all fully connected layers in the
            embeddings, layers, pooler and also the attention probabilities in
            the encoder.
        dec_layer: int
            The numner of hidden layers in the decoder.
        dec_hidden_size: int
            The size of the decoder's layers.
        dec_heads: int
            The number of attention heads for each attention layer in the decoder.
        dec_ff_size: int
            The size of the decoder's feed-forward layers.
        dec_dropout: int
            The dropout probabilitiy for all fully connected layers in the
            embeddings, layers, pooler and also the attention probabilities in
            the decoder.
    """

    pretrained_config_archive_map = BERTABS_FINETUNED_CONFIG_MAP
    model_type = "bertabs"

    def __init__(
        self,
        vocab_size=30522,
        max_pos=512,
        enc_layers=6,
        enc_hidden_size=512,
        enc_heads=8,
        enc_ff_size=512,
        enc_dropout=0.2,
        dec_layers=6,
        dec_hidden_size=768,
        dec_heads=8,
        dec_ff_size=2048,
        dec_dropout=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_pos = max_pos

        self.enc_layers = enc_layers
        self.enc_hidden_size = enc_hidden_size
        self.enc_heads = enc_heads
        self.enc_ff_size = enc_ff_size
        self.enc_dropout = enc_dropout

        self.dec_layers = dec_layers
        self.dec_hidden_size = dec_hidden_size
        self.dec_heads = dec_heads
        self.dec_ff_size = dec_ff_size
        self.dec_dropout = dec_dropout
