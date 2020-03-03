# coding=utf-8
# Copyright 2020 The Fairseq Authors and The HuggingFace Inc. team.
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
    "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/config.json",
    "bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-mnli/config.json",
    "bart-large-cnn": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/config.json",
}


class BartConfig(PretrainedConfig):
    r"""
        Configuration class for Bart. Parameters are renamed from the fairseq implementation
    """
    model_type = "bart"
    pretrained_config_archive_map = BART_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        activation_dropout=0.0,
        vocab_size=50265,
        pad_token_id=1,
        eos_token_id=2,
        d_model=1024,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        encoder_attention_heads=16,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        max_position_embeddings=1024,
        init_std=0.02,
        classifier_dropout=0.0,
        output_past=False,
        num_labels=3,
        bos_token_id=0,
        **common_kwargs
    ):
        r"""
            :class:`~transformers.BartConfig` is the configuration class for `BartModel`.
            Examples:
                config = BartConfig.from_pretrained('bart-large')
                model = BartModel(config)
        """
        super().__init__(
            num_labels=num_labels,
            output_past=output_past,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.eos_token_id = eos_token_id
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        # Classifier stuff
        self.classif_dropout = classifier_dropout

    @property
    def num_attention_heads(self):
        return self.encoder_attention_heads

    @property
    def hidden_size(self):
        return self.d_model
