# coding=utf-8
# Copyright 2010, The T5 Authors and HuggingFace Inc.
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
""" T5 model configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "t5-small": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-small-config.json",
    "t5-base": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-base-config.json",
    "t5-large": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-large-config.json",
    "t5-3b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-3b-config.json",
    "t5-11b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-11b-config.json",
}


class T5Config(PretrainedConfig):
    r"""
        :class:`~transformers.T5Config` is the configuration class to store the configuration of a
        `T5Model`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `T5Model`.
            d_model: Size of the encoder layers and the pooler layer. `d_model` can also accesed via the property `hidden_size`.
            num_layers: Number of hidden layers in the Transformer encoder. `num_layers` can also be accessed via the property `num_hidden_layers`.
            d_kv: Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model // num_heads`.
            d_ff: Size of the intermediate feed forward layer in each `T5Block`.
            num_heads: Number of attention heads for each attention layer in
                the Transformer encoder. `num_heads` can also be accessed via the property `num_attention_heads`.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            n_positions: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048). `n_positions` can also be accessed via the property `max_position_embeddings`.
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `T5Model`.
            initializer_factor: A factor for initializing all weight matrices (should be kept to 1.0, used for initialization testing).
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    model_type = "t5"

    def __init__(
        self,
        vocab_size=32128,
        n_positions=512,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers
