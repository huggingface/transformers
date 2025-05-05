# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Dia model configuration"""

from typing import TYPE_CHECKING

from ...configuration_utils import PretrainedConfig
from ...utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

class DiaEncoderConfig(PretrainedConfig):
    model_type = "dia_encoder"

    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        head_dim,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers  # was: n_layer
        self.hidden_size = hidden_size             # was: n_embd
        self.intermediate_size = intermediate_size # was: n_hidden
        self.num_attention_heads = num_attention_heads  # was: n_head
        self.head_dim = head_dim

        super().__init__(**kwargs)

class DiaDecoderConfig(PretrainedConfig):
    model_type = "dia_decoder"

    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        cross_query_heads,
        cross_head_dim,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers      # was: n_layer
        self.hidden_size = hidden_size                 # was: n_embd
        self.intermediate_size = intermediate_size     # was: n_hidden
        self.num_attention_heads = num_attention_heads  # was: gqa_query_heads
        self.num_key_value_heads = num_key_value_heads  # was: kv_heads
        self.head_dim = head_dim                      # was: gqa_head_dim
        self.cross_query_heads = cross_query_heads
        self.cross_head_dim = cross_head_dim

        super().__init__(**kwargs)


class DiaConfig(PretrainedConfig):
    model_type = "dia"
    sub_configs = {"encoder_config": DiaEncoderConfig, "decoder_config": DiaDecoderConfig}

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        src_vocab_size=128,
        tgt_vocab_size=1028,
        dropout=0.0,
        normalization_layer_epsilon=1e-5,
        weight_dtype="float32",
        rope_min_timescale=1,
        rope_max_timescale=10_000,
        is_encoder_decoder=True
        **kwargs,
    ):
        self.encoder_config = encoder
        self.decoder_config = decoder
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout
        self.normalization_layer_epsilon = normalization_layer_epsilon
        self.weight_dtype = weight_dtype
        self.rope_min_timescale = rope_min_timescale
        self.rope_max_timescale = rope_max_timescale
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


__all__ = ["DiaConfig"]
