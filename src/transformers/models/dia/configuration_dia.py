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
        num_hidden_layers=12,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        head_dim=128,
        norm_eps=1e-5,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        super().__init__(**kwargs)

class DiaDecoderConfig(PretrainedConfig):
    model_type = "dia_decoder"

    def __init__(
        self,
        num_hidden_layers=18,
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        cross_query_heads=16,
        cross_head_dim=128,
        norm_eps=1e-5,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size     
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.cross_query_heads = cross_query_heads
        self.cross_head_dim = cross_head_dim
        self.norm_eps = norm_eps
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
        norm_eps=1e-5,
        weight_dtype="float32",
        rope_min_timescale=1,
        rope_max_timescale=10_000,
        is_encoder_decoder=True,
        **kwargs,
    ):
        self.encoder_config = encoder
        self.decoder_config = decoder
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout
        self.norm_eps = norm_eps
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
