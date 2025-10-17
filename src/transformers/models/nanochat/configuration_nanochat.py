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

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


__all__ = ["NanoChatConfig"]


class NanoChatConfig(PretrainedConfig):
    r"""Configuration for the NanoChat model."""

    model_type = "nanochat"
    keys_to_ignore_at_inference = ["past_key_values"]


    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 768,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 1024,
        hidden_act: str = "relu2",
        attention_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        use_cache: bool = True,
        logits_soft_cap: float | None = 15.0,
        qkv_bias: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.resid_dropout = resid_dropout
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_cache = use_cache
        self.logits_soft_cap = logits_soft_cap
        self.qkv_bias = qkv_bias

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )