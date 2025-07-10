# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Optional

from ...configuration_utils import PretrainedConfig


class Lfm2Config(PretrainedConfig):
    model_type = "lfm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2560,
        num_hidden_layers: int = 32,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_embedding: bool = True,
        theta: float = 1000000.0,
        max_position_embeddings: int = 128_000,
        use_cache: bool = True,
        norm_eps: float = 0.00001,
        initializer_range: float = 0.02,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        conv_bias: bool = False,
        conv_dim: int = 2560,
        conv_L_cache: int = 3,
        block_dim: int = 2560,
        block_ff_dim: int = 12288,
        block_multiple_of: int = 256,
        block_ffn_dim_multiplier: float = 1.0,
        block_auto_adjust_ff_dim: bool = True,
        full_attn_idxs: Optional[list[int]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = theta
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range

        # attn operator config
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.full_attn_idxs = full_attn_idxs

        # custom operator config
        self.conv_bias = conv_bias
        self.conv_dim = conv_dim
        self.conv_L_cache = conv_L_cache

        # block config
        self.block_dim = block_dim
        self.block_ff_dim = block_ff_dim
        self.block_multiple_of = block_multiple_of
        self.block_ffn_dim_multiplier = block_ffn_dim_multiplier
        self.block_auto_adjust_ff_dim = block_auto_adjust_ff_dim

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_embedding,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return ["attention" if i in self.full_attn_idxs else "conv" for i in range(self.num_hidden_layers)]


__all__ = ["Lfm2Config"]
