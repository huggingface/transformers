# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Any, Optional

from ...configuration_utils import PretrainedConfig


class BLTTokenizerLocalConfig(PretrainedConfig):
    """
    Configuration class for BLT Tokenizer Local Model.
    """

    model_type = "blt_tokenizer_local"

    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        head_dim: Optional[int] = None,
        max_length: int = 2048,
        max_seqlen: Optional[int] = None,
        vocab_size: int = 32000,
        dropout: float = 0.0,
        patch_size: float = 1.0,
        dim_patch_emb: Optional[int] = None,
        dim_token_emb: Optional[int] = None,
        attn_impl: Optional[str] = "xformers",
        sliding_window: Optional[int] = None,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        rope_use_fp32_in_outer_product: bool = False,
        init_std_factor: str = "disabled",
        init_base_std: float = 0.02,
        n_kv_heads: Optional[int] = None,
        attn_bias_type: str = "local_block_causal",
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 1.0,
        patching_mode: Optional[str] = None,
        use_local_encoder_transformer: bool = False,
        downsampling_by_pooling: Optional[str] = None,
        encoder_hash_byte_group_size: Optional[Any] = None,
        cross_attn_encoder: bool = False,
        cross_attn_decoder: bool = False,
        cross_attn_k: Optional[int] = None,
        cross_attn_nheads: Optional[int] = None,
        cross_attn_all_layers_encoder: bool = False,
        cross_attn_all_layers_decoder: bool = False,
        cross_attn_init_by_pooling: bool = False,
        eos_id: int = 2,
        boe_id: int = 1,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.max_seqlen = max_seqlen
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.patch_size = patch_size
        self.dim_patch_emb = dim_patch_emb
        self.dim_token_emb = dim_token_emb
        self.attn_impl = attn_impl
        self.sliding_window = sliding_window
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product
        self.init_std_factor = init_std_factor
        self.init_base_std = init_base_std
        self.n_kv_heads = n_kv_heads
        self.attn_bias_type = attn_bias_type
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.patching_mode = patching_mode
        self.use_local_encoder_transformer = use_local_encoder_transformer
        self.downsampling_by_pooling = downsampling_by_pooling
        self.encoder_hash_byte_group_size = encoder_hash_byte_group_size
        self.cross_attn_encoder = cross_attn_encoder
        self.cross_attn_decoder = cross_attn_decoder
        self.cross_attn_k = cross_attn_k
        self.cross_attn_nheads = cross_attn_nheads
        self.cross_attn_all_layers_encoder = cross_attn_all_layers_encoder
        self.cross_attn_all_layers_decoder = cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = cross_attn_init_by_pooling
        self.eos_id = eos_id
        self.boe_id = boe_id
        self.norm_eps = norm_eps 