# coding=utf-8
# Copyright 2025 The HumanV Team. All rights reserved.
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
"""HumanV model configuration"""

from __future__ import annotations

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture.

    A reference checkpoint for this architecture is:
    [nebularesearchtrain/nilla-story](https://huggingface.co/nebularesearchtrain/nilla-story)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads (MHA if equals `num_attention_heads`, GQA otherwise).
        head_dim (`int`, *optional*, defaults to 32):
            Attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Weight initialization std.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            RMSNorm epsilon.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return key/value cache.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie input/output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE theta.
        rope_parameters (`dict`, *optional*):
            Optional dict; if provided and contains `rope_theta`, it overrides `rope_theta`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP projections.
        attn_implementation (`str`, *optional*, defaults to `"eager"`):
            Attention implementation. Supported: `"eager"`, `"sdpa"`.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type: `"full_attention"` or `"sliding_attention"`.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Enables sliding attention default behavior if `layer_types` is not provided.
        sliding_window (`int`, *optional*, defaults to 256):
            Sliding window length (tokens) for dense sliding attention (when sparse is disabled).
        use_sparse_attention (`bool`, *optional*, defaults to `False`):
            Enables sparse attention for `"sliding_attention"` layers.
        sparse_attention_impl (`str`, *optional*, defaults to `"local_global_block"`):
            Sparse attention implementation. Supported: `"local_global_block"`.
        sparse_block_size (`int`, *optional*, defaults to 64):
            Block size for sparse attention.
        sparse_local_num_blocks (`int`, *optional*, defaults to 4):
            Number of local blocks per query block.
        sparse_global_num_blocks (`int`, *optional*, defaults to 2):
            Number of global blocks per query block.
        sparse_global_block_stride (`int`, *optional*, defaults to 4):
            Stride for selecting global blocks.
        sparse_attention_window (`int`, *optional*, defaults to 256):
            Max context window for global block selection (tokens).
    """

    model_type = "humanv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = 8,
        head_dim: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        attn_implementation: str = "eager",
        layer_types: Optional[list[str]] = None,
        use_sliding_window: bool = False,
        sliding_window: int = 256,
        use_sparse_attention: bool = False,
        sparse_attention_impl: str = "local_global_block",
        sparse_block_size: int = 64,
        sparse_local_num_blocks: int = 4,
        sparse_global_num_blocks: int = 2,
        sparse_global_block_stride: int = 4,
        sparse_attention_window: int = 256,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.attn_implementation = attn_implementation

        self.rope_parameters = rope_parameters
        if isinstance(self.rope_parameters, dict) and "rope_theta" in self.rope_parameters:
            self.rope_theta = float(self.rope_parameters["rope_theta"])
        else:
            self.rope_theta = float(rope_theta)

        self.use_sliding_window = use_sliding_window
        self.sliding_window = int(sliding_window) if use_sliding_window else None

        if layer_types is None:
            if self.use_sliding_window and self.sliding_window is not None:
                self.layer_types = ["sliding_attention"] * self.num_hidden_layers
            else:
                self.layer_types = ["full_attention"] * self.num_hidden_layers
        else:
            if len(layer_types) != self.num_hidden_layers:
                raise ValueError(
                    f"`layer_types` must have length {self.num_hidden_layers}, got {len(layer_types)}."
                )
            self.layer_types = list(layer_types)

        self.use_sparse_attention = bool(use_sparse_attention)
        self.sparse_attention_impl = str(sparse_attention_impl)
        self.sparse_block_size = int(sparse_block_size)
        self.sparse_local_num_blocks = int(sparse_local_num_blocks)
        self.sparse_global_num_blocks = int(sparse_global_num_blocks)
        self.sparse_global_block_stride = int(sparse_global_block_stride)
        self.sparse_attention_window = int(sparse_attention_window)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["HumanVConfig"]
