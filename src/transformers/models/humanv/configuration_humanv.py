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

from typing import Optional, Union

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    nilla-story [nebularesearchtrain/nilla-story](https://huggingface.co/nebularesearchtrain/nilla-story).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

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
            Number of key/value heads for MHA/GQA/MQA.
        head_dim (`int`, *optional*, defaults to 32):
            Attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Non-linear activation used by the MLP.
        activation_backend (`str`, *optional*, defaults to `"silu"`):
            Activation backend selector (research switch).
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            RoPE configuration following the new `rope_parameters` format.
        use_selective_rope (`bool`, *optional*, defaults to `False`):
            Enables Selective RoPE (research path).
        selective_rope_scale (`float`, *optional*, defaults to 1.0):
            Scale factor for Selective RoPE gating.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Weight initialization std.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            RMSNorm epsilon.
        norm_backend (`str`, *optional*, defaults to `"rmsnorm"`):
            Norm backend selector (research switch).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Cache dtype selector: `"auto"`, `"fp32"`, `"bf16"`, `"fp16"`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Tie input/output embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP projections.
        attn_implementation (`str`, *optional*, defaults to `"eager"`):
            Attention implementation selector.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to enable sliding-window mode via default `layer_types`.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window size if enabled.
        max_window_layers (`int`, *optional*, defaults to 0):
            Number of initial layers using full attention when sliding window is enabled.
        layer_types (`list[str]`, *optional*):
            Per-layer attention pattern, e.g. `"full_attention"` / `"sliding_attention"`.
        use_sparse_attention (`bool`, *optional*, defaults to `False`):
            Enables sparse attention path.
        sparse_attention_impl (`str`, *optional*, defaults to `"local_global_block"`):
            Sparse attention implementation name.
        sparse_block_size (`int`, *optional*, defaults to 64):
            Block size for block-sparse attention.
        sparse_prefill_chunk_blocks (`int`, *optional*, defaults to 16):
            Chunking (in blocks) for long-context prefill.
        sparse_local_num_blocks (`int`, *optional*, defaults to 4):
            Number of local blocks to attend per query block.
        sparse_global_num_blocks (`int`, *optional*, defaults to 2):
            Number of global blocks sampled/pooled per query block.
        sparse_global_block_stride (`int`, *optional*, defaults to 4):
            Stride for choosing global blocks.
        sparse_attention_window (`int`, *optional*, defaults to 256):
            Max window span for sparse attention masking.

    ```python
    >>> from transformers import HumanVConfig, HumanVModel
    >>> config = HumanVConfig()
    >>> model = HumanVModel(config)
    >>> config = model.config
    ```"""

    model_type = "humanv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: Optional[int] = 50257,
        hidden_size: Optional[int] = 256,
        intermediate_size: Optional[int] = 1024,
        num_hidden_layers: Optional[int] = 8,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 8,
        head_dim: Optional[int] = 32,
        hidden_act: Optional[str] = "silu",
        activation_backend: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 1024,
        rope_parameters: Optional[Union[RopeParameters, dict]] = None,
        use_selective_rope: Optional[bool] = False,
        selective_rope_scale: Optional[float] = 1.0,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-5,
        norm_backend: Optional[str] = "rmsnorm",
        use_cache: Optional[bool] = True,
        kv_cache_dtype: Optional[str] = "auto",
        tie_word_embeddings: Optional[bool] = True,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        mlp_bias: Optional[bool] = False,
        attn_implementation: Optional[str] = "eager",
        use_sliding_window: Optional[bool] = False,
        sliding_window: Optional[int] = 4096,
        max_window_layers: Optional[int] = 0,
        layer_types: Optional[list[str]] = None,
        use_sparse_attention: Optional[bool] = False,
        sparse_attention_impl: Optional[str] = "local_global_block",
        sparse_block_size: Optional[int] = 64,
        sparse_prefill_chunk_blocks: Optional[int] = 16,
        sparse_local_num_blocks: Optional[int] = 4,
        sparse_global_num_blocks: Optional[int] = 2,
        sparse_global_block_stride: Optional[int] = 4,
        sparse_attention_window: Optional[int] = 256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.activation_backend = activation_backend

        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters
        self.use_selective_rope = use_selective_rope
        self.selective_rope_scale = selective_rope_scale

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.norm_backend = norm_backend

        self.use_cache = use_cache
        self.kv_cache_dtype = kv_cache_dtype

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        self.attn_implementation = attn_implementation

        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.layer_types = layer_types
        if self.layer_types is None:
            if self.use_sliding_window and self.sliding_window is not None:
                self.layer_types = [
                    "sliding_attention" if i >= self.max_window_layers else "full_attention"
                    for i in range(self.num_hidden_layers)
                ]
            else:
                self.layer_types = ["full_attention"] * self.num_hidden_layers
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_impl = sparse_attention_impl
        self.sparse_block_size = sparse_block_size
        self.sparse_prefill_chunk_blocks = sparse_prefill_chunk_blocks
        self.sparse_local_num_blocks = sparse_local_num_blocks
        self.sparse_global_num_blocks = sparse_global_num_blocks
        self.sparse_global_block_stride = sparse_global_block_stride
        self.sparse_attention_window = sparse_attention_window

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["HumanVConfig"]
