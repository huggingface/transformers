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
    HumanV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    nilla-story [nebularesearchtrain/nilla-story](https://huggingface.co/nebularesearchtrain/nilla-story).

    HumanV is a decoder-only Transformer designed with TPU-first constraints (shape-friendly dimensions, stable
    attention compute policy) and supports optional Local-Global Block-Sparse attention for long-context efficiency.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the HumanV model.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads for MHA/GQA/MQA.
        head_dim (`int`, *optional*, defaults to 32):
            Attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Non-linear activation function in the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length supported by the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Initialization std for weights.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            RMSNorm epsilon.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return KV cache.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Tie input/output embeddings.
        rope_parameters (`dict`, *optional*):
            RoPE parameters. Minimal supported key: `rope_theta` (float).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Bias for Q/K/V/O projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout for attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Bias for MLP projections.

        attn_implementation (`str`, *optional*, defaults to `"eager"`):
            Attention implementation hint. For TPU-first baseline, `"eager"` is recommended.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type. Supported: `"full_attention"`, `"sliding_attention"`.
            If not set, will be auto-derived from `use_sliding_window` settings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to enable sliding window attention in later layers (if `layer_types` is not set).
        sliding_window (`int`, *optional*, defaults to 256):
            Sliding window size.
        max_window_layers (`int`, *optional*, defaults to 0):
            Number of initial layers that use full attention before switching to sliding attention (if enabled).

        attention_backend (`str`, *optional*, defaults to `"eager"`):
            Backend selector (kept stable for future TPU-optimized kernels). Current: `"eager"`.
        attention_compute_dtype (`str`, *optional*, defaults to `"fp32"`):
            Compute dtype policy for attention internals. Recommended on TPU: `"fp32"` (QKᵀ/softmax/PV accumulate).

        use_sparse_attention (`bool`, *optional*, defaults to `False`):
            Enable sparse attention path.
        sparse_attention_impl (`str`, *optional*, defaults to `"local_global_block"`):
            Sparse attention algorithm. Current: `"local_global_block"`.
        sparse_block_size (`int`, *optional*, defaults to 64):
            Block size for block-sparse.
        sparse_local_num_blocks (`int`, *optional*, defaults to 4):
            Number of local blocks (lookback).
        sparse_global_num_blocks (`int`, *optional*, defaults to 2):
            Number of global blocks.
        sparse_global_block_stride (`int`, *optional*, defaults to 4):
            Stride for selecting global blocks.
        sparse_oow_enabled (`bool`, *optional*, defaults to `False`):
            Enable out-of-window residual path (RATTENTION-style summary).
        sparse_oow_gate_init (`float`, *optional*, defaults to 0.0):
            Init value for the learnable oow gate (per-head).
    """

    model_type = "humanv"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

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
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        attn_implementation: str = "eager",
        layer_types: Optional[list[str]] = None,
        use_sliding_window: bool = False,
        sliding_window: int = 256,
        max_window_layers: int = 0,
        attention_backend: str = "eager",
        attention_compute_dtype: str = "fp32",
        use_sparse_attention: bool = False,
        sparse_attention_impl: str = "local_global_block",
        sparse_block_size: int = 64,
        sparse_local_num_blocks: int = 4,
        sparse_global_num_blocks: int = 2,
        sparse_global_block_stride: int = 4,
        sparse_oow_enabled: bool = False,
        sparse_oow_gate_init: float = 0.0,
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

        if rope_parameters is None:
            rope_parameters = {"rope_theta": float(kwargs.pop("rope_theta", 10000.0))}
        self.rope_parameters = rope_parameters

        self.attn_implementation = attn_implementation
        self.use_sliding_window = use_sliding_window
        self.sliding_window = int(sliding_window) if use_sliding_window else None
        self.max_window_layers = int(max_window_layers)

        self.layer_types = layer_types
        if self.layer_types is None:
            if self.use_sliding_window and self.sliding_window is not None and self.max_window_layers >= 0:
                self.layer_types = [
                    "full_attention" if i < self.max_window_layers else "sliding_attention"
                    for i in range(self.num_hidden_layers)
                ]
            else:
                self.layer_types = ["full_attention"] * self.num_hidden_layers

        self.attention_backend = attention_backend
        self.attention_compute_dtype = attention_compute_dtype

        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_impl = sparse_attention_impl
        self.sparse_block_size = int(sparse_block_size)
        self.sparse_local_num_blocks = int(sparse_local_num_blocks)
        self.sparse_global_num_blocks = int(sparse_global_num_blocks)
        self.sparse_global_block_stride = int(sparse_global_block_stride)
        self.sparse_oow_enabled = sparse_oow_enabled
        self.sparse_oow_gate_init = float(sparse_oow_gate_init)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["HumanVConfig"]
