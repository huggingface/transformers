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
"""HumanV model configuration."""

from __future__ import annotations

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


def _validate_layer_types(layer_types: list[str], num_hidden_layers: int) -> None:
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            f"`layer_types` must have length == num_hidden_layers. Got {len(layer_types)} vs {num_hidden_layers}."
        )
    allowed = {"full_attention", "sliding_attention"}
    bad = [t for t in layer_types if t not in allowed]
    if bad:
        raise ValueError(f"Invalid entries in `layer_types`: {bad}. Allowed: {sorted(allowed)}.")


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of a HumanV checkpoint such as
    [nebularesearchtrain/nilla-story](https://huggingface.co/nebularesearchtrain/nilla-story).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

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
            Number of key/value heads (MHA if equal to `num_attention_heads`, MQA if 1, otherwise GQA).
        head_dim (`int`, *optional*, defaults to 32):
            The attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The base non-linear activation function used by the MLP.
        activation_backend (`str`, *optional*, defaults to `None`):
            Optional activation backend override (e.g. `"silu"`, `"gelu"`, `"sqrelu"`). If `None`, uses `hidden_act`.
        norm_backend (`str`, *optional*, defaults to `"rmsnorm"`):
            Normalization backend. Supported: `"rmsnorm"`, `"layernorm"`, `"scalenorm"`.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by normalization layers.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period for RoPE.
        use_selective_rope (`bool`, *optional*, defaults to `False`):
            Whether to enable Selective RoPE (content-dependent per-head phase adjustment).
        selective_rope_scale (`float`, *optional*, defaults to 0.25):
            Scale for the Selective RoPE phase adjustment.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated_normal_initializer for initializing weights.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache in generation.
        kv_cache_dtype (`str`, *optional*, defaults to `"bf16"`):
            DType used to store KV cache (`"bf16"`, `"fp16"`, `"fp32"`).
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether input and output word embeddings are tied.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Q/K/V/O projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout for attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP projections.
        attn_implementation (`str`, *optional*, defaults to `"eager"`):
            Attention backend hint (e.g. `"eager"`, `"sdpa"`). Backends depend on runtime support.
        layer_types (`list[str]`, *optional*, defaults to `None`):
            Per-layer attention type. Allowed values: `"full_attention"`, `"sliding_attention"`.
            If `None`, the pattern can be derived from `use_sliding_window/max_window_layers`.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to enable sliding-window pattern derivation when `layer_types` is `None`.
        max_window_layers (`int`, *optional*, defaults to 0):
            Number of initial layers using full attention when `use_sliding_window=True`.

        use_sparse_attention (`bool`, *optional*, defaults to `False`):
            Whether to enable sparse attention execution for `"sliding_attention"` layers.
        sparse_attention_impl (`str`, *optional*, defaults to `"local_global_block"`):
            Sparse attention implementation identifier.
        sparse_block_size (`int`, *optional*, defaults to 64):
            Block size for local-global block sparse attention.
        sparse_prefill_chunk_blocks (`int`, *optional*, defaults to 0):
            If > 0, chunks prefill computation in blocks to reduce peak memory.
        sparse_local_num_blocks (`int`, *optional*, defaults to 4):
            Number of local blocks per query block.
        sparse_global_num_blocks (`int`, *optional*, defaults to 2):
            Number of global blocks to attend.
        sparse_global_block_stride (`int`, *optional*, defaults to 4):
            Stride for selecting global blocks.
        sparse_attention_window (`int`, *optional*, defaults to 0):
            If > 0, limits local context window (in tokens) by reducing `sparse_local_num_blocks`.

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from transformers.models.humanv import HumanVConfig
    >>> config = HumanVConfig()
    >>> model = AutoModelForCausalLM.from_config(config)
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ```"""

    model_type = "humanv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        head_dim: int = 32,
        hidden_act: str = "silu",
        activation_backend: Optional[str] = None,
        norm_backend: str = "rmsnorm",
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10000.0,
        use_selective_rope: bool = False,
        selective_rope_scale: float = 0.25,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        kv_cache_dtype: str = "bf16",
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        attn_implementation: str = "eager",
        layer_types: Optional[list[str]] = None,
        use_sliding_window: bool = False,
        max_window_layers: int = 0,
        use_sparse_attention: bool = False,
        sparse_attention_impl: str = "local_global_block",
        sparse_block_size: int = 64,
        sparse_prefill_chunk_blocks: int = 0,
        sparse_local_num_blocks: int = 4,
        sparse_global_num_blocks: int = 2,
        sparse_global_block_stride: int = 4,
        sparse_attention_window: int = 0,
        **kwargs,
    ):
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if num_attention_heads * head_dim != hidden_size:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim "
                f"({hidden_size} != {num_attention_heads} * {head_dim})."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.hidden_act = hidden_act
        self.activation_backend = activation_backend

        self.norm_backend = norm_backend
        self.rms_norm_eps = rms_norm_eps

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_selective_rope = use_selective_rope
        self.selective_rope_scale = selective_rope_scale

        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.kv_cache_dtype = kv_cache_dtype

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        self.attn_implementation = attn_implementation

        self.use_sliding_window = use_sliding_window
        self.max_window_layers = max_window_layers

        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_impl = sparse_attention_impl
        self.sparse_block_size = sparse_block_size
        self.sparse_prefill_chunk_blocks = sparse_prefill_chunk_blocks
        self.sparse_local_num_blocks = sparse_local_num_blocks
        self.sparse_global_num_blocks = sparse_global_num_blocks
        self.sparse_global_block_stride = sparse_global_block_stride
        self.sparse_attention_window = sparse_attention_window

        self.layer_types = layer_types
        if self.layer_types is None:
            if self.use_sliding_window and self.max_window_layers > 0:
                self.layer_types = [
                    "full_attention" if i < self.max_window_layers else "sliding_attention"
                    for i in range(self.num_hidden_layers)
                ]
            else:
                self.layer_types = ["full_attention" for _ in range(self.num_hidden_layers)]
        _validate_layer_types(self.layer_types, self.num_hidden_layers)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HumanVConfig"]
