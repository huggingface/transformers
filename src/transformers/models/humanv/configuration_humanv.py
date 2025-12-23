# coding=utf-8
# Copyright 2025 The HumanV Team.
# Licensed under the Apache License, Version 2.0

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    [nebularesearchtrain/nilla-story](https://huggingface.co/nebularesearchtrain/nilla-story).

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
            Number of key/value heads. If equals `num_attention_heads`, the model uses MHA, otherwise GQA.
        head_dim (`int`, *optional*, defaults to 32):
            Dimension per attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Weight initialization range.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            RMSNorm epsilon.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache during generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE base theta.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether QKV/O projections use bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether MLP projections use bias.
        attn_implementation (`str`, *optional*, defaults to `"eager"`):
            Attention backend selector. Keep `"eager"` for TPU-first stability.
        layer_types (`list[str]`, *optional*):
            Per-layer attention pattern. Supported: `"full_attention"`, `"sliding_attention"`.
        use_sparse_attention (`bool`, *optional*, defaults to `False`):
            Enables sparse attention path for layers with `"sliding_attention"`.
        sparse_attention_impl (`str`, *optional*, defaults to `"local_global_block"`):
            Sparse attention implementation name.
        sparse_block_size (`int`, *optional*, defaults to 64):
            Block size (tokens per block) for block-sparse.
        sparse_local_num_blocks (`int`, *optional*, defaults to 4):
            Number of local (lookback) blocks per query block (causal).
        sparse_global_num_blocks (`int`, *optional*, defaults to 2):
            Number of global summary blocks per query block (causal, strided).
        sparse_global_block_stride (`int`, *optional*, defaults to 4):
            Stride in blocks for selecting global summary blocks.
        sparse_attention_window (`int`, *optional*, defaults to 256):
            Max token window for sliding/sparse layers. Used to cap local blocks (TPU-friendly).
        attention_compute_dtype (`str`, *optional*, defaults to `"fp32"`):
            Compute dtype for attention math (QK^T + softmax + PV). `"fp32"` recommended for stability on TPU.
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
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        attn_implementation: str = "eager",
        layer_types: Optional[list[str]] = None,
        use_sparse_attention: bool = False,
        sparse_attention_impl: str = "local_global_block",
        sparse_block_size: int = 64,
        sparse_local_num_blocks: int = 4,
        sparse_global_num_blocks: int = 2,
        sparse_global_block_stride: int = 4,
        sparse_attention_window: int = 256,
        attention_compute_dtype: str = "fp32",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.rope_theta = float(rope_theta)

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        self.attn_implementation = attn_implementation

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        self.layer_types = layer_types

        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_impl = sparse_attention_impl
        self.sparse_block_size = int(sparse_block_size)
        self.sparse_local_num_blocks = int(sparse_local_num_blocks)
        self.sparse_global_num_blocks = int(sparse_global_num_blocks)
        self.sparse_global_block_stride = int(sparse_global_block_stride)
        self.sparse_attention_window = int(sparse_attention_window)

        self.attention_compute_dtype = attention_compute_dtype

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["HumanVConfig"]
