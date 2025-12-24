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
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads. If equal to `num_attention_heads`, the model uses MHA. If set to `1`, the model
            uses MQA. Otherwise it uses GQA.
        head_dim (`int`, *optional*, defaults to 32):
            Dimension of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function used in the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length supported by the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation used for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon used by RMSNorm layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the key/value cache during generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the input and output embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias terms in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied to attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias terms in MLP projections.
        rope_parameters (`dict`, *optional*):
            RoPE parameters dict. Common key: `rope_theta`.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type. Supported values: `"full_attention"`, `"sliding_attention"`.
        attn_backend (`str`, *optional*, defaults to `"sdpa"`):
            Backend for dense attention. Supported values: `"sdpa"`, `"matmul"`.
        sliding_window (`int`, *optional*, defaults to 0):
            Sliding window size for `"sliding_attention"` layers when using dense mode.
        use_sparse_attention (`bool`, *optional*, defaults to `False`):
            Whether to enable sparse attention for `"sliding_attention"` layers.
        sparse_attention_impl (`str`, *optional*, defaults to `"local_global_block"`):
            Sparse attention implementation selector.
        sparse_block_size (`int`, *optional*, defaults to 64):
            Block size (tokens) for block-sparse attention.
        sparse_prefill_chunk_blocks (`int`, *optional*, defaults to 0):
            If > 0 and `use_cache=True`, splits long prefill into chunks of `sparse_prefill_chunk_blocks * sparse_block_size`.
        sparse_local_num_blocks (`int`, *optional*, defaults to 4):
            Number of local blocks per query block.
        sparse_global_num_blocks (`int`, *optional*, defaults to 2):
            Number of global blocks per query block.
        sparse_global_block_stride (`int`, *optional*, defaults to 4):
            Stride (in blocks) for selecting global blocks.
        sparse_attention_window (`int`, *optional*, defaults to 0):
            Optional window size (tokens) to limit selection range for sparse attention.

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
        vocab_size: int = 50257,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        head_dim: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        rope_parameters: Optional[dict] = None,
        layer_types: Optional[list[str]] = None,
        attn_backend: str = "sdpa",
        sliding_window: int = 0,
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
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.rope_parameters = rope_parameters
        self.layer_types = layer_types
        self.attn_backend = attn_backend
        self.sliding_window = sliding_window
        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_impl = sparse_attention_impl
        self.sparse_block_size = sparse_block_size
        self.sparse_prefill_chunk_blocks = sparse_prefill_chunk_blocks
        self.sparse_local_num_blocks = sparse_local_num_blocks
        self.sparse_global_num_blocks = sparse_global_num_blocks
        self.sparse_global_block_stride = sparse_global_block_stride
        self.sparse_attention_window = sparse_attention_window

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HumanVConfig"]
