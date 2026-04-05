from typing import Optional, Union

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
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
        rope_partial_rotary_factor: Optional[float] = 1.0,
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
        attn_backend: Optional[str] = "gqa_matmul",
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
        pad_token_id: Optional[int] = 50256,
        bos_token_id: Optional[int] = 50256,
        eos_token_id: Optional[int] = 50256,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = int(num_key_value_heads)

        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be > 0, got {self.num_key_value_heads}")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot exceed num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )

        self.head_dim = int(head_dim)
        self.hidden_act = str(hidden_act)
        self.activation_backend = str(activation_backend)

        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_parameters = rope_parameters
        self.rope_partial_rotary_factor = float(rope_partial_rotary_factor)
        self.use_selective_rope = bool(use_selective_rope)
        self.selective_rope_scale = float(selective_rope_scale)

        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.norm_backend = str(norm_backend)

        self.use_cache = bool(use_cache)
        self.kv_cache_dtype = str(kv_cache_dtype)

        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.mlp_bias = bool(mlp_bias)

        self.attn_implementation = str(attn_implementation)

        attn_backend = str(attn_backend).lower().strip()
        if attn_backend not in ("gqa_matmul", "sdpa"):
            attn_backend = "gqa_matmul"
        self.attn_backend = attn_backend

        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = int(sliding_window) if self.use_sliding_window else None
        self.max_window_layers = int(max_window_layers)

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

        self.use_sparse_attention = bool(use_sparse_attention)
        self.sparse_attention_impl = str(sparse_attention_impl)
        self.sparse_block_size = int(sparse_block_size)
        self.sparse_prefill_chunk_blocks = int(sparse_prefill_chunk_blocks)
        self.sparse_local_num_blocks = int(sparse_local_num_blocks)
        self.sparse_global_num_blocks = int(sparse_global_num_blocks)
        self.sparse_global_block_stride = int(sparse_global_block_stride)
        self.sparse_attention_window = int(sparse_attention_window)

        super().__init__(
            tie_word_embeddings=bool(tie_word_embeddings),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


__all__ = ["HumanVConfig"]
