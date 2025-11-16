"""Evo2 model configuration."""

from __future__ import annotations

from typing import Optional, Sequence

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import standardize_rope_params
from ...utils import logging


logger = logging.get_logger(__name__)

__all__ = ["Evo2Config"]


class Evo2Config(PretrainedConfig):
    r"""Configuration class for the Evo2 model."""

    model_type = "evo2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 2048,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        attn_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        layer_types: Optional[Sequence[str]] = None,
        hyena_filters: int = 256,
        hyena_kernel_size: int = 8,
        hyena_hidden_size: Optional[int] = None,
        hyena_order: int = 4,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id: Optional[int] = None,
        eos_token_id: int = 0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else hidden_size * 4
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attn_dropout = attn_dropout
        self.hidden_dropout = hidden_dropout
        self.mlp_dropout = mlp_dropout
        self.hyena_filters = hyena_filters
        self.hyena_kernel_size = hyena_kernel_size
        self.hyena_hidden_size = hyena_hidden_size if hyena_hidden_size is not None else hidden_size
        self.hyena_order = hyena_order
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        if layer_types is None:
            self.layer_types = ["attention"] * num_hidden_layers
        else:
            self.layer_types = list(layer_types)

        standardize_rope_params(self, rope_theta=self.rope_theta)

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "The length of `layer_types` must match `num_hidden_layers` (received"
                f" {len(self.layer_types)} and {self.num_hidden_layers})."
            )

        for layer_type in self.layer_types:
            if layer_type not in {"attention", "hyena"}:
                raise ValueError(f"Unsupported layer type: {layer_type}. Expected 'attention' or 'hyena'.")

        if self.num_attention_heads <= 0 or self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")

        if self.num_key_value_heads <= 0 or self.hidden_size % self.num_key_value_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_key_value_heads`.")

        logger.info("Initialized Evo2Config with %s layers (%s).", self.num_hidden_layers, ", ".join(self.layer_types))

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_head_dim(self) -> int:
        return self.hidden_size // self.num_key_value_heads

    @property
    def num_attention_layers(self) -> int:
        return sum(layer_type == "attention" for layer_type in self.layer_types)

    @property
    def num_hyena_layers(self) -> int:
        return sum(layer_type == "hyena" for layer_type in self.layer_types)

    def to_dict(self) -> dict:
        output = super().to_dict()
        output["layer_types"] = list(self.layer_types)
        return output
