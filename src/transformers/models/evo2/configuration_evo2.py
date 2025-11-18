"""Evo2 model configuration."""

from __future__ import annotations

from typing import Optional, Sequence, List, Dict, Any

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import standardize_rope_params
from ...utils import logging


logger = logging.get_logger(__name__)

__all__ = ["Evo2Config"]


class Evo2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Evo2Model`]. It is used to instantiate an Evo2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Evo2-1b-base model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the Evo2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Evo2Model`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use the same number of key/value heads as the number of
            query heads.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        attn_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden units.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the MLP layers.
        layer_types (`Sequence[str]`, *optional*):
            List of layer types ("attention" or "hyena") for each layer. If None, defaults to all "attention".
        hyena_filters (`int`, *optional*, defaults to 256):
            Number of Hyena filter groups.
        hyena_kernel_size (`int`, *optional*, defaults to 8):
            Kernel size for the short convolution in Hyena.
        hyena_hidden_size (`int`, *optional*):
            Hidden size for Hyena layers.
        hyena_order (`int`, *optional*, defaults to 4):
            Order of the Hyena recurrence.
        hyena_flip_x1x2 (`bool`, *optional*, defaults to False):
            Whether to flip x1 and x2 in the Hyena gating mechanism.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to True):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 0):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether to tie weight embeddings
    """

    model_type = "evo2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 512,
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
        hyena_flip_x1x2: bool = False,
        hyena_filter_configurations: Optional[List[Dict[str, Any]]] = None,
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
        self.hyena_flip_x1x2 = hyena_flip_x1x2
        self.hyena_filter_configurations = hyena_filter_configurations
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
