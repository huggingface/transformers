from typing import Optional

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
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
        rope_parameters: Optional[RopeParameters | dict] = None,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        self.layer_types = layer_types if layer_types is not None else ["full_attention"] * self.num_hidden_layers
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        elif isinstance(rope_parameters, dict):
            rp = dict(rope_parameters)
            rp.setdefault("rope_type", "default")
            rp.setdefault("rope_theta", rope_theta)
            rope_parameters = rp
        self.rope_parameters = rope_parameters

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["HumanVConfig"]
