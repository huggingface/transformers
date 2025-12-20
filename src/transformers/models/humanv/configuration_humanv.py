from typing import Optional, List
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, standardize_rope_params
from ...utils import logging

logger = logging.get_logger(__name__)

class HumanVConfig(PreTrainedConfig):
    model_type = "humanv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = 16,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: Optional[List[str]] = None,
        sparse_window_size: int = 512,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        
        self.sparse_window_size = sparse_window_size
        
        if layer_types is None:
            self.layer_types = ["full_attention"] * num_hidden_layers
        else:
            self.layer_types = layer_types
            
        self.rope_parameters = rope_parameters
        rope_theta = kwargs.get("rope_theta", 10000.0)
        standardize_rope_params(self, rope_theta=rope_theta)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ["HumanVConfig"]
