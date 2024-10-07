from typing import List

from transformers import PretrainedConfig, AutoTokenizer


class MolmoConfig(PretrainedConfig):
    model_type = "molmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        embedding_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        layer_norm_eps: float = 1e-5,
        rope_theta=10000.0,
        clip_qkv=None,
        qkv_bias: bool = False,
        weight_tying: bool = False,
        use_position_ids: bool=True,
        tie_word_embeddings: bool=True,
        attention_layer_norm: bool=False,
        norm_after: bool = False,
        layer_norm_type: str="rms",
        moe_num_experts=None,
        moe_top_k=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.weight_tying = weight_tying
        self.use_position_ids = use_position_ids
        self.attention_layer_norm = attention_layer_norm
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.clip_qkv = clip_qkv
        self.qkv_bias = qkv_bias
        self.norm_after = norm_after
        self.tie_word_embeddings = tie_word_embeddings
        self.layer_norm_type = layer_norm_type
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


MolmoConfig.register_for_auto_class()