# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

"""openPanguUltraMoE 718B model configuration"""

from ...configuration_utils import PreTrainedConfig

class OpenPanguMoEConfig(PreTrainedConfig):

    model_type = "pangu_ultra_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=153600,
        hidden_size=7680,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_mtp_layers=1,
        num_attention_heads=128,
        num_key_value_heads=128,
        num_shared_experts=1,
        num_routed_experts=256,
        routed_scaling_factor=2.5,
        attention_kv_lora_dim=512,
        attention_q_lora_dim=1536,
        attention_qk_rope_dim=64,
        attention_v_dim=128,
        attention_qk_dim=128,
        num_experts_per_tok=8,
        num_dense_layers=3,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=25600000,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.num_dense_layers = num_dense_layers
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.attention_kv_lora_dim = attention_kv_lora_dim
        self.attention_q_lora_dim = attention_q_lora_dim
        self.attention_qk_rope_dim = attention_qk_rope_dim
        self.attention_v_dim = attention_v_dim
        self.attention_qk_dim = attention_qk_dim
        self.attention_dropout = attention_dropout
        self.num_mtp_layers = num_mtp_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )