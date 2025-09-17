# coding=utf-8
# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved.
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

"""LongCat Flash model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class LongcatFlashConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongcatFlashModel`]. It is used to instantiate
    a LongCat Flash model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LongCat Flash architecture.
    e.g. [meituan-longcat/LongCat-Flash-Chat](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the LongCat Flash model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`LongcatFlashModel`]
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 56):
            Number of hidden layers in the Transformer decoder.
        num_layers (`int`, *optional*, defaults to 28):
            number of layers, each with 2 sublayers.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting from a multi-head checkpoint to a GQA checkpoint, each group key and value head should be
            constructed by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        ffn_hidden_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            The rank of the query LoRA projection in MLA (Multi-head Latent Attention).
        kv_lora_rank (`int`, *optional*, defaults to 512):
            The rank of the key-value LoRA projection in MLA.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            The dimension of the non-position encoding part of query/key heads.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            The dimension of the RoPE part of query/key heads.
        head_dim (`int`, *optional*, defaults to 64):
            Standard dimension of qk heads, unused except for CI.
        v_head_dim (`int`, *optional*, defaults to 128):
            The dimension of value heads.
        qk_head_dim (`int`, *optional*):
            The total dimension of query/key heads. If not specified, set to `qk_nope_head_dim + qk_rope_head_dim`.
        moe_topk (`int`, *optional*, defaults to 12):
            Number of experts to route to for each token in the MoE layer.
        n_routed_experts (`int`, *optional*, defaults to 512):
            Number of routed experts in the MoE layer.
        zero_expert_num (`int`, *optional*, defaults to 256):
            Number of zero experts (identity function) to add to the expert pool.
        expert_ffn_hidden_size (`int`, *optional*, defaults to 2048):
            Hidden size of individual expert FFN layers.
        routed_scaling_factor (`float`, *optional*, defaults to 6.0):
            Scaling factor applied to the routing weights.

    ```python
    >>> from transformers import LongcatFlashModel, LongcatFlashConfig

    >>> # Initializing a LongCat Flash style configuration
    >>> configuration = LongcatFlashConfig()

    >>> # Initializing a model from the configuration
    >>> model = LongcatFlashModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "longcat_flash"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.*.q_b_proj": "colwise",
        "layers.*.self_attn.*.kv_b_proj": "colwise",
        "layers.*.self_attn.*.o_proj": "rowwise",
        "layers.*.mlps.*.gate_proj": "colwise",
        "layers.*.mlps.*.up_proj": "colwise",
        "layers.*.mlps.*.down_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=6144,
        num_hidden_layers=56,
        num_layers=28,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        ffn_hidden_size=12288,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        head_dim=64,
        v_head_dim=128,
        qk_head_dim=None,
        moe_topk=12,
        n_routed_experts=512,
        zero_expert_num=256,
        expert_ffn_hidden_size=2048,
        routed_scaling_factor=6.0,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        if qk_head_dim is None:
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.ffn_hidden_size = ffn_hidden_size

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_head_dim
        self.head_dim = head_dim

        self.moe_topk = moe_topk
        self.n_routed_experts = n_routed_experts
        self.zero_expert_num = zero_expert_num
        self.expert_ffn_hidden_size = expert_ffn_hidden_size
        self.routed_scaling_factor = routed_scaling_factor

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if self.rope_scaling is not None:
            for key in ["beta_fast", "beta_slow", "factor"]:
                if key in self.rope_scaling:
                    self.rope_scaling[key] = float(self.rope_scaling[key])

        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["LongcatFlashConfig"]
