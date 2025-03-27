#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/deepseek_v2/modular_deepseek_V2.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_deepseek_V2.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 Baidu Inc and The HuggingFace Inc. team.
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


from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class DeepseekV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV2Model`]. It is used to instantiate a DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of DeepSeek-V2-Lite" [deepseek-ai/DeepSeek-V2-Lite"](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite").
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the DeepSeek model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`DeepseekV2Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            The number of key-value heads used to implement Grouped Query Attention (GQA). If
            `num_key_value_heads=num_attention_heads`, the model will use Multi-Head Attention (MHA). If
            `num_key_value_heads=1`, the model will use Multi-Query Attention (MQA). Otherwise, GQA is used.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon value used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/value attentions (useful for inference optimization).
        pad_token_id (`int`, *optional*):
            Padding token ID.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning-of-sequence token ID.
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token ID.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Tensor parallelism rank used during pretraining for efficient distributed training.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the Rotary Position Embeddings (RoPE).
        rope_scaling (`Dict`, *optional*):
            Configuration for scaling RoPE embeddings. Supports `linear` and `dynamic` scaling strategies.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value, and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to attention weights.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias term in the MLP layers.
        head_dim (`int`, *optional*, defaults to `qk_rope_head_dim`):
            The attention head dimension.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Weight coefficient for auxiliary loss in Mixture of Experts (MoE) models.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in the shallow layers before switching to MoE layers.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA decomposition for key-value projections.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the LoRA decomposition for query projections.
            Specifically, it determines the dimensionality to which the query (q) vectors are compressed before being expanded back to their original size.
            It reduces computational overhead while maintaining model performance.
        n_group (`int`, *optional*):
            Number of groups for routed experts.
        n_routed_experts (`int`, *optional*):
            Number of routed experts (None indicates a dense model).
        n_shared_experts (`int`, *optional*):
            Number of shared experts (None indicates a dense model).
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            The head dimension for the QK (query-key) projections when using NOPE (Neural Operator Position Encoding).
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            The head dimension for QK projections when using RoPE.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts in MoE models.
        seq_aux (`bool`, *optional*, defaults to `True`):
            Whether to compute the auxiliary loss for each individual sequence.
        topk_group (`int`, *optional*):
            Number of selected groups per token for expert selection.
        topk_method (`str`, *optional*, defaults to `"greedy"`):
            The method used for selecting top-k experts in the routed gate mechanism.
        v_head_dim (`int`, *optional*, defaults to 128):
            The dimension of value projections in the attention layers.
        num_experts_per_tok (`int`, *optional*):
            The number of experts selected per token. If `None`, the model behaves as a dense Transformer.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the probability distribution over top-k selected experts.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE (Mixture of Experts) representations.

    ```python
    >>> from transformers import DeepseekV2Model, DeepseekV2Config
    >>> # Initializing a DeepSeek-V2 style configuration
    >>> configuration = DeepseekV2Config()
    >>> # Accessing the model configuration
    >>> model = DeepseekV2Model(configuration)
    >>> print(model.config)
    ```
    """

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.q_a_proj": "colwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        aux_loss_alpha=0.001,
        first_k_dense_replace=0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        n_group=None,
        n_routed_experts=None,
        n_shared_experts=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        routed_scaling_factor=1.0,
        seq_aux=True,
        topk_group=None,
        topk_method="greedy",
        v_head_dim=128,
        num_experts_per_tok=None,
        norm_topk_prob=False,
        moe_intermediate_size=1407,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else qk_rope_head_dim
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.aux_loss_alpha = aux_loss_alpha
        self.first_k_dense_replace = first_k_dense_replace
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.seq_aux = seq_aux
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.v_head_dim = v_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.moe_intermediate_size = moe_intermediate_size


__all__ = ["DeepseekV2Config"]
