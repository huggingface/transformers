# coding=utf-8
# Copyright 2025 DeepSeek AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeek-V3.2-Exp implementation from DeepSeek AI.
# Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
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
"""DeepSeek V3.2 model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters


class DeepseekV32Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV32Model`]. It is used to instantiate
    a DeepSeek V3.2 model according to the specified arguments, defining the model architecture.

    DeepSeek V3.2 extends DeepSeek V3 with DeepSeek Sparse Attention (DSA), which uses a Lightning Indexer
    to select top-k tokens for sparse attention, reducing complexity from O(L^2) to O(L*k).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the DeepSeek V3.2 model.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 128):
            Number of key_value heads for Grouped Query Attention.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts (always active).
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of query/key heads without rotary position embeddings.
        n_group (`int`, *optional*, defaults to 8):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 4):
            Number of groups selected per token for expert routing.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts activated per token.
        first_k_dense_replace (`int`, *optional*, defaults to 3):
            Number of dense layers before switching to MoE layers.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Configuration for the RoPE embeddings.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings (for MLA).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of heads in the Lightning Indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each indexer head.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of tokens to select for sparse attention.
        use_sparse_attention (`bool`, *optional*, defaults to True):
            Whether to use sparse attention. Set to False for dense attention
            (useful for the dense warm-up training stage).
        detach_indexer_input (`bool`, *optional*, defaults to False):
            Whether to detach the indexer input from the computational graph.
            Used in Stage 2 training for separate optimization of indexer.

    Example:

    ```python
    >>> from transformers import DeepseekV32Model, DeepseekV32Config

    >>> # Initializing a DeepSeek V3.2 style configuration
    >>> configuration = DeepseekV32Config()

    >>> # Initializing a model from the configuration
    >>> model = DeepseekV32Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "deepseek_v32"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 129280,
        hidden_size: Optional[int] = 7168,
        intermediate_size: Optional[int] = 18432,
        moe_intermediate_size: Optional[int] = 2048,
        num_hidden_layers: Optional[int] = 61,
        num_attention_heads: Optional[int] = 128,
        num_key_value_heads: Optional[int] = 128,
        n_shared_experts: Optional[int] = 1,
        n_routed_experts: Optional[int] = 256,
        routed_scaling_factor: Optional[float] = 2.5,
        kv_lora_rank: Optional[int] = 512,
        q_lora_rank: Optional[int] = 1536,
        qk_rope_head_dim: Optional[int] = 64,
        v_head_dim: Optional[int] = 128,
        qk_nope_head_dim: Optional[int] = 128,
        n_group: Optional[int] = 8,
        topk_group: Optional[int] = 4,
        num_experts_per_tok: Optional[int] = 8,
        first_k_dense_replace: Optional[int] = 3,
        norm_topk_prob: Optional[bool] = True,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 4096,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        pretraining_tp: Optional[int] = 1,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        rope_interleave: Optional[bool] = True,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        # DeepSeek V3.2 specific: Lightning Indexer
        index_n_heads: Optional[int] = 64,
        index_head_dim: Optional[int] = 128,
        index_topk: Optional[int] = 2048,
        use_sparse_attention: Optional[bool] = True,
        detach_indexer_input: Optional[bool] = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        # DeepSeek V3.2 specific: Lightning Indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.use_sparse_attention = use_sparse_attention
        self.detach_indexer_input = detach_indexer_input

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation: Optional[set] = None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)

        # Convert to float because RoPE fn expect a float. Models on the hub were saved as int
        for key in ["beta_fast", "beta_slow", "factor"]:
            if key in self.rope_parameters:
                self.rope_parameters[key] = float(self.rope_parameters[key])
        return kwargs


__all__ = ["DeepseekV32Config"]
