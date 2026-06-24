# coding=utf-8
# Copyright 2024 The OpenPangu Team and The HuggingFace Inc. team. All rights reserved.
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
""" OpenPangu_v2 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OpenPanguV2Config(PretrainedConfig):
    model_type = "openpangu_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `OpenPangu_v2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "rowwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        moe_intermediate_size: int | None = None,
        num_hidden_layers: int | None = 0,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        use_mla: bool | None = False,
        n_shared_experts: int | None = None,
        n_routed_experts: int | None = None,
        routed_scaling_factor: float | None = None,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        qk_rope_head_dim: int | None = None,
        qk_nope_head_dim: int | None = None,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        num_experts_per_tok: int | None = None,
        first_k_dense_replace: int | None = 0,
        norm_topk_prob: bool | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = None,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        rope_interleave: bool | None = False,
        sliding_window: int | list[int] | None = None,
        swa_layers: list[str] | None = None,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        attention_bias: bool | None = False,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        param_sink_number: int | None = 0,
        attn_groupnorm: bool | None = False,
        attn_elementwise_gate: bool | None = False,
        router_sliding_window: int | None = 0,
        sandwich_norm: bool | None = False,
        block_post_layernorm_idx: list[int] | None = None,
        use_mhc: bool | None = False,
        mhc_use_gamma: bool | None = None,
        mhc_recur_norm: int | None = None,
        mhc_num_stream: int | None = None,
        vanilla_mlp: bool | None = False,
        attn_k_layernorm: bool | None = False,
        dsa_layers: list[str] | None = None,
        index_topk: int | None = None,
        index_head_dim: int | None = None,
        index_n_heads: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.layer_types = layer_types

        self.use_mla = use_mla
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        if qk_rope_head_dim is not None and qk_nope_head_dim is not None:
            self.head_dim = qk_rope_head_dim
            self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sliding_window = sliding_window
        self.swa_layers = swa_layers
        self.tie_word_embeddings = tie_word_embeddings

        self.param_sink_number = param_sink_number
        self.attn_groupnorm = attn_groupnorm
        self.attn_elementwise_gate = attn_elementwise_gate
        self.router_sliding_window = router_sliding_window
        self.sandwich_norm = sandwich_norm
        self.block_post_layernorm_idx = block_post_layernorm_idx
        self.use_mhc = use_mhc
        self.mhc_use_gamma = mhc_use_gamma
        self.mhc_recur_norm = mhc_recur_norm
        self.mhc_num_stream = mhc_num_stream
        self.vanilla_mlp = vanilla_mlp
        self.attn_k_layernorm = attn_k_layernorm
        
        # Indexer (DSA) parameters
        self.dsa_layers = dsa_layers
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        
        if self.layer_types is None:
            if self.swa_layers is not None:
                self.layer_types = [
                    "sliding_attention"
                    if i in self.swa_layers
                    else "full_attention"
                    for i in range(self.num_hidden_layers)
                ]
            else:
                self.layer_types = ["full_attention" for _ in range(self.num_hidden_layers)]
        
        if num_hidden_layers is not None and self.layer_types is not None and len(self.layer_types) != num_hidden_layers:
            raise ValueError(
                f"`num_hidden_layers` ({num_hidden_layers}) must be equal to the number of layer types "
                f"({len(layer_types)})"
            )

        super().__init__(**kwargs)
