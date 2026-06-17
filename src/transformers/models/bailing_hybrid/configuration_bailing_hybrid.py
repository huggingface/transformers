# Copyright 2025 InclusionAI and the HuggingFace Inc. team. All rights reserved.
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
"""BailingHybrid model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="inclusionAI/Ring-2.5-1T")
@strict
class BailingHybridConfig(PreTrainedConfig):
    r"""
    layer_group_size (`int`, *optional*, defaults to 8):
        Controls the hybrid layer pattern. Every `layer_group_size`-th layer uses full MLA attention,
        while the rest use lightning linear attention.
    n_group (`int`, *optional*, defaults to 8):
        Number of groups for routed experts in group-limited-greedy routing.
    first_k_dense_replace (`int`, *optional*, defaults to 4):
        Number of initial dense layers before switching to MoE.
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to interleave the rotary position embeddings.
    group_norm_size (`int`, *optional*, defaults to 8):
        Group size for group RMS normalization in linear attention layers.
    num_kv_heads_for_linear_attn (`int`, *optional*, defaults to 64):
        Number of key-value heads used in linear attention layers.
    linear_silu (`bool`, *optional*, defaults to `False`):
        Whether to apply SiLU activation on the gate in linear attention.
    moe_shared_expert_intermediate_size (`int`, *optional*, defaults to 2048):
        Intermediate size of the shared expert in MoE layers.
    topk_method (`str`, *optional*, defaults to `"noaux_tc"`):
        Method for selecting top-k experts in the MoE layer.
    scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
        Scoring function for the router in the MoE layer.
    partial_rotary_factor (`float`, *optional*, defaults to 0.5):
        Fraction of the head dimension to apply rotary position embeddings in linear attention layers.
    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
        Coefficient for the auxiliary load balancing loss from the router.

    Example:

    ```python
    >>> from transformers import BailingHybridModel, BailingHybridConfig

    >>> # Initializing a BailingHybrid style configuration
    >>> configuration = BailingHybridConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bailing_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
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
        "num_local_experts": "num_experts",
    }

    vocab_size: int = 157184
    hidden_size: int = 8192
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int | None = 64
    num_experts: int = 256
    num_shared_experts: int = 1
    num_experts_per_tok: int | None = 8
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int | None = 128
    qk_nope_head_dim: int = 128
    n_group: int | None = 8
    topk_group: int | None = 4
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    first_k_dense_replace: int | None = 4
    norm_topk_prob: bool | None = True
    layer_group_size: int = 8
    group_norm_size: int = 8
    num_kv_heads_for_linear_attn: int = 64
    linear_silu: bool = False
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 156892
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 156892
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    rope_interleave: bool | None = True
    partial_rotary_factor: float = 0.5
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    use_qk_norm: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim

        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.layer_group_size == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        self.standardize_rope_params()

        for key in ["beta_fast", "beta_slow", "factor"]:
            if key in self.rope_parameters:
                self.rope_parameters[key] = float(self.rope_parameters[key])
        return kwargs


__all__ = ["BailingHybridConfig"]
