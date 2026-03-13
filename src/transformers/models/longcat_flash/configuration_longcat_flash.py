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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="meituan-longcat/LongCat-Flash-Chat")
class LongcatFlashConfig(PreTrainedConfig):
    r"""
    ffn_hidden_size (`int`, *optional*, defaults to 12288):
        Dimension of the MLP representations.
    zero_expert_num (`int`, *optional*, defaults to 256):
        Number of zero experts (identity function) to add to the expert pool.
    expert_ffn_hidden_size (`int`, *optional*, defaults to 2048):
        Hidden size of individual expert FFN layers.
    qk_head_dim (`int`, *optional*):
        The total dimension of query/key heads. If not specified, set to `qk_nope_head_dim + qk_rope_head_dim`.
    moe_topk (`int`, *optional*, defaults to 12):
        Number of experts to route to for each token in the MoE layer.

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
    default_theta = 10000000.0
    base_model_tp_plan = {
        "layers.*.self_attn.*.q_b_proj": "colwise",
        "layers.*.self_attn.*.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.*.kv_b_proj": "colwise",
        "layers.*.self_attn.*.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts.identity_expert": "moe_identity_expert",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlps.*.gate_proj": "colwise",
        "layers.*.mlps.*.up_proj": "colwise",
        "layers.*.mlps.*.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 131072,
        hidden_size: int | None = 6144,
        num_hidden_layers: int | None = 56,
        num_layers: int | None = 28,
        num_attention_heads: int | None = 64,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        ffn_hidden_size: int | None = 12288,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int | None = 512,
        qk_nope_head_dim: int | None = 128,
        qk_rope_head_dim: int | None = 64,
        head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        qk_head_dim: int | None = None,
        moe_topk: int | None = 12,
        n_routed_experts: int | None = 512,
        zero_expert_num: int | None = 256,
        expert_ffn_hidden_size: int | None = 2048,
        routed_scaling_factor: float | None = 6.0,
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
        self.rope_parameters = rope_parameters

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation: set | None = None, **kwargs):
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


__all__ = ["LongcatFlashConfig"]
