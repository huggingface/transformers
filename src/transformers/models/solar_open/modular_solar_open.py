# Copyright 2026 Upstage and HuggingFace Inc. team.
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
"""PyTorch SolarOpen model."""

from torch import nn

from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoeMoE,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
)
from ..llama.modeling_llama import LlamaAttention, LlamaDecoderLayer


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="upstage/Solar-Open-100B")
class SolarOpenConfig(Glm4MoeConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    """

    model_type = "solar_open"
    default_theta = 1_000_000.0

    # Default tensor parallel plan for base model `SolarOpenModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    def __init__(
        self,
        vocab_size: int = 196608,
        hidden_size: int = 4096,
        moe_intermediate_size: int = 1280,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 128,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: int = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: RopeParameters | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        # Default partial_rotary_factor to 1.0 (instead of 0.5 in Glm4MoeConfig).
        # `setdefault` ensures this value is not overridden by subsequent calls.
        # This workaround is required due to modular inheritance limitations.
        kwargs.setdefault("partial_rotary_factor", 1.0)
        self.head_dim = head_dim

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            moe_hidden_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            n_shared_experts=n_shared_experts,
            n_routed_experts=n_routed_experts,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        del self.intermediate_size
        del self.first_k_dense_replace
        del self.use_qk_norm


class SolarOpenDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: SolarOpenConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = SolarOpenMoE(config)


class SolarOpenMoE(Glm4MoeMoE):
    pass


class SolarOpenAttention(LlamaAttention):
    def __init__(self, config: SolarOpenConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


class SolarOpenRMSNorm(Glm4MoeRMSNorm):
    pass


class SolarOpenPreTrainedModel(Glm4MoePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = None


class SolarOpenModel(Glm4MoeModel):
    pass


class SolarOpenForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "SolarOpenConfig",
    "SolarOpenPreTrainedModel",
    "SolarOpenModel",
    "SolarOpenForCausalLM",
]
