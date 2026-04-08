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

from huggingface_hub.dataclasses import strict
from torch import nn

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
@strict
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

    vocab_size: int = 196608
    moe_intermediate_size: int = 1280
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    head_dim: int = 128
    num_experts_per_tok: int = 8
    intermediate_size = AttributeError()
    first_k_dense_replace = AttributeError()
    use_qk_norm = AttributeError()

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 1.0)
        super().__post_init__(**kwargs)


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
