# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, is_torch_greater_or_equal
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from ..exaone4.modeling_exaone4 import Exaone4Attention
from ..glm4.modeling_glm4 import Glm4DecoderLayer
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeModel
from ..qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoePreTrainedModel,
    Qwen3MoeTopKRouter,
)


@auto_docstring(checkpoint="Aleph-Alpha/Kolibri-1")
@strict
class Kolibri1Config(Qwen3MoeConfig):
    r"""
    ```python
    >>> from transformers import Kolibri1Model, Kolibri1Config

    >>> configuration = Kolibri1Config()
    >>> model = Kolibri1Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "kolibri1"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }

    vocab_size: int = 128000
    hidden_size: int = 2560
    num_hidden_layers: int = 50
    num_attention_heads: int = 48
    head_dim: int = 128
    max_position_embeddings: int = 262144
    sliding_window: int | None = 513
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 6
    num_experts: int = 384
    layer_types: list[str] | None = None

    intermediate_size = AttributeError()
    use_sliding_window = AttributeError()
    decoder_sparse_step = AttributeError()
    mlp_only_layers = AttributeError()
    attention_bias = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        PreTrainedConfig.__post_init__(self, **kwargs)


class Kolibri1Attention(Exaone4Attention):
    def __init__(self, config: Kolibri1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.sliding_window_pattern


class Kolibri1MLP(Qwen3MoeMLP):
    pass


class Kolibri1Experts(Qwen3MoeExperts):
    pass


class Kolibri1TopKRouter(Qwen3MoeTopKRouter):
    def __init__(self, config):
        super().__init__(config)
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        if hidden_states.is_cuda and not torch.is_grad_enabled() and is_torch_greater_or_equal("2.8.0"):
            # Same fp32 logits as the upcast below (bf16/fp16 products are exact in fp32); CUDA-only, no autograd
            router_logits = torch.mm(hidden_states, self.weight.t(), out_dtype=torch.float32)
        else:
            router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        router_indices = torch.topk(router_logits + self.e_score_correction_bias, self.top_k, dim=-1)[1]
        router_scores = router_logits.gather(1, router_indices).sigmoid()
        if self.norm_topk_prob:
            router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)
        return router_logits, router_scores, router_indices


class Kolibri1SparseMoeBlock(DeepseekV3MoE):
    def __init__(self, config: Kolibri1Config):
        super().__init__(config)
        self.gate = Kolibri1TopKRouter(config)
        self.shared_experts = Kolibri1MLP(config, intermediate_size=config.shared_expert_intermediate_size)


class Kolibri1DecoderLayer(Glm4DecoderLayer):
    def __init__(self, config: Kolibri1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = Kolibri1SparseMoeBlock(config)


class Kolibri1PreTrainedModel(Qwen3MoePreTrainedModel):
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Kolibri1TopKRouter):
            init.zeros_(module.e_score_correction_bias)


class Kolibri1Model(Qwen2MoeModel):
    pass


class Kolibri1ForCausalLM(Qwen3MoeForCausalLM):
    pass


__all__ = [
    "Kolibri1Config",
    "Kolibri1ForCausalLM",
    "Kolibri1Model",
    "Kolibri1PreTrainedModel",
]
