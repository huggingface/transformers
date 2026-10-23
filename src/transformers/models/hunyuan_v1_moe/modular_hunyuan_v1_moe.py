# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HunYuanMoEV1 model."""

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2Model, DeepseekV2Moe
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1Attention, HunYuanDenseV1RotaryEmbedding
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForSequenceClassification,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ..mistral.modeling_mistral import MistralMLP
from ..mixtral.modeling_mixtral import MixtralExperts
from .configuration_hunyuan_v1_moe import HunYuanMoEV1Config


logger = logging.get_logger(__name__)


class HunYuanMoEV1RMSNorm(LlamaRMSNorm):
    pass


class HunYuanMoEV1MLP(MistralMLP):
    pass


class HunYuanMoEV1Attention(HunYuanDenseV1Attention):
    pass


class HunYuanMoEV1Gate(nn.Module):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        self.top_k = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[layer_idx]
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, dtype=torch.float32))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k_weights, top_k_index = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return router_logits, top_k_weights.to(router_logits.dtype), top_k_index


class HunYuanMoEV1Experts(MixtralExperts):
    pass


class HunYuanMoEV1Moe(DeepseekV2Moe):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int | None = None):
        super().__init__(config)
        self.gate = HunYuanMoEV1Gate(config, layer_idx=layer_idx)
        self.shared_experts = HunYuanMoEV1MLP(config)


class HunYuanMoEV1DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanMoEV1Attention(config=config, layer_idx=layer_idx)
        self.mlp = HunYuanMoEV1Moe(config, layer_idx=layer_idx)
        self.input_layernorm = HunYuanMoEV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanMoEV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx


class HunYuanMoEV1PreTrainedModel(LlamaPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": HunYuanMoEV1DecoderLayer,
        "attentions": HunYuanMoEV1Attention,
        "router_logits": OutputRecorder(HunYuanMoEV1Gate, index=0),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, HunYuanMoEV1Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, HunYuanMoEV1Gate):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class HunYuanMoEV1RotaryEmbedding(HunYuanDenseV1RotaryEmbedding):
    pass


class HunYuanMoEV1Model(DeepseekV2Model):
    pass


class HunYuanMoEV1ForCausalLM(DeepseekV2ForCausalLM):
    pass


class HunYuanMoEV1ForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "HunYuanMoEV1ForCausalLM",
    "HunYuanMoEV1Model",
    "HunYuanMoEV1PreTrainedModel",
    "HunYuanMoEV1ForSequenceClassification",
]
