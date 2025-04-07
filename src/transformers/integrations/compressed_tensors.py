# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import Llama4TextMLP


def skip(*args, **kwargs):
    pass


class CompressedExpertsLinear(nn.Module):
    """
    A module that implements a compressed version of a list of expert modules.
    This is specifically designed to work with Llama4TextExperts in MoE layers.
    """

    def __init__(self, config):
        # Skip random weight initialization for experts. Otherwise,
        # the init of this module would take over minutes. For a model
        # with tens of layers of experts, it would easily take over 20 minutes.
        nn.init.kaiming_uniform_ = skip
        nn.init.uniform_ = skip
        nn.init.normal_ = skip
        super().__init__()
        self.num_experts = config.num_local_experts
        self.expert_modules = nn.ModuleList([Llama4TextMLP(config) for _ in range(self.num_experts)])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.reshape(self.num_experts, -1, hidden_states.shape[-1])
        expert_routed_out_list = []
        for expert_idx in range(self.num_experts):
            expert_routed_out_list.append(self.expert_modules[expert_idx](hidden_states[expert_idx]))
        routed_out = torch.cat(expert_routed_out_list, dim=0)
        return routed_out
