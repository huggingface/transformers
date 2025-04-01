from typing import List, Optional, Tuple, Union

from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import Llama4TextMLP


class CompressedExpertsLinear(nn.Module):
    """
    A module that implements a compressed version of a list of expert modules.
    This is specifically designed to work with Llama4TextExperts in MoE layers.

    Instead of executing each expert separately, this module combines the weights
    of all experts and performs a single forward pass, which is more efficient
    when using compressed tensors.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.expert_modules = nn.ModuleList([Llama4TextMLP(config) for _ in range(self.num_experts)])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass that efficiently processes inputs through all experts.

        Args:
            hidden_states: Input tensor of shape [num_tokens, hidden_dim]
        Returns:
            Tensor of shape [num_tokens, hidden_dim] with expert outputs
        """
        hidden_states = hidden_states.reshape(self.num_experts, -1, hidden_states.shape[-1])
        expert_routed_out_list = []
        for expert_idx in range(self.num_experts):
            expert_routed_out_list.append(self.expert_modules[expert_idx](hidden_states[expert_idx]))
        routed_out = torch.cat(expert_routed_out_list, dim=0)
        return routed_out
