# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import unittest

import torch
from torch import nn

from transformers import PreTrainedConfig
from transformers.integrations.moe import (
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
    use_experts_implementation,
)
from transformers.testing_utils import require_torch


class RowNorm(nn.Module):
    """A per-row norm with a weight, the shape of a per-expert output norm."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


@use_experts_implementation
class ExpertsWithPostNorm(nn.Module):
    """Gated experts whose output goes through a per-row norm before the routing weights."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.randn(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = nn.SiLU()
        self.post_expert_norm = RowNorm(self.hidden_dim)

    def _apply_post_expert(self, proj_out):
        return self.post_expert_norm(proj_out)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # The eager reference: one expert at a time, the norm on each expert's rows
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_idx == top_k_index.T)
            if token_idx.numel() == 0:
                continue
            gate, up = nn.functional.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            out = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            out = self.post_expert_norm(out) * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, out)
        return final_hidden_states


@use_experts_implementation
class PlainExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(self.num_experts, 2 * config.intermediate_size, config.hidden_size)
        )
        self.down_proj = nn.Parameter(torch.randn(self.num_experts, config.hidden_size, config.intermediate_size))
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        raise NotImplementedError


@require_torch
class ExpertsImplementationsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = PreTrainedConfig(
            num_experts=8, hidden_size=16, intermediate_size=12, _experts_implementation="eager"
        )
        num_tokens, top_k = 10, 2
        self.hidden_states = torch.randn(num_tokens, self.config.hidden_size)
        self.top_k_index = torch.stack([torch.randperm(self.config.num_experts)[:top_k] for _ in range(num_tokens)])
        self.top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    def test_post_expert_hook_matches_the_eager_loop(self):
        experts = ExpertsWithPostNorm(self.config)
        expected = experts(self.hidden_states, self.top_k_index, self.top_k_weights)
        for forward in (batched_mm_experts_forward, grouped_mm_experts_forward):
            out = forward(experts, self.hidden_states, self.top_k_index, self.top_k_weights)
            torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_default_hook_is_the_identity(self):
        experts = PlainExperts(self.config)
        x = torch.randn(3, self.config.hidden_size)
        self.assertIs(experts._apply_post_expert(x), x)
