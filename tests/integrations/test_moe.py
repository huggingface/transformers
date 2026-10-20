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
"""Tests for the built-in experts implementations in `integrations/moe.py`.

Expert parallelism leaves each rank only its own experts, and the router marks the slots it does not own
with a sentinel expert id and a zero routing weight. `batched_mm_experts_forward` clamps those slots into
a real expert to keep the weight gather in bounds, so they have to be kept out of the output and of the
gradient explicitly.
"""

import subprocess
import sys
import types
import unittest

import torch

from transformers.activations import ACT2FN
from transformers.integrations.moe import (
    ALL_EXPERTS_FUNCTIONS,
    ExpertsInterface,
    ExpertsRequirements,
    batched_mm_experts_forward,
    use_experts_implementation,
)
from transformers.testing_utils import require_torch, torch_device


def make_experts(num_experts, hidden, inter, is_expert_parallel):
    """The attributes `batched_mm_experts_forward` reads off an experts module, with real weights."""
    act_fn = ACT2FN["silu"]

    def apply_gate(gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return act_fn(gate) * up

    return types.SimpleNamespace(
        num_experts=num_experts,
        has_gate=True,
        has_bias=False,
        is_transposed=False,
        act_fn=act_fn,
        _apply_gate=apply_gate,
        gate_up_proj=torch.randn(num_experts, 2 * inter, hidden, device=torch_device),
        gate_up_proj_bias=None,
        down_proj=torch.randn(num_experts, hidden, inter, device=torch_device),
        down_proj_bias=None,
        _is_expert_parallel=is_expert_parallel,
        has_post_expert_norm=False,
    )


NUM_EXPERTS = 4
# Slots (0, 1) and (2, 0) are sentinels: expert id `NUM_EXPERTS`, routing weight 0.
TOP_K_INDEX = [[0, NUM_EXPERTS], [1, 2], [NUM_EXPERTS, 3]]
TOP_K_WEIGHTS = [[0.7, 0.0], [0.4, 0.6], [0.0, 0.9]]


@require_torch
class BatchedMmExpertsForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # The sentinel slots only exist under expert parallelism, which is also what gates the handling.
        self.experts = make_experts(num_experts=NUM_EXPERTS, hidden=8, inter=16, is_expert_parallel=True)
        self.hidden_states = torch.randn(3, 8, device=torch_device)
        self.top_k_index = torch.tensor(TOP_K_INDEX, device=torch_device)

    def _weights(self, requires_grad=False):
        return torch.tensor(TOP_K_WEIGHTS, device=torch_device, dtype=torch.float32, requires_grad=requires_grad)

    def test_sentinel_slots_get_no_routing_weight_gradient(self):
        top_k_weights = self._weights(requires_grad=True)
        out = batched_mm_experts_forward(self.experts, self.hidden_states, self.top_k_index, top_k_weights)
        out.sum().backward()

        # The clamp leaves a sentinel slot a real expert output, so only the zero weight keeps it out of the
        # forward, and multiplying by zero does not stop the gradient reaching the weight on the other side.
        sentinel = self.top_k_index >= NUM_EXPERTS
        self.assertTrue(torch.all(top_k_weights.grad[sentinel] == 0))
        # The routed slots must still get one, or the assert above would pass on an all-zero gradient.
        self.assertTrue(torch.all(top_k_weights.grad[~sentinel] != 0))

    def test_sentinel_slots_do_not_reach_the_output(self):
        out = batched_mm_experts_forward(self.experts, self.hidden_states, self.top_k_index, self._weights())

        # Which expert a sentinel slot is clamped into cannot matter, so moving it leaves the output alone.
        moved = self.top_k_index.masked_fill(self.top_k_index >= NUM_EXPERTS, NUM_EXPERTS + 1)
        out_moved = batched_mm_experts_forward(self.experts, self.hidden_states, moved, self._weights())
        torch.testing.assert_close(out, out_moved)


@require_torch
class UseExpertsImplementationTest(unittest.TestCase):
    """Check that the class body and the instance can override the layout flags of the decorator."""

    def test_flags_set_by_the_class_or_the_instance_are_kept(self):
        @use_experts_implementation(has_bias=True, is_transposed=True)
        class Experts(torch.nn.Module):
            is_transposed = False  # the class body wins over the decorator

            def __init__(self, config, has_gate=True):
                super().__init__()
                self.has_gate = has_gate  # the instance wins over the class default

            def forward(self, hidden_states, top_k_index, top_k_weights):
                return hidden_states

        config = types.SimpleNamespace(_experts_implementation="eager")
        experts = Experts(config, has_gate=False)
        self.assertEqual((experts.has_gate, experts.has_bias, experts.is_transposed), (False, True, False))
        self.assertEqual((experts.is_concatenated, experts.has_post_expert_norm), (True, False))
        self.assertIs(experts.config, config)
        self.assertTrue(Experts(config).has_gate)


@require_torch
class ExpertsRequirementsTest(unittest.TestCase):
    """Check the requirements that experts implementations register in `ExpertsInterface`."""

    def test_names_are_known_without_importing_the_interface_that_serves_them(self):
        code = (
            "import sys, transformers.modeling_utils\n"
            "from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS\n"
            "assert 'deepgemm_megamoe' in ALL_EXPERTS_FUNCTIONS.valid_keys()\n"
            "assert not any(name.startswith('transformers.integrations.finegrained') for name in sys.modules)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def _register_load_time_only(self):
        requirements = ExpertsRequirements(switchable=False, parallel_styles={"moe_tp_experts": "megamoe_experts"})
        ExpertsInterface.register("load_time_only", batched_mm_experts_forward, requirements=requirements)
        self.addCleanup(ExpertsInterface._global_mapping.pop, "load_time_only", None)
        self.addCleanup(ExpertsInterface._global_requirements.pop, "load_time_only", None)
        return requirements

    def test_registering_declares_the_requirements(self):
        requirements = self._register_load_time_only()
        self.assertIs(ALL_EXPERTS_FUNCTIONS.requirements("load_time_only"), requirements)
        self.assertIn("load_time_only", ALL_EXPERTS_FUNCTIONS.valid_keys())

    def test_default_requirements(self):
        self.assertEqual(ALL_EXPERTS_FUNCTIONS.requirements("eager"), ExpertsRequirements())
        self.assertFalse(ALL_EXPERTS_FUNCTIONS.requirements("deepgemm_megamoe").switchable)

    def test_a_load_time_implementation_cannot_be_switched_to(self):
        from transformers import MixtralConfig, MixtralForCausalLM

        self._register_load_time_only()
        config = MixtralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=2,
        )
        model = MixtralForCausalLM(config)
        with self.assertRaisesRegex(RuntimeError, "load-time choice"):
            model.set_experts_implementation("load_time_only")
