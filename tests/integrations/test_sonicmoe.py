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

# Run the test: CUDA_VISIBLE_DEVICES=0 pytest -sv tests/integrations/test_sonicmoe.py
# Requires an SM90+ (Hopper/Blackwell) GPU with a working `nvidia-cutlass-dsl` toolchain; skipped otherwise.

import unittest

import torch

from transformers import Qwen3MoeConfig
from transformers.integrations.sonicmoe import SONIC_MOE_HANDLE
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
from transformers.testing_utils import require_kernels, require_torch_gpu, torch_device


@require_torch_gpu
@require_kernels
class SonicMoeCompileIntegrationTest(unittest.TestCase):
    """Exercises the sonicmoe experts path under `torch.compile`. Kept out of the common suite because it needs the
    real CuteDSL kernel; it skips wherever `sonicmoe_is_available` is False (e.g. non-Hopper GPUs or a driver/toolchain
    mismatch)."""

    def _build_tiny_experts(self):
        # Tiny but kernel-aligned dims (multiples of 128) so the CuteDSL GEMMs are happy.
        config = Qwen3MoeConfig(
            hidden_size=256,
            moe_intermediate_size=256,
            num_experts=8,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            hidden_act="silu",
            experts_implementation="sonicmoe",
        )
        experts = Qwen3MoeExperts(config).eval().to(torch_device, dtype=torch.bfloat16)
        with torch.no_grad():
            for param in experts.parameters():
                param.normal_(mean=0.0, std=0.02)
        return experts, config

    def test_sonicmoe_compiled_matches_eager(self):
        if not SONIC_MOE_HANDLE.sonicmoe_is_available:
            self.skipTest("sonic-moe kernel is not available on this machine")

        torch.manual_seed(0)
        experts, config = self._build_tiny_experts()

        num_tokens, top_k = 128, config.num_experts_per_tok
        hidden_states = torch.randn(num_tokens, config.hidden_size, device=torch_device, dtype=torch.bfloat16)
        router_logits = torch.randn(num_tokens, config.num_experts, device=torch_device)
        top_k_weights, top_k_index = torch.softmax(router_logits, dim=-1).topk(top_k, dim=-1)
        top_k_weights = top_k_weights.to(torch.bfloat16)

        with torch.no_grad():
            eager_out = experts(hidden_states, top_k_index, top_k_weights)

        # fullgraph=True is the point: the sonicmoe kernel must stay in-graph as one opaque node. If the
        # `allow_in_graph` shim is broken, this either errors here or graph-breaks — it cannot silently pass.
        torch._dynamo.reset()
        compiled_experts = torch.compile(experts, fullgraph=True)
        with torch.no_grad():
            compiled_out = compiled_experts(hidden_states, top_k_index, top_k_weights)

        self.assertEqual(compiled_out.shape, eager_out.shape)
        torch.testing.assert_close(compiled_out, eager_out, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
