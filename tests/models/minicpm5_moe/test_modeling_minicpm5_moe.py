# Copyright 2026 the Hugging Face team. All rights reserved.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, MiniCPM5Model, MiniCPM5MoEForCausalLM


class MiniCPM5ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniCPM5Model
        causal_lm_class = MiniCPM5MoEForCausalLM

    small_config_overrides = {"n_routed_experts": 4, "num_key_value_heads": 1}


@require_torch
class MiniCPM5ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPM5ModelTester

    def test_router_preserves_fp32_under_autocast(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_experts_per_tok = 1
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16)
        gate = model.model.layers[1].mlp.gate
        hidden_states = torch.ones(1, config.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            gate.weight.zero_()
            gate.weight[:2, 0] = 1
            gate.weight[1, 1] = 2**-10
            expected = gate(hidden_states)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = gate(hidden_states)
        self.assertEqual(expected[2].item(), 1)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)

    def test_grouped_router_excludes_masked_experts_when_scores_are_negative(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.n_group = 2
        config.topk_group = 1
        config.num_experts_per_tok = 2
        gate = AutoModelForCausalLM.from_config(config).model.layers[1].mlp.gate
        with torch.no_grad():
            gate.e_score_correction_bias.copy_(torch.tensor([-1.0, -1.0, -2.0, -2.0]))
            _, _, indices = gate(torch.zeros(1, config.hidden_size))
        self.assertEqual(sorted(indices[0].tolist()), [0, 1])

    def test_cached_decode_matches_full_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits[:, -1]
            prefix = model(input_ids=input_ids[:, :-1], use_cache=True)
            decoded = model(input_ids=input_ids[:, -1:], cache=prefix.cache, use_cache=True)
        torch.testing.assert_close(decoded.logits[:, -1], expected, atol=1e-5, rtol=1e-4)
