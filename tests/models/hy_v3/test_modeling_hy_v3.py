# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tests for HYV3 (MoE language model) configuration and modeling."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers.models.hy_v3.modeling_hy_v3 import HYV3ForCausalLM, HYV3Model

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class HYV3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HYV3Model


@require_torch
class HYV3ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HYV3ModelTester
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.8, 0.9]

    def test_router_logits_and_no_aux_loss(self):
        """HYV3 returns router_logits but does not compute aux_loss (always None)."""
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_router_logits = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                result = model(**input_dict)

            if hasattr(result, "router_logits") and result.router_logits is not None:
                num_moe_layers = sum(1 for t in config.mlp_layer_types if t == "sparse")
                self.assertEqual(len(result.router_logits), num_moe_layers)
                for rl in result.router_logits:
                    self.assertEqual(rl.shape[-1], config.num_experts)

            if hasattr(result, "aux_loss"):
                self.assertIsNone(result.aux_loss)


@slow
@require_torch
class HYV3IntegrationTest(unittest.TestCase):
    """Integration tests for HYV3 with a small randomized model."""

    model_id = "hf-internal-testing/HYV3-tiny-random"

    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_torch_accelerator
    def test_small_model_logits_batched(self):
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = HYV3ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(torch_device)

        EXPECTED_LOGITS_LEFT_UNPADDED = Expectations(
            {
                ("cuda", (8, 6)): [[0.0608, -0.0933, 0.1348], [-0.0688, -0.1099, 0.1396], [0.0199, -0.0913, 0.1641]],
                ("cuda", 9): [[0.063, -0.0938, 0.1348], [-0.0693, -0.1128, 0.1357], [0.0209, -0.0923, 0.1611]],
            }
        )
        expected_left_unpadded = torch.tensor(EXPECTED_LOGITS_LEFT_UNPADDED.get_expectation(), device=torch_device)

        EXPECTED_LOGITS_RIGHT_UNPADDED = Expectations(
            {
                ("cuda", (8, 6)): [[-0.0396, -0.1084, 0.0588], [-0.0100, -0.0903, 0.0747], [0.0645, -0.1172, 0.0508]],
                ("cuda", 9): [[-0.0378, -0.1089, 0.0581], [-0.0088, -0.0908, 0.0752], [0.064, -0.1167, 0.0483]],
            }
        )
        expected_right_unpadded = torch.tensor(EXPECTED_LOGITS_RIGHT_UNPADDED.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()

        torch.testing.assert_close(logits[0, -3:, :3], expected_left_unpadded, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, -3:, :3], expected_right_unpadded, atol=1e-3, rtol=1e-3)

    @require_torch_accelerator
    def test_small_model_generation(self):
        EXPECTED_TOKENS = Expectations(
            {
                ("cuda", 9): [1, 2, 3, 8754, 20977, 8754, 8754, 8754, 8754, 8754, 8754, 8754, 8372, 8754, 8372, 21393, 8754, 8372, 21393, 8754, 8372, 21393, 8754],
                ("cuda", (8, 6)): [1, 2, 3, 8754, 20977, 8754, 8754, 8754, 8754, 8754, 8754, 8754, 8372, 8754, 8372, 21393, 8754, 8372, 21393, 8754, 8372, 21393, 11262],
            }
        )  # fmt: skip
        expected_tokens = EXPECTED_TOKENS.get_expectation()

        model = HYV3ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(torch_device)
        input_ids = torch.LongTensor([[1, 2, 3]]).to(torch_device)

        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        self.assertEqual(generated_ids[0].tolist(), expected_tokens)
