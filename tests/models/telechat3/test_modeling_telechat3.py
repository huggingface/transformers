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
"""Testing suite for the PyTorch TeleChat3 model."""

import math
import unittest

from transformers import AutoTokenizer, TeleChat3Config, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import TeleChat3ForCausalLM, TeleChat3Model


class TeleChat3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = TeleChat3Model


@require_torch
class TeleChat3ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = TeleChat3ModelTester


class TeleChat3ConfigTest(unittest.TestCase):
    def test_telechat3_yarn_is_converted_to_yarn(self):
        # `rope_scaling` as found in the released checkpoints
        config = TeleChat3Config(
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            rope_scaling={
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "telechat3-yarn",
                "type": "telechat3-yarn",
            },
        )
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertNotIn("type", config.rope_parameters)
        self.assertEqual(config.rope_parameters["factor"], 4.0)
        self.assertAlmostEqual(config.rope_parameters["attention_factor"], 0.07 * math.log(4.0) + 1.0)

    def test_factor_follows_context_lengths(self):
        # The original code ignores `factor` when `original_max_position_embeddings` is set
        config = TeleChat3Config(
            max_position_embeddings=16384,
            rope_scaling={"factor": 4.0, "original_max_position_embeddings": 8192, "rope_type": "telechat3-yarn"},
        )
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertAlmostEqual(config.rope_parameters["attention_factor"], 0.07 * math.log(2.0) + 1.0)

    def test_explicit_attention_factor_is_kept(self):
        config = TeleChat3Config(
            rope_scaling={"factor": 4.0, "attention_factor": 1.5, "rope_type": "telechat3-yarn"},
        )
        self.assertEqual(config.rope_parameters["attention_factor"], 1.5)


@slow
@require_torch_accelerator
class TeleChat3IntegrationTest(unittest.TestCase):
    model_id = "Tele-AI/TeleChat3-36B-Thinking"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_36b_logits(self):
        model = TeleChat3ForCausalLM.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        input_ids = torch.tensor([[1, 1404, 1435, 1272, 1363, 1446, 334, 1279, 1306, 395]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor(EXPECTED_MEAN_PLACEHOLDER)
        EXPECTED_SLICE = torch.tensor(EXPECTED_SLICE_PLACEHOLDER)
        torch.testing.assert_close(logits.mean(-1)[0], EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    def test_model_36b_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = TeleChat3ForCausalLM.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertEqual(text, EXPECTED_TEXT_PLACEHOLDER)
