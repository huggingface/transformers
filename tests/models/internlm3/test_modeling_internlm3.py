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
"""Testing suite for the PyTorch InternLM3 model."""

import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import InternLM3ForCausalLM, InternLM3Model


class InternLM3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = InternLM3Model


@require_torch
class InternLM3ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = InternLM3ModelTester

    def test_qkv_bias_is_independent_of_bias(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.qkv_bias = True
        config.bias = False
        layer = InternLM3Model(config).layers[0]
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            self.assertIsNotNone(proj.bias)
        for proj in (layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            self.assertIsNone(proj.bias)


@slow
@require_torch_accelerator
class InternLM3IntegrationTest(unittest.TestCase):
    model_id = "internlm/internlm3-8b-instruct"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_8b_logits(self):
        model = InternLM3ForCausalLM.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        input_ids = torch.tensor([[1, 1404, 1435, 1272, 1363, 1446, 334, 1279, 1306, 395]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([-0.2702, 1.8108, 2.4337, 4.6791, 3.8414, 4.021, 4.3054, 4.6507, 4.841, 4.5961])
        EXPECTED_SLICE = torch.tensor(
            [-1.7656, -0.5547, 3.5625, -0.6719, 1.9141, 1.5781, 1.2344, 0.7188, 0.3398, 0.5039]
        )
        torch.testing.assert_close(logits.mean(-1)[0], EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    def test_model_8b_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = InternLM3ForCausalLM.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        EXPECTED_TEXT = "The capital of France is Paris.Paris is the capital of France. It is a city that is known"
        self.assertEqual(text, EXPECTED_TEXT)
