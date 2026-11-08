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
"""Testing suite for the PyTorch IQuestCoder model."""

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import AutoTokenizer, IQuestCoderConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import IQuestCoderForCausalLM, IQuestCoderModel


class IQuestCoderModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = IQuestCoderModel


@require_torch
class IQuestCoderModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = IQuestCoderModelTester

    def test_unsupported_options_are_rejected(self):
        # Released checkpoints set these to their disabled values, which must keep loading
        IQuestCoderConfig(clip_qkv=None, use_sliding_window=False, sliding_window=None, max_window_layers=0)
        with self.assertRaises(StrictDataclassClassValidationError):
            IQuestCoderConfig(clip_qkv=8.0)
        with self.assertRaises(StrictDataclassClassValidationError):
            IQuestCoderConfig(use_sliding_window=True, sliding_window=4096)


@slow
@require_torch_accelerator
class IQuestCoderIntegrationTest(unittest.TestCase):
    model_id = "IQuestLab/IQuest-Coder-V1-7B-Instruct"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_7b_logits(self):
        model = IQuestCoderForCausalLM.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        input_ids = torch.tensor([[1, 1404, 1435, 1272, 1363, 1446, 334, 1279, 1306, 395]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor(
            [23.4439, -0.9136, 6.2595, 9.6258, 8.6033, 13.4631, 6.3165, 8.7364, 11.1808, 6.8638]
        )
        EXPECTED_SLICE = torch.tensor([25.75, 30.75, 26.625, 28.625, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
        torch.testing.assert_close(logits.mean(-1)[0], EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    def test_model_7b_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = IQuestCoderForCausalLM.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        inputs = tokenizer("def fibonacci(n):", return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        EXPECTED_TEXT = "def fibonacci(n):\n    if n <= 0:\n     "
        self.assertEqual(text, EXPECTED_TEXT)
