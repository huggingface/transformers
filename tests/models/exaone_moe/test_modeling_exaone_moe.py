# Copyright 2026 The LG AI Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EXAONE MoE model."""

import unittest

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        ExaoneMoeForCausalLM,
        ExaoneMoeModel,
    )


class ExaoneMoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = ExaoneMoeModel


@require_torch
class ExaoneMoeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = ExaoneMoeModelTester
    model_split_percents = [0.5, 0.6]


@require_torch
class ExaoneMoeIntegrationTest(unittest.TestCase):
    TEST_MODEL_ID = "LGAI-EXAONE/K-EXAONE-236B-A23B"

    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)
    
    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = ExaoneMoeForCausalLM.from_pretrained(
                cls.TEST_MODEL_ID,
                device_map="auto",
                experts_implementation="eager",
            )

        return cls.model
    
    @slow
    @require_torch_large_accelerator
    def test_model_logits(self):
        input_ids = [405, 7584, 36608, 892, 95714, 2907, 1492, 758, 373, 582]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([[80.0818, 51.8579, 94.5935, 94.4710, 95.2955, 104.5337, 104.0203, 108.6814, 105.7278, 113.6849]])
        EXPECTED_SLICE = torch.tensor([86.0000, 80.5000, 88.0000, 81.5000, 90.5000, 89.0000, 87.5000, 88.5000, 87.0000, 89.0000])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)

    @slow
    def test_model_generation_sdpa(self):
        EXPECTED_TEXT = '<|user|>\nTell me about the Miracle on the Han river.<|endofturn|>\n<|assistant|>\n<think>\n\n</think>\n\nThe "Miracle on the Han River" refers to the rapid and remarkable economic transformation of South Korea (Republic of'
        prompt = "Tell me about the Miracle on the Han river."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = self.get_model()
        
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=False)
        input_ids = input_ids.to(model.model.embed_tokens.weight.device)

        with torch.no_grad():
            generated_ids = model.generate(**input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        self.assertEqual(EXPECTED_TEXT, text)

    @slow
    @require_torch_large_accelerator
    def test_model_generation_beyond_sliding_window_flash(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [21605, 2711]
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        input_ids = [72861, 2711] * 2048
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())