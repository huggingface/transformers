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

from pytest import mark

from transformers import (
    AutoTokenizer,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
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
    model_split_percents = [0.5, 0.8, 0.9]


@slow
@require_torch
class ExaoneMoeIntegrationTest(unittest.TestCase):
    TEST_MODEL_ID = "hf-internal-testing/EXAONE-MoE-Dummy-7B-A1B"

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

    def test_model_logits(self):
        input_ids = [405, 7584, 36608, 892, 95714, 2907, 1492, 758, 373, 582]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor(
            [[-2.2663, -3.0659, -3.0922, -3.2575, -3.2033, -3.4042, -3.1595, -3.2739, -3.8887, -0.6917]]
        )
        EXPECTED_SLICE = torch.tensor(
            [-2.3906, -3.0781, 2.7500, -3.0469, 0.5312, -1.4297, -1.8984, -2.6875, -1.7734, -2.1094]
        )
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)

    def test_model_generation_sdpa(self):
        EXPECTED_TEXT = "The deep learning is 100% accurate.\n\nThe 100% accurate is 100%"
        prompt = "The deep learning is "
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = self.get_model()

        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**input_ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        self.assertEqual(EXPECTED_TEXT, text)

    @require_flash_attn
    @mark.flash_attn_test
    def test_model_generation_beyond_sliding_window_flash(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [373, 686, 373, 115708, 373, 885]
        input_ids = [72861, 2711] + [21605, 2711] * 2048
        model = self.get_model()
        model.set_attn_implementation("flash_attention_2")
        input_ids = torch.tensor([input_ids]).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=6, do_sample=False)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-6:].tolist())
