# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch StableLm model."""

import unittest

import pytest

from transformers import BitsAndBytesConfig, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        StableLmForCausalLM,
        StableLmModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class StableLmModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = StableLmModel


@require_torch
class StableLmModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = StableLmModelTester


@require_torch
class StableLmModelIntegrationTest(unittest.TestCase):
    @slow
    def test_model_stablelm_3b_4e1t_logits(self):
        input_ids = {"input_ids": torch.tensor([[510, 8588, 310, 1900, 9386]], dtype=torch.long, device=torch_device)}

        model = StableLmForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t").to(torch_device)
        model.eval()

        output = model(**input_ids).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[2.9599, 2.4270, 1.5699, 1.4504, 2.6743]]).to(torch_device)
        torch.testing.assert_close(output.mean(dim=-1), EXPECTED_MEAN, rtol=5e-2, atol=5e-2)

        # Expected logits sliced from [0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([7.3438, -1.3438, 10.1875, 8.1250, 5.1875, 4.4375, 5.3125, 3.8594, 7.0625, 6.2188, 8.8125, 5.5625, 4.9375, 10.6250, 9.0625, 13.2500, 7.6562, 3.6562, 5.9688, 5.8438, 5.5312, 5.6562, 5.4062, 4.6875, 5.0625, 4.5312, 4.4375, 10.3125, 7.1562, 4.7500]).to(torch_device)  # fmt: skip
        torch.testing.assert_close(output[0, 0, :30], EXPECTED_SLICE, rtol=5e-2, atol=5e-2)

    @slow
    def test_model_stablelm_3b_4e1t_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
        model = StableLmForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t")
        input_ids = tokenizer.encode(
            "My favorite food has always been pizza, but lately",
            return_tensors="pt",
        )

        outputs = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        EXPECTED_TEXT_COMPLETION = """My favorite food has always been pizza, but lately I’ve been craving something different. I’ve been trying to eat healthier and I’ve"""
        self.assertEqual(text, EXPECTED_TEXT_COMPLETION)

    @slow
    def test_model_tiny_random_stablelm_2_logits(self):
        # Check parallel residual and qk layernorm forward pass
        input_ids = {"input_ids": torch.tensor([[510, 8588, 310, 1900, 9386]], dtype=torch.long, device=torch_device)}

        model = StableLmForCausalLM.from_pretrained("stabilityai/tiny-random-stablelm-2").to(torch_device)
        model.eval()

        output = model(**input_ids).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-2.7175, -3.6089, -2.6867, -3.2131, -3.9279]]).to(torch_device)
        torch.testing.assert_close(output.mean(dim=-1), EXPECTED_MEAN, rtol=5e-2, atol=5e-2)

        # Expected logits sliced from [0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([2.8594, 5.3750, 5.1875, 7.5625, 4.3125, 6.3125, 1.3672, 6.9062, 3.9531, 6.5000, 5.9062, 3.3281, 5.2812, 0.1279, 3.9688, 8.0000, 10.6875, 9.6875, 8.8750, 8.3750, 7.8750, 6.6250, 5.5938, 7.1250, 6.1250, 3.4062, 1.9453, 4.6250, 4.8125, 3.1875]).to(torch_device)  # fmt: skip
        torch.testing.assert_close(output[0, 0, :30], EXPECTED_SLICE, rtol=5e-2, atol=5e-2)

    @slow
    def test_model_tiny_random_stablelm_2_generation(self):
        # Check parallel residual and qk layernorm generation
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/tiny-random-stablelm-2")
        model = StableLmForCausalLM.from_pretrained("stabilityai/tiny-random-stablelm-2")
        input_ids = tokenizer.encode(
            "My favorite ride at the amusement park",
            return_tensors="pt",
        )

        outputs = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        EXPECTED_TEXT_COMPLETION = """My favorite ride at the amusement park is the 2000-mile roller coaster. It's a thrilling ride filled with roller coast"""
        self.assertEqual(text, EXPECTED_TEXT_COMPLETION)

    @require_bitsandbytes
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_3b_long_prompt(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [3, 3, 3]
        input_ids = [306, 338] * 2047
        model = StableLmForCausalLM.from_pretrained(
            "stabilityai/stablelm-3b-4e1t",
            device_map="auto",
            dtype="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-3:].tolist())
