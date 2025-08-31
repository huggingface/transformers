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

from transformers import StableLmConfig, is_torch_available
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
        StableLmForSequenceClassification,
        StableLmForTokenClassification,
        StableLmModel,
    )
    from transformers.models.stablelm.modeling_stablelm import StableLmRotaryEmbedding

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class StableLmModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = StableLmConfig
        base_model_class = StableLmModel
        causal_lm_class = StableLmForCausalLM
        sequence_class = StableLmForSequenceClassification
        token_class = StableLmForTokenClassification


@require_torch
class StableLmModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            StableLmModel,
            StableLmForCausalLM,
            StableLmForSequenceClassification,
            StableLmForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": StableLmModel,
            "text-classification": StableLmForSequenceClassification,
            "text-generation": StableLmForCausalLM,
            "zero-shot": StableLmForSequenceClassification,
            "token-classification": StableLmForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = StableLmModelTester
    rotary_embedding_layer = StableLmRotaryEmbedding  # Enables RoPE tests if set


@require_torch
class StableLmModelIntegrationTest(unittest.TestCase):
    @slow
    def test_model_stablelm_3b_4e1t_logits(self):
        input_ids = {"input_ids": torch.tensor([[510, 8588, 310, 1900, 9386]], dtype=torch.long, device=torch_device)}

        model = StableLmForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t").to(torch_device)
        model.eval()

        output = model(**input_ids).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[2.7146, 2.4245, 1.5616, 1.4424, 2.6790]]).to(torch_device)
        torch.testing.assert_close(output.mean(dim=-1), EXPECTED_MEAN, rtol=1e-4, atol=1e-4)

        # Expected logits sliced from [0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([7.1030, -1.4195,  9.9206,  7.7008,  4.9891,  4.2169,  5.5426,  3.7878, 6.7593,  5.7360,  8.4691,  5.5448,  5.0544, 10.4129,  8.5573, 13.0405, 7.3265,  3.5868,  6.1106,  5.9406,  5.6376,  5.7490,  5.4850,  4.8124, 5.1991,  4.6419,  4.5719,  9.9588,  6.7222,  4.5070]).to(torch_device)  # fmt: skip
        torch.testing.assert_close(output[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

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
        EXPECTED_MEAN = torch.tensor([[-2.7196, -3.6099, -2.6877, -3.1973, -3.9344]]).to(torch_device)
        torch.testing.assert_close(output.mean(dim=-1), EXPECTED_MEAN, rtol=1e-4, atol=1e-4)

        # Expected logits sliced from [0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([2.8364, 5.3811, 5.1659, 7.5485, 4.3219, 6.3315, 1.3967, 6.9147, 3.9679, 6.4786, 5.9176, 3.3067, 5.2917, 0.1485, 3.9630, 7.9947,10.6727, 9.6757, 8.8772, 8.3527, 7.8445, 6.6025, 5.5786, 7.0985,6.1369, 3.4259, 1.9397, 4.6157, 4.8105, 3.1768]).to(torch_device)  # fmt: skip
        torch.testing.assert_close(output[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

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
            load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-3:].tolist())
