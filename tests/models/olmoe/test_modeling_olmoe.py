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
"""Testing suite for the PyTorch OLMoE model."""

import unittest

from transformers import OlmoeConfig, is_torch_available
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers.testing_utils import (
    require_tokenizers,
    require_torch,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        OlmoeForCausalLM,
        OlmoeModel,
    )


class OlmoeModelTester(CausalLMModelTester):
    config_class = OlmoeConfig
    if is_torch_available():
        base_model_class = OlmoeModel
        causal_lm_class = OlmoeForCausalLM


@require_torch
class OlmoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (OlmoeModel, OlmoeForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": OlmoeModel,
            "text-generation": OlmoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = OlmoeModelTester

    @unittest.skip(reason="OLMoE does not support head pruning.")
    def test_headmasking(self):
        pass


@require_torch
class OlmoeIntegrationTest(unittest.TestCase):
    @slow
    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", device_map="auto")
        out = model(torch.tensor(input_ids)).logits.float()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-1.3814, -3.4450, -2.2990, -1.9542, -2.4387, -2.7941, -2.9312, -2.8309]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-2.3874, -2.4076, -2.4995, 4.2278, 1.4004, -0.0252, 0.4189, -2.7560, 0.3531, 1.6678, -0.7941, -1.1818, -0.2920, 0.7131, -1.4173, 1.6723, 0.5406, 0.1345, -0.1800, 0.2304, 1.2791, 0.7489, 0.6341, -0.0151, -1.3693, -1.2532, -2.3921, 0.7376, 1.6876, 0.5483])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @slow
    def test_model_7b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that \nthe speed of light is the same for all observers, no matter \nhow fast they are moving.  This is a very counter-intuitive \nconcept, and it took Einstein a long time to come up with \nthe theory.  The theory of relativity is based on two \npostulates"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", device_map="auto")

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @require_tokenizers
    def test_fast_special_tokens(self):
        fast_tokenizer = GPTNeoXTokenizerFast.from_pretrained("allenai/OLMoE-1B-7B-0924")

        original_add_eos_token = fast_tokenizer.add_eos_token

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test")
        self.assertEqual(fast, [34, 3410, 1071])

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test")
        self.assertEqual(fast, [34, 3410, 1071, 50279])

        fast_tokenizer.add_eos_token = original_add_eos_token

    @require_tokenizers
    def test_simple_encode_decode(self):
        rust_tokenizer = GPTNeoXTokenizerFast.from_pretrained("allenai/OLMoE-1B-7B-0924")

        self.assertEqual(rust_tokenizer.encode("This is a test"), [1552, 310, 247, 1071])
        self.assertEqual(rust_tokenizer.decode([1552, 310, 247, 1071], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [20025, 46549, 5225, 48561, 33656, 238, 12105])  # fmt: skip
        self.assertEqual(
            rust_tokenizer.decode([20025, 46549, 5225, 48561, 33656, 238, 12105], skip_special_tokens=True),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [12764, 50276, 12092])
        self.assertEqual(rust_tokenizer.decode([12764, 50276, 12092], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [12764, 50275, 12092])
        self.assertEqual(rust_tokenizer.decode([12764, 50275, 12092], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(rust_tokenizer.encode(""), [])

        self.assertEqual(rust_tokenizer.encode(" "), [209])

        self.assertEqual(rust_tokenizer.encode("  "), [50276])

        self.assertEqual(rust_tokenizer.encode(" Hello"), [24387])
