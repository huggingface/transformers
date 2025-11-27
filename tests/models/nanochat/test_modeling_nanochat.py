# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch NanoChat model."""

import unittest

from transformers import AutoTokenizer, NanoChatConfig, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        NanoChatForCausalLM,
        NanoChatModel,
    )


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class NanoChatModelTester(CausalLMModelTester):
    config_class = NanoChatConfig
    if is_torch_available():
        base_model_class = NanoChatModel
        causal_lm_class = NanoChatForCausalLM


@require_torch
class NanoChatModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = NanoChatModelTester


@require_torch
class NanoChatIntegrationTest(unittest.TestCase):
    """Integration tests for NanoChat models using real checkpoints."""

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_d20_logits(self):
        """Test that d20 model logits are computed correctly."""
        model_id = "nanochat-students/nanochat-d20"
        model = NanoChatForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Simple test input - "Hello world"
        test_text = "Hello world"
        input_ids = tokenizer.encode(test_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.float().cpu()

        # Basic shape checks
        self.assertEqual(logits.shape[0], 1)  # batch size
        self.assertEqual(logits.shape[1], input_ids.shape[1])  # sequence length
        self.assertEqual(logits.shape[2], model.config.vocab_size)  # vocab size 65536

        # Check logits are not NaN or Inf
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Check expected mean logits (with tolerance for numerical variation)
        EXPECTED_MEAN = torch.tensor([[-6.6607, -7.8095]])

        # Check first 10 logits at position [0,0,:10]
        EXPECTED_SLICE = torch.tensor(
            [-12.8750, -13.0625, -13.1875, -13.1875, -13.1875, -13.1875, -13.1875, -13.1875, -12.6250, -4.4062]
        )

        torch.testing.assert_close(logits.mean(-1), EXPECTED_MEAN, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-3, atol=1e-3)

    @slow
    def test_model_d20_generation(self):
        """Test that d20 model generates text correctly."""
        model_id = "nanochat-students/nanochat-d20"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = NanoChatForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

        # Test generation with chat template
        conversation = [
            [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "user", "content": "Tell me something."},
            ],
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            tokenizer_kwargs={"padding_side": "left"},
            return_tensors="pt",
        ).to(model.device)

        # Generate with greedy decoding for reproducibility
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        # Decode only the generated tokens
        generated_text = [
            tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
            tokenizer.decode(generated_ids[1, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
        ]

        EXPECTED_TEXT_COMPLETION = [
            "The capital of France is Paris.",
            "I'm ready to help. What's the first thing you'd like to know or discuss?",
        ]

        self.assertEqual(EXPECTED_TEXT_COMPLETION[0], generated_text[0])
        self.assertEqual(EXPECTED_TEXT_COMPLETION[1], generated_text[1])

    @slow
    def test_model_d32_logits(self):
        """Test that d32 model logits are computed correctly."""
        model_id = "karpathy/nanochat-d32"
        revision = "refs/pr/1"  # TODO: update when merged to hub
        model = NanoChatForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        # Simple test input - "Hello world"
        test_text = "Hello world"
        input_ids = tokenizer.encode(test_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.float().cpu()

        # Basic shape checks
        self.assertEqual(logits.shape[0], 1)  # batch size
        self.assertEqual(logits.shape[1], input_ids.shape[1])  # sequence length
        self.assertEqual(logits.shape[2], model.config.vocab_size)  # vocab size 65536

        # Check logits are not NaN or Inf
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Check expected mean logits (with tolerance for numerical variation)
        EXPECTED_MEAN = torch.tensor([[-5.5791, -8.3456]])

        # Check first 10 logits at position [0,0,:10]
        EXPECTED_SLICE = torch.tensor(
            [-12.3125, -13.1250, -12.8125, -13.1250, -13.1250, -13.1250, -13.1250, -13.1250, -11.8125, -1.4688]
        )

        torch.testing.assert_close(logits.mean(-1), EXPECTED_MEAN, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-3, atol=1e-3)

    @slow
    def test_model_d32_generation(self):
        """Test that d32 model generates text correctly."""
        model_id = "karpathy/nanochat-d32"
        revision = "refs/pr/1"  # TODO: update when merged to hub
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = NanoChatForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, revision=revision
        )

        # Test generation with chat template
        conversation = [
            [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "user", "content": "Tell me something."},
            ],
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            tokenizer_kwargs={"padding_side": "left"},
            return_tensors="pt",
        ).to(model.device)

        # Generate with greedy decoding for reproducibility
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        # Decode only the generated tokens
        generated_text = [
            tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
            tokenizer.decode(generated_ids[1, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
        ]

        EXPECTED_TEXT_COMPLETION = [
            "The capital of France is Paris.",
            "I'm here to help you explore your creative writing endeavors. What's been on your mind lately? Do you have a story idea you'd like to develop,",
        ]

        self.assertEqual(EXPECTED_TEXT_COMPLETION[0], generated_text[0])
        self.assertEqual(EXPECTED_TEXT_COMPLETION[1], generated_text[1])
