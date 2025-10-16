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

import gc
import unittest

from transformers import AutoTokenizer, NanoChatConfig, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
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
        EXPECTED_MEAN = torch.tensor([[-6.6598, -7.8072]])
        torch.testing.assert_close(logits.mean(-1), EXPECTED_MEAN, rtol=1e-3, atol=1e-2)

        # Check first 10 logits at position [0,0,:10]
        EXPECTED_SLICE = torch.tensor(
            [-12.875, -13.0625, -13.1875, -13.1875, -13.1875, -13.1875, -13.1875, -13.1875, -12.625, -4.21875]
        )
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-3, atol=1e-2)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_d20_generation(self):
        """Test that d20 model generates text correctly."""
        model_id = "nanochat-students/nanochat-d20"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = NanoChatForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

        # Test generation with chat template
        prompt = "What is the capital of France?"
        conversation = [
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        # Generate with greedy decoding for reproducibility
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        # Decode only the generated tokens
        generated_tokens = generated_ids[0, inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Check that text was generated
        self.assertGreater(len(generated_text), 0)
        self.assertGreater(len(generated_tokens), 0)

        # The model should generate a reasonable response (with greedy decoding this is deterministic)
        # Expected: "The capital of France is Paris."
        self.assertIn("Paris", generated_text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_d32_logits(self):
        """Test that d32 model logits are computed correctly."""
        model_id = "karpathy/nanochat-d32"
        revision = "refs/pr/1"
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
        EXPECTED_MEAN = torch.tensor([[-5.6796, -8.2629]])
        torch.testing.assert_close(logits.mean(-1), EXPECTED_MEAN, rtol=1e-3, atol=1e-2)

        # Check first 10 logits at position [0,0,:10]
        EXPECTED_SLICE = torch.tensor(
            [-12.4375, -13.1875, -12.875, -13.1875, -13.1875, -13.1875, -13.1875, -13.1875, -11.9375, -1.6328]
        )
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-3, atol=1e-2)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_d32_generation(self):
        """Test that d32 model generates text correctly."""
        model_id = "karpathy/nanochat-d32"
        revision = "refs/pr/1"
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = NanoChatForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, revision=revision
        )

        # Test generation with chat template
        prompt = "What is the capital of France?"
        conversation = [
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        # Generate with greedy decoding for reproducibility
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        # Decode only the generated tokens
        generated_tokens = generated_ids[0, inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Check that text was generated
        self.assertGreater(len(generated_text), 0)
        self.assertGreater(len(generated_tokens), 0)

        # The model should generate a reasonable response (with greedy decoding this is deterministic)
        # Expected: "The capital of France is Paris."
        self.assertIn("Paris", generated_text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()
