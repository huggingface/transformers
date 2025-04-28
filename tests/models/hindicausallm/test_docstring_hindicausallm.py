# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Team. All rights reserved.
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
"""Tests for the HindiCausalLM docstring examples."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        HindiCausalLMForCausalLM,
        HindiCausalLMTokenizer,
    )


@require_torch
class HindiCausalLMDocstringTest(TestCasePlus):
    @slow
    def test_hindicausallm_docstring(self):
        """Test the docstring example for HindiCausalLM."""
        # Load model and tokenizer
        model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-causal-lm")
        tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")
        model.to(torch_device)
        model.eval()

        # Generate text
        input_text = "भारत एक विशाल देश है"  # "India is a vast country"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(torch_device)

        # Generate with fixed seed for reproducibility
        torch.manual_seed(42)
        outputs = model.generate(
            input_ids, max_length=50, num_return_sequences=1, temperature=0.7, top_p=0.9, do_sample=True
        )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Verify output is not empty and contains the input text
        self.assertIsNotNone(generated_text)
        self.assertIn(input_text, generated_text)
        self.assertGreater(len(generated_text), len(input_text))


@require_torch
class HindiCausalLMModelForCausalLMDocTest(TestCasePlus):
    @slow
    def test_model_forward_example(self):
        """Test the forward pass example in HindiCausalLMForCausalLM docstring."""
        model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-causal-lm")
        tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")
        model.to(torch_device)
        model.eval()

        prompt = "भारत एक विशाल देश है"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Check output shapes
        self.assertEqual(outputs.logits.shape[0], 1)  # batch size
        self.assertEqual(outputs.logits.shape[2], model.config.vocab_size)  # vocab size

        # Generate
        torch.manual_seed(0)
        generate_ids = model.generate(inputs.input_ids, max_length=30)

        # Decode
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Verify the result contains the input
        self.assertIn(prompt, result)


if __name__ == "__main__":
    unittest.main()
