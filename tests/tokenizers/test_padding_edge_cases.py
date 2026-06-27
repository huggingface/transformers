# Copyright 2026 HuggingFace Inc.
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

import unittest

from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
)


if is_torch_available():
    from transformers import AutoModelForCausalLM, AutoTokenizer


@require_torch
class PaddingEdgeCaseTest(unittest.TestCase):
    def test_gpt2_left_padding_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"
        prompts = ["Hello world", "The quick brown fox jumps over the lazy dog"]
        inputs_left = tokenizer(prompts, padding=True, return_tensors="pt")

        tokenizer.padding_side = "right"
        inputs_right = tokenizer(prompts, padding=True, return_tensors="pt")

        left_attention = inputs_left["attention_mask"]
        right_attention = inputs_right["attention_mask"]

        self.assertNotEqual(
            left_attention[0, 0].item(),
            left_attention[0, -1].item(),
            "Left-padded mask should have 0s at the start",
        )
        self.assertEqual(
            right_attention[0, 0].item(),
            1,
            "Right-padded mask should have 1s at the start (shorter sequence first)",
        )

    @slow
    def test_gpt2_left_vs_right_padding_generation_diff(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        prompts = ["Hello world", "The quick brown fox jumps over the lazy dog"]

        tokenizer.padding_side = "left"
        inputs_left = tokenizer(prompts, padding=True, return_tensors="pt")
        outputs_left = model.generate(**inputs_left, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
        decoded_left = tokenizer.batch_decode(outputs_left, skip_special_tokens=True)

        tokenizer.padding_side = "right"
        inputs_right = tokenizer(prompts, padding=True, return_tensors="pt")
        outputs_right = model.generate(**inputs_right, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
        decoded_right = tokenizer.batch_decode(outputs_right, skip_special_tokens=True)

        self.assertEqual(len(decoded_left), 2)
        self.assertEqual(len(decoded_right), 2)

        left_first_prompt = (
            decoded_left[1] if len(inputs_left["input_ids"][0]) < len(inputs_left["input_ids"][1]) else decoded_left[0]
        )

        self.assertGreater(
            len(left_first_prompt),
            5,
            "Left-padded generation should produce a meaningful continuation",
        )

    @slow
    def test_llama_pad_token_eos_conflict(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.pad_token = tokenizer.eos_token

        prompts = ["The meaning of life is", "Once upon a time"]
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")

        self.assertEqual(inputs["input_ids"].shape[0], 2)
        self.assertTrue((inputs["attention_mask"][0] == 1).any())
        self.assertTrue((inputs["attention_mask"][1] == 1).all())

    def test_gpt2_padding_side_default(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.assertEqual(
            tokenizer.padding_side,
            "right",
            "GPT-2 tokenizer should default to right padding",
        )

    def test_pad_token_set_to_eos(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        self.assertIsNotNone(tokenizer.pad_token_id)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)

    def test_variable_length_batch_padding(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        prompts = ["Hi", "Hello world", "The quick brown fox"]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")

        self.assertEqual(inputs["input_ids"].shape[0], 3)
        max_len = max(len(tok) for tok in tokenizer(prompts)["input_ids"])
        self.assertEqual(inputs["input_ids"].shape[1], max_len)

        for i, seq_len in enumerate([len(tok) for tok in tokenizer(prompts)["input_ids"]]):
            pad_count = max_len - seq_len
            self.assertEqual(
                inputs["attention_mask"][i, :pad_count].sum().item(),
                0,
                f"Sequence {i} should have {pad_count} padding tokens at the start",
            )
            self.assertEqual(
                inputs["attention_mask"][i, pad_count:].sum().item(),
                seq_len,
                f"Sequence {i} should have {seq_len} unpadded tokens",
            )


@require_torch
class BatchedGenerationPaddingTest(unittest.TestCase):
    @slow
    def test_batched_generation_with_padding(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        prompts = ["Hello", "The quick brown fox"]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

        self.assertEqual(outputs.shape[0], 2)
        self.assertEqual(outputs.shape[1], inputs["input_ids"].shape[1] + 10)
