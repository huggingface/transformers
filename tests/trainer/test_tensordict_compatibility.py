# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""
Tests for TensorDict compatibility with data collators and tokenizers.

This module tests that dict-like objects (specifically TensorDict) work correctly
with transformers' padding and collation functionality. TensorDict implements
__iter__ to iterate over batch dimensions rather than keys, which requires
explicit .keys() calls in the codebase.
"""

import unittest

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    is_torch_available,
)
from transformers.testing_utils import require_tensordict, require_torch


if is_torch_available():
    import torch


@require_torch
@require_tensordict
class TensorDictCompatibilityTest(unittest.TestCase):
    """Test suite for TensorDict compatibility with data collators and tokenizers."""

    def setUp(self):
        """Set up test fixtures."""
        from tensordict import TensorDict

        self.TensorDict = TensorDict
        # Use a small, fast-loading tokenizer for tests
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def test_data_collator_with_padding_tensordict(self):
        """
        Test that DataCollatorWithPadding works correctly with TensorDict inputs.

        This is a regression test for issue where TensorDict.__iter__() iterates
        over batch dimensions instead of keys, causing RuntimeError: generator raised StopIteration.
        """
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Create batch with TensorDict objects of different lengths
        batch = [
            self.TensorDict(
                {"input_ids": torch.tensor([9, 8, 7]), "attention_mask": torch.tensor([1, 1, 1])},
                batch_size=[],
            ),
            self.TensorDict(
                {"input_ids": torch.tensor([6, 5]), "attention_mask": torch.tensor([1, 1])}, batch_size=[]
            ),
        ]

        # This should not raise RuntimeError
        result = collator(batch)

        # Verify the output is correctly padded (can be dict or Mapping like BatchEncoding)
        from collections.abc import Mapping
        self.assertIsInstance(result, Mapping)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

        # Check shapes - should be padded to max length (3)
        self.assertEqual(result["input_ids"].shape, torch.Size([2, 3]))
        self.assertEqual(result["attention_mask"].shape, torch.Size([2, 3]))

        # Check padding is correct
        expected_input_ids = torch.tensor([[9, 8, 7], [6, 5, self.tokenizer.pad_token_id]])
        expected_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        self.assertTrue(torch.equal(result["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(result["attention_mask"], expected_attention_mask))

    def test_data_collator_with_padding_tensordict_variable_lengths(self):
        """Test DataCollatorWithPadding with TensorDict inputs of highly variable lengths."""
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        batch = [
            self.TensorDict(
                {"input_ids": torch.tensor([1, 2, 3, 4, 5]), "attention_mask": torch.tensor([1, 1, 1, 1, 1])},
                batch_size=[],
            ),
            self.TensorDict(
                {"input_ids": torch.tensor([6]), "attention_mask": torch.tensor([1])}, batch_size=[]
            ),
            self.TensorDict(
                {"input_ids": torch.tensor([7, 8, 9]), "attention_mask": torch.tensor([1, 1, 1])}, batch_size=[]
            ),
        ]

        result = collator(batch)

        # Should be padded to max length (5)
        self.assertEqual(result["input_ids"].shape, torch.Size([3, 5]))
        self.assertEqual(result["attention_mask"].shape, torch.Size([3, 5]))

        # Check that shorter sequences are padded
        self.assertEqual(result["input_ids"][1, 1:].tolist(), [self.tokenizer.pad_token_id] * 4)
        self.assertEqual(result["attention_mask"][1, 1:].tolist(), [0] * 4)

    def test_data_collator_language_modeling_tensordict(self):
        """Test DataCollatorForLanguageModeling with TensorDict inputs."""
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        batch = [
            self.TensorDict(
                {"input_ids": torch.tensor([1, 2, 3, 4])},
                batch_size=[],
            ),
            self.TensorDict(
                {"input_ids": torch.tensor([5, 6])},
                batch_size=[],
            ),
        ]

        result = collator(batch)

        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        # Should be padded
        self.assertEqual(result["input_ids"].shape[0], 2)
        self.assertEqual(result["labels"].shape[0], 2)

    def test_tokenizer_pad_method_with_tensordict(self):
        """Test tokenizer.pad() method directly with TensorDict inputs."""
        # Create pre-tokenized inputs as TensorDict
        batch = [
            self.TensorDict(
                {
                    "input_ids": torch.tensor([101, 2023, 2003, 102]),
                    "attention_mask": torch.tensor([1, 1, 1, 1]),
                },
                batch_size=[],
            ),
            self.TensorDict(
                {
                    "input_ids": torch.tensor([101, 102]),
                    "attention_mask": torch.tensor([1, 1]),
                },
                batch_size=[],
            ),
        ]

        # This should not raise RuntimeError
        result = self.tokenizer.pad(batch, return_tensors="pt")

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(result["input_ids"].shape, torch.Size([2, 4]))

    def test_mixed_tensordict_and_dict_inputs(self):
        """Test that collator handles mixed TensorDict and regular dict inputs gracefully."""
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Mix of TensorDict and regular dict
        batch = [
            self.TensorDict(
                {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])}, batch_size=[]
            ),
            {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1])},
        ]

        result = collator(batch)

        self.assertEqual(result["input_ids"].shape, torch.Size([2, 3]))
        self.assertEqual(result["attention_mask"].shape, torch.Size([2, 3]))

    def test_tensordict_with_additional_fields(self):
        """Test TensorDict inputs with additional fields beyond input_ids and attention_mask."""
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        batch = [
            self.TensorDict(
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                    "token_type_ids": torch.tensor([0, 0, 0]),
                    "special_tokens_mask": torch.tensor([1, 0, 1]),
                },
                batch_size=[],
            ),
            self.TensorDict(
                {
                    "input_ids": torch.tensor([4, 5]),
                    "attention_mask": torch.tensor([1, 1]),
                    "token_type_ids": torch.tensor([0, 0]),
                    "special_tokens_mask": torch.tensor([1, 0]),
                },
                batch_size=[],
            ),
        ]

        result = collator(batch)

        # All fields should be present and padded
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("token_type_ids", result)
        self.assertIn("special_tokens_mask", result)

        # Check all are padded to same length
        for key in ["input_ids", "attention_mask", "token_type_ids", "special_tokens_mask"]:
            self.assertEqual(result[key].shape, torch.Size([2, 3]), f"Field {key} has wrong shape")

    def test_single_tensordict_input(self):
        """Test collator with a single TensorDict input."""
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        batch = [
            self.TensorDict(
                {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])}, batch_size=[]
            ),
        ]

        result = collator(batch)

        # Single input should not cause issues
        self.assertEqual(result["input_ids"].shape, torch.Size([1, 3]))
        self.assertEqual(result["attention_mask"].shape, torch.Size([1, 3]))


if __name__ == "__main__":
    unittest.main()
