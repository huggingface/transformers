# Copyright 2025 The HuggingFace Inc. team.
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
Tests for per-sample nested structure handling in trainer_pt_utils.
Fixes issue #43388: gather_for_metrics incorrectly truncates Mask2Former-style labels.
"""

import unittest

import numpy as np
import torch

from transformers.trainer_pt_utils import (
    flatten_per_sample_nested_batches,
    is_per_sample_nested,
)


class TestIsPerSampleNested(unittest.TestCase):
    """Tests for is_per_sample_nested function."""

    def test_tuple_of_lists_of_tensors(self):
        """Tuple of lists of tensors should be detected."""
        labels = ([torch.randn(5, 64), torch.randn(3, 64)], [torch.arange(5), torch.arange(3)])
        self.assertTrue(is_per_sample_nested(labels))

    def test_tuple_of_lists_of_numpy(self):
        """Tuple of lists of numpy arrays should be detected."""
        labels = ([np.random.randn(5, 64), np.random.randn(3, 64)], [np.arange(5), np.arange(3)])
        self.assertTrue(is_per_sample_nested(labels))

    def test_single_tensor(self):
        """Single tensor should not be detected."""
        self.assertFalse(is_per_sample_nested(torch.randn(10, 64)))

    def test_tuple_of_tensors(self):
        """Tuple of tensors (not lists) should not be detected."""
        self.assertFalse(is_per_sample_nested((torch.randn(10, 64), torch.randn(10, 32))))

    def test_empty_tuple(self):
        """Empty tuple should not be detected."""
        self.assertFalse(is_per_sample_nested(()))

    def test_list_not_tuple(self):
        """List (not tuple) should not be detected."""
        self.assertFalse(is_per_sample_nested([[torch.randn(5, 64)], [torch.arange(5)]]))


class TestFlattenPerSampleNestedBatches(unittest.TestCase):
    """Tests for flatten_per_sample_nested_batches function."""

    def test_flatten_multiple_batches(self):
        """Should flatten multiple batches and truncate."""
        batches = [
            ([torch.randn(5, 64), torch.randn(3, 64)], [torch.arange(5), torch.arange(3)]),
            ([torch.randn(7, 64), torch.randn(4, 64)], [torch.arange(7), torch.arange(4)]),
            ([torch.randn(2, 64)], [torch.arange(2)]),
        ]

        result = flatten_per_sample_nested_batches(batches, num_samples=5)

        self.assertEqual(len(result), 2)  # Two label types
        self.assertEqual(len(result[0]), 5)  # 5 images (truncated from 5)
        self.assertEqual(len(result[1]), 5)

    def test_flatten_preserves_shapes(self):
        """Should preserve individual tensor shapes."""
        batches = [
            ([torch.randn(5, 256, 256), torch.randn(3, 256, 256)], [torch.arange(5), torch.arange(3)]),
            ([torch.randn(7, 256, 256)], [torch.arange(7)]),
        ]

        result = flatten_per_sample_nested_batches(batches, num_samples=3)

        self.assertEqual(result[0][0].shape, torch.Size([5, 256, 256]))
        self.assertEqual(result[0][1].shape, torch.Size([3, 256, 256]))
        self.assertEqual(result[0][2].shape, torch.Size([7, 256, 256]))

    def test_truncate_to_one(self):
        """Should handle truncation to 1 sample (remainder=1 scenario)."""
        batches = [([torch.randn(3, 64)], [torch.arange(3)])]

        result = flatten_per_sample_nested_batches(batches, num_samples=1)

        self.assertEqual(len(result), 2)  # Both label types preserved
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)

    def test_empty_batches(self):
        """Should return None for empty batches."""
        self.assertIsNone(flatten_per_sample_nested_batches([], num_samples=5))


class TestMask2FormerScenario(unittest.TestCase):
    """End-to-end test simulating Mask2Former evaluation."""

    def test_full_evaluation_scenario(self):
        """Simulate full evaluation with multiple batches."""
        # 3 batches: 2+2+1 = 5 images, but dataset has 4 images
        batches = [
            ([torch.randn(5, 256, 256), torch.randn(3, 256, 256)],
             [torch.randint(0, 10, (5,)), torch.randint(0, 10, (3,))]),
            ([torch.randn(7, 256, 256), torch.randn(4, 256, 256)],
             [torch.randint(0, 10, (7,)), torch.randint(0, 10, (4,))]),
            ([torch.randn(2, 256, 256)],
             [torch.randint(0, 10, (2,))]),
        ]

        # Simulate what Trainer does
        result = flatten_per_sample_nested_batches(batches, num_samples=4)

        # Should have 4 images
        self.assertEqual(len(result[0]), 4)
        self.assertEqual(len(result[1]), 4)

        # Instance counts should be preserved
        self.assertEqual(result[0][0].shape[0], 5)  # First image: 5 instances
        self.assertEqual(result[0][1].shape[0], 3)  # Second image: 3 instances
        self.assertEqual(result[0][2].shape[0], 7)  # Third image: 7 instances
        self.assertEqual(result[0][3].shape[0], 4)  # Fourth image: 4 instances


if __name__ == "__main__":
    unittest.main()
