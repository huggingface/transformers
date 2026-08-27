# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch

from transformers.integrations.moe import _can_use_grouped_mm, _grouped_mm, _has_valid_grouped_mm_strides
from transformers.testing_utils import require_torch


@require_torch
class GroupedMmCompatibilityTest(unittest.TestCase):
    def test_rejects_unaligned_transposed_weight_stride(self):
        inputs = torch.empty(8, 1025, dtype=torch.bfloat16)
        weights = torch.empty(4, 512, 1025, dtype=torch.bfloat16).transpose(-2, -1)
        offsets = torch.tensor([2, 4, 6, 8], dtype=torch.int32)

        self.assertFalse(_has_valid_grouped_mm_strides(weights))
        self.assertFalse(_can_use_grouped_mm(inputs, weights, offsets))

    def test_accepts_aligned_transposed_weight_stride(self):
        weights = torch.empty(4, 512, 1024, dtype=torch.bfloat16).transpose(-2, -1)

        self.assertTrue(_has_valid_grouped_mm_strides(weights))

    def test_unaligned_stride_uses_fallback(self):
        inputs = torch.randn(4, 5)
        weights = torch.randn(2, 4, 5).transpose(-2, -1)
        offsets = torch.tensor([2, 4], dtype=torch.int32)

        result = _grouped_mm(inputs, weights, offsets)
        expected = torch.cat([inputs[:2] @ weights[0], inputs[2:] @ weights[1]])

        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
