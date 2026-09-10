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

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch

    from transformers.modeling_flash_attention_utils import _is_packed_sequence


@require_torch
class IsPackedSequenceTest(unittest.TestCase):
    def test_none_position_ids(self):
        self.assertFalse(_is_packed_sequence(None, batch_size=1))

    def test_1d_packed(self):
        # Pixtral-style `position_ids_in_meshgrid` output: flattened, multiple increasing runs.
        position_ids = torch.tensor([0, 1, 2, 0, 1])
        self.assertTrue(bool(_is_packed_sequence(position_ids, batch_size=1)))

    def test_1d_unpacked(self):
        position_ids = torch.arange(5)
        self.assertFalse(bool(_is_packed_sequence(position_ids, batch_size=1)))

    def test_2d_packed(self):
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
        self.assertTrue(bool(_is_packed_sequence(position_ids, batch_size=1)))

    def test_2d_unpacked_batch_size_one(self):
        position_ids = torch.tensor([[0, 1, 2, 3, 4]])
        self.assertFalse(bool(_is_packed_sequence(position_ids, batch_size=1)))

    def test_2d_unpacked_batch_size_greater_than_one(self):
        # Regular padded batch: never treated as packed, regardless of batch_size passed in.
        position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
        self.assertFalse(bool(_is_packed_sequence(position_ids, batch_size=2)))

    def test_3d_mrope_packed(self):
        # MRoPE-style [num_position_axes, batch_size, seq_len], all axes share the packing layout.
        position_ids = torch.stack([torch.tensor([[0, 1, 2, 0, 1, 2, 3]])] * 3)
        self.assertEqual(position_ids.ndim, 3)
        self.assertTrue(bool(_is_packed_sequence(position_ids, batch_size=1)))

    def test_3d_mrope_unpacked_not_misclassified(self):
        # Regression test: an unpacked, multi-example MRoPE batch must not be sent down the varlen path.
        position_ids = torch.stack([torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])] * 3)
        self.assertEqual(position_ids.ndim, 3)
        self.assertFalse(bool(_is_packed_sequence(position_ids, batch_size=2)))
