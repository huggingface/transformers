# Copyright 2026 The HuggingFace Team.
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

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from transformers.loss import fixed_cross_entropy

if is_torch_available():
    import torch
    import torch.nn as nn


@require_torch
class FixedCrossEntropyTester(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_ignores_unknown_kwargs(self):
        source = torch.randn(4, 10, requires_grad=True)
        target = torch.randint(0, 10, (4,))

        loss = fixed_cross_entropy(
            source,
            target,
            some_unknown_kwarg=123,
            another_one="ignored",
        )

        expected = nn.functional.cross_entropy(source, target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_sum_reduction_and_tensor_normalization(self):
        source = torch.randn(6, 5, requires_grad=True)
        target = torch.randint(0, 5, (6,))
        num_items = torch.tensor(6)

        loss = fixed_cross_entropy(
            source,
            target,
            num_items_in_batch=num_items,
        )

        expected = (
            nn.functional.cross_entropy(source, target, reduction="sum")
            / num_items
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_sum_reduction_and_int_normalization(self):
        source = torch.randn(8, 3, requires_grad=True)
        target = torch.randint(0, 3, (8,))
        num_items = 8

        loss = fixed_cross_entropy(
            source,
            target,
            num_items_in_batch=num_items,
        )

        expected = (
            nn.functional.cross_entropy(source, target, reduction="sum")
            / num_items
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_passes_valid_kwargs_only(self):
        source = torch.randn(5, 4, requires_grad=True)
        target = torch.randint(0, 4, (5,))

        weight = torch.rand(4)

        loss = fixed_cross_entropy(
            source,
            target,
            weight=weight,
            label_smoothing=0.1,
            invalid_kwarg=True,
        )

        expected = nn.functional.cross_entropy(
            source,
            target,
            weight=weight,
            label_smoothing=0.1,
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_loss_device_matches_input(self):
        source = torch.randn(4, 5)
        target = torch.randint(0, 5, (4,))
        num_items = torch.tensor(4)

        loss = fixed_cross_entropy(
            source,
            target,
            num_items_in_batch=num_items,
        )

        self.assertEqual(loss.device, source.device)
