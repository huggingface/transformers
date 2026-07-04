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

import random
import unittest

import torch

from transformers.integrations.deepspeed import _aggregate_weighted_sp_loss
from transformers.testing_utils import require_torch, torch_device


def _reference_aggregate(losses, good_tokens):
    total_loss = sum(loss * tokens for loss, tokens in zip(losses, good_tokens) if tokens > 0)
    total_good_tokens = sum(good_tokens)
    return total_loss / max(total_good_tokens, 1)


def _aggregate_tensors(losses, good_tokens, device):
    losses_per_rank = [torch.tensor(loss, device=device, dtype=torch.float32) for loss in losses]
    good_tokens_per_rank = [torch.tensor(count, device=device) for count in good_tokens]
    return _aggregate_weighted_sp_loss(losses_per_rank, good_tokens_per_rank)


@require_torch
class DeepspeedSpLossAggregationTest(unittest.TestCase):
    def _aggregate(self, losses, good_tokens):
        return _aggregate_tensors(losses, good_tokens, torch_device)

    def test_matches_reference_weighted_average(self):
        cases = [
            ([2.0, 4.0], [3, 1]),
            ([1.5, 0.0, 3.0], [2, 0, 4]),
            ([5.0, 6.0], [0, 0]),
            ([float("nan"), 2.0], [0, 5]),
            ([1.2, float("nan"), 3.4], [10, 0, 2]),
        ]
        for losses, good_tokens in cases:
            expected = _reference_aggregate(losses, good_tokens)
            actual = self._aggregate(losses, good_tokens).item()
            if expected != expected:
                self.assertTrue(actual != actual or actual == 0.0)
            else:
                self.assertAlmostEqual(actual, expected, places=5)

    def test_zero_token_rank_with_nan_is_ignored(self):
        result = self._aggregate([float("nan"), 2.0], [0, 4])
        self.assertEqual(result.item(), 2.0)

    def test_random_cases_match_reference(self):
        random.seed(0)
        for _ in range(200):
            rank_count = random.randint(1, 8)
            losses = [random.uniform(0, 10) for _ in range(rank_count)]
            good_tokens = [random.randint(0, 20) for _ in range(rank_count)]
            if random.random() < 0.1:
                losses[random.randrange(rank_count)] = float("nan")

            expected = _reference_aggregate(losses, good_tokens)
            actual = self._aggregate(losses, good_tokens).item()
            if expected != expected:
                self.assertTrue(actual != actual or actual == 0.0)
            else:
                self.assertAlmostEqual(actual, expected, places=5)

    def test_preserves_gradients_for_nonzero_token_ranks(self):
        device = torch_device
        losses = [
            torch.tensor(2.0, device=device, requires_grad=True),
            torch.tensor(4.0, device=device, requires_grad=True),
        ]
        good_tokens = [torch.tensor(3, device=device), torch.tensor(1, device=device)]
        result = _aggregate_weighted_sp_loss(losses, good_tokens)
        result.backward()
        self.assertAlmostEqual(losses[0].grad.item(), 0.75)
        self.assertAlmostEqual(losses[1].grad.item(), 0.25)

    def test_uses_tensor_ops_on_device(self):
        device = torch_device
        losses_per_rank = [
            torch.tensor(2.0, device=device),
            torch.tensor(4.0, device=device),
        ]
        good_tokens_per_rank = [
            torch.tensor(3, device=device),
            torch.tensor(1, device=device),
        ]

        loss = _aggregate_weighted_sp_loss(losses_per_rank, good_tokens_per_rank)
        self.assertEqual(loss.item(), _reference_aggregate([2.0, 4.0], [3, 1]))
        self.assertEqual(loss.device.type, torch.device(device).type)


if __name__ == "__main__":
    unittest.main()
