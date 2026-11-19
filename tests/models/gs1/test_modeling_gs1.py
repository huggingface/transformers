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

from transformers import GS1ForDecision
from transformers.loss.loss_gs1 import GS1ForDecisionLoss


class GS1DecisionLossTest(unittest.TestCase):
    all_model_classes = (GS1ForDecision,)

    def test_loss_matches_distribution_nll_and_expected_brier(self):
        logits = torch.tensor([0.2, -0.1, 1.0, 0.5, -0.5], requires_grad=True)
        labels = torch.tensor([0.75, 0.25, 0.2, 0.3, 0.5])
        counts = torch.tensor([2, 3])
        weights = torch.tensor([2.0, 0.5])

        loss = GS1ForDecisionLoss(
            logits,
            labels,
            counts,
            loss_weights=weights,
            brier_weight=0.25,
        )

        expected = []
        offset = 0
        for count, weight in zip((2, 3), weights, strict=True):
            scores = logits[offset : offset + count]
            target = labels[offset : offset + count]
            probability = scores.softmax(-1)
            nll = -(target * scores.log_softmax(-1)).sum()
            brier = (probability - target).square().sum() + 1 - target.square().sum()
            expected.append((nll + 0.25 * brier) * weight)
            offset += count
        torch.testing.assert_close(loss, torch.stack(expected).mean())

        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
