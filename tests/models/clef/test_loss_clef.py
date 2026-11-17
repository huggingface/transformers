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

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.loss.loss_clef import ClefForDecisionLoss
    from transformers.loss.loss_utils import LOSS_MAPPING


def naive_loss(logits, target, mask, is_score, ls, bw, ow):
    """The formula of the spec as an explicit loop over the questions, on the options that exist only."""
    base, brier, ordinal, num_score = 0.0, 0.0, 0.0, 0
    for q in range(logits.shape[0]):
        k = int(mask[q].sum())
        p = logits[q, :k].softmax(-1)
        t = target[q, :k]
        smoothed = (1 - ls) * t + ls / k if ls > 0 else t
        base = base - (smoothed * p.log()).sum()
        brier = brier + ((p - t) ** 2).sum()
        if is_score[q]:
            num_score += 1
            e = sum(p[i] * abs(i - j) * t[j] for i in range(k) for j in range(k))
            ordinal = ordinal + e / max(1, k - 1)
    count = logits.shape[0]
    total = base / count + bw * brier / count
    return total + ow * ordinal / max(1, num_score)


def random_case(questions, max_options, seed, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    widths = torch.randint(1, max_options + 1, (questions,), generator=generator)
    widths[0], widths[1] = max_options, 1  # a full row and a single option row
    mask = torch.arange(max_options)[None] < widths[:, None]
    logits = torch.randn(questions, max_options, generator=generator) * 2
    target = torch.zeros(questions, max_options)
    soft = torch.rand(questions, max_options, generator=generator) * mask
    target[: questions // 2] = soft[: questions // 2] / soft[: questions // 2].sum(-1, keepdim=True).clamp(min=1e-9)
    for q in range(questions // 2, questions):
        target[q, int(torch.randint(0, int(widths[q]), (1,), generator=generator))] = 1.0
    is_score = torch.rand(questions, generator=generator) < 0.5
    is_score[2] = True
    return logits.requires_grad_(requires_grad), target, mask, is_score


@require_torch
class ClefLossTest(unittest.TestCase):
    def test_registered(self):
        self.assertIs(LOSS_MAPPING["ClefForDecision"], ClefForDecisionLoss)

    @parameterized.expand([(0, 0, 0), (0.1, 0, 0), (0, 0.5, 0), (0, 0, 0.3), (0.05, 0.2, 0.1)])
    def test_matches_naive(self, ls, bw, ow):
        logits, target, mask, is_score = random_case(questions=9, max_options=5, seed=0)
        got = ClefForDecisionLoss(
            logits, target, mask, is_score, label_smoothing=ls, brier_weight=bw, ordinal_weight=ow
        )
        expected = naive_loss(logits.double(), target.double(), mask, is_score, ls, bw, ow)
        torch.testing.assert_close(got.double(), expected, atol=1e-6, rtol=0)

    def test_padding_does_not_matter(self):
        logits, target, mask, is_score = random_case(questions=6, max_options=4, seed=1)
        wide = torch.nn.functional.pad(logits, (0, 3), value=1e4)
        wide_target, wide_mask = torch.nn.functional.pad(target, (0, 3)), torch.nn.functional.pad(mask, (0, 3))
        kwargs = {"label_smoothing": 0.1, "brier_weight": 0.2, "ordinal_weight": 0.3}
        torch.testing.assert_close(
            ClefForDecisionLoss(wide, wide_target, wide_mask, is_score, **kwargs),
            ClefForDecisionLoss(logits, target, mask, is_score, **kwargs),
        )

    def test_hard_and_soft_labels_agree(self):
        logits, _, mask, is_score = random_case(questions=4, max_options=3, seed=2)
        mask = torch.ones(4, 3, dtype=torch.bool)
        hard = torch.tensor([0, 2, -100, 1])
        soft = torch.nn.functional.one_hot(hard.clamp(min=0), 3).float()
        soft[2] = torch.tensor([0.2, 0.3, 0.5])  # the ignored question: any distribution
        keep = hard != -100
        kwargs = {"brier_weight": 0.2, "ordinal_weight": 0.3, "label_smoothing": 0.1}
        a = ClefForDecisionLoss(logits, hard, mask, is_score, **kwargs)
        b = ClefForDecisionLoss(logits[keep], soft[keep], mask[keep], is_score[keep], **kwargs)
        torch.testing.assert_close(a, b)

    def test_single_option_questions_are_finite(self):
        logits = torch.tensor([[1.0, 0, 0, 0], [0.5, 1.0, -1.0, 2.0], [2.0, 0, 0, 0]], requires_grad=True)
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
        is_score = torch.tensor([True, True, False])
        loss = ClefForDecisionLoss(
            logits, torch.tensor([0, 3, 0]), mask, is_score, label_smoothing=0.1, brier_weight=0.5, ordinal_weight=0.5
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        # A question with one option has nothing to learn: no gradient reaches its logit
        self.assertEqual(logits.grad[0].abs().sum().item(), 0.0)
        self.assertNotEqual(logits.grad[1].abs().sum().item(), 0.0)

    def test_num_items_in_batch_normalises_sum(self):
        logits, target, mask, is_score = random_case(questions=6, max_options=4, seed=3)
        kwargs = {"brier_weight": 0.3}
        default = ClefForDecisionLoss(logits, target, mask, is_score, **kwargs)
        scaled = ClefForDecisionLoss(logits, target, mask, is_score, num_items_in_batch=18, **kwargs)
        torch.testing.assert_close(scaled, default * 6 / 18)

    def test_all_ignored_returns_zero_with_grad(self):
        logits, _, mask, is_score = random_case(questions=3, max_options=3, seed=4, requires_grad=True)
        loss = ClefForDecisionLoss(
            logits, torch.full((3,), -100), mask, is_score, brier_weight=0.5, ordinal_weight=0.5
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)

    def test_gradients_match_naive(self):
        kwargs = {"ls": 0.05, "bw": 0.2, "ow": 0.1}
        logits, target, mask, is_score = random_case(questions=9, max_options=5, seed=5)
        double = logits.double().requires_grad_(True)
        ClefForDecisionLoss(
            logits.requires_grad_(True),
            target,
            mask,
            is_score,
            label_smoothing=0.05,
            brier_weight=0.2,
            ordinal_weight=0.1,
        ).backward()
        naive_loss(double, target.double(), mask, is_score, **kwargs).backward()
        torch.testing.assert_close(logits.grad.double(), double.grad, atol=1e-6, rtol=0)

    def test_minimum_padding_is_finite_and_matches_naive(self):
        # The padding of `ClefDecisionOutput.logits`: two or more cells at the minimum of the dtype
        logits, target, mask, is_score = random_case(questions=6, max_options=5, seed=6)
        mask[2], mask[3] = (
            torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
            torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool),
        )
        target = target * mask
        target = target / target.sum(-1, keepdim=True).clamp(min=1e-9)
        padded = logits.detach().masked_fill(~mask, torch.finfo(torch.float32).min).requires_grad_(True)
        self.assertGreaterEqual(int((~mask).sum()), 2)
        for weights in ({}, {"label_smoothing": 0.1, "brier_weight": 0.5, "ordinal_weight": 0.3}):
            padded.grad = None
            loss = ClefForDecisionLoss(padded, target, mask, is_score, **weights)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(padded.grad).all())
            expected = naive_loss(
                logits.double(),
                target.double(),
                mask,
                is_score,
                weights.get("label_smoothing", 0),
                weights.get("brier_weight", 0),
                weights.get("ordinal_weight", 0),
            )
            torch.testing.assert_close(loss.double(), expected, atol=1e-6, rtol=0)

    def test_all_ignored_with_minimum_padding_is_zero(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).masked_fill(
            torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.bool), torch.finfo(torch.float32).min
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min).requires_grad_(True)
        loss = ClefForDecisionLoss(logits, torch.full((2,), -100), mask, torch.tensor([True, False]), brier_weight=1.0)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(logits.grad).all())
