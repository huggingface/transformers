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

import math
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.loss.loss_clef import ClefForDecisionLoss, clef_proper_reward
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


def naive_reward(q, t, is_score, sph=0.75, rps=1.0, log_floor=-9.21):
    """The proper reward of the spec for the probabilities `q` of the options of a question and their target `t`."""
    k = len(q)
    reward = sum(t[i] * max(math.log(max(q[i], 1e-12)), log_floor) for i in range(k))
    norm = max(math.sqrt(sum(x * x for x in q)), 1e-9)
    reward += sph * sum(t[i] * q[i] for i in range(k)) / norm
    if is_score:
        cdf_q = cdf_t = total = 0.0
        for i in range(k):
            cdf_q, cdf_t = cdf_q + q[i], cdf_t + t[i]
            total += (cdf_q - cdf_t) ** 2
        reward -= rps * total / (max(k, 2) - 1)
    return reward


def naive_rlcd(logits, targets, mask, is_score, eps_raw, sigma, weight, normalize, sph=0.75, rps=1.0):
    """
    `weight` times the RLCD policy term of the spec, as float64 loops over the samples, questions and options, from the
    raw normal draw `eps_raw` of shape `(samples, Q, K)`. It is differentiable in `logits` through `logp` only.
    """
    samples, questions = eps_raw.shape[0], logits.shape[0]
    detached = logits.detach()
    rewards = [[0.0] * questions for _ in range(samples)]
    logps = [[None] * questions for _ in range(samples)]
    for s in range(samples):
        for qi in range(questions):
            k = int(mask[qi].sum())
            raw = [float(eps_raw[s, qi, i]) * sigma for i in range(k)]
            mean = sum(raw) / k
            sample = [float(detached[qi, i]) + raw[i] - mean for i in range(k)]
            top = max(sample)
            exp = [math.exp(v - top) for v in sample]
            q = [v / sum(exp) for v in exp]
            t = [float(targets[qi, i]) for i in range(k)]
            rewards[s][qi] = naive_reward(q, t, bool(is_score[qi]), sph, rps)
            logps[s][qi] = -sum((sample[i] - logits[qi, i]) ** 2 for i in range(k)) / (2 * sigma**2)
    advantages = [[0.0] * questions for _ in range(samples)]
    for qi in range(questions):
        baseline = sum(rewards[s][qi] for s in range(samples)) / samples
        for s in range(samples):
            advantages[s][qi] = rewards[s][qi] - baseline
    if normalize:
        values = [a for row in advantages for a in row]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((a - mean) ** 2 for a in values) / (len(values) - 1)) + 1e-6
        advantages = [[a / std for a in row] for row in advantages]
    policy = 0.0
    for qi in range(questions):
        policy = policy + sum(-advantages[s][qi] * logps[s][qi] for s in range(samples)) / samples
    return weight * policy / questions


def oracle_noise(eps_raw, mask, sigma):
    """The noise of the spec from the raw draw: masked, scaled, and of zero mean over the options of each question."""
    m = mask.float()
    eps = eps_raw * sigma * m
    return (eps - eps.sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1)) * m


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

    @parameterized.expand([(32, False), (4, True)])
    def test_rlcd_matches_naive(self, samples, normalize):
        logits, targets, mask, is_score = random_case(questions=7, max_options=5, seed=1)
        self.assertIn(1, mask.sum(-1).tolist())
        g = torch.Generator().manual_seed(7)
        got = ClefForDecisionLoss(
            logits,
            targets,
            mask,
            is_score,
            rlcd_weight=0.25,
            rlcd_samples=samples,
            rlcd_normalize_advantage=normalize,
            generator=g,
        )
        eps_raw = torch.randn((samples, *logits.shape), generator=torch.Generator().manual_seed(7))
        base = ClefForDecisionLoss(logits, targets, mask, is_score)
        expected = base.double() + naive_rlcd(
            logits.double(), targets.double(), mask, is_score, eps_raw.double(), 0.4, 0.25, normalize
        )
        torch.testing.assert_close(got.double(), expected, atol=1e-5, rtol=0)

    def test_rlcd_gradient_is_the_score_function(self):
        logits, targets, mask, is_score = random_case(questions=7, max_options=5, seed=2)
        padded = logits.masked_fill(~mask, torch.finfo(torch.float32).min).requires_grad_(True)
        sigma, weight, samples = 0.4, 0.25, 32
        ClefForDecisionLoss(
            padded,
            targets,
            mask,
            is_score,
            rlcd_weight=weight,
            rlcd_samples=samples,
            generator=torch.Generator().manual_seed(3),
        ).backward()
        with_rlcd = padded.grad.clone()
        padded.grad = None
        ClefForDecisionLoss(padded, targets, mask, is_score).backward()
        policy_grad = with_rlcd - padded.grad
        eps_raw = torch.randn((samples, *logits.shape), generator=torch.Generator().manual_seed(3))
        eps = oracle_noise(eps_raw, mask, sigma)
        sample = logits + eps
        q = sample.masked_fill(~mask, torch.finfo(torch.float32).min).softmax(-1)
        reward = clef_proper_reward(q, targets * mask, mask, is_score)
        advantage = reward - reward.mean(0, keepdim=True)
        expected = weight * -(advantage[..., None] * eps).mean(0) / sigma**2 / logits.shape[0]
        torch.testing.assert_close(policy_grad, expected, atol=1e-5, rtol=0)
        self.assertTrue((policy_grad[~mask] == 0).all())
        self.assertTrue((with_rlcd[~mask] == 0).all())
        self.assertNotEqual(policy_grad.abs().sum().item(), 0.0)

    def test_zero_weight_is_identical_and_draws_nothing(self):
        logits, targets, mask, is_score = random_case(questions=7, max_options=5, seed=3)
        kwargs = {"label_smoothing": 0.1, "brier_weight": 0.2, "ordinal_weight": 0.3}
        previous = ClefForDecisionLoss(logits, targets, mask, is_score, **kwargs)
        generator = torch.Generator().manual_seed(11)
        generator_state, global_state = generator.get_state(), torch.get_rng_state()
        zero = ClefForDecisionLoss(
            logits,
            targets,
            mask,
            is_score,
            rlcd_weight=0.0,
            rlcd_sigma=0.7,
            rlcd_samples=8,
            rlcd_normalize_advantage=True,
            generator=generator,
            **kwargs,
        )
        unseeded = ClefForDecisionLoss(logits, targets, mask, is_score, rlcd_weight=0.0, **kwargs)
        self.assertTrue(torch.equal(zero, previous))
        self.assertTrue(torch.equal(unseeded, previous))
        self.assertTrue(torch.equal(generator.get_state(), generator_state))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_state))
        # A positive weight without a generator draws from the global generator
        ClefForDecisionLoss(logits, targets, mask, is_score, rlcd_weight=0.25, **kwargs)
        self.assertFalse(torch.equal(torch.get_rng_state(), global_state))

    def test_rlcd_invalid_settings_raise(self):
        # A negative weight would push the logits away from the samples with a high reward
        logits, targets, mask, is_score = random_case(questions=4, max_options=3, seed=5)
        for kwargs in (
            {"rlcd_weight": -0.25},
            {"rlcd_weight": -1e-6},
            {"rlcd_weight": 0.25, "rlcd_sigma": 0.0},
            {"rlcd_weight": 0.25, "rlcd_sigma": -0.4},
            {"rlcd_weight": 0.25, "rlcd_samples": 1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ClefForDecisionLoss(logits, targets, mask, is_score, **kwargs)
        # Out of training the weight is zero, and the other settings are not read
        loss = ClefForDecisionLoss(logits, targets, mask, is_score, rlcd_weight=0.0, rlcd_sigma=0.0, rlcd_samples=1)
        self.assertTrue(torch.isfinite(loss))

    def test_rlcd_single_option_and_one_level_score_are_finite(self):
        logits = torch.tensor([[1.0, 0, 0, 0], [0.5, 1.0, -1.0, 2.0], [2.0, 0, 0, 0]])
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
        logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min).requires_grad_(True)
        is_score = torch.tensor([True, True, False])
        for normalize in (False, True):
            logits.grad = None
            loss = ClefForDecisionLoss(
                logits,
                torch.tensor([0, 3, 0]),
                mask,
                is_score,
                brier_weight=0.5,
                ordinal_weight=0.5,
                rlcd_weight=0.25,
                rlcd_normalize_advantage=normalize,
                generator=torch.Generator().manual_seed(0),
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
            # A question with one option has nothing to learn, from the policy term either
            self.assertEqual(logits.grad[0].abs().sum().item(), 0.0)
            self.assertEqual(logits.grad[2].abs().sum().item(), 0.0)
            self.assertEqual(logits.grad[~mask].abs().sum().item(), 0.0)
            self.assertNotEqual(logits.grad[1].abs().sum().item(), 0.0)

    @parameterized.expand([(False,), (True,)])
    def test_rlcd_ignores_minus_100_questions(self, normalize):
        logits, targets, mask, is_score = random_case(questions=7, max_options=5, seed=4)
        hard = targets.argmax(-1)
        hard[[0, 3, 5]] = -100
        keep = hard != -100
        padded = logits.masked_fill(~mask, torch.finfo(torch.float32).min).requires_grad_(True)
        samples = 16
        rlcd = {"rlcd_weight": 0.25, "rlcd_samples": samples, "rlcd_normalize_advantage": normalize}
        got = ClefForDecisionLoss(
            padded, hard, mask, is_score, brier_weight=0.2, generator=torch.Generator().manual_seed(5), **rlcd
        )
        got.backward()
        # The loss on the kept questions alone, with the kept slice of the same noise
        eps_raw = torch.randn((samples, *logits.shape), generator=torch.Generator().manual_seed(5))
        kept = logits[keep].double().requires_grad_(True)
        onehot = torch.nn.functional.one_hot(hard[keep], logits.shape[-1]).double()
        expected = ClefForDecisionLoss(kept, hard[keep], mask[keep], is_score[keep], brier_weight=0.2).double()
        expected = expected + naive_rlcd(
            kept, onehot, mask[keep], is_score[keep], eps_raw[:, keep].double(), 0.4, 0.25, normalize
        )
        expected.backward()
        torch.testing.assert_close(got.double(), expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(padded.grad[keep].double() * mask[keep], kept.grad * mask[keep], atol=1e-5, rtol=0)
        self.assertEqual(padded.grad[~keep].abs().sum().item(), 0.0)
        self.assertEqual(padded.grad[~mask].abs().sum().item(), 0.0)

    def test_rlcd_num_items_in_batch(self):
        logits, targets, mask, is_score = random_case(questions=6, max_options=4, seed=6)

        def policy(**kwargs):
            generator = torch.Generator().manual_seed(9)
            total = ClefForDecisionLoss(
                logits, targets, mask, is_score, rlcd_weight=0.25, generator=generator, **kwargs
            )
            return total - ClefForDecisionLoss(logits, targets, mask, is_score, **kwargs)

        default, scaled = policy(), policy(num_items_in_batch=18)
        self.assertNotEqual(default.item(), 0.0)
        torch.testing.assert_close(scaled, default * 6 / 18, atol=1e-6, rtol=1e-5)

    def test_reward_is_proper(self):
        generator = torch.Generator().manual_seed(12)
        width = 4
        mask = torch.ones(1, width, dtype=torch.bool)
        outcomes = torch.eye(width)[:, None]  # every one-hot outcome, as the target of one question
        for score in (False, True):
            is_score = torch.tensor([score])
            for _ in range(5):
                t = torch.rand(width, generator=generator) + 0.05
                t = t / t.sum()

                def expected_reward(q, t=t, is_score=is_score):
                    # E_{y ~ t}[reward(q, onehot(y))] for each row of `q`, of shape (n, width)
                    rewards = torch.stack([clef_proper_reward(q[:, None], y, mask, is_score)[:, 0] for y in outcomes])
                    return (t[:, None] * rewards).sum(0)

                q = (torch.randn(200, width, generator=generator) * 1.5).softmax(-1)
                self.assertTrue((expected_reward(q) < expected_reward(t[None])).all())

    def test_reward_matches_laya_reference_values(self):
        uniform = torch.full((1, 4), 0.25)
        onehot = torch.tensor([[1.0, 0, 0, 0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        # log(1/4) + 0.75 * (1/4) / (1/2)
        choice = clef_proper_reward(uniform, onehot, mask, torch.tensor([False]))
        torch.testing.assert_close(choice, torch.tensor([math.log(0.25) + 0.375]))
        # minus the ranked probability score ((3/4)^2 + (1/2)^2 + (1/4)^2 + 0) / (4 - 1)
        score = clef_proper_reward(uniform, onehot, mask, torch.tensor([True]))
        torch.testing.assert_close(score, torch.tensor([math.log(0.25) + 0.375 - 0.875 / 3]))
        # No probability on the right option hits the floor of the log; the padded cell of `q` is ignored
        q = torch.tensor([[0.0, 0.5, 0.5, 0.3]])
        ragged = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        floored = clef_proper_reward(q, onehot, ragged, torch.tensor([True]), spherical_weight=0.5, rps_weight=2.0)
        torch.testing.assert_close(floored, torch.tensor([-9.21 - 2.0 * (1.0 + 0.25 + 0.0) / 2]))
        # Leading sample dimensions broadcast
        batched = clef_proper_reward(uniform.expand(3, 2, 1, 4), onehot, mask, torch.tensor([True]))
        self.assertEqual(batched.shape, (3, 2, 1))
