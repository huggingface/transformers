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

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from torch.nn import functional as F

    from transformers.decision import SparseCosineDecisionHead


@require_torch
class SparseCosineDecisionHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head = SparseCosineDecisionHead(hidden_size=8, width=8)
        self.questions = [(0, 1, 4), (0, 4, 6), (1, 0, 3)]
        self.options = [(0, 6, 8), (0, 8, 12), (0, 1, 3), (0, 3, 6), (0, 6, 9), (1, 3, 5), (1, 5, 9)]
        self.owners = [0, 0, 1, 1, 1, 2, 2]
        self.hidden = torch.randn(2, 15, 8)

    def reference(self, head, hidden):
        query = torch.stack([hidden[b, start:end].mean(0) for b, start, end in self.questions])
        options = torch.stack([hidden[b, start:end].mean(0) for b, start, end in self.options])
        query = F.normalize(head.question_norm(head.question_projection(query)).float(), dim=-1)
        options = F.normalize(head.option_norm(head.option_projection(options)).float(), dim=-1)
        return (query[self.owners] * options).sum(-1) * head.logit_scale.exp()

    def test_logits_and_gradients_match_span_mean_reference(self):
        reference = copy.deepcopy(self.head)
        actual_hidden = self.hidden.clone().requires_grad_()
        reference_hidden = self.hidden.clone().requires_grad_()
        actual = self.head(actual_hidden, self.questions, self.options, self.owners)
        expected = self.reference(reference, reference_hidden)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(actual_hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-5)
        for parameter, ref_parameter in zip(self.head.parameters(), reference.parameters()):
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=1e-5, atol=1e-5)

    def test_option_permutation_padding_and_noncontiguous_input(self):
        reference = self.head(self.hidden, self.questions, self.options, self.owners)
        perm = [4, 3, 2, 1, 0, 6, 5]
        permuted = self.head(
            self.hidden, self.questions, [self.options[i] for i in perm], [self.owners[i] for i in perm]
        )
        torch.testing.assert_close(permuted, reference[perm])
        changed = self.hidden.clone()
        changed[0, 12:] = 10000
        changed[1, 9:] = -10000
        torch.testing.assert_close(self.head(changed, self.questions, self.options, self.owners), reference)
        noncontiguous = torch.stack((self.hidden, self.hidden), dim=2)[:, :, 0]
        self.assertFalse(noncontiguous.is_contiguous())
        torch.testing.assert_close(self.head(noncontiguous, self.questions, self.options, self.owners), reference)

    def test_metadata_validation(self):
        for bad in [(-1, 1, 3), (2, 1, 3), (0, -1, 2), (0, 2, 2), (0, 2, 16), (0, 1), (0, 1.5, 3)]:
            with self.subTest(span=bad), self.assertRaises(ValueError):
                self.head(self.hidden, [bad, *self.questions[1:]], self.options, self.owners)
        for owners in [[-1, *self.owners[1:]], [3, *self.owners[1:]], [2, *self.owners[1:]], self.owners[:-1]]:
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                self.head(self.hidden, self.questions, self.options, owners)
        with self.assertRaises(TypeError):
            self.head(self.hidden, torch.tensor(self.questions), self.options, self.owners)
        with self.assertRaises(ValueError):
            self.head(self.hidden, [], self.options, self.owners)

    def test_parameter_count_and_scale_bound(self):
        self.assertEqual(sum(p.numel() for p in SparseCosineDecisionHead(512).parameters()), 527361)
        self.assertAlmostEqual(self.head.logit_scale.exp().item(), 12.0, places=4)
        with torch.no_grad():
            self.head.logit_scale.fill_(20.0)
        self.assertLessEqual(
            self.head(self.hidden, self.questions, self.options, self.owners).abs().max().item(), 100.001
        )
        for scale in [0.0, -1.0, 101.0, float("nan"), float("inf")]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                SparseCosineDecisionHead(8, initial_scale=scale)

    def test_local_checkpoint_roundtrip_and_type_guard(self):
        expected = self.head(self.hidden, self.questions, self.options, self.owners)
        with tempfile.TemporaryDirectory() as tmp:
            self.head.save_pretrained(tmp)
            restored = SparseCosineDecisionHead.from_pretrained(tmp, local_files_only=True)
            self.assertEqual(restored.config, self.head.config)
            torch.testing.assert_close(
                restored(self.hidden, self.questions, self.options, self.owners), expected, rtol=0, atol=0
            )
            config_path = Path(tmp) / "joint_head_config.json"
            config = json.loads(config_path.read_text())
            config["head_type"] = "clef"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "cosine_span_v1"):
                SparseCosineDecisionHead.from_pretrained(tmp, local_files_only=True)
