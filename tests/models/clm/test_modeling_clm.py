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
import tempfile
import unittest

from transformers import AutoConfig, AutoTokenizer, ClmConfig, is_torch_available, pipeline
from transformers.testing_utils import require_torch

from ...test_decision_head_common import DecisionHeadTesterMixin, render, tiny_qwen2


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import AutoModelForDecision, ClmForDecision, Trainer, TrainingArguments


@require_torch
class ClmForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (ClmForDecision,) if is_torch_available() else ()

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        cls.config = ClmConfig(
            text_config=tiny_qwen2(cls.tokenizer).config,
            hidden_size=24,
            projection_dim=8,
            num_hidden_layers=3,
        )
        cls.config._attn_implementation = "eager"
        cls.config.text_config._attn_implementation = "eager"
        cls.decision_config = ClmForDecision(cls.config).decision_config
        request = {
            "state": {"name": "Account", "facts": [True, {"balance": 12}]},
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Choose a team.",
                    "criteria": {"z": "Refunds", "a": None},
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "severity": {"type": "score", "criteria": ["Low", "High"]},
            },
        }
        expected_state = "name: Account\n\nfacts:\n  - true\n  -\n    balance: 12"
        expected_texts = [
            [expected_state + "\n\nChoose a team.", "Refunds", "a"],
            [
                expected_state + "\n\nIs it urgent?",
                "false: No. This is false: Is it urgent?",
                "true: Yes. This is true: Is it urgent?",
            ],
            [expected_state, "Low", "High"],
        ]
        cls.test = {"request": request, "sequences": []}
        for texts in expected_texts:
            sequences = cls.tokenizer(texts)["input_ids"]
            cls.test["sequences"].append(
                {
                    "input_ids": [token for sequence in sequences for token in sequence],
                    "marks": torch.tensor([len(sequence) for sequence in sequences]).cumsum(0).tolist(),
                }
            )
        cls.prompts = render(cls.tokenizer, request, cls.decision_config)

    def setUp(self):
        self.config = copy.deepcopy(type(self).config)
        self.inputs = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]),
            "marks": torch.tensor([[2, 4, 6, 8, 10, 12]]),
            "num_options": torch.tensor([[1, 1, 1]]),
        }

    def test_paired_contrastive_loss(self):
        model = ClmForDecision(self.config)
        labels = torch.arange(3)
        for groups in (None, torch.tensor([0, 0, 1])):
            with self.subTest(groups=groups):
                output = model(**self.inputs, labels=labels, group_ids=groups)
                logits = output.logits
                if groups is not None:
                    mask = groups[:, None] == groups[None]
                    mask.fill_diagonal_(False)
                    logits = logits.masked_fill(mask, -torch.inf)
                expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                torch.testing.assert_close(output.loss, expected)
                model.zero_grad()
                output.loss.backward()
                self.assertTrue(torch.isfinite(model.state_head.input_projection.weight.grad).all())
                self.assertGreater(model.action_head.output_projection.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model(**self.inputs).loss)
        torch.testing.assert_close(model(**self.inputs, return_dict=False)[0], model(**self.inputs).logits)

    def test_soft_targets_and_unmatched_candidates(self):
        model = ClmForDecision(self.config)
        labels = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        output = model(**self.inputs, labels=labels)
        reverse = labels.T[:2]
        reverse = reverse / reverse.sum(-1, keepdim=True)
        expected = (F.cross_entropy(output.logits, labels) + F.cross_entropy(output.logits.T[:2], reverse)) / 2
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_save_load(self):
        for model_type in ("qwen2", "llama"):
            with self.subTest(model_type=model_type):
                self.config.text_config = AutoConfig.for_model(
                    model_type,
                    **{key: value for key, value in self.config.text_config.to_dict().items() if key != "model_type"},
                )
                model = ClmForDecision(self.config).eval()
                expected = model(**self.inputs, labels=torch.arange(3))
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    loaded = AutoModelForDecision.from_pretrained(directory).eval()
                output = loaded(**self.inputs, labels=torch.arange(3))
                torch.testing.assert_close(output.loss, expected.loss)
                torch.testing.assert_close(output.logits, expected.logits)

    def test_bfloat16_checkpoint(self):
        model = ClmForDecision(self.config)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = AutoModelForDecision.from_pretrained(directory, dtype=torch.bfloat16).eval()
        self.assertEqual(loaded.model.dtype, torch.bfloat16)
        self.assertEqual(loaded.state_head.input_projection.weight.dtype, torch.float32)
        self.assertEqual(loaded.action_head.output_projection.weight.dtype, torch.float32)
        self.assertEqual(loaded.logit_scale.dtype, torch.float32)
        output = loaded(**self.inputs, labels=torch.arange(3))
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))

    def test_candidate_permutation(self):
        model = ClmForDecision(self.config).eval()
        expected = model(**self.inputs).logits
        segments = self.inputs["input_ids"].reshape(6, 2)
        permuted = segments[torch.tensor([0, 3, 2, 5, 4, 1])].reshape(1, -1)
        actual = model(**{**self.inputs, "input_ids": permuted}).logits
        torch.testing.assert_close(actual, expected[:, [1, 2, 0]])

    def test_trainer(self):
        model = ClmForDecision(self.config)
        model.base_model.requires_grad_(False)
        before = model.state_head.input_projection.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                    max_steps=1,
                    report_to="none",
                    remove_unused_columns=False,
                    save_strategy="no",
                ),
                train_dataset=[0, 1],
                data_collator=lambda features: {**self.inputs, "labels": torch.arange(3)},
            )
            result = trainer.train()
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
        self.assertFalse(torch.equal(before, model.state_head.input_projection.weight))
        self.assertTrue(all(p.grad is None for p in model.base_model.parameters()))

    def test_typed_decisions(self):
        model = ClmForDecision(self.config).eval()
        request = {
            "state": "A refund request",
            "questions": {
                "intent": {"type": "choice", "criteria": {str(index): None for index in range(32)}},
                "truth": {"type": "noul", "instructions": "Is this a refund?"},
                "quality": {"type": "score", "criteria": ["bad", "good"], "values": [1, 5]},
            },
        }
        prompts = render(self.tokenizer, request, model.decision_config)
        scores = [model.decide(**prompt.inputs, decision_questions=prompt.questions)[0] for prompt in prompts]
        self.assertEqual([len(score) for score in scores], [32, 2, 2])
        decide = pipeline("decision", model=model, tokenizer=self.tokenizer, device="cpu")
        answers = decide(request)
        self.assertAlmostEqual(sum(answers["intent"]["probabilities"].values()), 1, places=6)
        self.assertAlmostEqual(answers["truth"]["noul"], scores[1].softmax(-1)[1].item(), places=6)
        self.assertAlmostEqual(
            answers["quality"]["score"], (scores[2].softmax(-1) * torch.tensor([1, 5])).sum().item(), places=6
        )
        request["questions"]["intent"]["allow_null"] = True
        with self.assertRaisesRegex(ValueError, "does not predict null"):
            decide(request)

    def test_template_and_pipeline(self):
        tokenizer = self.tokenizer
        model = ClmForDecision(self.config).eval()
        request = self.test["request"]
        decide = pipeline("decision", model=model, tokenizer=tokenizer, device="cpu")
        answers = decide(request)
        self.assertEqual(set(answers), set(request["questions"]))
        self.assertEqual(list(answers["intent"]["probabilities"]), ["z", "a"])
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            model.decision_config.save_pretrained(directory)
            restored = pipeline(
                "decision", model=directory, model_kwargs={"attn_implementation": "eager"}, device="cpu"
            )
            reloaded = restored(request)
        for name in answers:
            field = "noul" if name == "urgent" else "probabilities"
            torch.testing.assert_close(reloaded[name][field], answers[name][field], rtol=1e-5, atol=1e-6)
