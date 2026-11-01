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

import tempfile
import unittest

from transformers import AutoConfig, AutoTokenizer, ClmConfig, is_torch_available, pipeline
from transformers.testing_utils import require_torch

from ...test_decision_head_common import render, tiny_qwen2


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import AutoModelForDecision, ClmForDecision, Trainer, TrainingArguments


@require_torch
class ClmForDecisionTest(unittest.TestCase):
    all_model_classes = (ClmForDecision,) if is_torch_available() else ()

    def setUp(self):
        self.config = ClmConfig(
            text_config=AutoConfig.for_model(
                "qwen2",
                vocab_size=128,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            ),
            hidden_size=24,
            projection_dim=8,
            num_hidden_layers=3,
        )
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
        model = ClmForDecision(self.config).eval()
        expected = model(**self.inputs, labels=torch.arange(3))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = AutoModelForDecision.from_pretrained(directory).eval()
        output = loaded(**self.inputs, labels=torch.arange(3))
        torch.testing.assert_close(output.loss, expected.loss)
        torch.testing.assert_close(output.logits, expected.logits)

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

    def test_template_and_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.config.text_config = tiny_qwen2(tokenizer).config
        model = ClmForDecision(self.config).eval()
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
        for prompt, texts in zip(render(tokenizer, request, model.decision_config), expected_texts):
            boundaries = [0, *prompt.marks]
            self.assertEqual(
                [prompt.input_ids[a:b] for a, b in zip(boundaries, boundaries[1:])],
                tokenizer(texts)["input_ids"],
            )
        decide = pipeline("decision", model=model, tokenizer=tokenizer, device="cpu")
        answers = decide(request)
        self.assertEqual(set(answers), set(request["questions"]))
        self.assertEqual(list(answers["intent"]["probabilities"]), ["z", "a"])
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            model.decision_config.save_pretrained(directory)
            restored = pipeline("decision", model=directory, device="cpu")
            reloaded = restored(request)
        for name in answers:
            field = "noul" if name == "urgent" else "probabilities"
            torch.testing.assert_close(reloaded[name][field], answers[name][field])
