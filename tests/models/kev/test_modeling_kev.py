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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    import torch

    from transformers import AutoModelForDecision, KevConfig, KevForDecision, Qwen2Config, Trainer, TrainingArguments
    from transformers.models.kev.modeling_kev import kev_loss


@require_torch
class KevLossTest(unittest.TestCase):
    def test_upstream_loss_and_gradients(self):
        # Golden values from kev.train.question_loss at jaredpalmer/kev@2074cc4, with all optional modifiers disabled.
        for labels, expected_loss, expected_grad in (
            (
                torch.tensor([[0, 2]]),
                0.5144830346107483,
                [-0.2127787471, 0.2127787322, 0.0344668254, 0.1544695944, -0.1889364421],
            ),
            (
                torch.tensor([[0.75, 0.25, 0.1, 0.3, 0.6]]),
                0.7669830322265625,
                [-0.0877787471, 0.0877787322, -0.0155331753, 0.0044695884, 0.0110635459],
            ),
        ):
            with self.subTest(soft_labels=labels.is_floating_point()):
                logits = torch.tensor([0.1, -0.2, -1.0, 0.5, 1.2], requires_grad=True)
                loss = kev_loss(logits, labels, torch.tensor([[2, 3]]))
                torch.testing.assert_close(loss, torch.tensor(expected_loss))
                loss.backward()
                torch.testing.assert_close(logits.grad, torch.tensor(expected_grad))

    def test_ignored_questions(self):
        logits = torch.tensor([0.1, -0.2, -1.0, 0.5, 1.2], requires_grad=True)
        loss = kev_loss(logits, torch.tensor([[0, -100]]), torch.tensor([[2, 3]]))
        loss.backward()
        torch.testing.assert_close(logits.grad[2:], torch.zeros(3))
        ignored_loss = kev_loss(logits, torch.tensor([[-100, -100]]), torch.tensor([[2, 3]]))
        self.assertEqual(ignored_loss.item(), 0.0)
        self.assertTrue(ignored_loss.requires_grad)

    def get_model_and_inputs(self):
        model = KevForDecision(
            KevConfig(
                hidden_size=16,
                head_dim=8,
                text_config=Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                ),
            )
        ).eval()
        inputs = {
            "input_ids": torch.arange(14)[None],
            "marks": torch.tensor([[2, 4, 6, 8, 10, 12, 14]]),
            "num_options": torch.tensor([[2, 3]]),
        }
        return model, inputs

    def test_loss_backpropagates_through_head_and_backbone(self):
        model, inputs = self.get_model_and_inputs()
        output = model(**inputs, labels=torch.tensor([[0, 2]]))
        torch.testing.assert_close(output.logits, model(**inputs).logits)
        tuple_output = model(**inputs, labels=torch.tensor([[0, 2]]), return_dict=False)
        torch.testing.assert_close(tuple_output[0], output.loss)
        output.loss.backward()
        for parameter in (model.key.weight, model.query.weight, model.model.get_input_embeddings().weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_save_load(self):
        model, inputs = self.get_model_and_inputs()
        labels = torch.tensor([[0, 2]])
        expected = model(**inputs, labels=labels)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = AutoModelForDecision.from_pretrained(directory).eval()
        output = loaded(**inputs, labels=labels)
        torch.testing.assert_close(output.loss, expected.loss)
        torch.testing.assert_close(output.logits, expected.logits)
        self.assertIsNone(loaded(**inputs).loss)

    def test_bfloat16_loss(self):
        model, inputs = self.get_model_and_inputs()
        model = model.to(torch.bfloat16)
        output = model(**inputs, labels=torch.tensor([[0, 2]]))
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        )

    def test_trainer(self):
        model, inputs = self.get_model_and_inputs()
        before = model.key.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    report_to="none",
                    remove_unused_columns=False,
                    save_strategy="no",
                ),
                train_dataset=[0],
                data_collator=lambda features: {**inputs, "labels": torch.tensor([[0, 2]])},
            )
            result = trainer.train()
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
        self.assertEqual(trainer.state.global_step, 1)
        self.assertFalse(torch.equal(before, model.key.weight))


@require_torch
class KevForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (KevForDecision,) if is_torch_available() else ()
    decision_repo = "hf-internal-testing/decision-kev"
    tiny_sizes = {"hidden_size": 16, "head_dim": 8}


@slow
@require_torch_accelerator
class KevIntegrationTest(unittest.TestCase):
    def test_kev_4b(self):
        expected = {
            "intent": {"refund": 0.8130, "track": 0.0053, "other": 0.1817},
            "urgent": 0.7992,
            "anger": {"0": 0.0519, "1": 0.3996, "2": 0.5484},
        }
        check_decision(self, "jaredpalmer/kev-4b", "ArthurZ/kev-4b", expected, dtype="float32")
