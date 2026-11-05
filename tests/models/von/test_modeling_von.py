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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForDecision,
        ModernBertConfig,
        ModernBertModel,
        Trainer,
        TrainingArguments,
        VonConfig,
        VonForDecision,
    )
    from transformers.models.von.modeling_von import von_loss


@require_torch
class VonLossTest(unittest.TestCase):
    def test_upstream_loss_and_gradients(self):
        # Golden values from compute_marker_rlcd_loss at wfzyx/von@fd577a8, including its log(p + 1e-8).
        for labels, expected_loss, expected_grad in (
            (torch.tensor([[2]]), 0.5587821006774902, [0.2896187305, 0.2060268223, -0.4956454635]),
            (torch.tensor([[0.1, 0.3, 0.6]]), 1.0041191577911377, [0.1294842809, -0.1697471738, 0.0402629897]),
        ):
            with self.subTest(soft_labels=labels.is_floating_point()):
                logits = torch.tensor([0.1, -0.2, 1.2], requires_grad=True)
                loss = von_loss(logits, labels)
                torch.testing.assert_close(loss, torch.tensor(expected_loss))
                loss.backward()
                torch.testing.assert_close(logits.grad, torch.tensor(expected_grad))

    def test_saturated_predictions_preserve_upstream_epsilon(self):
        logits = torch.tensor([-100.0, 100.0], requires_grad=True)
        loss = von_loss(logits, torch.tensor([[0]]))
        torch.testing.assert_close(loss, torch.tensor(19.42068099975586))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros(2))

    def test_ignored_question(self):
        logits = torch.randn(3, requires_grad=True)
        loss = von_loss(logits, torch.tensor([[-100]]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros(3))

    def get_model_and_inputs(self):
        model = VonForDecision(
            VonConfig(
                hidden_size=16,
                noul_prior={"a": 0.0, "b": 2.0},
                text_config=ModernBertConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    local_attention=8,
                    pad_token_id=0,
                    bos_token_id=1,
                    eos_token_id=2,
                    cls_token_id=1,
                    sep_token_id=2,
                ),
            )
        ).eval()
        inputs = {
            "input_ids": torch.tensor([[3, 4, 5, 6, 1, 9, 1, 7, 8, 2] * 2]),
            "marks": torch.tensor([[5, 7, 10, 15, 17, 20]]),
            "num_options": torch.tensor([[2]]),
        }
        return model, inputs

    def test_loss_uses_raw_logits_and_backpropagates(self):
        model, inputs = self.get_model_and_inputs()
        output = model(**inputs, labels=torch.tensor([[0]]))
        corrected = model(**inputs).logits
        torch.testing.assert_close(corrected, output.logits - torch.tensor([2.0, 0.0]))
        tuple_output = model(**inputs, labels=torch.tensor([[0]]), return_dict=False)
        torch.testing.assert_close(tuple_output[0], output.loss)
        output.loss.backward()
        for parameter in (model.out_proj.weight, model.model.get_input_embeddings().weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_save_load(self):
        model, inputs = self.get_model_and_inputs()
        labels = torch.tensor([[0]])
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
        output = model(**inputs, labels=torch.tensor([[0]]))
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        )

    def test_trainer(self):
        model, inputs = self.get_model_and_inputs()
        before = model.out_proj.weight.detach().clone()
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
                data_collator=lambda features: {**inputs, "labels": torch.tensor([[0]])},
            )
            result = trainer.train()
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
        self.assertEqual(trainer.state.global_step, 1)
        self.assertFalse(torch.equal(before, model.out_proj.weight))


@require_torch
class VonForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (VonForDecision,) if is_torch_available() else ()
    decision_repo = "hf-internal-testing/decision-von"
    tiny_sizes = {"hidden_size": 16}
    backbone_class = ModernBertModel if is_torch_available() else None

    def test_options_only_see_the_prompt_and_themselves(self):
        # Two blocks of a prompt `3 4 5 6`, a `[MASK]` (1) before each option and `[SEP]` (2), where the option `7 8`
        # comes second then first: the prompt, `[SEP]` and that option keep their states if options only attend to the
        # prompt of their block and to themselves, at positions that restart after it (also for the sliding window)
        config = copy.deepcopy(self.config)
        config.text_config = ModernBertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            local_attention=8,
        )
        model = VonForDecision(config).eval()
        input_ids = torch.tensor([3, 4, 5, 6, 1, 9, 1, 7, 8, 2] + [3, 4, 5, 6, 1, 7, 8, 1, 10, 2])
        marks = torch.tensor([5, 7, 10, 15, 18, 20])
        # The first question of the request has two options
        num_options = torch.tensor([len(question.options) for question in self.prompts[0].questions])
        states = []
        for implementation in ("sdpa", "eager"):
            model.model.set_attn_implementation(implementation)
            with torch.no_grad():
                backbone_inputs = model.backbone_inputs(input_ids, marks, num_options)
                states.append(model.model(**backbone_inputs).last_hidden_state[0])
            first, second = states[-1].split(10)
            torch.testing.assert_close(first[[0, 1, 2, 3, 6, 7, 8, 9]], second[[0, 1, 2, 3, 4, 5, 6, 9]])
        torch.testing.assert_close(*states)


@slow
@require_torch_accelerator
class VonIntegrationTest(unittest.TestCase):
    def test_von(self):
        expected = {
            "intent": {"refund": 0.9158, "other": 0.0842},
            "urgent": 0.8436,
            "tone": {"0": 0.7911, "1": 0.2089},
        }
        check_decision(self, "wfzyx/von", "ArthurZ/von", expected, dtype="float32")
