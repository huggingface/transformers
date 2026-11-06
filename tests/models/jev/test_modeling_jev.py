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

import json
import tempfile
import unittest
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from transformers import AutoProcessor, is_torch_available, pipeline
from transformers.decision import DecisionConfig, Question
from transformers.testing_utils import require_torch, require_torch_accelerator, require_vision, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision, render


if is_torch_available():
    import torch

    from transformers import AutoModelForDecision, JevConfig, JevForDecision, Qwen2Config, Trainer, TrainingArguments
    from transformers.loss.loss_jev import JevForDecisionLoss


@require_torch
class JevLossTest(unittest.TestCase):
    def setUp(self):
        self.config = JevConfig(
            hidden_size=16,
            num_verbalizers=16,
            offsets={"noul": 8, "score": 0, "choice": 10},
            softcap=2.0,
            text_config=Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            ),
        )
        self.inputs = {
            "input_ids": torch.arange(12)[None],
            "marks": torch.tensor([[4, 8, 12]]),
            "num_options": torch.tensor([[2, 6, 4]]),
            "question_type_ids": torch.tensor(
                [[Question.questions_types.index(k) for k in ("noul", "score", "choice")]]
            ),
            "labels": torch.tensor([[1, 3, 2]]),
            "loss_weights": torch.tensor([[0.25, 2.0, 0.5]]),
        }

    def test_reference_loss_and_gradients(self):
        # Golden values from trainjev.losses.kl_rps at shibing624/train-jev@88bcc9d.
        soft_labels = torch.tensor([[0.2, 0.8, 0.05, 0.1, 0.2, 0.3, 0.25, 0.1, 0.0, 0.2, 0.0, 0.8]])
        cases = (
            (
                torch.tensor([[1, 3, 2]]),
                0.0,
                None,
                1.6074589490890503,
                [
                    0.1914808452,
                    -0.1914808452,
                    0.0134539399,
                    0.0598560162,
                    0.1156741679,
                    -0.3125763834,
                    0.0453034379,
                    0.0782888085,
                    0.0779418349,
                    0.0316887870,
                    -0.2381349802,
                    0.1285043508,
                ],
            ),
            (
                soft_labels,
                0.0,
                None,
                0.4593898057937622,
                [
                    0.1248141751,
                    -0.1248141825,
                    -0.0036421223,
                    0.0251717903,
                    0.0497519299,
                    -0.0774605125,
                    -0.0377272293,
                    0.0439061336,
                    0.0779418349,
                    -0.0349778831,
                    0.0951983631,
                    -0.1381623298,
                ],
            ),
            (
                soft_labels,
                0.2,
                self.inputs["loss_weights"],
                0.4107740819454193,
                [
                    0.0373059139,
                    -0.0373059213,
                    -0.0080373567,
                    0.0508070849,
                    0.1073135063,
                    -0.1603680551,
                    -0.0818812922,
                    0.0921660811,
                    0.0431385003,
                    -0.0179415010,
                    0.0536751598,
                    -0.0788721740,
                ],
            ),
        )
        for labels, brier_weight, weights, expected_loss, expected_grad in cases:
            with self.subTest(soft_labels=labels.is_floating_point(), brier_weight=brier_weight):
                logits = torch.tensor(
                    [0.1, -0.2, -1.0, 0.5, 1.2, -0.4, 0.3, 0.8, 0.4, -0.5, 0.6, 0.9], requires_grad=True
                )
                loss = JevForDecisionLoss(
                    logits,
                    labels,
                    self.inputs["num_options"],
                    torch.tensor([False, True, False]),
                    brier_weight=brier_weight,
                    loss_weights=weights,
                )
                torch.testing.assert_close(loss, torch.tensor(expected_loss))
                loss.backward()
                torch.testing.assert_close(logits.grad, torch.tensor(expected_grad))

    def test_loss_and_save_load(self):
        model = JevForDecision(self.config).eval()
        self.assertIs(model.loss_function, JevForDecisionLoss)
        output = model(**self.inputs)
        torch.testing.assert_close(model(**self.inputs, return_dict=False)[0], output.loss)
        output.loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.model.get_input_embeddings().weight.grad.abs().sum().item(), 0)
        # Custom offsets: these head rows are not options of any question and must receive no supervised gradient.
        torch.testing.assert_close(model.proj.weight.grad[[6, 7, 14, 15]], torch.zeros(4, 16))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = AutoModelForDecision.from_pretrained(directory).eval()
        result = loaded(**self.inputs)
        torch.testing.assert_close(result.loss, output.loss)
        torch.testing.assert_close(result.logits, output.logits)
        self.assertIsNone(model(**{k: v for k, v in self.inputs.items() if k != "labels"}).loss)

    def test_accumulated_loss(self):
        model = JevForDecision(self.config).eval()
        model(**self.inputs).loss.backward()
        reference = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()
        for index in range(3):
            sample = {k: v if k == "input_ids" else v[:, index : index + 1] for k, v in self.inputs.items()}
            model(**sample, num_items_in_batch=3).loss.backward()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, reference[name], rtol=1e-4, atol=1e-6)

    def test_bfloat16_loss(self):
        model = JevForDecision(self.config).to(torch.bfloat16)
        output = model(**self.inputs)
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_ignored_and_singleton_questions(self):
        logits = torch.tensor([1.0, -0.5, 0.5], requires_grad=True)
        loss = JevForDecisionLoss(logits, torch.tensor([[0, -100]]), torch.tensor([1, 2]), torch.tensor([True, False]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros(3))

    def test_trainer(self):
        model = JevForDecision(self.config)
        before = model.proj.weight.detach().clone()
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
                data_collator=lambda features: self.inputs,
            )
            result = trainer.train()
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
        self.assertFalse(torch.equal(before, model.proj.weight))


@require_torch
class JevForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (JevForDecision,) if is_torch_available() else ()
    decision_repo = "hf-internal-testing/decision-gev"
    tiny_sizes = {"hidden_size": 16}

    @require_vision
    def test_chat_template_reproduces_the_model_code(self):
        super().test_chat_template_reproduces_the_model_code()
        # The converted decider heads (`DECIDERS`), whose templates see images
        for repo in DECIDERS:
            with self.subTest(repo):
                test = json.loads(Path(hf_hub_download(repo, "decision_test.json")).read_bytes())
                prompts = render(
                    AutoProcessor.from_pretrained(repo), test["request"], DecisionConfig.from_pretrained(repo)
                )
                sequences = [{"input_ids": prompt.input_ids, "marks": prompt.marks} for prompt in prompts]
                self.assertEqual(sequences, test["sequences"])

    def test_from_a_vllm_decision_checkpoint(self):
        # A vLLM decision model ships its head (`readout.safetensors`) and settings (`decision_config.json`) next to the
        # weights of its model: read as they are, without any file of ours
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(0)
            backbone = self.backbone_class(self.config.text_config)
            backbone.save_pretrained(tmp)
            self.tokenizer.save_pretrained(tmp)
            codes, readout = ["A", "B", "C"], torch.randn(3, self.config.text_config.hidden_size)
            settings = {
                "codes": codes,
                "temperature": 2.0,
                "attention_mode": "noncausal_full_attention",
                "prompt": "d3",
            }
            Path(tmp, "decision_config.json").write_text(json.dumps(settings), encoding="utf-8")
            save_file({"weight": readout}, f"{tmp}/readout.safetensors", metadata={"format": "pt"})
            model = JevForDecision.from_pretrained(tmp)
            decide = pipeline("decision", model=tmp)
        self.assertEqual(model.config.num_verbalizers, 3)
        self.assertFalse(model.config.text_config.is_causal)
        torch.testing.assert_close(model.proj.weight, readout)
        self.assertIsInstance(decide.model, JevForDecision)
        self.assertEqual(decide.decision_config.temperature, dict.fromkeys(["noul", "choice", "score"], 2.0))

    def test_config_json_cannot_add_extra_files(self):
        # Only the config classes of decision models name the files read next to the weights, never a `config.json`
        with tempfile.TemporaryDirectory() as tmp:
            self.backbone_class(self.config.text_config).save_pretrained(tmp)
            config = json.loads(Path(tmp, "config.json").read_text(encoding="utf-8"))
            Path(tmp, "config.json").write_text(
                json.dumps({**config, "extra_files": ["evil.safetensors"]}), encoding="utf-8"
            )
            save_file(
                {"model.norm.weight": torch.full_like(torch.empty(config["hidden_size"]), 7.0)},
                f"{tmp}/evil.safetensors",
            )
            model = self.backbone_class.from_pretrained(tmp)
        self.assertFalse(torch.all(model.model.norm.weight == 7.0))


# Code readouts of a Qwen3.5 checkpoint, read with the files of its repo, and the answers each gives to the request of its
# `decision_test.json`. One per prompt and attention mode: the other fine-tunes (d3-*, Darwin-*, PhysicalEye) share them
DECIDERS = {
    "ArthurZ/pplx-decider-v1.1-27b": {
        "refund": 0.9993,
        "spam": 0.0003,
        "tone": {"calm": 0.1327, "angry": 0.8002, "sad": 0.0671},
        "urgency": {"0": 0.0008, "1": 0.0058, "2": 0.7808, "3": 0.2126},
    },
    "ArthurZ/pplx-decider-v1-27b": {
        "refund": 0.9946,
        "spam": 0.0208,
        "tone": {"calm": 0.1099, "angry": 0.7974, "sad": 0.0927},
        "urgency": {"0": 0.0034, "1": 0.0051, "2": 0.9301, "3": 0.0614},
    },
    "ArthurZ/d3": {
        "refund": 0.9908,
        "spam": 0.0006,
        "tone": {"calm": 0.1407, "angry": 0.8094, "sad": 0.05},
        "urgency": {"0": 0.0007, "1": 0.0026, "2": 0.724, "3": 0.2727},
    },
}


@slow
@require_torch_accelerator
class JevIntegrationTest(unittest.TestCase):
    def test_jev_9b(self):
        expected = {
            "intent": {"refund": 0.9999, "track": 0.0000, "other": 0.0001},
            "urgent": 0.5419,
            "anger": {"0": 0.0165, "1": 0.1113, "2": 0.5163, "3": 0.3126, "4": 0.0419, "5": 0.0013},
        }
        check_decision(self, "ArthurZ/JEV-9B", "ArthurZ/JEV-9B", expected, atol=1e-2, dtype="bfloat16")

    def test_jev_35b(self):
        # System 1 merged into the weights: its head is rows of the `lm_head`
        expected = {
            "intent": {"refund": 0.9981, "track": 0.0001, "other": 0.0018},
            "urgent": 0.7539,
            "anger": {"0": 0.0155, "1": 0.0859, "2": 0.4147, "3": 0.3896, "4": 0.0872, "5": 0.007},
        }
        check_decision(self, "ArthurZ/JEV-35B", "ArthurZ/JEV-35B", expected, atol=1e-2)

    def test_deciders(self):
        for repo, expected in DECIDERS.items():
            with self.subTest(repo):
                check_decision(self, None, repo, expected, atol=1e-2)
