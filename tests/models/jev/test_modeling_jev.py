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

    from transformers import JevForDecision
    from transformers.loss.loss_jev import JevForDecisionLoss


@require_torch
class JevLossTest(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [1.0, 0.1, -0.2, -1.0, 0.5, 1.2, -0.4, 0.3, 0.8, 0.4, -0.5, 0.6, 0.9], requires_grad=True
        )
        self.num_options = torch.tensor([1, 2, 6, 4])
        self.is_score = torch.tensor([True, False, True, False])
        self.labels = torch.tensor([0, 1, 3, 2])
        self.soft_labels = torch.tensor([1.0, 0.2, 0.8, 0.05, 0.1, 0.2, 0.3, 0.25, 0.1, 0.0, 0.2, 0.0, 0.8])
        self.weights = torch.tensor([0.5, 0.25, 2.0, 0.5])

    def test_model_card_loss_and_gradients(self):
        # Independent float64 categorical KL and ordinal RPS reference, including ragged and singleton questions.
        sizes = self.num_options.tolist()
        ignored = self.labels.clone()
        ignored[-1] = -100
        for labels in (self.labels, self.soft_labels, ignored, torch.full_like(self.labels, -100)):
            soft_labels = labels.is_floating_point()
            for weights in (None, self.weights):
                for rps_weight in (0.0, 0.5):
                    with self.subTest(labels=labels.tolist(), weighted=weights is not None, rps_weight=rps_weight):
                        reference_logits = self.logits.detach().double().requires_grad_()
                        targets = labels.split(sizes) if soft_labels else labels
                        terms = []
                        for index, (scores, target) in enumerate(zip(reference_logits.split(sizes), targets)):
                            if not soft_labels:
                                if target == -100:
                                    continue
                                target = torch.nn.functional.one_hot(target, scores.numel())
                            teacher = torch.distributions.Categorical(probs=target.double())
                            student = torch.distributions.Categorical(logits=scores)
                            term = torch.distributions.kl_divergence(teacher, student)
                            if self.is_score[index] and scores.numel() > 1:
                                difference = student.probs.cumsum(0)[:-1] - teacher.probs.cumsum(0)[:-1]
                                term = term + rps_weight * difference.square().mean()
                            terms.append(term if weights is None else term * weights[index])
                        expected = torch.stack(terms).mean() if terms else reference_logits.sum() * 0
                        actual = JevForDecisionLoss(
                            self.logits,
                            labels,
                            self.num_options,
                            self.is_score,
                            rps_weight=rps_weight,
                            loss_weights=weights,
                        )
                        torch.testing.assert_close(actual.double(), expected)
                        actual_gradient = torch.autograd.grad(actual, self.logits)[0]
                        expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
                        torch.testing.assert_close(actual_gradient.double(), expected_gradient)

    def test_accumulated_loss(self):
        loss = JevForDecisionLoss(
            self.logits, self.soft_labels, self.num_options, self.is_score, loss_weights=self.weights
        )
        expected_gradient = torch.autograd.grad(loss, self.logits)[0]
        start = 0
        for index, count in enumerate(self.num_options.tolist()):
            end = start + count
            JevForDecisionLoss(
                self.logits[start:end],
                self.soft_labels[start:end],
                self.num_options[index : index + 1],
                self.is_score[index : index + 1],
                loss_weights=self.weights[index : index + 1],
                num_items_in_batch=4,
            ).backward()
            start = end
        torch.testing.assert_close(self.logits.grad, expected_gradient)

    def test_confident_prediction_gradients(self):
        # Standard fp32 log_softmax rounds the winning loss and gradient to zero at this margin.
        logits = torch.tensor([9.0, -9.0, -9.0], requires_grad=True)
        loss = JevForDecisionLoss(logits, torch.tensor([0]), torch.tensor([3]), torch.tensor([False]))
        mass = 2 * torch.exp(torch.tensor(-18.0, dtype=torch.float64))
        non_target_gradient = mass / (1 + mass) / 2
        expected_gradient = torch.stack([-2 * non_target_gradient, non_target_gradient, non_target_gradient])
        loss.backward()
        torch.testing.assert_close(loss.double(), torch.log1p(mass), rtol=1e-6, atol=0)
        torch.testing.assert_close(logits.grad.double(), expected_gradient, rtol=1e-6, atol=0)


@require_torch
class JevForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (JevForDecision,) if is_torch_available() else ()
    decision_repo = "hf-internal-testing/decision-gev"
    tiny_sizes = {"hidden_size": 16}

    def test_loss_backward(self):
        inputs = {
            "input_ids": torch.arange(12)[None],
            "marks": torch.tensor([[4, 8, 12]]),
            "num_options": torch.tensor([[2, 6, 4]]),
            "question_type_ids": torch.tensor(
                [[Question.questions_types.index(kind) for kind in ("noul", "score", "choice")]]
            ),
            "labels": torch.tensor([[1, 3, 2]]),
        }
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = JevForDecision(self.config).to(dtype).eval()
                output = model(**inputs)
                self.assertEqual(output.loss.dtype, torch.float32)
                output.loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertGreater(model.proj.weight.grad.abs().sum(), 0)
                self.assertGreater(model.model.get_input_embeddings().weight.grad.abs().sum(), 0)
                active = torch.zeros(self.config.num_verbalizers, dtype=torch.bool)
                for kind, count in zip(("noul", "score", "choice"), inputs["num_options"][0]):
                    start = self.config.offsets[kind]
                    active[start : start + count] = True
                inactive_gradients = model.proj.weight.grad[~active]
                torch.testing.assert_close(inactive_gradients, torch.zeros_like(inactive_gradients))

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
        with tempfile.TemporaryDirectory() as tmp:  # saved as a decision model, with its decision config
            model.save_pretrained(tmp)
            reloaded = JevForDecision.from_pretrained(tmp)
        self.assertEqual(reloaded.decision_config, model.decision_config)
        torch.testing.assert_close(reloaded.proj.weight, readout)
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
    "hf-internal-testing/pplx-decider-v1.1-27b": {
        "refund": 0.9993,
        "spam": 0.0003,
        "tone": {"calm": 0.1327, "angry": 0.8002, "sad": 0.0671},
        "urgency": {"0": 0.0008, "1": 0.0058, "2": 0.7808, "3": 0.2126},
    },
    "hf-internal-testing/pplx-decider-v1-27b": {
        "refund": 0.9946,
        "spam": 0.0208,
        "tone": {"calm": 0.1099, "angry": 0.7974, "sad": 0.0927},
        "urgency": {"0": 0.0034, "1": 0.0051, "2": 0.9301, "3": 0.0614},
    },
    "hf-internal-testing/d3": {
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
        check_decision(
            self, "hf-internal-testing/JEV-9B", "hf-internal-testing/JEV-9B", expected, atol=1e-2, dtype="bfloat16"
        )

    def test_jev_35b(self):
        # System 1 merged into the weights: its head is rows of the `lm_head`
        expected = {
            "intent": {"refund": 0.9981, "track": 0.0001, "other": 0.0018},
            "urgent": 0.7539,
            "anger": {"0": 0.0155, "1": 0.0859, "2": 0.4147, "3": 0.3896, "4": 0.0872, "5": 0.007},
        }
        check_decision(self, "hf-internal-testing/JEV-35B", "hf-internal-testing/JEV-35B", expected, atol=1e-2)

    def test_deciders(self):
        for repo, expected in DECIDERS.items():
            with self.subTest(repo):
                check_decision(self, None, repo, expected, atol=1e-2)
