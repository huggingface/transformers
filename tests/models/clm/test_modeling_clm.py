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
from pathlib import Path

from transformers import AutoTokenizer, ClmConfig, is_torch_available, pipeline
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...test_decision_head_common import DecisionHeadTesterMixin, render, tiny_qwen2


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import AutoModelForDecision, ClmForDecision
    from transformers.loss.loss_clm import ClmForDecisionLoss


@require_torch
class ClmLossTest(unittest.TestCase):
    def test_official_loss_and_gradients(self):
        logits = torch.tensor([[1.0, -0.3, 0.2], [0.4, 1.2, -0.5], [0.1, 0.6, 0.7]], requires_grad=True)
        cases = [
            (torch.arange(3), None),
            (torch.arange(3), torch.tensor([0, 0, 1])),
            (torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), None),
        ]
        for labels, group_ids in cases:
            with self.subTest(soft_labels=labels.is_floating_point(), group_ids=group_ids):
                reference_logits = logits.detach().double().requires_grad_()
                scores = reference_logits
                targets = F.one_hot(labels, 3).double() if labels.ndim == 1 else labels.double()
                if group_ids is not None:
                    same_group = group_ids[:, None] == group_ids[None]
                    same_group.fill_diagonal_(False)
                    scores = scores.masked_fill(same_group, -torch.inf)
                forward = (
                    F.cross_entropy(scores, targets.argmax(-1))
                    if labels.ndim == 1
                    else F.cross_entropy(scores, targets)
                )
                reverse = targets.T
                positive = reverse.sum(-1) > 0
                reverse = reverse[positive] / reverse[positive].sum(-1, keepdim=True)
                backward = (
                    F.cross_entropy(scores.T[positive], reverse.argmax(-1))
                    if labels.ndim == 1
                    else F.cross_entropy(scores.T[positive], reverse)
                )
                expected = (forward + backward) / 2
                actual = ClmForDecisionLoss(logits, labels, group_ids=group_ids)
                torch.testing.assert_close(actual.double(), expected)
                torch.testing.assert_close(
                    torch.autograd.grad(actual, logits)[0].double(), torch.autograd.grad(expected, reference_logits)[0]
                )

    def test_confident_prediction_gradients(self):
        logits = torch.tensor([[9.0, -9.0], [-9.0, 9.0]], requires_grad=True)
        loss = ClmForDecisionLoss(logits, torch.arange(2))
        mass = torch.exp(torch.tensor(-18.0, dtype=torch.float64))
        gradient = mass / (1 + mass) / 2
        expected_gradient = torch.stack([-gradient, gradient, gradient, -gradient]).reshape(2, 2)
        loss.backward()
        torch.testing.assert_close(loss.double(), torch.log1p(mass), rtol=1e-6, atol=0)
        torch.testing.assert_close(logits.grad.double(), expected_gradient, rtol=1e-6, atol=0)


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

    def test_loss_backward(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = ClmForDecision(self.config).to(dtype).eval()
                output = model(**self.inputs, labels=torch.arange(3))
                self.assertEqual(output.loss.dtype, torch.float32)
                output.loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertGreater(model.action_head.output_projection.weight.grad.abs().sum(), 0)
                self.assertGreater(model.model.get_input_embeddings().weight.grad.abs().sum(), 0)

    def test_candidate_permutation(self):
        model = ClmForDecision(self.config).eval()
        expected = model(**self.inputs).logits
        segments = self.inputs["input_ids"].reshape(6, 2)
        permuted = segments[torch.tensor([0, 3, 2, 5, 4, 1])].reshape(1, -1)
        actual = model(**{**self.inputs, "input_ids": permuted}).logits
        torch.testing.assert_close(actual, expected[:, [1, 2, 0]])

    def test_tokenizer_attention_mask(self):
        model = ClmForDecision(self.config).eval()
        expected = model(**self.inputs).logits
        output = model(**self.inputs, attention_mask=torch.ones_like(self.inputs["input_ids"]))
        torch.testing.assert_close(output.logits, expected)
        padded = {
            **self.inputs,
            "input_ids": F.pad(self.inputs["input_ids"], (3, 2)),
            "attention_mask": F.pad(torch.ones_like(self.inputs["input_ids"]), (3, 2)),
            "marks": self.inputs["marks"] + 3,
        }
        torch.testing.assert_close(model(**padded).logits, expected)

    def test_checkpoint_conversion(self):
        from transformers.models.clm.convert_clm_weights_to_hf import convert_checkpoint

        model = ClmForDecision(self.config).eval()
        expected = model(**self.inputs, labels=torch.arange(3))
        names = {"input_projection": "inp", "layers": "hidden", "layer_norms": "norms", "output_projection": "out"}
        weights = {"logit_scale": model.logit_scale.detach()}
        for head_name in ("state_head", "action_head"):
            state = {}
            for key, value in model.get_submodule(head_name).state_dict().items():
                module, parameter = key.split(".", 1)
                state[f"{names[module]}.{parameter}"] = value
            weights[head_name] = state
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model.base_model.save_pretrained(path / "encoder")
            self.tokenizer.save_pretrained(path / "encoder")
            weights["cfg"] = {
                "model": str(path / "encoder"),
                "width": self.config.hidden_size,
                "depth": self.config.num_hidden_layers,
                "projection_dim": self.config.projection_dim,
                "activation": self.config.hidden_act,
                "layernorm": self.config.layer_norm,
                "residual": self.config.residual,
            }
            torch.save(weights, path / "CLM_v0.1-8B.pt")
            convert_checkpoint(str(path), str(path / "converted"))
            loaded = AutoModelForDecision.from_pretrained(path / "converted").eval()
        output = loaded(**self.inputs, labels=torch.arange(3))
        torch.testing.assert_close(output.logits, expected.logits)
        torch.testing.assert_close(output.loss, expected.loss)
        torch.testing.assert_close(loaded.logit_scale, model.logit_scale)

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


@slow
@require_torch_accelerator
class ClmIntegrationTest(unittest.TestCase):
    def test_clm_8b(self):
        from transformers.models.clm.convert_clm_weights_to_hf import convert_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            convert_checkpoint(
                "Contrastive-LM/CLM-v0.1-8B",
                directory,
                revision="e939398d4556fcd9400c76fa8c5a513202f42b0a",
                encoder_revision="b968826d9c46dd6066d109eabc6255188de91218",
            )
            decide = pipeline("decision", model=directory, dtype="float32", model_kwargs={"device_map": torch_device})
            answers = decide(
                {
                    "state": "Customer: my invoice was charged twice and nobody answers the phone!",
                    "questions": {
                        "urgency": {"type": "noul", "instructions": "Is this urgent?"},
                        "department": {
                            "type": "choice",
                            "instructions": "Which team should handle this?",
                            "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"},
                        },
                        "frustration": {
                            "type": "score",
                            "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Frustrated", "Very angry"],
                        },
                    },
                }
            )
        torch.testing.assert_close(answers["urgency"]["noul"], 0.841952, rtol=0, atol=1e-4)
        torch.testing.assert_close(
            answers["department"]["probabilities"], {"billing": 0.987977, "technical": 0.012023}, rtol=0, atol=1e-4
        )
        self.assertEqual(answers["department"]["choice"], "billing")
        torch.testing.assert_close(answers["frustration"]["score"], 1.999977, rtol=0, atol=1e-4)
