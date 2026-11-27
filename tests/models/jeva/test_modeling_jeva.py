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
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from transformers import AutoTokenizer, JevaConfig, is_torch_available, pipeline
from transformers.testing_utils import require_peft, require_torch

from ...test_decision_head_common import DecisionHeadTesterMixin, render, tiny_qwen2


if is_torch_available():
    import torch

    from transformers import AutoModelForDecision, JevaForDecision
    from transformers.loss.loss_jeva import JevaForDecisionLoss
    from transformers.models.jeva.modeling_jeva import _log_softmax


class JevaModelTester:
    is_training = True

    def __init__(self, config):
        self.config = config

    def prepare_inputs(self):
        return {
            "input_ids": torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "option_input_ids": torch.tensor([[2, 3, 0], [4, 5, 6], [7, 8, 9], [3, 4, 5], [6, 7, 0]]),
            "option_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]]),
            "num_options": torch.tensor([2, 3]),
            "option_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "labels": torch.tensor([[0.2, 0.3, 0.5, 0.0], [0.0, 0.1, 0.3, 0.6]]),
            "allow_null": torch.tensor([True, False]),
        }

    def prepare_config_and_inputs_for_training(self):
        return copy.deepcopy(self.config), self.prepare_inputs()


@require_torch
class JevaLossTest(unittest.TestCase):
    def test_candidate_and_coverage_loss(self):
        logits = torch.tensor([[0.0, 0.0, -torch.inf], [0.0, 0.0, 0.0]])
        coverage_logits = torch.zeros(2, requires_grad=True)
        loss = JevaForDecisionLoss(
            logits=_log_softmax(logits),
            labels=torch.tensor([[0.25, 0.5, 0.25, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            coverage_logits=coverage_logits,
            option_mask=torch.isfinite(logits),
            allow_null=torch.tensor([True, False]),
            num_items_in_batch=3,
        )

        expected = (0.75 * math.log(2) + math.log(2) + math.log(3)) / 3
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        torch.testing.assert_close(coverage_logits.grad, torch.tensor([-0.25 / 3, 0.0]))

    def test_confident_prediction_gradients(self):
        logits = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)
        coverage = torch.tensor([18.0], requires_grad=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        loss = JevaForDecisionLoss(
            _log_softmax(logits), torch.tensor([[0.0, 1.0, 0.0, 0.0]]), coverage, mask, allow_null=torch.tensor([True])
        )
        mass = torch.exp(torch.tensor(-18.0, dtype=torch.float64))
        option_gradient = mass / (1 + 2 * mass)
        expected_gradient = torch.stack([-2 * option_gradient, option_gradient, option_gradient])[None]
        loss.backward()
        torch.testing.assert_close(loss.double(), torch.log1p(2 * mass) + torch.log1p(mass), rtol=1e-6, atol=0)
        torch.testing.assert_close(logits.grad.double(), expected_gradient, rtol=1e-6, atol=0)
        torch.testing.assert_close(coverage.grad.double(), (-mass / (1 + mass))[None], rtol=1e-6, atol=0)


@require_torch
class JevaForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (JevaForDecision,) if is_torch_available() else ()

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        cls.config = JevaConfig(
            text_config=tiny_qwen2(cls.tokenizer).config,
            projection_dim=16,
            decision_num_heads=2,
        )
        cls.decision_config = JevaForDecision(cls.config).decision_config
        cls.request = {
            "state": "My card has been charged twice.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which intent?",
                    "criteria": {"refund": None, "delivery": None},
                }
            },
        }
        cls.prompts = render(cls.tokenizer, cls.request, cls.decision_config)

    def setUp(self):
        self.model_tester = JevaModelTester(type(self).config)
        self.config = copy.deepcopy(type(self).config)
        self.inputs = self.model_tester.prepare_inputs()

    def test_option_chunking(self):
        model = JevaForDecision(self.config).eval()
        output = model(**self.inputs)
        chunked = model(**self.inputs, option_chunk_size=2)
        torch.testing.assert_close(chunked.loss, output.loss)
        torch.testing.assert_close(chunked.logits, output.logits)
        torch.testing.assert_close(chunked.coverage_logits, output.coverage_logits)

    def test_chat_template_reproduces_the_model_code(self):
        # The official collator encodes the state/question and each option independently.
        texts = [
            "Decision type: choice\nQuestion: Which intent?\nState:\n" + self.request["state"],
            "Option for choice: refund",
            "Option for choice: delivery",
        ]
        segments = self.tokenizer(texts)["input_ids"]
        prompt = self.prompts[0]
        self.assertEqual(prompt.input_ids, [token for segment in segments for token in segment])
        self.assertEqual(prompt.marks, torch.tensor([len(segment) for segment in segments]).cumsum(0).tolist())

    def test_typed_decisions(self):
        model = JevaForDecision(self.config).eval()
        request = {
            "state": "A refund request",
            "questions": {
                "intent": {"type": "choice", "criteria": {str(i): "An intent" for i in range(32)}, "allow_null": True},
                "truth": {"type": "noul", "instructions": "Is this a refund?"},
                "quality": {"type": "score", "criteria": ["bad", "good"], "values": [1, 5]},
            },
        }
        prompts = render(self.tokenizer, request, model.decision_config)
        scores = [model.decide(**prompt.inputs, decision_questions=prompt.questions)[0] for prompt in prompts]
        self.assertEqual([len(score) for score in scores], [33, 2, 2])
        answers = pipeline("decision", model=model, tokenizer=self.tokenizer, device="cpu")(request)
        intent = answers["intent"]
        self.assertAlmostEqual(sum(intent["probabilities"].values()) + intent["null_probability"], 1, places=6)
        best = max(intent["probabilities"], key=intent["probabilities"].get)
        self.assertEqual(
            intent["choice"], None if intent["null_probability"] >= intent["probabilities"][best] else best
        )
        probabilities = scores[2].softmax(-1)
        self.assertAlmostEqual(
            answers["quality"]["score"], (probabilities * torch.tensor([1, 5])).sum().item(), places=6
        )
        self.assertAlmostEqual(answers["truth"]["noul"], scores[1].softmax(-1)[1].item(), places=6)

    @require_peft
    def test_checkpoint_conversion(self):
        from peft import LoraConfig, get_peft_model

        from transformers.models.jeva.convert_jeva_weights_to_hf import convert_checkpoint

        model = JevaForDecision(self.config).eval()
        with tempfile.TemporaryDirectory() as checkpoint, tempfile.TemporaryDirectory() as converted:
            path = Path(checkpoint)
            model.model.save_pretrained(path / "backbone")
            self.tokenizer.save_pretrained(path / "tokenizer")
            model.model = get_peft_model(model.model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
            for name, parameter in model.model.named_parameters():
                if "lora_B" in name:
                    torch.nn.init.normal_(parameter, std=0.01)
            model.model.save_pretrained(path / "backbone_adapter")
            settings = {"projection_dim": 16, "decision_num_heads": 2}
            (path / "config.yaml").write_text(yaml.safe_dump({"model": settings}))
            head = {
                name: {
                    key: value.squeeze() if value.numel() == 1 else value
                    for key, value in model.get_submodule(name).state_dict().items()
                }
                for name in ("context_proj", "option_proj", "core")
            }
            torch.save(head, path / "head.pt")
            expected = model(**self.inputs)
            convert_checkpoint(checkpoint, converted)
            loaded = AutoModelForDecision.from_pretrained(converted).eval()
            output = loaded(**self.inputs)
        torch.testing.assert_close(output.logits, expected.logits)
        torch.testing.assert_close(output.coverage_logits, expected.coverage_logits)
        torch.testing.assert_close(output.loss, expected.loss)
