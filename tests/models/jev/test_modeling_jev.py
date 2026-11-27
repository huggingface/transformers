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

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from transformers import AutoProcessor, is_torch_available, pipeline
from transformers.decision import DecisionConfig, Question
from transformers.testing_utils import require_torch, require_torch_accelerator, require_vision, slow

from ...test_decision_head_common import (
    DecisionHeadTesterMixin,
    check_decision,
    render,
)


if is_torch_available():
    import torch

    from transformers import JevForDecision


class JevModelTester:
    def __init__(self, config, prompt):
        self.config = config
        self.prompt = prompt

    def prepare_config_and_inputs_for_training(self):
        prompt = self.prompt
        counts = torch.tensor([[len(q.options) for q in prompt.questions]])
        types = torch.tensor([[Question.questions_types.index(q.type) for q in prompt.questions]])
        generator = torch.Generator().manual_seed(0)
        labels = (torch.rand(counts.shape, generator=generator) * counts).long()
        inputs = {
            **prompt.inputs,
            "marks": prompt.inputs["marks"][None],
            "num_options": counts,
            "question_type_ids": types,
            "labels": labels,
            "loss_weights": torch.full_like(counts, 0.5, dtype=torch.float32),
        }
        return copy.deepcopy(self.config), inputs


@require_torch
class JevForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (JevForDecision,) if is_torch_available() else ()
    is_training = True
    decision_repo = "hf-internal-testing/decision-gev"
    tiny_sizes = {"hidden_size": 16}

    def setUp(self):
        self.model_tester = JevModelTester(self.config, self.prompts[0])

    def test_training_loss_on_packed_questions(self):
        model = JevForDecision(copy.deepcopy(self.config)).train()
        questions = [
            Question.from_request(
                "intent", {"type": "choice", "criteria": {"refund": None, "tracking": None}}, self.decision_config
            ),
            Question.from_request(
                "urgency", {"type": "score", "criteria": ["Low", "Medium", "High"]}, self.decision_config
            ),
        ]
        prompt = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]), "marks": torch.tensor([3, 6])}
        targets = [[0.0, 1.0, 0.0], [0.0, 0.25, 0.75, 0.0]]
        batch = model.prepare_inputs_for_decision([(prompt, questions)], targets=targets)
        outputs = model(
            **batch,
            loss_weights=torch.tensor([[0.5, 2.0]]),
            num_items_in_batch=3,
        )

        choice_target, score_target = (torch.tensor(target[1:]) for target in targets)
        choice_logits, score_logits = outputs.logits.detach().split((2, 3))
        choice_kl = torch.nn.functional.cross_entropy(choice_logits, choice_target)
        score_log_probs = score_logits.log_softmax(-1)
        score_kl = torch.nn.functional.kl_div(score_log_probs, score_target, reduction="sum")
        score_rps = (score_log_probs.exp().cumsum(0)[:-1] - score_target.cumsum(0)[:-1]).square().mean()
        expected_loss = (0.5 * choice_kl + 2 * (score_kl + 0.5 * score_rps)) / 3
        torch.testing.assert_close(outputs.loss, expected_loss)

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
