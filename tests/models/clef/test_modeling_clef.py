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

from safetensors.torch import load_file, save_file

from transformers import is_torch_available, pipeline
from transformers.decision import Question
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision


if is_torch_available():
    import torch

    from transformers import ClefForDecision
    from transformers.models.clef.modeling_clef import ClefDecisionOutput


@require_torch
class ClefForDecisionTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (ClefForDecision,) if is_torch_available() else ()
    decision_repo = "hf-internal-testing/decision-clef"
    # The backbone is wider than the head
    tiny_sizes = {
        "backbone_hidden_size": 24,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_evidence_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }

    def model_inputs(self, prompt):
        """The inputs of `forward` for a prompt, with a batch of one prompt."""
        questions = prompt.questions
        return {
            "input_ids": prompt.inputs["input_ids"],
            "marks": prompt.inputs["marks"][None],
            "num_options": torch.tensor([[len(question.options) for question in questions]]),
            "question_type_ids": torch.tensor([[Question.questions_types.index(q.type) for q in questions]]),
        }

    def labels(self, prompt):
        """The first option of every question as the right one."""
        return torch.zeros(len(prompt.questions), dtype=torch.long)

    def test_forward_with_labels_returns_loss(self):
        torch.manual_seed(0)
        config = copy.deepcopy(self.config)
        config.brier_weight, config.ordinal_weight = 0.5, 0.5
        model = ClefForDecision(config)
        prompt = self.prompts[0]
        output = model(**self.model_inputs(prompt), labels=self.labels(prompt))
        self.assertIsInstance(output, ClefDecisionOutput)
        self.assertTrue(torch.isfinite(output.loss))
        widths = [len(question.options) for question in prompt.questions]
        self.assertEqual(output.logits.shape, (len(widths), max(widths)))
        self.assertEqual(output.option_mask.sum(-1).tolist(), widths)
        output.loss.backward()
        # Every parameter the loss depends on has a gradient (the lm head only reads its weight through the lexical prior)
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        self.assertEqual(missing, [])

    def test_forward_with_ragged_options_gives_finite_loss(self):
        # Questions of 1, 4 and 2 options: five padded cells, all at the minimum of the dtype
        torch.manual_seed(0)
        config = copy.deepcopy(self.config)
        config.brier_weight, config.ordinal_weight, config.label_smoothing = 0.5, 0.5, 0.1
        model = ClefForDecision(config)
        input_ids = self.prompts[0].inputs["input_ids"]
        widths, marks = [1, 4, 2], torch.arange(1, 2 * (3 + 7) + 1)
        self.assertLess(int(marks[-1]), input_ids.shape[-1])
        output = model(
            input_ids=input_ids,
            marks=marks[None],
            num_options=torch.tensor([widths]),
            question_type_ids=torch.tensor([[0, 2, 1]]),
            labels=torch.tensor([0, 3, 1]),
        )
        self.assertEqual(output.logits.shape, (3, 4))
        self.assertEqual((~output.option_mask).sum().item(), 5)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_forward_without_labels_unchanged(self):
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        prompt = self.prompts[0]
        inputs = self.model_inputs(prompt)
        with torch.no_grad():
            # The scores of the head modules, as `forward` produced them before it returned an output
            hidden_states = model.model(input_ids=inputs["input_ids"], use_cache=False).last_hidden_state
            num_options, question_type_ids = inputs["num_options"][0], inputs["question_type_ids"][0]
            option_question_ids = torch.repeat_interleave(num_options)
            memory, options, fields, field_embeddings, priors = model.embeddings(
                hidden_states,
                inputs["input_ids"],
                inputs["marks"][0],
                num_options,
                question_type_ids,
                model.lm_head.weight,
            )
            options, fields = model.encoder(memory, options, fields, field_embeddings, option_question_ids)
            expected = model.scorer(fields[:, option_question_ids], options, priors)[0]
            output = model(**inputs)
        self.assertIsNone(output.loss)
        torch.testing.assert_close(output.logits[output.option_mask], expected)
        padded = output.logits[~output.option_mask]
        self.assertTrue((padded == torch.finfo(output.logits.dtype).min).all())

    def test_decide_and_pipeline_unchanged(self):
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        for prompt, scores in zip(self.prompts, self.scores(model)):
            self.assertEqual([len(s) for s in scores], [len(question.options) for question in prompt.questions])
        decide = pipeline("decision", model=model, tokenizer=self.tokenizer, decision_config=self.decision_config)
        answers = decide(self.test["request"])
        self.assertEqual(list(answers), list(self.test["request"]["questions"]))
        for answer in answers.values():
            probabilities = answer.get("probabilities")
            if probabilities is not None:
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    def test_loss_weights_come_from_config(self):
        prompt = self.prompts[0]
        losses = []
        for brier_weight in (0.0, 0.5):
            torch.manual_seed(0)
            config = copy.deepcopy(self.config)
            config.brier_weight = brier_weight
            model = ClefForDecision(config).eval()
            with torch.no_grad():
                losses.append(model(**self.model_inputs(prompt), labels=self.labels(prompt)).loss)
        self.assertGreater(losses[1].item(), losses[0].item())

    def test_from_a_clef_checkpoint(self):
        # A clef checkpoint ships its head (`joint_head.safetensors`, original names) and its config
        # (`joint_head_config.json`) next to the weights of its model: read as they are, without any file of ours
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        with tempfile.TemporaryDirectory() as saved, tempfile.TemporaryDirectory() as repo:
            model.save_pretrained(saved)  # under the names of the original checkpoint
            state_dict = load_file(f"{saved}/model.safetensors")
            head = {key: value for key, value in state_dict.items() if not key.startswith(("model.", "lm_head."))}
            save_file(head, f"{repo}/joint_head.safetensors", metadata={"format": "pt"})
            backbone = self.backbone_class(self.config.text_config)
            backbone.load_state_dict({k: v for k, v in state_dict.items() if k not in head}, strict=False)
            backbone.save_pretrained(repo)
            self.tokenizer.save_pretrained(repo)
            config = self.config
            sizes = {
                "hidden_size": config.backbone_hidden_size,
                "width": config.hidden_size,
                "routing_layers": config.num_evidence_layers,
                "layers": config.num_hidden_layers,
                "heads": config.num_attention_heads,
                "feedforward": config.intermediate_size,
            }
            Path(repo, "joint_head_config.json").write_text(json.dumps(sizes), encoding="utf-8")
            loaded = ClefForDecision.from_pretrained(repo, attn_implementation="eager").eval()
            decide = pipeline("decision", model=repo, model_kwargs={"attn_implementation": "eager"})
        for scores, loaded_scores in zip(self.scores(model), self.scores(loaded)):
            torch.testing.assert_close(loaded_scores, scores)
        self.assertIsInstance(decide.model, ClefForDecision)
        self.assertTrue(decide.decision_config.sort_choices)


@slow
@require_torch_accelerator
class ClefIntegrationTest(unittest.TestCase):
    def test_clef_flash(self):
        expected = {
            "intent": {"other": 0.0093, "refund": 0.9907},
            "urgent": 0.0817,
            "tone": {"0": 0.9143, "1": 0.0857},
        }
        check_decision(self, "Cloudflare/clef-flash", "ArthurZ/clef-flash", expected, dtype="bfloat16")
