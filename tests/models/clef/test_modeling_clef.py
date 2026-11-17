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
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import load_file, save_file

from transformers import (
    AutoTokenizer,
    BertConfig,
    ClefConfig,
    ModernBertConfig,
    is_torch_available,
    logging,
    pipeline,
)
from transformers.decision import Question
from transformers.models.clef.configuration_clef import CLEF_PLAIN_TEMPLATE, CLEF_TEMPLATE
from transformers.testing_utils import CaptureLogger, require_torch, require_torch_accelerator, slow

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision, render


if is_torch_available():
    import torch
    from torch import nn

    from transformers import ClefForDecision
    from transformers.modeling_layers import GradientCheckpointingLayer
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

    # A prompt scores the same alone, in a batch and padded on either side: the spans pool their own tokens only, so
    # what is left is the float32 rounding of the backbone, which batches and pads its matmuls differently
    batch_tolerance = {"atol": 2e-6, "rtol": 0}

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

    def synthetic_prompt(self, length, options, seed):
        """
        The inputs of a prompt of `length` random tokens with a batch of one prompt: per question, the span of its
        instruction then of each of its options, of 1 to 3 tokens and one token apart.
        """
        generator = torch.Generator().manual_seed(seed)
        input_ids = torch.randint(len(self.tokenizer), (1, length), generator=generator)
        num_spans = sum(width + 1 for width in options)
        sizes = 1 + torch.arange(num_spans) % 3
        starts = 1 + torch.cat([torch.zeros(1, dtype=torch.long), (sizes + 1).cumsum(0)[:-1]])
        marks = torch.stack([starts, starts + sizes], dim=-1).flatten()
        self.assertLess(int(marks[-1]), length)
        return {
            "input_ids": input_ids,
            "marks": marks[None],
            "num_options": torch.tensor([options]),
            "question_type_ids": torch.tensor([[index % 3 for index in range(len(options))]]),
        }

    @staticmethod
    def collate(prompts, padding_side="right", pad_token_id=0):
        """A batch of prompts: tokens padded on `padding_side`, marks with -1, `num_options` and types with 0."""
        length = max(prompt["input_ids"].shape[-1] for prompt in prompts)
        num_marks = max(prompt["marks"].shape[-1] for prompt in prompts)
        num_questions = max(prompt["num_options"].shape[-1] for prompt in prompts)
        batch = {"input_ids": [], "attention_mask": [], "marks": [], "num_options": [], "question_type_ids": []}
        for prompt in prompts:
            input_ids, marks = prompt["input_ids"][0], prompt["marks"][0]
            padding = length - input_ids.shape[-1]
            pad = torch.full((padding,), pad_token_id)
            if padding_side == "left":
                batch["input_ids"].append(torch.cat([pad, input_ids]))
                batch["attention_mask"].append(torch.cat([torch.zeros(padding), torch.ones_like(input_ids)]))
                marks = marks + padding
            else:
                batch["input_ids"].append(torch.cat([input_ids, pad]))
                batch["attention_mask"].append(torch.cat([torch.ones_like(input_ids), torch.zeros(padding)]))
            batch["marks"].append(nn.functional.pad(marks, (0, num_marks - marks.shape[-1]), value=-1))
            for key in ("num_options", "question_type_ids"):
                values = prompt[key][0]
                batch[key].append(nn.functional.pad(values, (0, num_questions - values.shape[-1]), value=0))
        return {key: torch.stack(values).long() for key, values in batch.items()}

    def ragged_prompts(self):
        """Three prompts of different lengths, numbers of questions and of options."""
        return [
            self.synthetic_prompt(70, [2, 5, 1], seed=1),
            self.synthetic_prompt(40, [4], seed=2),
            self.synthetic_prompt(55, [3, 3], seed=3),
        ]

    def test_batch_equals_singles(self):
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        prompts = self.ragged_prompts()
        with torch.no_grad():
            singles = [model(**prompt).logits for prompt in prompts]
            output = model(**self.collate(prompts))
        row = 0
        for single in singles:
            for question in single:
                width = int((question > torch.finfo(question.dtype).min).sum())
                torch.testing.assert_close(output.logits[row, :width], question[:width], **self.batch_tolerance)
                self.assertEqual(output.option_mask[row].sum().item(), width)
                row += 1
        self.assertEqual(row, output.logits.shape[0])
        self.assertEqual(output.logits.shape[1], 5)

    def test_left_and_right_padding_agree(self):
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        prompts = self.ragged_prompts()
        with torch.no_grad():
            right = model(**self.collate(prompts, padding_side="right", pad_token_id=0)).logits
            left = model(**self.collate(prompts, padding_side="left", pad_token_id=7)).logits
        torch.testing.assert_close(left, right, **self.batch_tolerance)

    def test_batched_loss_equals_mean_over_questions(self):
        torch.manual_seed(0)
        config = copy.deepcopy(self.config)
        config.brier_weight, config.label_smoothing = 0.5, 0.1
        model = ClefForDecision(config).eval()
        prompts = self.ragged_prompts()
        generator = torch.Generator().manual_seed(0)
        labels = [
            torch.cat([torch.randint(n, (1,), generator=generator) for n in p["num_options"][0]]) for p in prompts
        ]
        with torch.no_grad():
            singles = torch.stack([model(**p, labels=label).loss for p, label in zip(prompts, labels)])
            batched = model(**self.collate(prompts), labels=torch.cat(labels)).loss
        counts = torch.tensor([float(label.numel()) for label in labels])
        torch.testing.assert_close(batched, (singles * counts).sum() / counts.sum(), atol=1e-6, rtol=0)

    def test_padded_question_slots_produce_no_rows(self):
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        prompt = self.synthetic_prompt(40, [3], seed=4)
        padded = {**prompt, "num_options": torch.tensor([[3, 0, 0]]), "question_type_ids": torch.tensor([[0, 0, 0]])}
        with torch.no_grad():
            single, output = model(**prompt), model(**padded)
        self.assertEqual(output.logits.shape, (1, 3))
        self.assertEqual(output.option_mask.tolist(), [[True, True, True]])
        torch.testing.assert_close(output.logits, single.logits)
        # In a batch, the record of one question contributes one row whatever the number of question slots
        with torch.no_grad():
            batched = model(**self.collate([self.synthetic_prompt(70, [2, 5, 1], seed=1), prompt]))
        self.assertEqual(batched.logits.shape, (4, 5))
        torch.testing.assert_close(batched.logits[3, :3], single.logits[0], **self.batch_tolerance)
        self.assertEqual(batched.option_mask[3].tolist(), [True, True, True, False, False])

    def test_batched_backward_and_bf16(self):
        torch.manual_seed(0)
        config = copy.deepcopy(self.config)
        config.brier_weight, config.ordinal_weight = 0.5, 0.5
        model = ClefForDecision(config)
        inputs = self.collate(self.ragged_prompts())
        labels = torch.tensor([1, 4, 0, 3, 2, 0])
        model(**inputs, labels=labels).loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(**inputs, labels=labels)
        self.assertEqual(output.logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        head = [name for name, _ in model.named_parameters() if not name.startswith(("model.", "lm_head."))]
        self.assertGreater(len(head), 0)

    def test_span_pooling(self):
        # The mean of each span over its own tokens, in float32; an empty (padded) span pools to exactly zero
        from transformers.models.clef.modeling_clef import pool_spans

        hidden_states = torch.randn(2, 30, 8)
        starts, ends = torch.tensor([[0, 5, 0], [3, 29, 0]]), torch.tensor([[1, 12, 0], [9, 30, 0]])
        pooled = pool_spans(hidden_states, starts, ends)
        self.assertEqual(pooled.shape, (2, 3, 8))
        self.assertEqual(pooled.dtype, torch.float32)
        for b in range(2):
            for s in range(2):
                expected = hidden_states[b, starts[b, s] : ends[b, s]].mean(0)
                torch.testing.assert_close(pooled[b, s], expected, atol=1e-6, rtol=0)
            self.assertTrue((pooled[b, 2] == 0).all())
        # Half precision states pool in float32
        self.assertEqual(pool_spans(hidden_states.bfloat16(), starts, ends).dtype, torch.float32)
        # The lexical pooling reads the rows of the table of embeddings of the tokens of the spans
        table, input_ids = torch.randn(50, 8), torch.randint(50, (2, 30))
        torch.testing.assert_close(
            pool_spans(table, starts, ends, input_ids=input_ids), pool_spans(table[input_ids], starts, ends)
        )
        # Its gradients reach the states (and the table) of the tokens of the spans only
        hidden_states.requires_grad_(True)
        pool_spans(hidden_states, starts, ends).sum().backward()
        in_a_span = torch.zeros(2, 30, dtype=torch.bool)
        for b in range(2):
            for s in range(2):
                in_a_span[b, starts[b, s] : ends[b, s]] = True
        self.assertTrue((hidden_states.grad[~in_a_span] == 0).all())
        self.assertTrue((hidden_states.grad[in_a_span] != 0).all())

    def test_span_pooling_precision(self):
        # Short spans at the end of a long sequence of large states: the mean of their own tokens, not the difference
        # of two running sums over the whole sequence (which float32 rounds to ~1e-2 here)
        from transformers.models.clef.modeling_clef import pool_spans

        generator = torch.Generator().manual_seed(0)
        hidden_states = 100 + torch.randn(1, 4096, 64, generator=generator)
        starts = torch.tensor([[4090, 4000, 4093, 4095, 0, 2048]])
        ends = torch.tensor([[4091, 4003, 4096, 4096, 1, 2050]])
        pooled = pool_spans(hidden_states, starts, ends)
        reference = torch.stack(
            [hidden_states[0, start:end].double().mean(0) for start, end in zip(starts[0], ends[0])]
        )
        self.assertLess((pooled[0].double() - reference).abs().max().item(), 1e-5)

    def test_span_pooling_memory_is_linear(self):
        # Many short spans that tile a long sequence, as the marks of a prompt do: memory linear in their tokens, no
        # `[spans, length, dim]` tensor (2 GB here)
        from transformers.models.clef.modeling_clef import pool_spans

        try:
            import resource
        except ImportError:
            self.skipTest("resource is not available on this platform")
        length, dim = 4096, 64
        generator = torch.Generator().manual_seed(0)
        hidden_states = torch.randn(1, length, dim, generator=generator)
        sizes = 1 + torch.randint(3, (length,), generator=generator)
        ends = sizes.cumsum(0)
        ends = torch.cat([ends[ends < length], torch.tensor([length])])
        starts = torch.cat([torch.zeros(1, dtype=torch.long), ends[:-1]])
        starts, ends = starts[None], ends[None]
        num_spans = starts.shape[-1]
        self.assertGreater(num_spans, 1500)
        self.assertEqual(int((ends - starts).sum()), length)  # they do not overlap and cover every token
        self.assertEqual(pool_spans(hidden_states, starts[:, :1], ends[:, :1]).shape, (1, 1, dim))
        # `ru_maxrss` is in bytes on macOS and in kilobytes elsewhere
        unit = 1 if sys.platform == "darwin" else 1024
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        pooled = pool_spans(hidden_states, starts, ends)
        growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit - before
        self.assertEqual(pooled.shape, (1, num_spans, dim))
        torch.testing.assert_close(pooled[0, -1], hidden_states[0, starts[0, -1] :].mean(0), atol=1e-6, rtol=0)
        self.assertLess(growth, 256 * 1024**2)

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
            num_options, question_type_ids = inputs["num_options"], inputs["question_type_ids"]
            option_question_ids = torch.repeat_interleave(num_options[0])[None]
            is_option, is_question = torch.ones_like(option_question_ids, dtype=torch.bool), num_options > 0
            memory, options, fields, field_embeddings, priors = model.embeddings(
                hidden_states,
                inputs["input_ids"],
                inputs["marks"],
                num_options,
                question_type_ids,
                model.lm_head.weight,
                option_question_ids,
                is_option,
            )
            options, fields = model.encoder(
                memory, options, fields, field_embeddings, option_question_ids, is_option, is_question
            )
            expected = model.scorer(fields[:, option_question_ids[0]], options, priors)[0]
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

    def test_rlcd_only_in_training_mode(self):
        torch.manual_seed(0)
        config = copy.deepcopy(self.config)
        config.brier_weight = 0.2
        model = ClefForDecision(config)
        inputs = self.collate(self.ragged_prompts())
        labels = torch.tensor([1, 4, 0, 3, 2, 0])

        def loss(rlcd_weight, seed=0):
            model.config.rlcd_weight = rlcd_weight
            torch.manual_seed(seed)
            with torch.no_grad():
                return model(**inputs, labels=labels).loss

        model.eval()
        supervised = loss(0.0)
        self.assertTrue(torch.equal(loss(0.25), supervised))
        self.assertTrue(torch.equal(loss(0.25, seed=1), supervised))
        model.train()
        # The backbone and the head have no dropout: in training mode only the policy term is random
        self.assertTrue(torch.equal(loss(0.0, seed=1), supervised))
        trained = loss(0.25)
        self.assertTrue(torch.isfinite(trained))
        self.assertNotEqual(trained.item(), supervised.item())
        self.assertTrue(torch.equal(loss(0.25), trained))
        self.assertNotEqual(loss(0.25, seed=1).item(), trained.item())

    def test_rlcd_config_validation(self):
        config = ClefConfig()
        self.assertEqual(
            (
                config.rlcd_weight,
                config.rlcd_sigma,
                config.rlcd_samples,
                config.rlcd_spherical_weight,
                config.rlcd_rps_weight,
                config.rlcd_normalize_advantage,
            ),
            (0.0, 0.4, 32, 0.75, 1.0, False),
        )
        for kwargs in ({"rlcd_sigma": 0.0}, {"rlcd_samples": 1}, {"rlcd_weight": -1.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ClefConfig(**kwargs)

    def test_rlcd_backward_reaches_every_parameter(self):
        torch.manual_seed(0)
        config = copy.deepcopy(self.config)
        config.rlcd_weight, config.rlcd_samples, config.rlcd_normalize_advantage = 0.25, 8, True
        model = ClefForDecision(config).train()
        output = model(**self.collate(self.ragged_prompts()), labels=torch.tensor([1, 4, 0, 3, 2, 0]))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

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

    def test_from_pretrained_with_loading_info(self):
        # `output_loading_info` gives the model, with its decision config, and the weights it initialized anew
        from transformers import AutoModelForDecision

        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded, info = AutoModelForDecision.from_pretrained(directory, output_loading_info=True)
            self.assertIsInstance(loaded, ClefForDecision)
            self.assertEqual(loaded.decision_config, model.decision_config)
            self.assertEqual(set(info["missing_keys"]), set())
            state = load_file(f"{directory}/model.safetensors")
            removed = next(key for key in state if key.startswith("model.layers."))
            del state[removed]
            save_file(state, f"{directory}/model.safetensors", metadata={"format": "pt"})
            loaded, info = ClefForDecision.from_pretrained(directory, output_loading_info=True)
        self.assertEqual(set(info["missing_keys"]), {removed})
        self.assertEqual(loaded.decision_config, model.decision_config)

    def test_any_number_of_options(self):
        # Clef pools spans and reads no label token: its decision configs cap no question type
        many = {
            "state": "s",
            "questions": {
                "pick": {"type": "choice", "criteria": {f"option {index:02d}": None for index in range(40)}},
                "rate": {"type": "score", "criteria": [f"level {index}" for index in range(30)]},
                "yes": {"type": "noul", "instructions": "Yes?"},
            },
        }
        uncapped = {"noul": None, "choice": None, "score": None}
        checkpoint_config = ClefConfig.decision_config({"joint_head_config": {}}, {})
        self.assertEqual(checkpoint_config.labels, uncapped)
        for head_type in ("joint", "cosine"):
            config = copy.deepcopy(self.config)
            config.head_type, config.cosine_width = head_type, 8
            torch.manual_seed(0)
            model = ClefForDecision(config).eval()
            self.assertEqual(model.decision_config.labels, uncapped)
            for decision_config in (checkpoint_config, model.decision_config):
                (prompt,) = render(self.tokenizer, many, decision_config)
                self.assertEqual([len(q.options) for q in prompt.questions], [40, 30, 2])
                self.assertEqual(len(prompt.marks), 2 * (40 + 30 + 2 + 3))
                with torch.no_grad():
                    output = model(**self.model_inputs(prompt))
                self.assertEqual(output.logits.shape, (3, 40))
                self.assertEqual(output.option_mask.sum(-1).tolist(), [40, 30, 2])
            decide = pipeline("decision", model=model, tokenizer=self.tokenizer)
            answers = decide(many)
            self.assertEqual(len(answers["pick"]["probabilities"]), 40)
            self.assertEqual(len(answers["rate"]["probabilities"]), 30)
            self.assertAlmostEqual(sum(answers["pick"]["probabilities"].values()), 1.0, places=5)

    @classmethod
    def encoder_tokenizer(cls):
        """The tokenizer of the tests with a `[CLS]` and a `[SEP]`, like the tokenizers of encoders."""
        if "_encoder_tokenizer" not in cls.__dict__:
            tokenizer = copy.deepcopy(cls.tokenizer)
            tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]"})
            cls._encoder_tokenizer = tokenizer
        return cls._encoder_tokenizer

    def encoder_model(self, backbone="bert", head_type="joint", text_config_kwargs=None, **config_kwargs):
        """A tiny Clef on a tiny bidirectional backbone (`bert`: absolute positions, `modernbert`: rotary)."""
        sizes = {
            "vocab_size": len(self.encoder_tokenizer()),
            "hidden_size": self.config.backbone_hidden_size,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            **(text_config_kwargs or {}),
        }
        text_config = BertConfig(**sizes) if backbone == "bert" else ModernBertConfig(**sizes, pad_token_id=0)
        defaults = {"cosine_width": 8, "global_pooling": "first", "prompt_style": "plain", **self.tiny_sizes}
        config = ClefConfig(text_config=text_config, head_type=head_type, **{**defaults, **config_kwargs})
        config._attn_implementation, config.dtype = "eager", torch.float32
        torch.manual_seed(0)
        return ClefForDecision(config).eval()

    encoders = [(backbone, head) for backbone in ("bert", "modernbert") for head in ("joint", "cosine")]

    def test_encoder_backbone_forward_backward(self):
        for backbone, head_type in self.encoders:
            with self.subTest(backbone=backbone, head_type=head_type):
                model = self.encoder_model(backbone, head_type).train()
                inputs = self.collate(self.ragged_prompts())
                output = model(**inputs, labels=torch.tensor([1, 4, 0, 3, 2, 0]))
                self.assertEqual(output.logits.shape, (6, 5))
                self.assertEqual(output.option_mask.sum(-1).tolist(), [2, 5, 1, 4, 3, 3])
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                # Every parameter but BERT's pooler (which the head does not read) gets a finite gradient
                for name, parameter in model.named_parameters():
                    if name.startswith("model.pooler."):
                        self.assertIsNone(parameter.grad, name)
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                head = model.cosine_head if head_type == "cosine" else model.scorer
                self.assertGreater(max(p.grad.abs().max().item() for p in head.parameters()), 0)

    def test_encoder_save_reload_exact(self):
        prompts = render(self.encoder_tokenizer(), self.test["request"], self.encoder_model().decision_config)
        for backbone, head_type in self.encoders:
            with self.subTest(backbone=backbone, head_type=head_type):
                model = self.encoder_model(backbone, head_type)
                inputs = self.collate(self.ragged_prompts())
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    reloaded = ClefForDecision.from_pretrained(directory, attn_implementation="eager").eval()
                self.assertEqual(reloaded.config.global_pooling, "first")
                self.assertEqual(reloaded.config.prompt_style, "plain")
                self.assertEqual(type(reloaded.model), type(model.model))
                # It reloads with the decision config it was saved with
                self.assertEqual(reloaded.decision_config, model.decision_config)
                with torch.no_grad():
                    difference = (model(**inputs).logits - reloaded(**inputs).logits).abs().max()
                self.assertEqual(difference.item(), 0)
                for prompt in prompts:
                    with torch.no_grad():
                        scores = model.decide(**prompt.inputs, decision_questions=prompt.questions)
                        reloaded_scores = reloaded.decide(**prompt.inputs, decision_questions=prompt.questions)
                    for score, reloaded_score in zip(scores, reloaded_scores):
                        torch.testing.assert_close(reloaded_score, score, atol=0, rtol=0)

    def test_saved_decision_config_wins_over_the_default(self):
        # A model saved with another decision config reloads with it, not with the default of its prompt style
        model = self.encoder_model()
        model.decision_config.temperature = {"choice": 2.0}
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = ClefForDecision.from_pretrained(directory)
        self.assertEqual(reloaded.decision_config.temperature, {"choice": 2.0})
        self.assertEqual(reloaded.decision_config.chat_template, CLEF_PLAIN_TEMPLATE)

    def test_first_pooling_uses_first_attended_token(self):
        # The global state (which the fields and the lexical priors read) is the first attended token of each prompt
        # with "first" and the last with "last"; the memory of the joint head reads every token whatever the pooling
        inputs = self.collate(self.ragged_prompts(), padding_side="left")
        attention_mask = inputs["attention_mask"]
        positions = torch.arange(attention_mask.shape[-1])
        first = (attention_mask * (attention_mask.shape[-1] - positions)).argmax(-1)
        last = (attention_mask * positions).argmax(-1)
        self.assertGreater(int(first.max()), 0)  # left padding: the first attended token is not at 0
        batch = torch.arange(len(first))

        def global_inputs(model, hidden_states, attention_mask=attention_mask):
            num_options = inputs["num_options"]
            option_ends = num_options.cumsum(-1)
            positions = torch.arange(int(option_ends[:, -1].max())).expand(len(num_options), -1).contiguous()
            option_question_ids = torch.searchsorted(option_ends, positions, right=True).clamp(
                max=num_options.shape[-1] - 1
            )
            _, _, _, field_embeddings, priors = model.embeddings(
                hidden_states,
                inputs["input_ids"],
                inputs["marks"],
                num_options,
                inputs["question_type_ids"],
                model.model.get_input_embeddings().weight,
                option_question_ids,
                positions < option_ends[:, -1:],
                attention_mask,
            )
            return torch.cat([field_embeddings.flatten(), priors.flatten()])

        for pooling, read, unread in (("first", first, last), ("last", last, first)):
            with self.subTest(pooling=pooling):
                model = self.encoder_model("modernbert", global_pooling=pooling)
                hidden_states = torch.randn(*inputs["input_ids"].shape, self.config.backbone_hidden_size)
                changed_read, changed_unread = hidden_states.clone(), hidden_states.clone()
                # Not a constant: the head normalizes the hidden states
                changed_read[batch, read] += torch.randn(len(batch), self.config.backbone_hidden_size)
                changed_unread[batch, unread] += torch.randn(len(batch), self.config.backbone_hidden_size)
                with torch.no_grad():
                    reference = global_inputs(model, hidden_states)
                    # Exactly: the spans pool their own tokens only, and the token is in none
                    self.assertEqual((global_inputs(model, changed_unread) - reference).abs().max().item(), 0)
                    self.assertGreater((global_inputs(model, changed_read) - reference).abs().max().item(), 1e-2)
        # Without an attention mask: the first and the last token of the sequence
        model = self.encoder_model("modernbert", global_pooling="first")
        hidden_states = torch.randn(*inputs["input_ids"].shape, self.config.backbone_hidden_size)
        changed = hidden_states.clone()
        changed[:, -1] += torch.randn(len(batch), self.config.backbone_hidden_size)
        with torch.no_grad():
            reference = global_inputs(model, hidden_states, None)
            self.assertEqual((global_inputs(model, changed, None) - reference).abs().max().item(), 0)
            changed[:, 0] += torch.randn(len(batch), self.config.backbone_hidden_size)
            self.assertGreater((global_inputs(model, changed, None) - reference).abs().max().item(), 1e-2)

    def test_plain_template_spans(self):
        tokenizer = self.encoder_tokenizer()
        decision_config = self.encoder_model().decision_config
        self.assertEqual(decision_config.chat_template, CLEF_PLAIN_TEMPLATE)
        (prompt,) = render(tokenizer, self.test["request"], decision_config)
        self.assertEqual(prompt.input_ids[0], tokenizer.cls_token_id)
        self.assertEqual(prompt.input_ids[-1], tokenizer.sep_token_id)
        for token in ("<|im_start|>", "<|im_end|>"):
            self.assertNotIn(tokenizer.convert_tokens_to_ids(token), prompt.input_ids)

        def text(value):
            return value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"))

        expected = []
        for question in prompt.questions:
            expected.append(text(question.name if question.instructions in (None, "") else question.instructions))
            for option, description in zip(question.options, question.descriptions):
                if question.type == "noul" and description is None:
                    description = {
                        "true": "The proposition is true or the answer is yes.",
                        "false": "The proposition is false or the answer is no.",
                    }[option]
                option = (
                    {"option_id": option} if description is None else {"option_id": option, "description": description}
                )
                expected.append(text(option))
        spans = [
            tokenizer.decode(prompt.input_ids[start:end]) for start, end in zip(prompt.marks[::2], prompt.marks[1::2])
        ]
        self.assertEqual(spans, expected)
        # The same marked spans as the ChatML prompt of the checkpoints
        (chatml,) = render(tokenizer, self.test["request"], ClefConfig.decision_config({"joint_head_config": {}}, {}))
        chatml_spans = [chatml.input_ids[start:end] for start, end in zip(chatml.marks[::2], chatml.marks[1::2])]
        self.assertEqual([prompt.input_ids[s:e] for s, e in zip(prompt.marks[::2], prompt.marks[1::2])], chatml_spans)
        # Without a [CLS] and a [SEP], the beginning and end of sequence tokens, and nothing without them either
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>"})
        (prompt,) = render(tokenizer, self.test["request"], decision_config)
        self.assertEqual((prompt.input_ids[0], prompt.input_ids[-1]), (tokenizer.bos_token_id, tokenizer.eos_token_id))

    def test_encoder_has_no_lm_head_and_ties_nothing_wrong(self):
        for backbone in ("bert", "modernbert"):
            with self.subTest(backbone=backbone):
                model = self.encoder_model(backbone)
                embeddings = model.model.get_input_embeddings().weight
                if backbone == "modernbert":
                    # No causal language modeling head class: the lexical prior reads the input embeddings
                    self.assertFalse(hasattr(model, "lm_head"))
                    self.assertFalse(any(key.startswith("lm_head") for key in model.state_dict()))
                    self.assertEqual(model._tied_weights_keys, {})
                else:
                    # BERT has a causal language modeling head class: tied to its real input embeddings
                    self.assertEqual(
                        model._tied_weights_keys, {"lm_head.weight": "model.embeddings.word_embeddings.weight"}
                    )
                    self.assertIs(model.lm_head.weight, embeddings)
                self.assertIs(model.lexical_embeddings(), embeddings)
                logger = logging.get_logger("transformers")
                with tempfile.TemporaryDirectory() as directory:
                    with CaptureLogger(logger) as saving:
                        model.save_pretrained(directory)
                    with CaptureLogger(logger) as loading:
                        reloaded = ClefForDecision.from_pretrained(directory)
                self.assertNotIn("tie", saving.out.lower())
                self.assertNotIn("tie", loading.out.lower())
                self.assertNotIn("LOAD REPORT", loading.out)
                self.assertNotIn("missing", loading.out.lower())
                self.assertNotIn("unexpected", loading.out.lower())
                if backbone == "bert":
                    self.assertIs(reloaded.lm_head.weight, reloaded.model.get_input_embeddings().weight)
                    self.assertTrue(torch.equal(reloaded.lm_head.weight, model.lm_head.weight))

    def test_encoder_pipeline(self):
        tokenizer = self.encoder_tokenizer()
        for backbone, head_type in self.encoders:
            with self.subTest(backbone=backbone, head_type=head_type):
                decide = pipeline("decision", model=self.encoder_model(backbone, head_type), tokenizer=tokenizer)
                answers = decide(self.test["request"])
                self.assertEqual(list(answers), list(self.test["request"]["questions"]))
                for answer in answers.values():
                    probabilities = answer.get("probabilities")
                    if probabilities is not None:
                        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    def test_causal_defaults_unchanged(self):
        config = ClefConfig()
        self.assertEqual((config.global_pooling, config.prompt_style), ("last", "chatml"))
        # The template of the checkpoints, as it was before encoder backbones
        digest = hashlib.sha256(CLEF_TEMPLATE.encode()).hexdigest()
        self.assertEqual(digest, "a6188c1c50d7eab117a506fad8ebc9c36c5f1d9b75ec757f8cb1268e73a7ad83")
        torch.manual_seed(0)
        model = ClefForDecision(self.config)
        decision_config = model.decision_config
        self.assertEqual(decision_config.chat_template, CLEF_TEMPLATE)
        self.assertEqual((decision_config.noul_options, decision_config.sort_choices), (["true", "false"], True))
        self.assertEqual(ClefConfig.decision_config({"joint_head_config": {}}, {}), decision_config)
        self.assertEqual(model._tied_weights_keys, {"lm_head.weight": "model.embed_tokens.weight"})
        self.assertIs(model.lexical_embeddings(), model.lm_head.weight)

    def test_invalid_pooling_and_prompt_style(self):
        with self.assertRaises(ValueError):
            ClefConfig(global_pooling="mean")
        with self.assertRaises(ValueError):
            ClefConfig(prompt_style="llama")

    def test_absolute_positions_left_and_right_padding_agree(self):
        # BERT adds absolute position embeddings: the positions must skip the left padding
        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                model = self.encoder_model("bert", head_type)
                prompts = self.ragged_prompts()
                with torch.no_grad():
                    right = model(**self.collate(prompts, padding_side="right", pad_token_id=0)).logits
                    left = model(**self.collate(prompts, padding_side="left", pad_token_id=7)).logits
                    single = model(**prompts[1]).logits
                    # Positions that count the padding shift every left-padded prompt
                    left_inputs = self.collate(prompts, padding_side="left", pad_token_id=7)
                    counting = torch.arange(left_inputs["input_ids"].shape[-1]).expand_as(left_inputs["input_ids"])
                    shifted = model(**left_inputs, position_ids=counting).logits
                self.assertGreater((shifted - right).abs().max().item(), 1e-3)
                torch.testing.assert_close(left, right, **self.batch_tolerance)
                # And a padded prompt scores as it does alone
                torch.testing.assert_close(right[3:4, :4], single, **self.batch_tolerance)

    def checkpointing_models(self):
        """Pairs of the same tiny model in train mode, for each head and a causal and a bidirectional backbone."""
        no_dropout = {"hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0}
        for name, build in (
            ("qwen2-joint", lambda: ClefForDecision(self.config)),
            ("qwen2-cosine", lambda: self.cosine_model()),
            ("modernbert-joint", lambda: self.encoder_model("modernbert", "joint")),
            ("bert-cosine", lambda: self.encoder_model("bert", "cosine", text_config_kwargs=no_dropout)),
        ):
            torch.manual_seed(0)
            model = build().train()
            yield name, model, copy.deepcopy(model)

    def test_activation_checkpointing_enable(self):
        for name, model, _ in self.checkpointing_models():
            with self.subTest(name):
                self.assertTrue(model.supports_gradient_checkpointing)
                self.assertFalse(model.is_activation_checkpointing)
                model.activation_checkpointing_enable()
                self.assertTrue(model.is_activation_checkpointing)
                layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
                backbone = [m for m in model.model.modules() if isinstance(m, GradientCheckpointingLayer)]
                self.assertGreater(len(backbone), 0)
                self.assertTrue(all(layer.gradient_checkpointing for layer in layers))
                if name.endswith("joint"):  # the layers of the head too
                    self.assertGreater(len(layers), len(backbone))
                model.activation_checkpointing_disable()
                self.assertFalse(model.is_activation_checkpointing)
                # The deprecated name the Trainer of older versions calls
                with self.assertWarns(FutureWarning):
                    model.gradient_checkpointing_enable()
                self.assertTrue(model.is_activation_checkpointing)

    def test_activation_checkpointing_matches(self):
        inputs = self.collate(self.ragged_prompts())
        labels = torch.tensor([1, 4, 0, 3, 2, 0])
        for name, model, checkpointed in self.checkpointing_models():
            with self.subTest(name):
                checkpointed.activation_checkpointing_enable()
                calls = []
                checkpointed.model.get_input_embeddings().register_forward_hook(lambda *_: calls.append(1))
                outputs = []
                for each in (model, checkpointed):
                    output = each(**inputs, labels=labels)
                    output.loss.backward()
                    outputs.append(output)
                # The checkpointed backbone runs its embeddings once, and its layers twice
                self.assertEqual(len(calls), 1)
                torch.testing.assert_close(outputs[1].logits, outputs[0].logits, atol=1e-6, rtol=0)
                torch.testing.assert_close(outputs[1].loss, outputs[0].loss, atol=1e-6, rtol=0)
                parameters = dict(checkpointed.named_parameters())
                for key, parameter in model.named_parameters():
                    if parameter.grad is None:
                        self.assertIsNone(parameters[key].grad, key)
                        continue
                    torch.testing.assert_close(parameters[key].grad, parameter.grad, atol=1e-6, rtol=0, msg=key)

    def test_activation_checkpointing_recomputes_the_layers(self):
        model = ClefForDecision(self.config).train()
        model.activation_checkpointing_enable()
        counts = {}
        for name, module in model.named_modules():
            if isinstance(module, GradientCheckpointingLayer):  # its hooks run once: count those of its MLP
                module.mlp.register_forward_pre_hook(
                    lambda *_, name=name: counts.update({name: counts.get(name, 0) + 1})
                )
        model(**self.collate(self.ragged_prompts()), labels=torch.tensor([1, 4, 0, 3, 2, 0])).loss.backward()
        # Each layer, of the backbone and of the head, runs again in the backward pass
        self.assertGreater(len(counts), 0)
        self.assertTrue(any(name.startswith("encoder.") for name in counts))
        self.assertEqual(set(counts.values()), {2}, counts)

    def test_activation_checkpointing_needs_backbone_support(self):
        model = ClefForDecision(self.config).train()
        model.model.supports_gradient_checkpointing = False
        with self.assertRaisesRegex(ValueError, "backbone .*Qwen2Model.* does not support activation checkpointing"):
            model.activation_checkpointing_enable()
        self.assertFalse(model.is_activation_checkpointing)
        model.activation_checkpointing_disable()  # nothing to disable, no error

    @classmethod
    def ultrabert_tokenizer(cls):
        """
        The tokenizer of the tests made like UltraBERT's: no `[CLS]`, one token for the beginning and end of sequence and
        the padding, and a separator.
        """
        if "_ultrabert_tokenizer" not in cls.__dict__:
            tokenizer = copy.deepcopy(cls.tokenizer)
            end = "<|endoftext|>"
            tokenizer.add_special_tokens(
                {"bos_token": end, "eos_token": end, "pad_token": end, "sep_token": "<|sep|>"}
            )
            cls._ultrabert_tokenizer = tokenizer
        return cls._ultrabert_tokenizer

    def ultrabert_model(self, head_type="joint", sliding_window=16, **text_config_kwargs):
        """
        A tiny Clef on a tiny UltraBERT: rotary positions, two local layers of radius `sliding_window` (with their own
        RoPE base) and one global layer, padding, beginning and end of sequence all id 0.
        """
        from transformers import UltrabertConfig

        text_config = UltrabertConfig(
            vocab_size=len(self.ultrabert_tokenizer()),
            hidden_size=self.config.backbone_hidden_size,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            sliding_window=sliding_window,
            **text_config_kwargs,
        )
        self.assertEqual((text_config.pad_token_id, text_config.bos_token_id, text_config.eos_token_id), (0, 0, 0))
        config = ClefConfig(
            text_config=text_config,
            head_type=head_type,
            **{"cosine_width": 8, "global_pooling": "first", "prompt_style": "plain", **self.tiny_sizes},
        )
        config._attn_implementation, config.dtype = "eager", torch.float32
        torch.manual_seed(0)
        return ClefForDecision(config).eval()

    def ultrabert_prompts(self, lengths=(70, 40, 55)):
        """The ragged prompts, each starting with the beginning of sequence token, id 0 like the padding."""
        options = ([2, 5, 1], [4], [3, 3])
        prompts = [self.synthetic_prompt(n, o, seed=seed) for seed, (n, o) in enumerate(zip(lengths, options), 1)]
        for prompt in prompts:
            prompt["input_ids"][:, 0] = 0
        return prompts

    def assert_batch_equals_singles(self, model, prompts, padding_side="right", pad_token_id=0):
        with torch.no_grad():
            singles = [model(**prompt).logits for prompt in prompts]
            output = model(**self.collate(prompts, padding_side=padding_side, pad_token_id=pad_token_id))
        row = 0
        for single in singles:
            for question in single:
                width = int((question > torch.finfo(question.dtype).min).sum())
                torch.testing.assert_close(output.logits[row, :width], question[:width], **self.batch_tolerance)
                self.assertEqual(output.option_mask[row].sum().item(), width)
                row += 1
        self.assertEqual(row, output.logits.shape[0])

    def test_ultrabert_backbone_forward_backward(self):
        from transformers import UltrabertModel

        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                model = self.ultrabert_model(head_type).train()
                self.assertIsInstance(model.model, UltrabertModel)
                # No causal language modeling head class: the lexical prior reads the input embeddings
                self.assertFalse(hasattr(model, "lm_head"))
                self.assertFalse(any(key.startswith("lm_head") for key in model.state_dict()))
                self.assertEqual(model._tied_weights_keys, {})
                self.assertIs(model.lexical_embeddings(), model.model.get_input_embeddings().weight)
                output = model(**self.collate(self.ultrabert_prompts()), labels=torch.tensor([1, 4, 0, 3, 2, 0]))
                self.assertEqual(output.logits.shape, (6, 5))
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertTrue(parameter.requires_grad, name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                # The backbone and the head learn (one bias of the scorer shifts every option alike: its gradient is 0)
                for module in (model.model, model.cosine_head if head_type == "cosine" else model.scorer):
                    self.assertGreater(max(p.grad.abs().max().item() for p in module.parameters()), 0)
                for name, parameter in model.model.named_parameters():
                    self.assertGreater(parameter.grad.abs().max().item(), 0, name)

    def test_ultrabert_batch_equals_singles(self):
        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                self.assert_batch_equals_singles(self.ultrabert_model(head_type), self.ultrabert_prompts())

    def test_ultrabert_left_and_right_padding_agree(self):
        prompts = self.ultrabert_prompts()
        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                model = self.ultrabert_model(head_type)
                batches = {
                    side: self.collate(prompts, padding_side=side, pad_token_id=0) for side in ("right", "left")
                }
                with torch.no_grad():
                    right = model(**batches["right"]).logits
                    left = model(**batches["left"]).logits
                    single = model(**prompts[1]).logits
                    # The positions skip the padding: the backbone gives each real token the same hidden state
                    hidden = {}
                    for side, batch in batches.items():
                        mask = batch["attention_mask"]
                        positions = (mask.long().cumsum(-1) - 1).clamp(min=0)
                        states = model.model(
                            input_ids=batch["input_ids"], attention_mask=mask, position_ids=positions
                        ).last_hidden_state
                        hidden[side] = [states[b][mask[b].bool()] for b in range(len(prompts))]
                for b in range(len(prompts)):
                    torch.testing.assert_close(hidden["left"][b], hidden["right"][b], atol=1e-6, rtol=0)
                torch.testing.assert_close(left, right, **self.batch_tolerance)
                torch.testing.assert_close(right[3:4, :4], single, **self.batch_tolerance)
                self.assert_batch_equals_singles(model, prompts, padding_side="left")

    def test_every_backbone_batch_and_padding_side_agree(self):
        # Causal, BERT-style and UltraBERT backbones, both heads: a prompt scores the same alone and in a batch padded
        # on either side, up to the float32 rounding of the backbone only
        for backbone in ("qwen2", "bert", "modernbert", "ultrabert"):
            for head_type in ("joint", "cosine"):
                with self.subTest(backbone=backbone, head_type=head_type):
                    if backbone == "qwen2":
                        model = ClefForDecision(self.config).eval() if head_type == "joint" else self.cosine_model()
                    elif backbone == "ultrabert":
                        model = self.ultrabert_model(head_type)
                    else:
                        model = self.encoder_model(backbone, head_type)
                    prompts = self.ultrabert_prompts() if backbone == "ultrabert" else self.ragged_prompts()
                    for padding_side in ("right", "left"):
                        self.assert_batch_equals_singles(model, prompts, padding_side)
                    with torch.no_grad():
                        right = model(**self.collate(prompts, padding_side="right", pad_token_id=0)).logits
                        left = model(**self.collate(prompts, padding_side="left", pad_token_id=0)).logits
                    torch.testing.assert_close(left, right, **self.batch_tolerance)

    def test_ultrabert_prompt_longer_than_local_window(self):
        # Prompts several times longer than the attention radius of the local layers
        prompts = self.ultrabert_prompts(lengths=(90, 40, 64))
        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                model = self.ultrabert_model(head_type, sliding_window=4)
                self.assertGreater(min(prompt["input_ids"].shape[-1] for prompt in prompts), 4 * 8)
                for padding_side in ("right", "left"):
                    self.assert_batch_equals_singles(model, prompts, padding_side)
                # The window is used: the same weights with a global radius score differently
                wide = self.ultrabert_model(head_type, sliding_window=1024)
                wide.load_state_dict(model.state_dict())
                with torch.no_grad():
                    difference = (model(**prompts[0]).logits - wide(**prompts[0]).logits).abs().max()
                self.assertGreater(difference.item(), 1e-4)

    def test_ultrabert_save_reload_exact(self):
        from transformers import UltrabertModel

        inputs = self.collate(self.ultrabert_prompts())
        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                model = self.ultrabert_model(head_type, sliding_window=8)
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    reloaded = ClefForDecision.from_pretrained(directory, attn_implementation="eager").eval()
                self.assertIsInstance(reloaded.model, UltrabertModel)
                self.assertEqual(reloaded.config.head_type, head_type)
                self.assertEqual((reloaded.config.global_pooling, reloaded.config.prompt_style), ("first", "plain"))
                text_config = reloaded.config.text_config
                self.assertEqual(text_config.layer_types, model.config.text_config.layer_types)
                self.assertEqual(text_config.sliding_window, 8)
                self.assertEqual(text_config.rope_parameters, model.config.text_config.rope_parameters)
                self.assertEqual(reloaded.decision_config, model.decision_config)
                self.assertFalse(hasattr(reloaded, "lm_head"))
                with torch.no_grad():
                    difference = (model(**inputs).logits - reloaded(**inputs).logits).abs().max()
                self.assertEqual(difference.item(), 0)

    def test_ultrabert_activation_checkpointing_parity(self):
        inputs = self.collate(self.ultrabert_prompts())
        labels = torch.tensor([1, 4, 0, 3, 2, 0])
        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                model = self.ultrabert_model(head_type, sliding_window=8).train()
                checkpointed = copy.deepcopy(model)
                checkpointed.activation_checkpointing_enable()
                self.assertTrue(checkpointed.is_activation_checkpointing)
                backbone = [m for m in checkpointed.model.modules() if isinstance(m, GradientCheckpointingLayer)]
                self.assertEqual(len(backbone), 3)
                self.assertTrue(all(layer.gradient_checkpointing for layer in backbone))
                outputs = []
                for each in (model, checkpointed):
                    output = each(**inputs, labels=labels)
                    output.loss.backward()
                    outputs.append(output)
                torch.testing.assert_close(outputs[1].logits, outputs[0].logits, atol=1e-6, rtol=0)
                torch.testing.assert_close(outputs[1].loss, outputs[0].loss, atol=1e-6, rtol=0)
                parameters = dict(checkpointed.named_parameters())
                for key, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, key)
                    torch.testing.assert_close(parameters[key].grad, parameter.grad, atol=1e-6, rtol=0, msg=key)

    def test_ultrabert_pipeline_with_tokenizer_without_cls(self):
        tokenizer = self.ultrabert_tokenizer()
        self.assertIsNone(tokenizer.cls_token)
        self.assertEqual(tokenizer.bos_token_id, tokenizer.pad_token_id)
        self.assertNotEqual(tokenizer.sep_token_id, tokenizer.bos_token_id)
        for head_type in ("joint", "cosine"):
            with self.subTest(head_type=head_type):
                model = self.ultrabert_model(head_type)
                self.assertEqual(model.decision_config.chat_template, CLEF_PLAIN_TEMPLATE)
                # The plain prompt starts with the beginning of sequence token and ends with the separator
                (prompt,) = render(tokenizer, self.test["request"], model.decision_config)
                self.assertEqual(prompt.input_ids[0], tokenizer.bos_token_id)
                self.assertEqual(prompt.input_ids[-1], tokenizer.sep_token_id)
                self.assertNotIn(tokenizer.sep_token_id, prompt.input_ids[:-1])
                decide = pipeline("decision", model=model, tokenizer=tokenizer)
                answers = decide(self.test["request"])
                self.assertEqual(list(answers), list(self.test["request"]["questions"]))
                for answer in answers.values():
                    probabilities = answer.get("probabilities")
                    if probabilities is not None:
                        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    def cosine_model(self, **config_kwargs):
        config = copy.deepcopy(self.config)
        config.head_type = "cosine"
        config.cosine_width = 8
        for key, value in config_kwargs.items():
            setattr(config, key, value)
        torch.manual_seed(0)
        return ClefForDecision(config).eval()

    def test_cosine_matches_naive(self):
        model = self.cosine_model()
        head = model.cosine_head
        prompts = self.ragged_prompts()
        inputs = self.collate(prompts)
        with torch.no_grad():
            output = model(**inputs)
            hidden = model.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).last_hidden_state
            hidden = hidden.float()
            scale = min(head.logit_scale.item(), math.log(100.0))
            expected = []
            for b, prompt in enumerate(prompts):
                marks = prompt["marks"][0].tolist()
                span = 0
                for count in prompt["num_options"][0].tolist():
                    start, end = marks[2 * span : 2 * span + 2]
                    question = head.question_norm(head.question_projection(hidden[b, start:end].mean(0)))
                    row = []
                    for option in range(1, count + 1):
                        start, end = marks[2 * (span + option) : 2 * (span + option) + 2]
                        vector = head.option_norm(head.option_projection(hidden[b, start:end].mean(0)))
                        cosine = (question * vector).sum() / (question.norm() * vector.norm())
                        row.append(math.exp(scale) * cosine)
                    expected.append(row)
                    span += count + 1
        self.assertEqual(output.logits.shape[0], len(expected))
        for row, scores in zip(output.logits, expected):
            torch.testing.assert_close(row[: len(scores)], torch.stack(scores), atol=1e-5, rtol=0)
            self.assertTrue((row[len(scores) :] == torch.finfo(torch.float32).min).all())

    def test_cosine_loads_tern_state_dict(self):
        model = self.cosine_model()
        width, size = 8, self.config.backbone_hidden_size
        tern = {
            "question_projection.weight": torch.randn(width, size),
            "question_projection.bias": torch.randn(width),
            "question_norm.weight": torch.randn(width),
            "question_norm.bias": torch.randn(width),
            "option_projection.weight": torch.randn(width, size),
            "option_projection.bias": torch.randn(width),
            "option_norm.weight": torch.randn(width),
            "option_norm.bias": torch.randn(width),
            "logit_scale": torch.tensor(1.5),
        }
        model.cosine_head.load_state_dict(tern, strict=True)
        self.assertEqual(model.cosine_head.logit_scale.item(), 1.5)

    def test_cosine_scale_clamped(self):
        model = self.cosine_model()
        with torch.no_grad():
            model.cosine_head.logit_scale.fill_(math.log(1e6))
            logits = model(**self.collate(self.ragged_prompts())).logits
        real = logits[logits > torch.finfo(torch.float32).min]
        self.assertLessEqual(real.abs().max().item(), 100 + 1e-4)

    def test_cosine_batch_equals_singles(self):
        model = self.cosine_model()
        prompts = self.ragged_prompts()
        with torch.no_grad():
            singles = [model(**prompt).logits for prompt in prompts]
            output = model(**self.collate(prompts))
            left = model(**self.collate(prompts, padding_side="left"))
        row = 0
        for single in singles:
            for question in single:
                width = int((question > torch.finfo(question.dtype).min).sum())
                torch.testing.assert_close(output.logits[row, :width], question[:width], **self.batch_tolerance)
                torch.testing.assert_close(left.logits[row, :width], question[:width], **self.batch_tolerance)
                row += 1
        self.assertEqual(row, output.logits.shape[0])

    def test_cosine_train_save_reload(self):
        model = self.cosine_model()
        inputs = self.collate(self.ragged_prompts())
        labels = torch.tensor([1, 4, 0, 3, 2, 0])
        model(**inputs, labels=labels).loss.backward()
        for name, parameter in model.cosine_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().max().item(), 0, name)
        self.assertEqual(len(list(model.cosine_head.parameters())), 9)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = ClefForDecision.from_pretrained(
                directory, attn_implementation=model.config._attn_implementation
            ).eval()
        self.assertEqual(reloaded.config.head_type, "cosine")
        with torch.no_grad():
            difference = (model(**inputs).logits - reloaded(**inputs).logits).abs().max()
        self.assertEqual(difference.item(), 0)

    def test_cosine_has_no_joint_modules(self):
        model = self.cosine_model()
        for name in ("lm_head", "embeddings", "encoder", "scorer"):
            self.assertFalse(hasattr(model, name), name)
        backbone = sum(p.numel() for p in model.model.parameters())
        head = sum(p.numel() for p in model.cosine_head.parameters())
        self.assertEqual(sum(p.numel() for p in model.parameters()), backbone + head)
        for parameter in model.cosine_head.parameters():
            self.assertEqual(parameter.dtype, torch.float32)

    def test_joint_default_unchanged(self):
        self.assertEqual(ClefConfig().head_type, "joint")
        torch.manual_seed(0)
        model = ClefForDecision(self.config)
        keys = list(model.state_dict())
        # The keys of the model at 71872b2023, before the cosine head, for this tiny config
        self.assertEqual({key.split(".")[0] for key in keys}, {"model", "lm_head", "embeddings", "encoder", "scorer"})
        self.assertEqual(len(keys), 93)
        self.assertEqual(len([key for key in keys if key.startswith("model.")]), 26)

    def test_invalid_head_type(self):
        with self.assertRaises(ValueError):
            ClefConfig(head_type="bilinear")
        with self.assertRaises(ValueError):
            ClefConfig(head_type="cosine", cosine_width=0)

    def test_cosine_head_stays_float32_in_half_precision(self):
        model = self.cosine_model()
        inputs = self.collate(self.ragged_prompts())
        labels = torch.tensor([1, 4, 0, 3, 2, 0])
        with torch.no_grad():
            expected = model(**inputs).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = ClefForDecision.from_pretrained(directory, dtype=torch.bfloat16)
        for name, parameter in loaded.cosine_head.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in loaded.model.parameters()))
        output = loaded(**inputs, labels=labels)
        self.assertEqual(output.logits.dtype, torch.float32)
        output.loss.backward()
        for name, parameter in loaded.cosine_head.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        torch.testing.assert_close(output.logits, expected, atol=0.5, rtol=0.1)
        # A model cast afterwards with `.to()` still runs: the head computes in float32 whatever its weights
        model.to(torch.bfloat16)
        self.assertEqual(model(**inputs).logits.dtype, torch.float32)

    def test_cosine_under_autocast(self):
        model = self.cosine_model()
        inputs = self.collate(self.ragged_prompts())
        labels = torch.tensor([1, 4, 0, 3, 2, 0])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(**inputs, labels=labels)
        self.assertEqual(output.logits.dtype, torch.float32)
        output.loss.backward()
        for name, parameter in model.cosine_head.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_pipeline_with_cosine(self):
        model = self.cosine_model()
        decide = pipeline("decision", model=model, tokenizer=self.tokenizer, decision_config=self.decision_config)
        answers = decide(self.test["request"])
        self.assertEqual(list(answers), list(self.test["request"]["questions"]))
        for answer in answers.values():
            probabilities = answer.get("probabilities")
            if probabilities is not None:
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    # Conversion of the checkpoints of clef and of the folders derived from them, each synthesised from tiny models

    uncapped = {"noul": None, "choice": None, "score": None}
    # The sizes of a new joint head on a backbone of `backbone_hidden_size`
    joint_sizes = {
        key: value for key, value in tiny_sizes.items() if key not in ("backbone_hidden_size", "num_key_value_heads")
    }
    converter_logger = "transformers.models.clef.convert_clef_weights_to_hf"

    @staticmethod
    def head_state(model):
        """The weights of the head of `model`, under the names of the original checkpoints, and all of its weights."""
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            state = load_file(f"{directory}/model.safetensors")
        return {key: value for key, value in state.items() if not key.startswith(("model.", "lm_head."))}, state

    @staticmethod
    def joint_head_config(config):
        """The `joint_head_config.json` of a clef checkpoint with the head of `config`."""
        return {
            "hidden_size": config.backbone_hidden_size,
            "width": config.hidden_size,
            "routing_layers": config.num_evidence_layers,
            "layers": config.num_hidden_layers,
            "heads": config.num_attention_heads,
            "feedforward": config.intermediate_size,
            "dropout": 0.0,
        }

    @staticmethod
    def write_source(directory, backbone, tokenizer, head=None, head_config=None, unsloth=None, auto_map=None):
        """A folder of a decision checkpoint: its backbone, tokenizer, and the files of its head when given."""
        backbone.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        if auto_map is not None:
            config = json.loads(Path(directory, "config.json").read_text(encoding="utf-8"))
            Path(directory, "config.json").write_text(json.dumps({**config, "auto_map": auto_map}), encoding="utf-8")
        if head is not None:
            save_file(head, f"{directory}/joint_head.safetensors", metadata={"format": "pt"})
        for name, content in (("joint_head_config.json", head_config), ("unsloth_decision_config.json", unsloth)):
            if content is not None:
                Path(directory, name).write_text(json.dumps(content), encoding="utf-8")

    def convert(self, source, cli=False, **kwargs):
        """Converts `source` to a new folder, removed after the test, and loads it with `AutoModelForDecision`."""
        from transformers import AutoModelForDecision
        from transformers.models.clef.convert_clef_weights_to_hf import convert_clef_checkpoint, main

        output = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, output, ignore_errors=True)
        if cli:
            flags = [[f"--{key}", str(value)] for key, value in kwargs.items() if value is not None]
            main(["--source", str(source), "--output", output, *[item for flag in flags for item in flag]])
        else:
            convert_clef_checkpoint(source, output, **kwargs)
        model, info = AutoModelForDecision.from_pretrained(
            output, output_loading_info=True, attn_implementation="eager"
        )
        self.assertIsInstance(model, ClefForDecision)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            self.assertEqual(set(info[key]), set(), key)
        self.assertEqual(json.loads(Path(output, "config.json").read_text(encoding="utf-8"))["model_type"], "clef")
        self.assertEqual(model.decision_config.labels, self.uncapped)
        return output, model.eval()

    def test_convert_tern_cosine_roundtrip(self):
        # A Tern cosine checkpoint: its causal backbone (`save_pretrained`, native names), its tokenizer, the state dict
        # of its cosine head with the config of that head, its calibration and runtime files
        model = self.cosine_model(cosine_initial_scale=7.0)
        with torch.no_grad():
            model.cosine_head.logit_scale.fill_(1.5)
        head, state = self.head_state(model)
        head = {key.removeprefix("cosine_head."): value for key, value in head.items()}
        self.assertEqual(len(head), 9)
        backbone = self.backbone_class(model.config.text_config)
        missing, unexpected = backbone.load_state_dict(
            {key: value for key, value in state.items() if key.startswith("model.")}, strict=False
        )
        self.assertEqual((missing, unexpected), (["lm_head.weight"], []))
        head_config = {"head_type": "cosine_span_v1", "hidden_size": 24, "width": 8, "initial_scale": 7.0}
        unsloth = {"layout": "clef", "max_len": 512, "temperature": [1.0, 1.0, 1.0], "base_model": "tiny"}
        with tempfile.TemporaryDirectory() as source:
            self.write_source(source, backbone, self.tokenizer, head, head_config, {**unsloth, "fine_tuned": True})
            Path(source, "native_klm_runtime.json").write_text(json.dumps({"weights": "float32"}), encoding="utf-8")
            with CaptureLogger(logging.get_logger(self.converter_logger)) as captured:
                _, converted = self.convert(source)
        # The language modeling head of the backbone, which the cosine head does not read, is dropped
        self.assertIn("lm_head.weight", captured.out)
        config = converted.config
        self.assertEqual((config.head_type, config.cosine_width, config.cosine_initial_scale), ("cosine", 8, 7.0))
        self.assertEqual((config.global_pooling, config.prompt_style), ("last", "chatml"))
        self.assertEqual(converted.decision_config.chat_template, CLEF_TEMPLATE)
        self.assertEqual(converted.decision_config.temperature, {"noul": 1.0, "choice": 1.0, "score": 1.0})
        for name, parameter in converted.cosine_head.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertTrue(torch.equal(parameter, head[name]), name)
        # The logits of the converted model are those of the head of the source on its backbone, exactly
        inputs = self.collate(self.ragged_prompts())
        with torch.no_grad():
            difference = (converted(**inputs).logits - model(**inputs).logits).abs().max()
        self.assertEqual(difference.item(), 0)

    def test_convert_joint_with_temperatures(self):
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        head, state = self.head_state(model)
        backbone = self.backbone_class(self.config.text_config)
        backbone.load_state_dict({key: value for key, value in state.items() if key not in head}, strict=True)
        # A head saved in bfloat16: converted to float32, it scores as the head of the model rounded to bfloat16
        head = {key: value.bfloat16() for key, value in head.items()}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not name.startswith(("model.", "lm_head.")):
                    parameter.copy_(parameter.bfloat16().float())
        inputs = self.collate(self.ragged_prompts())
        temperature, head_temperature = [0.75, 1.25, 2.0], 0.5  # noul, choice, score
        for unsloth, expected in (
            (
                {"layout": "clef", "max_len": 321, "temperature": temperature, "head_temperature": head_temperature},
                {"noul": 0.75 * 0.5, "choice": 1.25 * 0.5, "score": 2.0 * 0.5},
            ),
            (
                {"layout": "clef", "max_len": 321, "temperature": temperature},
                {"noul": 0.75, "choice": 1.25, "score": 2.0},
            ),
        ):
            with self.subTest(head_temperature="head_temperature" in unsloth), tempfile.TemporaryDirectory() as source:
                self.write_source(source, backbone, self.tokenizer, head, self.joint_head_config(self.config), unsloth)
                output, converted = self.convert(source)
                saved = json.loads(Path(output, "decision.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["temperature"], expected)
                self.assertEqual(saved["labels"], self.uncapped)
                self.assertEqual(converted.decision_config.temperature, expected)
                self.assertEqual(Path(output, "decision_template.jinja").read_text(encoding="utf-8"), CLEF_TEMPLATE)
                self.assertEqual(AutoTokenizer.from_pretrained(output).model_max_length, 321)
                self.assertEqual(converted.config.head_type, "joint")
                self.assertEqual((converted.config.global_pooling, converted.config.prompt_style), ("last", "chatml"))
                # The head is saved in float32, under the names of the original checkpoints
                weights = load_file(f"{output}/model.safetensors")
                saved_head = {
                    key: value for key, value in weights.items() if not key.startswith(("model.", "lm_head."))
                }
                self.assertEqual(set(saved_head), set(head))
                for key, value in saved_head.items():
                    self.assertEqual(value.dtype, torch.float32, key)
                    self.assertTrue(torch.equal(value, head[key].float()), key)
                with torch.no_grad():
                    difference = (converted(**inputs).logits - model(**inputs).logits).abs().max()
                self.assertEqual(difference.item(), 0)

    def reranker_package(self, directory, config, state):
        """
        A remote-code reranker package like `ericlewis/ultrabert-150m-reranker`: the original config of a native model
        with the `auto_map` and labels of its remote classes, `state` as its weights, its tokenizer, and Python files that
        must never be imported.
        """
        remote = {
            "architectures": ["UltrabertForReranking"],
            "auto_map": {
                "AutoConfig": "configuration_ultrabert.UltrabertConfig",
                "AutoModelForSequenceClassification": "modeling_reranker.UltrabertForReranking",
            },
            "num_labels": 1,
            "id2label": {"0": "relevance"},
            "label2id": {"relevance": 0},
            "problem_type": "regression",
        }
        Path(directory, "config.json").write_text(json.dumps({**config, **remote}), encoding="utf-8")
        save_file(state, f"{directory}/model.safetensors", metadata={"format": "pt"})
        for name in ("configuration_ultrabert.py", "modeling_reranker.py"):
            Path(directory, name).write_text(
                'raise RuntimeError("the remote code of the package ran")\n', encoding="utf-8"
            )
        self.ultrabert_tokenizer().save_pretrained(directory)

    def test_convert_reranker_backbone(self):
        # The weights of a native UltraBERT under a single `backbone.` prefix next to a task head: the prefix is
        # stripped, the backbone loaded strictly, the task head dropped, and a new decision head initialized
        from transformers import AutoConfig, UltrabertModel

        config = {
            "model_type": "ultrabert",
            "vocab_size": len(self.ultrabert_tokenizer()),
            "hidden_size": 24,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 12,
            "attention_windows": [8, 8, None],  # the original format of the package
            "rope_theta": [10000.0, 10000.0, 160000.0],
            "pad_token_id": 0,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "dtype": "float32",
        }
        torch.manual_seed(0)
        backbone = UltrabertModel(AutoConfig.for_model(**config)).eval()
        self.assertEqual(backbone.config.layer_types, ["sliding_attention", "sliding_attention", "full_attention"])
        state = {f"backbone.{key}": value.contiguous() for key, value in backbone.state_dict().items()}
        size = config["hidden_size"]
        task_head = {
            "pair_head.0.weight": torch.randn(size, size),
            "pair_head.2.weight": torch.randn(size),
            "pair_head.2.bias": torch.randn(size),
            "pair_head.3.weight": torch.randn(1, size),
            "pair_head.3.bias": torch.randn(1),
        }
        inputs = self.collate(self.ultrabert_prompts())
        with tempfile.TemporaryDirectory() as source:
            self.reranker_package(source, config, {**state, **task_head})
            for head_type, sizes, cli in ((None, self.joint_sizes, False), ("cosine", {"cosine_width": 8}, True)):
                with self.subTest(head_type=head_type):
                    with CaptureLogger(logging.get_logger(self.converter_logger)) as captured:
                        output, converted = self.convert(source, cli=cli, head_type=head_type, **sizes)
                    for key in task_head:
                        self.assertIn(key, captured.out)
                    self.assertEqual(converted.config.head_type, head_type or "joint")
                    self.assertIsInstance(converted.model, UltrabertModel)
                    text_config = converted.config.text_config
                    self.assertEqual(text_config.layer_types, backbone.config.layer_types)
                    self.assertEqual(text_config.sliding_window, 8)
                    self.assertEqual(text_config.rope_parameters, backbone.config.rope_parameters)
                    saved = json.loads(Path(output, "config.json").read_text(encoding="utf-8"))
                    for key in ("auto_map", "architectures", "id2label"):
                        self.assertNotIn(key, saved["text_config"])
                    self.assertNotIn("auto_map", saved)
                    self.assertFalse(any(Path(output).glob("*.py")))
                    self.assertEqual(
                        (converted.config.global_pooling, converted.config.prompt_style), ("first", "plain")
                    )
                    self.assertEqual(converted.decision_config.chat_template, CLEF_PLAIN_TEMPLATE)
                    # Every tensor of the backbone is the one of the source
                    converted_state = converted.model.state_dict()
                    self.assertEqual(set(converted_state), set(backbone.state_dict()))
                    for key, value in backbone.state_dict().items():
                        self.assertTrue(torch.equal(converted_state[key], value), key)
                    self.assertFalse(any(key.startswith("lm_head") for key in converted.state_dict()))
                    if head_type == "cosine":
                        self.assertEqual(converted.config.cosine_width, 8)
                        self.assertAlmostEqual(converted.cosine_head.logit_scale.item(), math.log(12.0), places=6)
                    else:
                        self.assertEqual(converted.config.hidden_size, self.joint_sizes["hidden_size"])
                    with torch.no_grad():
                        logits = converted(**inputs).logits
                    self.assertTrue(torch.isfinite(logits[logits > torch.finfo(logits.dtype).min]).all())
        # The repair of the prefix is refused when it is ambiguous or incomplete
        first_key = next(iter(state))
        refused = {
            "ambiguous": (
                {**state, **{key.replace("backbone.", "encoder.", 1): v.clone() for key, v in state.items()}},
                "ambiguous",
            ),
            "incomplete": (
                {key: value for key, value in state.items() if key != first_key},
                first_key.removeprefix("backbone."),
            ),
            "unexpected": ({**state, "backbone.extra.weight": torch.zeros(2)}, "extra.weight"),
        }
        for case, (weights, message) in refused.items():
            with self.subTest(refused=case), tempfile.TemporaryDirectory() as source:
                self.reranker_package(source, config, {**weights, **task_head})
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    self.convert(source)

    def test_convert_sequence_classification_backbone(self):
        # A joint head next to a backbone whose config registers a (remote) sequence-classification class only: the bare
        # native model is the backbone, whether saved without prefix or as its native sequence-classification class
        from transformers import UltrabertForSequenceClassification

        model = self.ultrabert_model("joint")
        head, state = self.head_state(model)
        auto_map = {"AutoModelForSequenceClassification": "modeling_reranker.UltrabertForReranking"}
        inputs = self.collate(self.ultrabert_prompts())
        for layout in ("bare", "sequence_classification"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as source:
                if layout == "bare":
                    backbone = model.model
                else:
                    torch.manual_seed(0)
                    backbone = UltrabertForSequenceClassification(model.config.text_config)
                    missing, unexpected = backbone.load_state_dict(
                        {key: value for key, value in state.items() if key.startswith("model.")}, strict=False
                    )
                    self.assertEqual(unexpected, [])
                    self.assertTrue(missing and not any(key.startswith("model.") for key in missing), missing)
                head_config = self.joint_head_config(model.config)
                self.write_source(source, backbone, self.ultrabert_tokenizer(), head, head_config, auto_map=auto_map)
                with CaptureLogger(logging.get_logger(self.converter_logger)) as captured:
                    _, converted = self.convert(source)
                if layout == "sequence_classification":
                    for key in missing:  # the task head of the source, dropped
                        self.assertIn(key, captured.out)
                self.assertEqual((converted.config.global_pooling, converted.config.prompt_style), ("first", "plain"))
                with torch.no_grad():
                    difference = (converted(**inputs).logits - model(**inputs).logits).abs().max()
                self.assertEqual(difference.item(), 0)

    def test_incomplete_source_raises(self):
        from transformers.models.clef.convert_clef_weights_to_hf import convert_clef_checkpoint

        model = self.cosine_model()
        head, state = self.head_state(model)
        head = {key.removeprefix("cosine_head."): value for key, value in head.items()}
        backbone = self.backbone_class(model.config.text_config)
        backbone.load_state_dict({k: v for k, v in state.items() if k.startswith("model.")}, strict=False)
        head_config = {"head_type": "cosine_span_v1", "hidden_size": 24, "width": 8, "initial_scale": 12.0}
        # One head file without the other
        for present, missing in (
            ("joint_head.safetensors", "joint_head_config.json"),
            ("joint_head_config.json", None),
        ):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as source:
                self.write_source(source, backbone, self.tokenizer, head, head_config)
                missing = missing or "joint_head.safetensors"
                Path(source, missing).unlink()
                with tempfile.TemporaryDirectory() as output:
                    with self.assertRaisesRegex(ValueError, re.escape(missing)):
                        convert_clef_checkpoint(source, output)
                    self.assertEqual(list(Path(output).iterdir()), [])
        # A head of another type, sizes or a head type that contradict the head of the source, other layouts
        with tempfile.TemporaryDirectory() as source:
            self.write_source(source, backbone, self.tokenizer, head, head_config)
            for kwargs, message in (
                ({"head_type": "joint"}, "cosine"),
                ({"cosine_width": 16}, "cosine_width"),
                ({"safe_serialization": False}, "safetensors"),
            ):
                with self.subTest(**kwargs), tempfile.TemporaryDirectory() as output:
                    with self.assertRaisesRegex(ValueError, message):
                        convert_clef_checkpoint(source, output, **kwargs)
            Path(source, "joint_head_config.json").write_text(
                json.dumps({**head_config, "head_type": "bilinear_v9"}), encoding="utf-8"
            )
            with tempfile.TemporaryDirectory() as output, self.assertRaisesRegex(ValueError, "bilinear_v9"):
                convert_clef_checkpoint(source, output)
            Path(source, "joint_head_config.json").write_text(json.dumps(head_config), encoding="utf-8")
            Path(source, "unsloth_decision_config.json").write_text(
                json.dumps({"layout": "per_question"}), encoding="utf-8"
            )
            with tempfile.TemporaryDirectory() as output, self.assertRaisesRegex(ValueError, "per_question"):
                convert_clef_checkpoint(source, output)

    def test_converted_checkpoint_loads_with_auto(self):
        # Converted with the command line, the checkpoint loads with `AutoModelForDecision` and runs in the pipeline
        torch.manual_seed(0)
        model = ClefForDecision(self.config).eval()
        head, state = self.head_state(model)
        backbone = self.backbone_class(self.config.text_config)
        backbone.load_state_dict({key: value for key, value in state.items() if key not in head}, strict=True)
        with tempfile.TemporaryDirectory() as source:
            self.write_source(source, backbone, self.tokenizer, head, self.joint_head_config(self.config))
            output, converted = self.convert(source, cli=True)
        names = {path.name for path in Path(output).iterdir()}
        native = {
            "config.json",
            "model.safetensors",
            "decision.json",
            "decision_template.jinja",
            "tokenizer_config.json",
        }
        self.assertLessEqual(native, names)
        self.assertFalse(names & {"joint_head.safetensors", "joint_head_config.json", "unsloth_decision_config.json"})
        for scores, converted_scores in zip(self.scores(model), self.scores(converted)):
            torch.testing.assert_close(converted_scores, scores, atol=0, rtol=0)
        decide = pipeline("decision", model=output, model_kwargs={"attn_implementation": "eager"})
        self.assertIsInstance(decide.model, ClefForDecision)
        self.assertEqual(decide.decision_config.labels, self.uncapped)
        answers = decide(self.test["request"])
        self.assertEqual(list(answers), list(self.test["request"]["questions"]))
        for answer in answers.values():
            probabilities = answer.get("probabilities")
            if probabilities is not None:
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)


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
