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
import sys
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import load_file, save_file

from transformers import BertConfig, ClefConfig, ModernBertConfig, is_torch_available, logging, pipeline
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
                torch.testing.assert_close(output.logits[row, :width], question[:width], atol=1e-5, rtol=0)
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
        torch.testing.assert_close(left, right, atol=1e-5, rtol=0)

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
        torch.testing.assert_close(batched.logits[3, :3], single.logits[0], atol=1e-5, rtol=0)
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

    def test_span_pooling_memory_is_linear(self):
        # The means of the spans are differences of prefix sums: no `[spans, length, dim]` tensor
        from transformers.models.clef.modeling_clef import pool_spans

        hidden_states = torch.randn(2, 30, 8)
        starts, ends = torch.tensor([[0, 5, 0], [3, 29, 0]]), torch.tensor([[1, 12, 0], [9, 30, 0]])
        pooled = pool_spans(hidden_states, starts, ends)
        self.assertEqual(pooled.shape, (2, 3, 8))
        self.assertEqual(pooled.dtype, torch.float32)
        for b in range(2):
            for s in range(2):
                expected = hidden_states[b, starts[b, s] : ends[b, s]].mean(0)
                torch.testing.assert_close(pooled[b, s], expected, atol=1e-5, rtol=0)
            self.assertTrue((pooled[b, 2] == 0).all())  # an empty (padded) span pools to zero

        try:
            import resource
        except ImportError:
            self.skipTest("resource is not available on this platform")
        length, dim, num_spans = 4096, 64, 5000
        generator = torch.Generator().manual_seed(0)
        hidden_states = torch.randn(1, length, dim, generator=generator)
        starts = torch.randint(length - 1, (1, num_spans), generator=generator)
        ends = starts + 1 + torch.randint(length, (1, num_spans), generator=generator) % (length - starts)
        self.assertEqual(pool_spans(hidden_states, starts[:, :1], ends[:, :1]).shape, (1, 1, dim))
        # `ru_maxrss` is in bytes on macOS and in kilobytes elsewhere; a `[spans, length, dim]` float tensor is ~5 GB
        unit = 1 if sys.platform == "darwin" else 1024
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        pooled = pool_spans(hidden_states, starts, ends)
        growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit - before
        self.assertEqual(pooled.shape, (1, num_spans, dim))
        self.assertLess(growth, 512 * 1024**2)

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
                    # Up to the rounding of the prefix sums of the spans, which every token goes through
                    self.assertLess((global_inputs(model, changed_unread) - reference).abs().max().item(), 1e-5)
                    self.assertGreater((global_inputs(model, changed_read) - reference).abs().max().item(), 1e-2)
        # Without an attention mask: the first and the last token of the sequence
        model = self.encoder_model("modernbert", global_pooling="first")
        hidden_states = torch.randn(*inputs["input_ids"].shape, self.config.backbone_hidden_size)
        changed = hidden_states.clone()
        changed[:, -1] += torch.randn(len(batch), self.config.backbone_hidden_size)
        with torch.no_grad():
            reference = global_inputs(model, hidden_states, None)
            self.assertLess((global_inputs(model, changed, None) - reference).abs().max().item(), 1e-5)
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
                torch.testing.assert_close(left, right, atol=1e-5, rtol=0)
                # And a padded prompt scores as it does alone
                torch.testing.assert_close(right[3:4, :4], single, atol=1e-5, rtol=0)

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
                torch.testing.assert_close(output.logits[row, :width], question[:width], atol=1e-5, rtol=0)
                torch.testing.assert_close(left.logits[row, :width], question[:width], atol=1e-5, rtol=0)
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
