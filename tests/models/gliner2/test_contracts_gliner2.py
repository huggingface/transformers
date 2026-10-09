# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import unittest
from types import SimpleNamespace

from transformers import AutoConfig, Gliner2Config, is_torch_available
from transformers.testing_utils import require_torch
from transformers.utils import is_tokenizers_available

from .test_modeling_gliner2 import _word_tokenizer


if is_torch_available():
    import torch

    from transformers import Gliner2ForSchemaExtraction


def _encoder():
    config = AutoConfig.for_model(
        "deberta-v2",
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        type_vocab_size=0,
    )
    config._attn_implementation = "eager"
    return config


def _boundary(**overrides):
    config = {
        "boundary_dim": 16,
        "pair_dim": 16,
        "boundary_attention_layers": 0,
        "candidate_attention_layers": 0,
        "query_attention_layers": 0,
        "candidate_pool": "per_query",
        "enable_records": False,
        "enable_relations": False,
        "pool_size": 4,
        "candidate_budget": 4,
        "training_candidate_budget": 4,
        "start_top_k": 2,
        "end_top_k": 2,
        "ends_per_start": 2,
        "starts_per_end": 2,
    }
    config.update(overrides)
    return config


def _boundary_inputs():
    return {
        "input_ids": torch.randint(1, 50, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "text_word_indices": torch.tensor([[1, 2, 3, 4]]),
        "text_word_mask": torch.ones(1, 4, dtype=torch.bool),
        "query_marker_indices": torch.tensor([[0, 1]]),
        "query_marker_mask": torch.ones(1, 2, dtype=torch.bool),
        "query_group_index": torch.zeros(1, 2, dtype=torch.long),
        "prompt_marker_indices": torch.tensor([[2]]),
        "prompt_marker_mask": torch.ones(1, 1, dtype=torch.bool),
        "prompt_group_index": torch.zeros(1, 1, dtype=torch.long),
        "cls_marker_indices": torch.tensor([[5]]),
        "cls_marker_mask": torch.ones(1, 1, dtype=torch.bool),
        "cls_group_index": torch.zeros(1, 1, dtype=torch.long),
        "task_type_ids": torch.tensor([[1]]),
        "group_mask": torch.ones(1, 1, dtype=torch.bool),
    }


def _span_inputs():
    batch = _boundary_inputs()
    batch["cls_marker_indices"] = torch.tensor([[5]])
    batch["cls_marker_mask"] = torch.ones(1, 1, dtype=torch.bool)
    batch["cls_group_index"] = torch.zeros(1, 1, dtype=torch.long)
    return batch


@require_torch
class ContractBranchTest(unittest.TestCase):
    def test_count_lstm_v2_and_training_step(self):
        config = Gliner2Config(encoder_config=_encoder(), max_width=4, counting_layer="count_lstm_v2")
        model = Gliner2ForSchemaExtraction(config).train()
        labels = {"span_structures": [[[1, [[(0, 0)]]]]]}
        output = model(**_span_inputs(), labels=labels)
        self.assertEqual(output.span_logits[0][0].shape[0], 1)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.count_embed.pos_embedding.weight.grad)

    def test_gradient_checkpointing_training_step(self):
        config = Gliner2Config(encoder_config=_encoder(), max_width=4, counting_layer="count_lstm_v2")
        model = Gliner2ForSchemaExtraction(config).train()
        model.encoder.gradient_checkpointing_enable()
        labels = {"span_structures": [[[1, [[(0, 0)]]]]]}
        output = model(**_span_inputs(), labels=labels)
        output.loss.backward()
        self.assertIsNotNone(next(model.span_rep.parameters()).grad)

    def test_boundary_branches_return_tensors(self):
        for export_mode in ("auto", "streaming", "vectorized"):
            for pool in ("per_query", "shared"):
                model = Gliner2ForSchemaExtraction(
                    Gliner2Config(
                        encoder_config=_encoder(),
                        architecture="boundary",
                        boundary_config=_boundary(export_mode=export_mode, candidate_pool=pool),
                    )
                ).eval()
                output = model(**_boundary_inputs())
                self.assertIsNotNone(output.boundary.candidates.pair_logits)
                self.assertIsNone(getattr(output, "relation_scorer", None))
                self.assertIsNone(output.relation_pairs)
        for flag in ({"enable_rotary_endpoints": True}, {"enable_span_content": True}):
            model = Gliner2ForSchemaExtraction(
                Gliner2Config(encoder_config=_encoder(), architecture="boundary", boundary_config=_boundary(**flag))
            ).eval()
            output = model(**_boundary_inputs())
            self.assertTrue(torch.isfinite(output.boundary.candidates.pair_logits).all())

    def test_attribute_rescoring(self):
        model = Gliner2ForSchemaExtraction(
            Gliner2Config(encoder_config=_encoder(), architecture="boundary", boundary_config=_boundary())
        ).eval()
        output = model(**_boundary_inputs())
        scores = model.score_spans(
            output.text_states,
            output.text_mask,
            output.query_states,
            output.query_mask,
            torch.tensor([[[[0, 2]], [[0, 2]]]], dtype=torch.long),
        )
        self.assertEqual(tuple(scores.shape), (1, 2, 1))

    def test_inference_relations_and_records_are_tensors(self):
        if not is_tokenizers_available():
            self.skipTest("tokenizers is required")
        from transformers import Gliner2Processor

        tokenizer = _word_tokenizer()
        processor = Gliner2Processor(tokenizer, word_splitter="whitespace")
        schema = {
            "entities": {"person": {"dtype": "list", "threshold": 0.4, "validators": []}},
            "relations": [{"wrote": {"head": {}, "tail": {}}}],
            "json_structures": [{"note": {"title": {"dtype": "str", "choices": ["notes"]}, "body": "list"}}],
            "record_metadata": {"note": {"mode": "anchorless"}},
            "classifications": [
                {"task": "topic", "labels": ["math", "art"], "multi_label": False, "class_act": "softmax"}
            ],
        }
        text = "Ada wrote notes."
        labels = {
            "entities": {"person": ["Ada"]},
            "relations": {"wrote": [{"head": "Ada", "tail": "notes"}]},
            "json_structures": {"note": [{"title": "notes", "body": "notes"}]},
            "classifications": {"topic": ["math"]},
        }
        encoded = processor(text, schema=schema, labels=labels, architecture="boundary", return_tensors="pt")
        targets = encoded["targets"]
        for key in (
            "span_structures",
            "relation_gold_pairs",
            "relation_gold_mask",
            "relation_routing",
            "record_groups",
            "classification_targets",
        ):
            self.assertIn(key, targets)
        self.assertNotIn("structure_labels", targets)
        self.assertNotIn("relation_edges", targets)
        self.assertNotIn("records", targets)
        groups = encoded["metadata"][0]["groups"]
        entity = next(group for group in groups if group.task == "entities")
        person = next(field for field in entity.fields if field.name == "person")
        self.assertEqual(person.threshold, 0.4)
        note = next(group for group in groups if group.name.startswith("note"))
        self.assertEqual(note.record.mode, "anchorless")
        title = next(field for field in note.fields if field.name == "title")
        self.assertEqual(title.choices, ("notes",))
        topic = next(group for group in groups if group.task == "classifications")
        self.assertEqual(topic.activation, "softmax")
        config = Gliner2Config(
            encoder_config=_encoder(),
            architecture="boundary",
            boundary_config=_boundary(
                enable_records=True, enable_relations=True, record_dim=16, record_instance_queries=2
            ),
        )
        config.encoder_config.vocab_size = tokenizer.vocab_size
        model = Gliner2ForSchemaExtraction(config).eval()
        batch = {key: value for key, value in encoded.items() if key != "metadata"}
        output = model(**batch)
        self.assertIsNone(getattr(output, "relation_scorer", None))
        self.assertIsInstance(output.relation_pairs, torch.Tensor)
        self.assertIsInstance(output.relation_logits, torch.Tensor)
        self.assertEqual(output.relation_pairs.shape[-1], 6)
        self.assertIsInstance(output.record_logits, list)
        self.assertTrue(output.record_logits[0])
        self.assertTrue(torch.is_tensor(output.record_logits[0][0].object_logits))
        model.train()
        trained = model(**batch)
        self.assertTrue(torch.isfinite(trained.loss))
        self.assertIn("relation_loss", trained.losses)
        self.assertIn("record_object_loss", trained.losses)


@require_torch
class ContractDecodeTest(unittest.TestCase):
    def _schema(self):
        def label(task, name):
            return {"type": "LabelRef", "task": task, "label": name}

        return {
            "tasks": {
                "topic": {"labels": ["math", "art"], "min_labels": 1, "max_labels": 1},
                "mood": {"labels": ["calm", "tense"], "default": "calm"},
                "level": {"labels": ["low", "high"], "min_labels": 1, "max_labels": 1, "ordered": True},
            },
            "constraints": [
                {"type": "Implies", "cond": label("topic", "math"), "then": label("mood", "calm")},
                {"type": "Not", "child": label("mood", "tense")},
                {"type": "Excludes", "left": label("topic", "art"), "right": label("mood", "tense")},
                {"type": "Iff", "left": label("topic", "math"), "right": label("mood", "calm")},
                {"type": "And", "children": [label("topic", "math")]},
                {"type": "Or", "children": [label("mood", "calm"), label("level", "low")]},
                {"type": "ExactlyOneOf", "children": [label("level", "low"), label("level", "high")]},
                {"type": "Cardinality", "task": "mood", "minimum": 1, "maximum": None},
                {"type": "Cardinality", "task": "mood", "minimum": 0, "maximum": 1},
                {"type": "Cardinality", "task": "mood", "minimum": 1, "maximum": 1},
                {"type": "AnySelected", "task": "mood"},
                {"type": "AnyOtherSelected", "task": "mood"},
                {"type": "IsDefault", "task": "mood"},
                {"type": "MinLevel", "task": "level", "level": "low"},
                {"type": "MaxLevel", "task": "level", "level": "high"},
                {"type": "AtLevel", "task": "level", "level": "low"},
            ],
        }

    def test_constraint_operators_and_decoders(self):
        from transformers.models.gliner2.decoding_gliner2 import decode_constrained_classification

        schema = self._schema()
        logits = {
            "topic": {"math": 2.0, "art": -2.0},
            "mood": {"calm": 1.5, "tense": -1.5},
            "level": {"low": 1.0, "high": -1.0},
        }
        for decoder in ("auto", "independent", "exact", "beam"):
            decoded = decode_constrained_classification(logits, schema, 1.0, decoder=decoder, on_infeasible="relax")
            self.assertIn("topic", decoded)
        relaxed = decode_constrained_classification(
            logits, schema, 1.0, decoder="exact", on_infeasible="min_violations"
        )
        self.assertIn("topic", relaxed)

    def test_overlap_policies(self):
        from transformers.models.gliner2.decoding_gliner2 import resolve_overlaps

        spans = [(0.9, 0, 3), (0.8, 1, 2), (0.2, 3, 5)]
        for policy in ("allow", "nested", "flat", "longest"):
            kept = resolve_overlaps(
                spans, policy, score=lambda item: item[0], start=lambda item: item[1], end=lambda item: item[2]
            )
            self.assertTrue(kept)
            self.assertLessEqual(len(kept), len(spans))

    def test_joint_optimizers(self):
        from transformers.models.gliner2.decoding_gliner2 import decode_joint_sample
        from transformers.models.gliner2.processing_gliner2 import Field, FieldGroup

        schema = {
            "entities": {"person": {}},
            "relations": {"wrote": {"head": "person", "tail": "person"}},
        }
        row = {
            "groups": (FieldGroup(task="entities", name="entities", fields=(Field(name="person"),)),),
            "schema": schema,
            "text": "Ada wrote notes",
            "start": [0, 4, 10],
            "end": [3, 9, 15],
            "prefix_len": 0,
            "architecture": "span",
        }
        sample = {"span_logits": [torch.zeros(1, 1, 3, 2)], "counts": [1]}
        for optimizer in ("greedy", "beam"):
            decoded = decode_joint_sample(sample, row, optimizer=optimizer)
            self.assertIsInstance(decoded, dict)

    def test_long_text_merge(self):
        from transformers.models.gliner2.processing_gliner2 import merge_chunk_results

        text = "Ada wrote notes."
        merged = merge_chunk_results(
            text,
            [SimpleNamespace(start_char=0), SimpleNamespace(start_char=4)],
            [
                {"entities": {"person": [{"text": "Ada", "start": 0, "end": 3, "confidence": 0.9}]}},
                {"entities": {"person": [{"text": "wrote", "start": 0, "end": 5, "confidence": 0.4}]}},
            ],
            include_spans=True,
            include_confidence=True,
        )
        self.assertIn("entities", merged)


@require_torch
class ContractTypeTest(unittest.TestCase):
    def test_loss_registration(self):
        from transformers.loss.loss_gliner2 import ForSchemaExtractionLoss
        from transformers.loss.loss_utils import LOSS_MAPPING

        self.assertIs(LOSS_MAPPING["ForSchemaExtraction"], ForSchemaExtractionLoss)
        model = Gliner2ForSchemaExtraction(Gliner2Config(encoder_config=_encoder()))
        bound = model.loss_function
        if isinstance(bound, property):
            bound = bound.fget(model)
        self.assertIs(bound, ForSchemaExtractionLoss)
