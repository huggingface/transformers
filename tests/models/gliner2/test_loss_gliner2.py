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

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoConfig, Gliner2Config, Gliner2ForSchemaExtraction
from transformers.models.gliner2.decoding_gliner2 import linear_sum_assignment
from transformers.models.gliner2.loss_gliner2 import (
    TargetCapacityError,
    abstention_loss,
    asymmetric_focal_loss,
    balanced_multilabel_bce,
    clamp_gold_count,
    compute_record_group_loss,
    count_log_rate_loss,
    head_touch,
    select_hard_negative_candidates,
    span_count_loss,
    span_structure_loss,
    sparse_relation_loss,
    supervises_count,
)
from transformers.models.gliner2.modeling_gliner2 import count_conditioned_scores


def _encoder():
    encoder = AutoConfig.for_model(
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
    encoder._attn_implementation = "eager"
    return encoder


def _span_inputs(task_id, fields=1):
    batch, length, words = 1, 8, 3
    return {
        "input_ids": torch.randint(1, 50, (batch, length)),
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
        "text_word_indices": torch.tensor([[1, 2, 3]]),
        "text_word_mask": torch.ones(batch, words, dtype=torch.bool),
        "query_marker_indices": torch.tensor([[0] * fields]),
        "query_marker_mask": torch.ones(batch, fields, dtype=torch.bool),
        "query_group_index": torch.zeros(batch, fields, dtype=torch.long),
        "cls_marker_indices": torch.zeros(batch, 1, dtype=torch.long),
        "cls_marker_mask": torch.zeros(batch, 1, dtype=torch.bool),
        "cls_group_index": torch.zeros(batch, 1, dtype=torch.long),
        "prompt_marker_indices": torch.zeros(batch, 1, dtype=torch.long),
        "prompt_marker_mask": torch.ones(batch, 1, dtype=torch.bool),
        "prompt_group_index": torch.zeros(batch, 1, dtype=torch.long),
        "task_type_ids": torch.tensor([[task_id]]),
        "group_mask": torch.ones(batch, 1, dtype=torch.bool),
    }


class LossMathTest(unittest.TestCase):
    def test_gold_count_einsum_and_clamp(self):
        self.assertEqual(clamp_gold_count(25), 19)
        self.assertEqual(clamp_gold_count(0), 0)
        self.assertFalse(supervises_count(1))
        self.assertTrue(supervises_count(2))
        span = torch.randn(4, 3, 5)
        projected = torch.randn(2, 2, 5)
        scores = count_conditioned_scores(span, projected)
        self.assertEqual(tuple(scores.shape), (2, 2, 4, 3))
        self.assertTrue(torch.allclose(scores[1, 0, 2, 1], torch.dot(span[2, 1], projected[1, 0])))

    def test_span_bce_drops_negatives_randomly(self):
        scores = torch.tensor([[[[0.2, -0.4], [1.5, -1.0]]]], requires_grad=True)
        structure = [1, [[(0, 0)]]]
        mask = torch.tensor([False, False, False, True])
        held = span_structure_loss(scores, structure, mask, training=False)
        dropped = []
        for seed in range(12):
            torch.manual_seed(seed)
            dropped.append(float(span_structure_loss(scores, structure, mask, training=True)))
        self.assertGreater(len({round(value, 6) for value in dropped}), 1)
        labels = torch.zeros_like(scores)
        labels[0, 0, 0, 0] = 1
        element = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
        constant = element * torch.where(labels > 0, 1.0, 0.5)
        constant = (constant.reshape(1, 1, -1) * (~mask).float()).sum()
        self.assertTrue(any(abs(value - float(constant)) > 1e-5 for value in dropped))
        self.assertLessEqual(max(dropped), float(held) + 1e-5)
        positive = torch.ones_like(scores)
        self.assertTrue(
            torch.allclose(
                span_structure_loss(positive, positive, mask, training=True),
                span_structure_loss(positive, positive, mask, training=False),
            )
        )

    def test_span_label_errors(self):
        scores = torch.zeros(1, 1, 2, 2)
        with self.assertRaises(ValueError):
            span_structure_loss(scores, torch.zeros(1, 1, 2, 1), torch.zeros(4, dtype=torch.bool), training=False)
        with self.assertRaises(ValueError):
            clamp_gold_count(-1)
        logits = torch.randn(2, 20)
        loss = span_count_loss(logits, torch.tensor([3, 19]))
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss))

    def test_boundary_objectives_and_hungarian(self):
        logits = torch.tensor([[0.0, 2.0]])
        targets = torch.tensor([[1.0, 0.0]])
        keep = torch.ones(1, 2, dtype=torch.bool)
        bce = balanced_multilabel_bce(logits, targets, keep, negative_weight=0.5)
        focal = asymmetric_focal_loss(logits, targets, keep, negative_weight=0.5)
        self.assertTrue(torch.isfinite(bce) and torch.isfinite(focal))
        self.assertNotAlmostEqual(float(bce), float(focal), places=4)
        pair = torch.tensor([[[0.2, 3.0, -1.0]]])
        labels = torch.tensor([[[0.0, 1.0, 0.0]]])
        valid = torch.ones(1, 1, 3, dtype=torch.bool)
        chosen = select_hard_negative_candidates(pair, labels, valid, negatives_per_positive=1, minimum_negatives=1)
        self.assertTrue(bool(chosen[0, 0, 1]))
        query = torch.tensor([[0.0, 1.0]])
        mentions = torch.tensor([[[False], [True]]])
        self.assertGreater(float(abstention_loss(query, mentions, torch.ones(1, 2, dtype=torch.bool))), 0.0)
        rate = torch.zeros(1, 1)
        mask = torch.tensor([[[True, False]]])
        self.assertTrue(torch.isfinite(count_log_rate_loss(rate, mask, torch.ones(1, 1, dtype=torch.bool))))
        rows, cols = linear_sum_assignment(torch.tensor([[5.0, 0.1], [0.2, 4.0]]))
        self.assertEqual(rows.tolist(), [0, 1])
        self.assertEqual(cols.tolist(), [1, 0])
        with self.assertRaises(ValueError):
            linear_sum_assignment(torch.tensor([[float("nan"), 0.0]]))

    def test_record_assignment_and_head_touch(self):
        spans = [torch.tensor([[0, 2]])]
        spec = type("Spec", (), {"mode": "latent", "task_index": 0, "anchor_query_id": None})()
        field = type("Field", (), {"query_id": 0, "cardinality": type("Card", (), {"is_scalar": True})()})()
        record = type(
            "Record",
            (),
            {"field_for_query": lambda self, query_id: type("T", (), {"values": [[(0, 2)]]})()},
        )()
        group = type("Group", (), {})()
        group.object_logits = torch.tensor([1.5, -1.0], requires_grad=True)
        group.assign_logits = [torch.tensor([[0.0, 2.0], [0.0, -2.0]], requires_grad=True)]
        group.field_spans = spans
        group.field_query_ids = [0]
        group.field_specs = [field]
        group.spec = spec
        group.instance_seed = [(0, 0), (0, 0)]
        group.num_instances = 2
        losses = compute_record_group_loss(group, [record])
        total = losses["object_loss"] + losses["field_loss"]
        total.backward()
        self.assertIsNotNone(group.object_logits.grad)
        group.num_instances = 1
        group.object_logits = torch.zeros(1)
        with self.assertRaises(TargetCapacityError):
            compute_record_group_loss(group, [record, record])
        layer = nn.Linear(2, 2)
        touched = head_touch([layer])
        self.assertEqual(float(touched), 0.0)
        touched.backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertEqual(int(layer.weight.grad.count_nonzero()), 0)
        pairs = type("Pairs", (), {})()
        pairs.relation_index = torch.tensor([0, 0])
        pairs.batch_index = torch.tensor([0, 0])
        pairs.head_start = torch.tensor([0, 1])
        pairs.head_end = torch.tensor([2, 2])
        pairs.tail_start = torch.tensor([2, 3])
        pairs.tail_end = torch.tensor([4, 4])
        pairs.pair_mask = torch.tensor([True, True])
        rel = sparse_relation_loss(
            torch.zeros(2, requires_grad=True),
            pairs,
            torch.tensor([[[[0, 2, 2, 4]]]]),
            torch.tensor([[[True]]]),
            1.0,
        )
        self.assertTrue(torch.isfinite(rel))
        rel.backward()


class ForwardLossTest(unittest.TestCase):
    def _span_model(self):
        config = Gliner2Config(encoder_config=_encoder(), max_width=4, counting_layer="count_lstm")
        model = Gliner2ForSchemaExtraction(config)
        last = model.count_pred[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[0] = 5.0
        return model

    def test_inference_does_not_require_labels(self):
        model = self._span_model().eval()
        output = model(**_span_inputs(2))
        self.assertIsNone(output.loss)
        self.assertEqual(output.counts[0][0], 0)
        self.assertEqual(output.span_logits[0][0].shape[0], 0)

    def test_training_uses_gold_count_and_skips_entity_count_ce(self):
        model = self._span_model().train()
        structure = [3, [[(0, 0)], [(1, 1)], [(0, 1)]]]
        labels = {"span_structures": [[structure]]}
        torch.manual_seed(0)
        entity = model(**_span_inputs(1), labels=labels)
        self.assertEqual(entity.span_logits[0][0].shape[0], 3)
        self.assertEqual(float(entity.losses["count_loss"]), 0.0)
        self.assertTrue(torch.isfinite(entity.losses["structure_loss"]))
        torch.manual_seed(0)
        first = model(**_span_inputs(2), labels=labels)
        torch.manual_seed(1)
        second = model(**_span_inputs(2), labels=labels)
        self.assertEqual(first.span_logits[0][0].shape[0], 3)
        self.assertGreater(float(first.losses["count_loss"]), 0.0)
        self.assertNotAlmostEqual(
            float(first.losses["structure_loss"]), float(second.losses["structure_loss"]), places=5
        )
        first.loss.backward()
        self.assertIsNotNone(model.count_embed.pos_embedding.weight.grad)
        with self.assertRaises(ValueError):
            model(**_span_inputs(4), labels={"classification_targets": [[torch.zeros(3)]]})

    def test_boundary_loss_and_optional_head_touch(self):
        boundary = {
            "boundary_dim": 16,
            "pair_dim": 16,
            "boundary_attention_layers": 0,
            "candidate_attention_layers": 0,
            "query_attention_layers": 0,
            "candidate_pool": "per_query",
            "enable_records": True,
            "enable_relations": True,
            "record_dim": 16,
            "record_instance_queries": 2,
            "pool_size": 4,
            "candidate_budget": 4,
            "training_candidate_budget": 4,
            "start_top_k": 2,
            "end_top_k": 2,
            "ends_per_start": 2,
            "starts_per_end": 2,
        }
        config = Gliner2Config(encoder_config=_encoder(), architecture="boundary", boundary_config=boundary)
        model = Gliner2ForSchemaExtraction(config).train()
        batch = {
            "input_ids": torch.randint(1, 50, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "text_word_indices": torch.tensor([[1, 2, 3]]),
            "text_word_mask": torch.ones(1, 3, dtype=torch.bool),
            "query_marker_indices": torch.tensor([[0]]),
            "query_marker_mask": torch.ones(1, 1, dtype=torch.bool),
            "query_group_index": torch.zeros(1, 1, dtype=torch.long),
            "cls_marker_indices": torch.tensor([[4, 5]]),
            "cls_marker_mask": torch.ones(1, 2, dtype=torch.bool),
            "cls_group_index": torch.zeros(1, 2, dtype=torch.long),
            "task_type_ids": torch.tensor([[4]]),
            "group_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        bare = model(**batch)
        self.assertIsNone(bare.loss)
        labels = {
            "mention_pairs": torch.tensor([[[[0, 2]]]]),
            "mention_mask": torch.ones(1, 1, 1, dtype=torch.bool),
            "classification_targets": [[torch.tensor([1.0, 0.0])]],
            "record_groups": [
                [
                    {
                        "mode": "anchorless",
                        "field_query_ids": [0],
                        "field_scalar": [True],
                        "records": [{"fields": {0: [[(0, 1)]]}}],
                    }
                ]
            ],
        }
        output = model(**batch, labels=labels)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIn("start_loss", output.losses)
        self.assertIn("record_object_loss", output.losses)
        self.assertIn("classification_loss", output.losses)
        output.loss.backward()
        self.assertIsNotNone(model.record_decoder.object_head.weight.grad)
        self.assertIsNotNone(model.relation_scorer.mlp[0].weight.grad)
        self.assertEqual(int(model.relation_scorer.mlp[0].weight.grad.count_nonzero()), 0)
        with self.assertRaises(ValueError):
            model(**batch, labels={"mention_pairs": torch.zeros(1, 1, 1)})
