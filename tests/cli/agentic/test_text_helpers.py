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
"""
Pure-function tests for ``text._aggregate_entities``.

The aggregator merges B-/I- tagged sub-word predictions back into whole
entities. It runs on the output of ``AutoModelForTokenClassification`` and is
small enough to test exhaustively without any model.
"""

from __future__ import annotations

from transformers.cli.agentic.text import _aggregate_entities


def _ent(label, score, start, end):
    return {"entity": label, "score": score, "start": start, "end": end}


def test_aggregate_entities_empty():
    assert _aggregate_entities([], "anything") == []


def test_aggregate_entities_single_entity_gets_word_filled_in():
    text = "Tim Cook"
    out = _aggregate_entities([_ent("B-PER", 0.99, 0, 3)], text)
    assert out == [{"entity_group": "PER", "score": 0.99, "start": 0, "end": 3, "word": "Tim"}]


def test_aggregate_entities_merges_continuation_tokens():
    text = "Tim Cook"
    entities = [
        _ent("B-PER", 0.99, 0, 3),
        _ent("I-PER", 0.95, 4, 8),
    ]
    [merged] = _aggregate_entities(entities, text)
    assert merged["entity_group"] == "PER"
    assert (merged["start"], merged["end"]) == (0, 8)
    assert merged["word"] == "Tim Cook"
    # Score is the minimum across merged tokens (worst-confidence convention).
    assert merged["score"] == 0.95


def test_aggregate_entities_does_not_merge_across_groups():
    text = "Apple Tim"
    entities = [
        _ent("B-ORG", 0.9, 0, 5),
        _ent("B-PER", 0.8, 6, 9),
    ]
    out = _aggregate_entities(entities, text)
    assert [e["entity_group"] for e in out] == ["ORG", "PER"]
    assert [e["word"] for e in out] == ["Apple", "Tim"]


def test_aggregate_entities_continuation_without_matching_group_starts_new_entity():
    # I- tag whose group doesn't match the current entity should start a new one.
    text = "Apple Tim"
    entities = [
        _ent("B-ORG", 0.9, 0, 5),
        _ent("I-PER", 0.8, 6, 9),
    ]
    out = _aggregate_entities(entities, text)
    assert [e["entity_group"] for e in out] == ["ORG", "PER"]


def test_aggregate_entities_handles_label_without_dash():
    # Some models output bare labels (e.g., "PER" not "B-PER").
    out = _aggregate_entities([_ent("PER", 0.7, 0, 3)], "Tim")
    assert out[0]["entity_group"] == "PER"
    assert out[0]["word"] == "Tim"
