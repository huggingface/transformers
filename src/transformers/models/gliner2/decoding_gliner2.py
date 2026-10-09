# Copyright 2026 The HuggingFace Inc. team and the GLiNER2 authors. All rights reserved.
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

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from itertools import combinations, product
from types import SimpleNamespace
from typing import Any

import torch


logger = logging.getLogger(__name__)

_RESERVED = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "(",
    ")",
)
_ACTIVATIONS = ("auto", "sigmoid", "softmax")
_DECODERS = ("auto", "independent", "exact", "beam")
_ON_INFEASIBLE = ("relax", "min_violations", "raise")
_AGGREGATIONS = ("max", "mean", "first")
_MODEL_KEYS = (
    "json_structures",
    "classifications",
    "entities",
    "relations",
    "json_descriptions",
    "entity_descriptions",
)
_L = "[L]"
_OVERLAP_ALIASES = {
    "allow": "allow",
    "all": "allow",
    "none": "allow",
    "nested": "nested",
    "allow_nested": "nested",
    "flat": "disallow",
    "disallow": "disallow",
    "no_overlap": "disallow",
    "non_overlapping": "disallow",
    "longest": "longest",
    "keep_longest": "longest",
}
_EXPR_FIELDS = {
    "LabelRef": ("task", "label"),
    "AnySelected": ("task",),
    "AnyOtherSelected": ("task",),
    "IsDefault": ("task",),
    "Cardinality": ("task", "minimum", "maximum"),
    "MinLevel": ("task", "level"),
    "MaxLevel": ("task", "level"),
    "AtLevel": ("task", "level"),
    "Not": ("child",),
    "And": ("children",),
    "Or": ("children",),
    "ExactlyOneOf": ("children",),
    "Implies": ("cond", "then"),
    "Iff": ("left", "right"),
    "Excludes": ("left", "right"),
}
_JOINT_FIELDS = {
    "TypedEndpoints": ("relation", "head_types", "tail_types"),
    "NoSelfLoops": ("relation",),
    "UniqueRelationPair": ("relation", "directed"),
    "UniqueRelationSlot": ("relation", "slot"),
    "EntityOverlapPolicy": ("policy",),
    "MaxRelationsPerHead": ("limit", "relation"),
    "MaxRelationsPerTail": ("limit", "relation"),
    "SymmetricRelation": ("relation",),
    "InverseRelation": ("relation", "inverse"),
    "AcyclicRelation": ("relation",),
}


class SchemaError(ValueError):
    """Invalid schema, task, label, or constraint."""


class InfeasibleError(RuntimeError):
    """No assignment satisfies the constraints."""

    def __init__(self, message: str, violations=()):
        super().__init__(message)
        self.violations = tuple(violations)


class _BudgetExceeded(Exception):
    """Exact search expanded more nodes than the budget allows."""


def _number(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def probability_to_logit(probability: float) -> float:
    """Return log-odds, accepting exact zero and one."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf
    return math.log(probability) - math.log1p(-probability)


def center_logit(logit: float, threshold: float = 0.5) -> float:
    """Center a logit so positive utility means passing ``threshold``."""
    return float(logit) - probability_to_logit(threshold)


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid."""
    value = float(value)
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(values: Sequence[float]) -> list:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [item / total for item in exps]


def _geometric_mean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("confidence components must be probabilities in [0, 1]")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    result: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        result.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(result)


def _clean(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{what} must be a non-empty string")
    for token in _RESERVED:
        if token in value:
            raise SchemaError(
                f"{what} may not contain {token!r}: label and prompt strings are "
                f"injected into the model prompt verbatim, and this would corrupt "
                f"logit-to-label alignment"
            )
    return value


def _prob(value: float | None, field_name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise SchemaError(f"{field_name} must be in [0, 1]")


def normalize_overlap_policy(policy: str | None, *, default: str | None = None) -> str:
    """Return a canonical overlap policy. ``flat`` means disallow."""
    selected = default if policy is None else policy
    if selected is None:
        raise ValueError("overlap_policy=None requires an architecture default")
    if not isinstance(selected, str):
        raise TypeError("overlap_policy must be a string or None")
    key = selected.strip().lower().replace("-", "_")
    try:
        return _OVERLAP_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown overlap_policy {selected!r}; expected one of: allow, nested, flat/disallow, longest"
        ) from exc


def _item_score(item: Any) -> float:
    if isinstance(item, Mapping):
        for key in ("score", "confidence", "probability"):
            if key in item and item[key] is not None:
                return float(item[key])
        return 0.0
    for key in ("score", "confidence", "probability"):
        if hasattr(item, key) and getattr(item, key) is not None:
            return float(getattr(item, key))
    return 0.0


def _item_bound(item: Any, primary: str, aliases: Sequence[str]) -> int:
    keys = (primary, *aliases)
    if isinstance(item, Mapping):
        for key in keys:
            if key in item and item[key] is not None:
                return int(item[key])
    for key in keys:
        if hasattr(item, key) and getattr(item, key) is not None:
            return int(getattr(item, key))
    raise ValueError(f"span is missing a {primary} offset")


def resolve_overlaps(
    spans,
    policy,
    *,
    score: Callable[[Any], float] | None = None,
    start: Callable[[Any], int] | None = None,
    end: Callable[[Any], int] | None = None,
    default: str | None = None,
):
    """Resolve half-open spans for flat, nested, or longest overlap.

    Flat (and disallow) is the maximum-total-score non-overlapping set.
    Ties prefer more spans, then the rank key. Nested keeps containment and
    rejects crossings. Longest drops spans strictly inside another candidate.
    """
    score_fn = score or _item_score
    start_fn = start or (lambda item: _item_bound(item, "start", ("char_start", "token_start")))
    end_fn = end or (lambda item: _item_bound(item, "end", ("char_end", "token_end")))
    canonical = normalize_overlap_policy(policy, default=default)
    if not spans:
        return []
    indexed = list(enumerate(spans))

    def rank_key(row):
        index, item = row
        return (-float(score_fn(item)), int(start_fn(item)), int(end_fn(item)), index)

    ranked = sorted(indexed, key=rank_key)
    distinct = []
    seen_boundaries = set()
    for row in ranked:
        item = row[1]
        boundaries = (int(start_fn(item)), int(end_fn(item)))
        if boundaries in seen_boundaries:
            continue
        seen_boundaries.add(boundaries)
        distinct.append(row)
    if canonical == "allow":
        return [item for _, item in distinct]
    if canonical == "nested":
        kept = []
        for row in distinct:
            candidate = row[1]
            candidate_start = int(start_fn(candidate))
            candidate_end = int(end_fn(candidate))
            crossing = False
            for _, existing in kept:
                existing_start = int(start_fn(existing))
                existing_end = int(end_fn(existing))
                overlaps = candidate_start < existing_end and existing_start < candidate_end
                contains = (candidate_start <= existing_start and existing_end <= candidate_end) or (
                    existing_start <= candidate_start and candidate_end <= existing_end
                )
                if overlaps and not contains:
                    crossing = True
                    break
            if not crossing:
                kept.append(row)
        return [item for _, item in kept]
    if canonical == "longest":
        kept = []
        for row in distinct:
            candidate = row[1]
            candidate_start = int(start_fn(candidate))
            candidate_end = int(end_fn(candidate))
            strictly_contained = any(
                int(start_fn(other)) <= candidate_start
                and candidate_end <= int(end_fn(other))
                and (int(start_fn(other)) < candidate_start or candidate_end < int(end_fn(other)))
                for _, other in distinct
            )
            if not strictly_contained:
                kept.append(row)
        return [item for _, item in kept]
    by_end = sorted(
        distinct,
        key=lambda row: (
            int(end_fn(row[1])),
            int(start_fn(row[1])),
            -float(score_fn(row[1])),
            row[0],
        ),
    )
    ends = [int(end_fn(item)) for _, item in by_end]
    predecessors = [
        bisect.bisect_right(ends, int(start_fn(item)), 0, index) - 1 for index, (_, item) in enumerate(by_end)
    ]
    best: list[tuple] = [(0.0, ())]

    def selection_key(selection: tuple):
        rows = [by_end[index] for index in selection]
        return tuple(rank_key(row) for row in sorted(rows, key=rank_key))

    for index, (_, item) in enumerate(by_end):
        previous_score, previous_selection = best[predecessors[index] + 1]
        with_item = (previous_score + float(score_fn(item)), previous_selection + (index,))
        without_item = best[index]
        if with_item[0] > without_item[0]:
            best.append(with_item)
        elif with_item[0] < without_item[0]:
            best.append(without_item)
        elif len(with_item[1]) > len(without_item[1]):
            best.append(with_item)
        elif len(with_item[1]) < len(without_item[1]):
            best.append(without_item)
        else:
            best.append(with_item if selection_key(with_item[1]) < selection_key(without_item[1]) else without_item)
    selected = [by_end[index] for index in best[-1][1]]
    selected.sort(key=rank_key)
    return [item for _, item in selected]


def _char_span(start, end, offset, start_map, end_map, text):
    """Map a half-open word span onto the normalized text."""
    token_start, token_end = start - offset, end - offset
    if not (0 <= token_start < token_end <= len(end_map)):
        return None
    char_start = int(start_map[token_start])
    char_end = int(end_map[token_end - 1])
    surface = text[char_start:char_end].strip()
    if not surface:
        return None
    return surface, char_start, char_end


def _query_specs(meta):
    """Extractive fields in marker order, which is the candidate query axis."""
    specs = []
    for group in meta.get("groups") or []:
        if group["task_type"] == "classifications":
            continue
        for field_name in group["fields"]:
            specs.append(
                {
                    "task_type": group["task_type"],
                    "task_name": group["name"],
                    "field_name": field_name,
                }
            )
    return specs


def _group_candidates(candidates, threshold, temperature):
    """Threshold pair logits into per-query (score, start, end) lists."""
    probs = torch.sigmoid(candidates.pair_logits / temperature)
    eligible = candidates.valid_mask & candidates.query_mask.unsqueeze(-1)
    keep = eligible & (probs >= threshold)
    grouped = [[[] for _ in range(candidates.indices.shape[1])] for _ in range(candidates.indices.shape[0])]
    batch, query, candidate = keep.nonzero(as_tuple=True)
    for index in range(batch.shape[0]):
        b = int(batch[index])
        q = int(query[index])
        c = int(candidate[index])
        grouped[b][q].append(
            (
                float(probs[b, q, c]),
                int(candidates.indices[b, q, c, 0]),
                int(candidates.indices[b, q, c, 1]),
            )
        )
    for sample in grouped:
        for scored in sample:
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return grouped


def _kept_spans(scored, policy):
    return resolve_overlaps(
        scored,
        policy or "disallow",
        score=lambda item: item[0],
        start=lambda item: item[1],
        end=lambda item: item[2],
    )


def _format_span(surface, score, char_start, char_end, include_confidence, include_spans):
    if include_spans and include_confidence:
        return {"text": surface, "confidence": score, "start": char_start, "end": char_end}
    if include_spans:
        return {"text": surface, "start": char_start, "end": char_end}
    if include_confidence:
        return {"text": surface, "confidence": score}
    return surface


def _relation_specs(meta, query_states, directional):
    """One spec and state per relation group, using its head and tail markers."""
    states = query_states[0] if query_states.dim() == 3 else query_states
    specs = []
    packed = []
    query_id = 0
    limit = int(states.shape[0])
    for group in meta.get("groups") or []:
        if group["task_type"] == "classifications":
            continue
        fields = list(group["fields"])
        if group["task_type"] == "relations" and len(fields) >= 2 and query_id + 1 < limit:
            head_id, tail_id = query_id, query_id + 1
            role = states[[head_id, tail_id]]
            if directional:
                packed.append(torch.cat((role[0], role[1]), dim=-1))
            else:
                packed.append(role.mean(dim=0))
            specs.append(
                SimpleNamespace(
                    relation_type=group["name"],
                    head_query_ids=(head_id,),
                    tail_query_ids=(tail_id,),
                    allow_self=False,
                )
            )
        query_id += len(fields)
    return specs, packed


def _deduplicate_relation_edges(edges):
    """Collapse contained mentions and repeated head/tail text."""
    if len(edges) < 2:
        return edges

    def canonical_mentions(side):
        mentions = {(edge[side][1], edge[side][2]): edge[side] for edge in edges}
        canonical = {}
        for coordinates in mentions:
            start, end = coordinates
            containing = [
                candidate for candidate in mentions.values() if candidate[1] <= start and candidate[2] >= end
            ]
            canonical[coordinates] = max(containing, key=lambda candidate: (candidate[2] - candidate[1], -candidate[1]))
        return canonical

    head_canonical = canonical_mentions("head")
    tail_canonical = canonical_mentions("tail")
    exact = {}
    for edge in edges:
        head = head_canonical[(edge["head"][1], edge["head"][2])]
        tail = tail_canonical[(edge["tail"][1], edge["tail"][2])]
        normalized = {**edge, "head": head, "tail": tail}
        key = (head[1], head[2], tail[1], tail[2])
        previous = exact.get(key)
        if previous is None or edge["score"] > previous["score"]:
            exact[key] = normalized

    def semantic_text(value):
        return " ".join(value.casefold().split())

    def rank(candidate):
        _, head_start, head_end = candidate["head"]
        _, tail_start, tail_end = candidate["tail"]
        distance = max(head_start - tail_end, tail_start - head_end, 0)
        return (distance, -candidate["score"], head_start, tail_start)

    semantic = {}
    for edge in exact.values():
        key = (semantic_text(edge["head"][0]), semantic_text(edge["tail"][0]))
        previous = semantic.get(key)
        if previous is None or rank(edge) < rank(previous):
            semantic[key] = edge
    values = list(semantic.values())
    kept = []
    for edge in values:
        head_tokens = set(semantic_text(edge["head"][0]).split())
        tail_tokens = set(semantic_text(edge["tail"][0]).split())
        dominated = False
        for other in values:
            if other is edge:
                continue
            other_head = set(semantic_text(other["head"][0]).split())
            other_tail = set(semantic_text(other["tail"][0]).split())
            if (head_tokens < other_head and tail_tokens == other_tail) or (
                tail_tokens < other_tail and head_tokens == other_head
            ):
                dominated = True
                break
        if not dominated:
            kept.append(edge)
    return sorted(kept, key=lambda edge: (edge["head"][1], edge["tail"][1], -edge["score"]))


def _format_relation(edge, include_confidence, include_spans):
    """Public head/tail payload for one kept relation edge."""
    score = edge["score"]
    head, head_start, head_end = edge["head"]
    tail, tail_start, tail_end = edge["tail"]
    if include_spans:
        value = {
            "head": {"text": head, "start": head_start, "end": head_end},
            "tail": {"text": tail, "start": tail_start, "end": tail_end},
        }
        if include_confidence:
            value["head"]["confidence"] = score
            value["tail"]["confidence"] = score
        return value
    if include_confidence:
        return {
            "head": {"text": head, "confidence": score},
            "tail": {"text": tail, "confidence": score},
        }
    return (head, tail)


def _decode_relations(sample_out, meta, threshold, include_confidence, include_spans):
    """Score typed relation pairs and map them onto character offsets."""
    scorer = sample_out.get("relation_scorer")
    generator = sample_out.get("relation_pair_generator")
    candidates = sample_out.get("candidates")
    text_states = sample_out.get("text_states")
    query_states = sample_out.get("query_states")
    if scorer is None or generator is None or candidates is None or text_states is None or query_states is None:
        return {}
    directional = sample_out.get("directional_relation_states")
    if directional is None:
        directional = int(getattr(scorer, "relation_query_dim", 0)) == int(text_states.shape[-1]) * 2
    specs, packed = _relation_specs(meta, query_states, bool(directional))
    if not specs:
        return {}
    relation_states = torch.stack(packed).unsqueeze(0)
    with torch.inference_mode():
        pairs = generator.generate(candidates, [None], specs)
        if len(pairs) == 0:
            return {}
        temperature = float(sample_out.get("relation_temperature") or 1.0)
        logits = scorer(text_states[:1], relation_states, candidates, pairs)
        probabilities = torch.sigmoid(logits / temperature)
    aliases = {
        f"{name}: {description}": name
        for name, description in (meta.get("relation_descriptions") or {}).items()
    }
    relation_metadata = meta.get("relation_metadata") or {}
    offset = int(meta.get("prefix_len") or 0)
    start_map = list(meta.get("start") or [])
    end_map = list(meta.get("end") or [])
    text = meta.get("text") or ""
    edges = {}
    for pair_index, probability in enumerate(probabilities):
        relation_type = aliases.get(pairs.relation_types[pair_index], pairs.relation_types[pair_index])
        relation_threshold = relation_metadata.get(relation_type, {}).get("threshold", threshold)
        if relation_threshold is None:
            relation_threshold = threshold
        score = float(probability.detach())
        if score < relation_threshold:
            continue
        head = _char_span(
            int(pairs.head_start[pair_index]),
            int(pairs.head_end[pair_index]),
            offset,
            start_map,
            end_map,
            text,
        )
        tail = _char_span(
            int(pairs.tail_start[pair_index]),
            int(pairs.tail_end[pair_index]),
            offset,
            start_map,
            end_map,
            text,
        )
        if head is None or tail is None:
            continue
        edges.setdefault(relation_type, []).append({"score": score, "head": head, "tail": tail})
    formatted = {}
    for relation_type, relation_edges in edges.items():
        formatted[relation_type] = [
            _format_relation(edge, include_confidence, include_spans)
            for edge in _deduplicate_relation_edges(relation_edges)
        ]
    return formatted


def decode_boundary(
    sample_out,
    meta,
    *,
    threshold=0.5,
    include_confidence=False,
    include_spans=False,
    overlap_policy="disallow",
    temperature=1.0,
):
    """Decode one boundary sample into the pre-format result dict.

    Args:
        sample_out: Candidate batch row, or a mapping with `grouped_candidates`.
        meta: Processor metadata for one example.
        threshold: Pair-probability cutoff.
        include_confidence: Keep scores on formatted spans.
        include_spans: Keep character offsets.
        overlap_policy: One of the shared overlap policies. `flat` means disallow.
        temperature: Positive divisor applied to pair logits before the sigmoid.

    Returns:
        A dict whose `entities` value is a one-item list, matching the
        formatter used by the public extraction API. Relation groups are
        filled from `relation_scorer` when the sample carries query states.
    """
    specs = _query_specs(meta)
    if sample_out.get("grouped_candidates") is not None:
        grouped = sample_out["grouped_candidates"]
    else:
        candidates = sample_out["candidates"]
        grouped = _group_candidates(candidates, threshold, temperature)[0]
    offset = int(meta.get("prefix_len") or 0)
    start_map = list(meta.get("start") or [])
    end_map = list(meta.get("end") or [])
    text = meta.get("text") or ""
    null_logits = sample_out.get("null_logits")
    abstain = float(meta.get("abstention_threshold") or 0.5)
    entities = OrderedDict()
    structures = OrderedDict()
    for query_id, spec in enumerate(specs):
        if query_id >= len(grouped):
            break
        if null_logits is not None and float(torch.sigmoid(null_logits[query_id])) > abstain:
            scored = []
        else:
            scored = _kept_spans(grouped[query_id], overlap_policy)
        spans = []
        for score, start, end in scored:
            mapped = _char_span(start, end, offset, start_map, end_map, text)
            if mapped is None:
                continue
            surface, char_start, char_end = mapped
            spans.append((surface, score, char_start, char_end))
        if spec["task_type"] == "entities":
            entities[spec["field_name"]] = [
                _format_span(surface, score, char_start, char_end, include_confidence, include_spans)
                for surface, score, char_start, char_end in spans
            ]
        elif spec["task_type"] == "json_structures":
            instance = structures.setdefault(spec["task_name"], OrderedDict())
            formatted = [
                _format_span(surface, score, char_start, char_end, include_confidence, include_spans)
                for surface, score, char_start, char_end in spans
            ]
            instance[spec["field_name"]] = formatted[0] if formatted else None
    result = {}
    if entities:
        result["entities"] = [entities]
    for name, instance in structures.items():
        if any(value is not None and value != [] for value in instance.values()):
            result[name] = [instance]
    result.update(
        _decode_relations(sample_out, meta, threshold, include_confidence, include_spans)
    )
    for group in meta.get("groups") or []:
        if group["task_type"] == "relations" and group["name"] not in result:
            result[group["name"]] = []
    return result


def decode_records(
    sample_out,
    meta,
    *,
    threshold=0.5,
    include_confidence=False,
    include_spans=False,
):
    """Decode record-head outputs when a sample carries `records`.

    Args:
        sample_out: Mapping with decoded `records`.
        meta: Processor metadata. Unused by the already-decoded path.
        threshold: Unused when records are pre-decoded.
        include_confidence: Unused when records are pre-decoded.
        include_spans: Unused when records are pre-decoded.

    Returns:
        The record mapping stored on the sample.
    """
    del meta, threshold, include_confidence, include_spans
    return dict(sample_out.get("records") or {})


class Expr:
    """One node of the classification constraint AST."""

    def __init__(self, kind: str, **fields):
        self.type = kind
        self.fields = fields

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Expr) and self.type == other.type and self.fields == other.fields

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={self.fields[name]!r}" for name in _EXPR_FIELDS[self.type])
        return f"{self.type}({parts})"

    def __getattr__(self, name: str):
        try:
            return self.fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def references(self):
        return _walk(self)[0]

    def to_dict(self) -> dict:
        payload = {"type": self.type}
        for name in _EXPR_FIELDS[self.type]:
            value = self.fields[name]
            if isinstance(value, Expr):
                value = value.to_dict()
            elif isinstance(value, tuple) and value and isinstance(value[0], Expr):
                value = [item.to_dict() for item in value]
            payload[name] = value
        return payload


def _walk(expr: Expr, schema=None):
    """Collect task, label, set, and count references, optionally checking them."""
    kind = expr.type
    fields = expr.fields
    tasks, labels, sets, counts = set(), set(), set(), set()

    def absorb(child):
        child_tasks, child_labels, child_sets, child_counts = _walk(child, schema)
        tasks.update(child_tasks)
        labels.update(child_labels)
        sets.update(child_sets)
        counts.update(child_counts)

    def spec_of(task):
        getter = getattr(schema, "task_spec", None)
        if getter is not None:
            return getter(task)
        return schema.task(task)

    if kind == "LabelRef":
        tasks.add(fields["task"])
        labels.add((fields["task"], fields["label"]))
        if schema is not None:
            spec = spec_of(fields["task"])
            if fields["label"] not in spec.label_names:
                raise SchemaError(f"{fields['label']!r} is not a label of {fields['task']!r}")
    elif kind in {"AnySelected", "AnyOtherSelected", "IsDefault"}:
        tasks.add(fields["task"])
        sets.add(fields["task"])
        if schema is not None:
            spec = spec_of(fields["task"])
            if kind != "AnySelected" and spec.default is None:
                if kind == "AnyOtherSelected":
                    raise SchemaError(f"any_other_selected requires task {fields['task']!r} to declare a default")
                raise SchemaError(f"is_default requires task {fields['task']!r} to declare a default")
    elif kind == "Cardinality":
        tasks.add(fields["task"])
        counts.add(fields["task"])
        if schema is not None:
            spec = spec_of(fields["task"])
            count = len(spec.label_names)
            if not 0 <= fields["minimum"] <= count:
                raise SchemaError(
                    f"cardinality minimum {fields['minimum']} outside [0, {count}] for {fields['task']!r}"
                )
            if fields["maximum"] is not None:
                if not 0 <= fields["maximum"] <= count:
                    raise SchemaError(
                        f"cardinality maximum {fields['maximum']} outside [0, {count}] for {fields['task']!r}"
                    )
                if fields["minimum"] > fields["maximum"]:
                    raise SchemaError(
                        f"cardinality minimum {fields['minimum']} exceeds maximum "
                        f"{fields['maximum']} for {fields['task']!r}"
                    )
    elif kind in {"MinLevel", "MaxLevel", "AtLevel"}:
        tasks.add(fields["task"])
        if schema is not None:
            spec = spec_of(fields["task"])
            if not spec.ordered:
                raise SchemaError(f"ordinal op requires an ordered task; {fields['task']!r} is unordered")
            if fields["level"] not in spec.label_names:
                raise SchemaError(f"{fields['level']!r} is not a label of {fields['task']!r}")
    elif kind == "Not":
        absorb(fields["child"])
    elif kind in {"And", "Or", "ExactlyOneOf"}:
        for child in fields["children"]:
            absorb(child)
    else:
        left = fields.get("cond", fields.get("left"))
        right = fields.get("then", fields.get("right"))
        absorb(left)
        absorb(right)
    return frozenset(tasks), frozenset(labels), frozenset(sets), frozenset(counts)


def _k_not(value):
    return None if value is None else (not value)


def _k_and(values):
    seen_none = False
    for value in values:
        if value is False:
            return False
        if value is None:
            seen_none = True
    return None if seen_none else True


def _k_or(values):
    seen_none = False
    for value in values:
        if value is True:
            return True
        if value is None:
            seen_none = True
    return None if seen_none else False


def _evaluate(expr: Expr, assignment) -> bool | None:
    """Return true, false, or unknown for one constraint node."""
    kind = expr.type
    fields = expr.fields
    if kind == "LabelRef":
        return assignment.holds(fields["task"], fields["label"])
    if kind == "AnySelected":
        task = fields["task"]
        labels = assignment.selected(task) | assignment.domain(task)
        return _k_or([assignment.holds(task, label) for label in labels])
    if kind == "AnyOtherSelected":
        task = fields["task"]
        default = assignment.default(task)
        labels = assignment.selected(task) | assignment.domain(task)
        return _k_or([assignment.holds(task, label) for label in labels if label != default])
    if kind == "IsDefault":
        task = fields["task"]
        default = assignment.default(task)
        if default is None:
            return False
        return assignment.holds(task, default)
    if kind == "Cardinality":
        selected = assignment.selected(fields["task"])
        domain = assignment.domain(fields["task"])
        lo = len(selected)
        hi = len(selected | domain)
        maximum = hi if fields["maximum"] is None else fields["maximum"]
        if lo > maximum:
            return False
        if hi < fields["minimum"]:
            return False
        if lo >= fields["minimum"] and hi <= maximum:
            return True
        return None
    if kind in {"MinLevel", "MaxLevel", "AtLevel"}:
        levels = assignment.levels(fields["task"])
        if not levels:
            return False
        target = assignment.index(fields["task"], fields["level"])
        if kind == "MinLevel":
            if min(levels) >= target:
                return True
            if max(levels) < target:
                return False
            return None
        if kind == "MaxLevel":
            if max(levels) <= target:
                return True
            if min(levels) > target:
                return False
            return None
        if levels == {target}:
            return True
        if target not in levels:
            return False
        return None
    if kind == "Not":
        return _k_not(_evaluate(fields["child"], assignment))
    if kind == "And":
        return _k_and([_evaluate(child, assignment) for child in fields["children"]])
    if kind == "Or":
        return _k_or([_evaluate(child, assignment) for child in fields["children"]])
    if kind == "ExactlyOneOf":
        vals = [_evaluate(child, assignment) for child in fields["children"]]
        trues = sum(1 for value in vals if value is True)
        nones = sum(1 for value in vals if value is None)
        if trues >= 2:
            return False
        if trues == 1:
            return True if nones == 0 else None
        return False if nones == 0 else None
    if kind == "Implies":
        return _k_or([_k_not(_evaluate(fields["cond"], assignment)), _evaluate(fields["then"], assignment)])
    if kind == "Iff":
        left = _evaluate(fields["left"], assignment)
        right = _evaluate(fields["right"], assignment)
        if left is None or right is None:
            return None
        return left == right
    left = _evaluate(fields["left"], assignment)
    right = _evaluate(fields["right"], assignment)
    return _k_not(_k_and([left, right]))


def _expr_from_dict(data: Mapping) -> Expr:
    values = dict(data)
    kind = values.pop("type", None)
    if kind not in _EXPR_FIELDS:
        raise SchemaError(f"unknown constraint type {kind!r}")
    fields = {}
    for key, value in values.items():
        if isinstance(value, Mapping) and "type" in value:
            fields[key] = _expr_from_dict(value)
        elif isinstance(value, (list, tuple)):
            fields[key] = tuple(
                _expr_from_dict(item) if isinstance(item, Mapping) and "type" in item else item for item in value
            )
        else:
            fields[key] = value
    return Expr(kind, **fields)


def _as_expr(value) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, str) for item in value):
        return Expr("LabelRef", task=value[0], label=value[1])
    raise SchemaError(
        f"cannot interpret {value!r} as a constraint expression; use a ('task', 'label') tuple or a DSL constructor"
    )


def _int(value, what) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"{what} must be a non-negative int")
    return value


def _task_name(value, what) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{what} must be a task name string")
    return value


def all_of(*exprs) -> Expr:
    """Conjunction of constraint expressions."""
    return Expr("And", children=tuple(_as_expr(item) for item in exprs))


def any_of(*exprs) -> Expr:
    """Disjunction of constraint expressions."""
    return Expr("Or", children=tuple(_as_expr(item) for item in exprs))


def not_(expr) -> Expr:
    """Negation of one constraint expression."""
    return Expr("Not", child=_as_expr(expr))


def implies(cond, then) -> Expr:
    """``cond`` implies ``then``."""
    return Expr("Implies", cond=_as_expr(cond), then=_as_expr(then))


def iff(left, right) -> Expr:
    """Both sides are equivalent."""
    return Expr("Iff", left=_as_expr(left), right=_as_expr(right))


def excludes(left, right) -> Expr:
    """The two sides cannot both hold."""
    return Expr("Excludes", left=_as_expr(left), right=_as_expr(right))


def exactly_one_of(*exprs) -> Expr:
    """Exactly one child holds."""
    return Expr("ExactlyOneOf", children=tuple(_as_expr(item) for item in exprs))


def at_least(task, count) -> Expr:
    """The task selects at least ``count`` labels."""
    return Expr(
        "Cardinality", task=_task_name(task, "at_least task"), minimum=_int(count, "at_least count"), maximum=None
    )


def at_most(task, count) -> Expr:
    """The task selects at most ``count`` labels."""
    return Expr("Cardinality", task=_task_name(task, "at_most task"), minimum=0, maximum=_int(count, "at_most count"))


def exactly(task, count) -> Expr:
    """The task selects exactly ``count`` labels."""
    number = _int(count, "exactly count")
    return Expr("Cardinality", task=_task_name(task, "exactly task"), minimum=number, maximum=number)


def at_level(task, level) -> Expr:
    """An ordered task is exactly ``level``."""
    return Expr("AtLevel", task=_task_name(task, "at_level task"), level=level)


def min_level(task, level) -> Expr:
    """An ordered task is at least ``level``."""
    return Expr("MinLevel", task=_task_name(task, "min_level task"), level=level)


def max_level(task, level) -> Expr:
    """An ordered task is at most ``level``."""
    return Expr("MaxLevel", task=_task_name(task, "max_level task"), level=level)


def between_level(task, lo, hi) -> Expr:
    """An ordered task lies between ``lo`` and ``hi``."""
    name = _task_name(task, "between_level task")
    return Expr("And", children=(Expr("MinLevel", task=name, level=lo), Expr("MaxLevel", task=name, level=hi)))


def any_selected(task) -> Expr:
    """At least one label of ``task`` is selected."""
    return Expr("AnySelected", task=_task_name(task, "any_selected task"))


def any_other_selected(task) -> Expr:
    """At least one non-default label of ``task`` is selected."""
    return Expr("AnyOtherSelected", task=_task_name(task, "any_other_selected task"))


def is_default(task) -> Expr:
    """The declared default label is selected."""
    return Expr("IsDefault", task=_task_name(task, "is_default task"))


def label(task, name) -> Expr:
    """A single (task, label) literal."""
    return Expr("LabelRef", task=_task_name(task, "label task"), label=name)


@dataclass(frozen=True)
class LabelSpec:
    """One classification label."""

    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        _clean(self.name, "label name")
        if self.description is not None:
            _clean(self.description, f"description of label {self.name!r}")


@dataclass(frozen=True)
class TaskSpec:
    """Cardinality, order, and calibration for one classification task."""

    name: str
    labels: tuple
    min_labels: int = 0
    max_labels: int | None = None
    ordered: bool = False
    threshold: float = 0.5
    candidate_threshold: float | None = None
    activation: str = "auto"
    temperature: float = 1.0
    default: str | None = None
    instruction: str | None = None
    examples: tuple = ()

    def __post_init__(self) -> None:
        _clean(self.name, "task name")
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "examples", tuple(tuple(pair) for pair in self.examples))
        if not self.labels:
            raise SchemaError(f"task {self.name!r} must declare at least one label")
        names = [item.name for item in self.labels]
        if len(set(names)) != len(names):
            raise SchemaError(f"task {self.name!r} has duplicate label names")
        if not isinstance(self.min_labels, int) or self.min_labels < 0:
            raise SchemaError(f"task {self.name!r}: min_labels must be a non-negative int")
        if self.max_labels is not None:
            if not isinstance(self.max_labels, int) or self.max_labels < 0:
                raise SchemaError(f"task {self.name!r}: max_labels must be a non-negative int")
            if self.max_labels > len(self.labels):
                raise SchemaError(
                    f"task {self.name!r}: max_labels ({self.max_labels}) exceeds the "
                    f"number of labels ({len(self.labels)})"
                )
        if self.default is not None:
            if self.default not in names:
                raise SchemaError(f"task {self.name!r}: default {self.default!r} is not one of its labels")
            if self.min_labels < 1:
                object.__setattr__(self, "min_labels", 1)
        if self.max_labels is not None and self.min_labels > self.max_labels:
            raise SchemaError(
                f"task {self.name!r}: min_labels ({self.min_labels}) exceeds max_labels ({self.max_labels})"
            )
        if self.ordered and len(self.labels) < 2:
            raise SchemaError(f"ordered task {self.name!r} requires at least two labels")
        if not isinstance(self.threshold, (int, float)) or not 0 < self.threshold < 1:
            raise SchemaError(f"task {self.name!r}: threshold must be in (0, 1)")
        _prob(self.candidate_threshold, f"task {self.name!r}: candidate_threshold")
        if self.activation not in _ACTIVATIONS:
            raise SchemaError(f"task {self.name!r}: activation must be one of {_ACTIVATIONS}")
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise SchemaError(f"task {self.name!r}: temperature must be positive")
        if self.instruction is not None:
            _clean(self.instruction, f"instruction of task {self.name!r}")
        for pair in self.examples:
            if len(pair) != 2:
                raise SchemaError(f"task {self.name!r}: each example must be a (text, label) pair")
            _, label_name = pair
            if label_name not in names:
                raise SchemaError(f"task {self.name!r}: example label {label_name!r} is not one of its labels")

    @property
    def label_names(self) -> tuple:
        return tuple(item.name for item in self.labels)

    @property
    def is_exclusive(self) -> bool:
        return self.min_labels == 1 and self.max_labels == 1

    def effective_max_labels(self) -> int:
        return len(self.labels) if self.max_labels is None else self.max_labels


def _coerce_labels(labels: Any) -> tuple:
    if isinstance(labels, Mapping):
        return tuple(LabelSpec(name, desc) for name, desc in labels.items())
    if isinstance(labels, str):
        raise SchemaError("labels must be a list or a {label: description} mapping, not a str")
    coerced = []
    for item in labels:
        if isinstance(item, LabelSpec):
            coerced.append(item)
        elif isinstance(item, str):
            coerced.append(LabelSpec(item))
        else:
            raise SchemaError(f"invalid label entry {item!r}")
    return tuple(coerced)


def _partition(constraints, active):
    keep, pure_drop, mixed = [], [], []
    for constraint in constraints:
        refs = frozenset(constraint.references())
        if refs <= active:
            keep.append(constraint)
        elif refs & active:
            mixed.append(constraint)
        else:
            pure_drop.append(constraint)
    return keep, pure_drop, mixed


class ClassificationSchema:
    """Mutable builder for tasks and hard constraints."""

    def __init__(self) -> None:
        self._tasks: dict = {}
        self._constraints: list = []

    @property
    def task_specs(self) -> tuple:
        return tuple(self._tasks.values())

    @property
    def task_order(self) -> tuple:
        return tuple(self._tasks.keys())

    @property
    def constraints(self) -> tuple:
        return tuple(self._constraints)

    def task_spec(self, name: str) -> TaskSpec:
        try:
            return self._tasks[name]
        except KeyError:
            raise SchemaError(f"unknown task {name!r}") from None

    def task(
        self,
        name,
        labels,
        *,
        min_labels=0,
        max_labels=None,
        ordered=False,
        threshold=0.5,
        candidate_threshold=None,
        activation="auto",
        temperature=1.0,
        default=None,
        instruction=None,
        examples=(),
    ):
        if name in self._tasks:
            raise SchemaError(f"task {name!r} is already defined")
        self._tasks[name] = TaskSpec(
            name=name,
            labels=_coerce_labels(labels),
            min_labels=min_labels,
            max_labels=max_labels,
            ordered=ordered,
            threshold=threshold,
            candidate_threshold=candidate_threshold,
            activation=activation,
            temperature=temperature,
            default=default,
            instruction=instruction,
            examples=tuple(examples),
        )
        return self

    def single(self, name, labels, **kwargs):
        kwargs.setdefault("min_labels", 1)
        kwargs.setdefault("max_labels", 1)
        return self.task(name, labels, **kwargs)

    def multi(self, name, labels, **kwargs):
        return self.task(name, labels, **kwargs)

    def ordinal(self, name, labels, **kwargs):
        kwargs.setdefault("min_labels", 1)
        kwargs.setdefault("max_labels", 1)
        kwargs["ordered"] = True
        return self.task(name, labels, **kwargs)

    def constrain(self, *expressions):
        """Append constraints validated against tasks declared so far."""
        declared = set(self._tasks)
        for expr in expressions:
            if not hasattr(expr, "references"):
                raise SchemaError(
                    f"constraint {expr!r} is not a constraint expression; build one with the constraints DSL"
                )
            missing = frozenset(expr.references()) - declared
            if missing:
                name = sorted(missing)[0]
                raise SchemaError(
                    f"constraint references undeclared task {name!r}; declare task {name!r} before constraining it"
                )
            self._constraints.append(expr)
        return self

    def subset(self, *names, keep_constraints="strict"):
        """Return a new schema containing only ``names``."""
        return self._narrow(self._resolve_names(names), keep_constraints)

    def drop(self, *names, keep_constraints="strict"):
        """Return a new schema with ``names`` removed."""
        removed = self._resolve_names(names)
        return self._narrow(frozenset(self._tasks) - removed, keep_constraints)

    def _resolve_names(self, names) -> frozenset:
        requested = frozenset(names)
        unknown = requested - set(self._tasks)
        if unknown:
            raise SchemaError(f"unknown task(s): {sorted(unknown)}")
        return requested

    def _narrow(self, active: frozenset, keep_constraints: str):
        if keep_constraints not in ("strict", "prune", "error_on_mixed"):
            raise SchemaError("keep_constraints must be 'strict', 'prune' or 'error_on_mixed'")
        keep, pure_drop, mixed = _partition(self._constraints, active)
        if mixed:
            raise SchemaError(
                "cannot narrow: constraint(s) reference both kept and removed tasks; "
                "a half-applied invariant is never safe"
            )
        if keep_constraints == "strict" and pure_drop:
            raise SchemaError(
                "cannot narrow under keep_constraints='strict': would drop "
                f"{len(pure_drop)} constraint(s); use 'prune' or 'error_on_mixed'"
            )
        if keep_constraints == "prune" and pure_drop:
            warnings.warn(
                f"subset/drop dropped {len(pure_drop)} constraint(s) referencing only "
                f"removed tasks; the prompt changes, so remaining scores are a new "
                f"measurement",
                stacklevel=3,
            )
        out = ClassificationSchema()
        for name, spec in self._tasks.items():
            if name in active:
                out._tasks[name] = spec
        out._constraints = list(keep)
        return out

    def to_dict(self) -> dict:
        return {
            "version": 3,
            "tasks": {name: _task_to_dict(spec) for name, spec in self._tasks.items()},
            "constraints": [constraint.to_dict() for constraint in self._constraints],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping) -> ClassificationSchema:
        schema = cls()
        for name, spec in data.get("tasks", {}).items():
            schema.task(name, **_task_kwargs_from_dict(spec))
        for raw in data.get("constraints", ()):
            schema.constrain(_expr_from_dict(raw))
        return schema

    @classmethod
    def from_json(cls, value: str) -> ClassificationSchema:
        return cls.from_dict(json.loads(value))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ClassificationSchema):
            return NotImplemented
        return self._tasks == other._tasks and list(self._constraints) == list(other._constraints)

    def __repr__(self) -> str:
        return f"ClassificationSchema(tasks={list(self._tasks)}, constraints={len(self._constraints)})"


def _task_to_dict(spec: TaskSpec) -> dict:
    descs = {item.name: item.description for item in spec.labels if item.description is not None}
    labels: Any = {item.name: item.description for item in spec.labels} if descs else list(spec.label_names)
    out: dict = {"labels": labels}
    if spec.min_labels:
        out["min_labels"] = spec.min_labels
    if spec.max_labels is not None:
        out["max_labels"] = spec.max_labels
    if spec.ordered:
        out["ordered"] = spec.ordered
    if spec.threshold != 0.5:
        out["threshold"] = spec.threshold
    if spec.candidate_threshold is not None:
        out["candidate_threshold"] = spec.candidate_threshold
    if spec.activation != "auto":
        out["activation"] = spec.activation
    if spec.temperature != 1.0:
        out["temperature"] = spec.temperature
    if spec.default is not None:
        out["default"] = spec.default
    if spec.instruction is not None:
        out["instruction"] = spec.instruction
    if spec.examples:
        out["examples"] = [list(pair) for pair in spec.examples]
    return out


def _task_kwargs_from_dict(spec: Mapping) -> dict:
    kwargs = dict(spec)
    kwargs["labels"] = kwargs.pop("labels")
    examples = kwargs.pop("examples", None)
    if examples is not None:
        kwargs["examples"] = tuple(tuple(pair) for pair in examples)
    return kwargs


class Assignment:
    """Kleene view of decided labels and still-open domains."""

    def __init__(self, specs, chosen, decided, *, possible=None, always=None):
        self.specs = specs
        self.chosen = chosen
        self.decided = frozenset(decided)
        self.possible = possible
        self.always = always
        self.search = possible is not None

    def is_decided(self, task):
        return task in self.decided

    def selected(self, task):
        if self.search:
            if task in self.decided:
                return self.chosen[task].labels
            return self.always[task]
        return self.chosen.get(task, frozenset())

    def domain(self, task):
        if self.search:
            if task in self.decided:
                return self.chosen[task].labels
            return self.possible[task]
        if task in self.decided:
            return self.chosen.get(task, frozenset())
        return frozenset(self.specs[task].label_names)

    def holds(self, task, label_name):
        if label_name in self.selected(task):
            return True
        if label_name in self.domain(task):
            return False if self.is_decided(task) else None
        return False

    def levels(self, task):
        names = self.specs[task].label_names
        index = {name: i for i, name in enumerate(names)}
        return frozenset(index[label_name] for label_name in self.domain(task) if label_name in index)

    def index(self, task, label_name):
        return self.specs[task].label_names.index(label_name)

    def default(self, task):
        return self.specs[task].default


def _spec_map(task_specs) -> dict:
    return {spec.name: spec for spec in task_specs}


@dataclass(frozen=True)
class CompiledClassificationSchema:
    """Model schema dict plus the constraint set the decoder searches."""

    model_schema: dict
    task_specs: tuple
    constraints: tuple
    task_order: tuple
    fingerprint: str

    def task(self, name) -> TaskSpec:
        for spec in self.task_specs:
            if spec.name == name:
                return spec
        raise SchemaError(f"unknown task {name!r}")

    def task_spec(self, name) -> TaskSpec:
        return self.task(name)

    def build(self) -> dict:
        return self.model_schema


def _classification_entry(spec: TaskSpec) -> dict:
    entry = {
        "task": spec.name,
        "labels": list(spec.label_names),
        "multi_label": not spec.is_exclusive,
        "cls_threshold": spec.threshold,
        "class_act": spec.activation,
    }
    if spec.instruction:
        entry["prompt"] = spec.instruction
    descs = {item.name: item.description for item in spec.labels if item.description}
    if descs:
        entry["label_descriptions"] = descs
    if spec.examples:
        entry["examples"] = [list(pair) for pair in spec.examples]
    return entry


def _has_reserved(value: str) -> bool:
    return any(token in value for token in _RESERVED)


def _assert_model_schema(model: dict) -> None:
    for key in _MODEL_KEYS:
        if key not in model:
            raise SchemaError(f"compiled model schema is missing key {key!r}")
    if not isinstance(model["classifications"], list):
        raise SchemaError("compiled 'classifications' must be a list")
    for entry in model["classifications"]:
        task = entry.get("task")
        if not isinstance(task, str) or not task:
            raise SchemaError("classification entry has an invalid 'task'")
        for required in ("labels", "multi_label", "cls_threshold", "class_act"):
            if required not in entry:
                raise SchemaError(f"classification task {task!r} is missing {required!r}")
        if not isinstance(entry["labels"], list) or not entry["labels"]:
            raise SchemaError(f"classification task {task!r} has invalid 'labels'")
        if not isinstance(entry["multi_label"], bool):
            raise SchemaError(f"classification task {task!r} has non-bool 'multi_label'")
        strings = [task] + list(entry["labels"])
        if "prompt" in entry:
            strings.append(entry["prompt"])
        strings += list(entry.get("label_descriptions", {}).keys())
        strings += list(entry.get("label_descriptions", {}).values())
        for pair in entry.get("examples", []):
            strings += [str(item) for item in pair]
        for value in strings:
            if isinstance(value, str) and _has_reserved(value):
                raise SchemaError(
                    f"classification task {task!r} emitted a reserved marker token in "
                    f"{value!r}; this would corrupt logit-to-label alignment"
                )


def _check_prefix_collisions(task_order) -> None:
    for left in task_order:
        for right in task_order:
            if left == right:
                continue
            if right.startswith(left) and len(right) > len(left) and right[len(left)] in (":", " "):
                raise SchemaError(
                    f"task name {left!r} is a boundary-prefix of {right!r}; the prompt "
                    f"resolver could confuse them. Rename one."
                )


def _static_feasibility(schema, task_specs) -> None:
    constraints = schema.constraints
    if not constraints:
        return
    specs = _spec_map(task_specs)
    undetermined = Assignment(specs, {}, ())
    for constraint in constraints:
        if _evaluate(constraint, undetermined) is False:
            raise SchemaError("constraint set is unsatisfiable on the declared label sets")
    for spec in task_specs:
        if not spec.is_exclusive:
            continue
        touching = [constraint for constraint in constraints if spec.name in constraint.references()]
        if not touching:
            continue
        reachable = False
        for label_name in spec.label_names:
            assignment = Assignment(specs, {spec.name: frozenset({label_name})}, [spec.name])
            if all(_evaluate(constraint, assignment) is not False for constraint in touching):
                reachable = True
                break
        if not reachable:
            raise SchemaError(
                f"no label of exclusive task {spec.name!r} satisfies its constraints; "
                f"the task is pinned to the empty set"
            )


def _lower_defaults(schema, task_specs) -> tuple:
    constraints = list(schema.constraints)
    for spec in task_specs:
        if spec.default is not None:
            constraints.append(
                Expr(
                    "Iff",
                    left=Expr("IsDefault", task=spec.name),
                    right=Expr("Not", child=Expr("AnyOtherSelected", task=spec.name)),
                )
            )
    return tuple(constraints)


def _fingerprint(schema: ClassificationSchema) -> str:
    payload = json.dumps(schema.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compile_classification(schema) -> CompiledClassificationSchema:
    if isinstance(schema, CompiledClassificationSchema):
        return schema
    if not isinstance(schema, ClassificationSchema):
        raise SchemaError(f"compile_schema expects a ClassificationSchema, got {type(schema).__name__}")
    task_specs = schema.task_specs
    if not task_specs:
        raise SchemaError("cannot compile a schema with no tasks")
    for constraint in schema.constraints:
        _walk(constraint, schema)
    _check_prefix_collisions(schema.task_order)
    _static_feasibility(schema, task_specs)
    model = {
        "json_structures": [],
        "classifications": [_classification_entry(spec) for spec in task_specs],
        "entities": {},
        "relations": [],
        "json_descriptions": {},
        "entity_descriptions": {},
    }
    _assert_model_schema(model)
    return CompiledClassificationSchema(
        model_schema=model,
        task_specs=tuple(task_specs),
        constraints=_lower_defaults(schema, task_specs),
        task_order=tuple(schema.task_order),
        fingerprint=_fingerprint(schema),
    )


def task_utilities(spec, logits) -> dict:
    """Centered log-odds. Positive means the label clears the threshold."""
    return {label_name: center_logit(value / spec.temperature, spec.threshold) for label_name, value in logits.items()}


def task_probabilities(spec, logits) -> dict:
    """Sigmoid retention probabilities. Presentation may still be softmax."""
    return {label_name: sigmoid(value / spec.temperature) for label_name, value in logits.items()}


def retain(spec, logits, *, candidate_threshold, cap, rescued) -> frozenset:
    """Keep labels above the retention floor, plus every rescued label."""
    rescued = frozenset(rescued)
    floor = spec.candidate_threshold if spec.candidate_threshold is not None else candidate_threshold
    finite = {label_name: value for label_name, value in logits.items() if math.isfinite(value)}
    probs = task_probabilities(spec, finite)
    utils = task_utilities(spec, finite)
    keep = {label_name for label_name, prob in probs.items() if prob >= floor} | (rescued & set(finite))
    if len(keep) > cap:
        ranked = sorted(keep, key=lambda label_name: (label_name not in rescued, -utils[label_name], label_name))
        keep = set(ranked[: max(cap, len(rescued & set(finite)))])
    return frozenset(keep)


@dataclass(frozen=True)
class LocalAssignment:
    """One cardinality-feasible label set for a single task."""

    task: str
    labels: frozenset
    utility: float

    def __post_init__(self):
        object.__setattr__(self, "labels", frozenset(self.labels))


def _utility_of(labels, utilities) -> float:
    return float(sum(utilities[label_name] for label_name in labels))


def _all_subsets(items):
    items = sorted(items)
    for size in range(len(items) + 1):
        for combo in combinations(items, size):
            yield frozenset(combo)


def _sorted_locals(locals_):
    return sorted(locals_, key=lambda local: (-local.utility, tuple(sorted(local.labels))))


def enumerate_locals(spec, retained, utilities, *, bound_labels, set_coupled=False, count_coupled=False) -> list:
    """Enumerate cardinality-feasible locals, highest utility first."""
    retained = frozenset(retained)
    min_labels = spec.min_labels
    max_labels = min(spec.effective_max_labels(), len(retained))
    if spec.is_exclusive:
        locals_ = [
            LocalAssignment(spec.name, frozenset({label_name}), utilities[label_name]) for label_name in retained
        ]
        return _sorted_locals(locals_)
    if set_coupled:
        locals_ = []
        for base in _all_subsets(retained):
            if min_labels <= len(base) <= max_labels:
                locals_.append(LocalAssignment(spec.name, base, _utility_of(base, utilities)))
        return _sorted_locals(locals_)
    bound = retained & frozenset(bound_labels)
    free = retained - bound
    free_ranked = sorted(free, key=lambda label_name: (-utilities[label_name], label_name))
    locals_ = []
    for base in _all_subsets(bound):
        if len(base) > max_labels:
            continue
        if count_coupled:
            lo = max(min_labels, len(base))
            hi = min(max_labels, len(base) + len(free))
            for count in range(lo, hi + 1):
                need = count - len(base)
                selected = frozenset(base) | frozenset(free_ranked[:need])
                if len(selected) == count:
                    locals_.append(LocalAssignment(spec.name, selected, _utility_of(selected, utilities)))
            continue
        selected = set(base)
        for free_label in free_ranked:
            if len(selected) >= max_labels:
                break
            if utilities[free_label] > 0:
                selected.add(free_label)
        if len(selected) < min_labels:
            for free_label in free_ranked:
                if len(selected) >= min_labels:
                    break
                if free_label not in selected:
                    selected.add(free_label)
        if not (min_labels <= len(selected) <= max_labels):
            continue
        locals_.append(LocalAssignment(spec.name, frozenset(selected), _utility_of(selected, utilities)))
    return _sorted_locals(locals_)


class ClassificationScores:
    """Raw per-label logits plus the schema fingerprint."""

    def __init__(self, text, tasks, fingerprint, specs):
        self.text = text
        self.tasks = {task: dict(values) for task, values in tasks.items()}
        self.fingerprint = fingerprint
        self.specs = specs

    def _spec(self, task):
        try:
            return self.specs[task]
        except KeyError:
            raise SchemaError(f"unknown task {task!r}") from None

    def logit(self, task: str, label_name: str) -> float:
        return self.tasks[task][label_name]

    def probability(self, task: str, label_name: str) -> float:
        """Temperature then sigmoid, or max-subtracted softmax when exclusive."""
        spec = self._spec(task)
        temp = spec.temperature
        activation = spec.activation
        if activation == "auto":
            activation = "sigmoid" if not spec.is_exclusive else "softmax"
        if activation == "softmax":
            names = list(self.tasks[task])
            scaled = [self.tasks[task][name] / temp for name in names]
            probs = _softmax(scaled)
            return probs[names.index(label_name)]
        return sigmoid(self.tasks[task][label_name] / temp)

    def utility(self, task: str, label_name: str) -> float:
        """Temperature then threshold centering. This is the search objective."""
        spec = self._spec(task)
        return center_logit(self.tasks[task][label_name] / spec.temperature, spec.threshold)


@dataclass
class Solution:
    """Chosen locals, objective, and any violated constraints."""

    assignments: dict
    score: float
    violations: tuple = ()
    exact: bool = True
    decoder: str = "independent"

    @property
    def feasible(self) -> bool:
        return not self.violations


class DecodeProblem:
    """Per-task locals and the constraints that couple them."""

    def __init__(self, schema, task_order, locals_map, constraints):
        self.schema = schema
        self.task_order = tuple(task_order)
        self.locals = locals_map
        self.constraints = tuple(constraints)
        self.specs = _spec_map(schema.task_specs)
        self.possible = {}
        self.always = {}
        self.touching = {}
        walked = [_walk(constraint) for constraint in self.constraints]
        for task in self.task_order:
            local_list = self.locals[task]
            union = frozenset().union(*[local.labels for local in local_list]) if local_list else frozenset()
            inter = frozenset.intersection(*[local.labels for local in local_list]) if local_list else frozenset()
            self.possible[task] = union
            self.always[task] = inter
            self.touching[task] = tuple(
                constraint for constraint, (tasks, _, _, _) in zip(self.constraints, walked) if task in tasks
            )

    def assignment(self, chosen, decided):
        return Assignment(
            self.specs,
            chosen,
            decided,
            possible=self.possible,
            always=self.always,
        )

    def violations_of(self, chosen):
        assignment = self.assignment(chosen, self.task_order)
        return tuple(constraint for constraint in self.constraints if _evaluate(constraint, assignment) is False)


def _fallback_locals(spec, retained, utils):
    ranked = sorted(retained, key=lambda label_name: (-utils[label_name], label_name))
    max_labels = min(spec.effective_max_labels(), len(ranked))
    count = max(spec.min_labels, 1)
    count = min(count, max_labels) if max_labels else 0
    chosen = frozenset(ranked[:count])
    return [LocalAssignment(spec.name, chosen, float(sum(utils[label_name] for label_name in chosen)))]


def build_problem(compiled, scores, config, *, active=None, full_retention_tasks=()):
    """Retain candidates, enumerate locals, and drop constraints outside ``active``."""
    order = compiled.task_order
    if active is not None:
        active_set = set(active)
        order = tuple(task for task in order if task in active_set)
    active_set = set(order)
    full_retention = set(full_retention_tasks)
    constraints = []
    label_refs: dict = {}
    set_tasks: set = set()
    count_tasks: set = set()
    for constraint in compiled.constraints:
        tasks, labels, sets, counts = _walk(constraint)
        if not tasks <= active_set:
            continue
        constraints.append(constraint)
        for task, label_name in labels:
            label_refs.setdefault(task, set()).add(label_name)
        set_tasks |= set(sets)
        count_tasks |= set(counts)
    locals_map: dict = {}
    for task in order:
        spec = compiled.task(task)
        logits = scores.tasks[task]
        rescued = label_refs.get(task, set())
        if task in full_retention:
            retained = frozenset(spec.label_names)
        else:
            retained = retain(
                spec,
                logits,
                candidate_threshold=config.candidate_threshold,
                cap=config.max_candidates_per_task,
                rescued=rescued,
            )
        if len(retained) < spec.min_labels:
            retained = frozenset(spec.label_names)
        utils = task_utilities(spec, {label_name: logits[label_name] for label_name in retained})
        set_coupled = task in set_tasks
        count_coupled = (task in count_tasks) and not set_coupled
        locals_ = enumerate_locals(
            spec,
            retained,
            utils,
            bound_labels=label_refs.get(task, set()),
            set_coupled=set_coupled,
            count_coupled=count_coupled,
        )
        if not locals_:
            locals_ = _fallback_locals(spec, retained, utils)
        locals_map[task] = locals_
    return DecodeProblem(compiled, order, locals_map, constraints)


def _search_order(problem):
    return sorted(
        problem.task_order,
        key=lambda task: (-len(problem.touching[task]), len(problem.locals[task]), task),
    )


def _accepts(problem, order, index, chosen, local) -> bool:
    task = order[index]
    chosen[task] = local
    assignment = problem.assignment(chosen, order[: index + 1])
    ok = all(_evaluate(constraint, assignment) is not False for constraint in problem.touching[task])
    del chosen[task]
    return ok


def _suffix_max(order, problem):
    suffix = [0.0] * (len(order) + 1)
    for index in range(len(order) - 1, -1, -1):
        locals_ = problem.locals[order[index]]
        best = max((local.utility for local in locals_), default=0.0)
        suffix[index] = best + suffix[index + 1]
    return suffix


def _recursive_search(problem, *, mode: str, budget: int) -> Solution | None:
    """Depth-first search shared by exact and min-violations.

    Exact prunes with the suffix bound and keeps a partial only when it is not
    yet false. Equal totals keep the incumbent. Min-violations walks the same
    locals and ranks complete assignments by violation count, then utility.
    """
    order = _search_order(problem)
    suffix = _suffix_max(order, problem) if mode == "exact" else None
    best = {"assign": None, "score": -math.inf, "weight": math.inf}
    nodes = {"n": 0}

    def dfs(index, chosen, score):
        nodes["n"] += 1
        if nodes["n"] > budget:
            raise _BudgetExceeded
        if mode == "exact" and score + suffix[index] <= best["score"]:
            return
        if index == len(order):
            if mode == "exact":
                best["assign"] = dict(chosen)
                best["score"] = score
                return
            violated = problem.violations_of(chosen)
            weight = float(len(violated))
            key = (weight, -score)
            current = (best["weight"], -best["score"])
            if key < current:
                best["assign"] = dict(chosen)
                best["weight"] = weight
                best["score"] = score
            return
        task = order[index]
        for local in problem.locals[task]:
            if mode == "exact" and score + local.utility + suffix[index + 1] <= best["score"]:
                break
            if mode == "exact" and not _accepts(problem, order, index, chosen, local):
                continue
            chosen[task] = local
            dfs(index + 1, chosen, score + local.utility)
            del chosen[task]

    if mode == "min_violations":
        exhausted = False
        try:
            dfs(0, {}, 0.0)
        except _BudgetExceeded:
            exhausted = True
        assign = best["assign"] or {task: problem.locals[task][0] for task in problem.task_order}
        return Solution(
            assignments=dict(assign),
            score=sum(local.utility for local in assign.values()),
            violations=problem.violations_of(assign),
            exact=not exhausted,
            decoder="min_violations",
        )
    try:
        dfs(0, {}, 0.0)
    except _BudgetExceeded:
        raise
    if best["assign"] is None:
        return None
    return Solution(
        assignments=best["assign"],
        score=best["score"],
        violations=(),
        exact=True,
        decoder="exact",
    )


def _signature(chosen, order):
    return tuple((task, tuple(sorted(chosen[task].labels))) for task in order if task in chosen)


def _beam_search(problem, beam_size: int) -> Solution | None:
    """Same local expansion as exact search, without the suffix bound."""
    order = _search_order(problem)
    beams = [(0.0, {})]
    for index, task in enumerate(order):
        expanded = []
        for score, chosen in beams:
            for local in problem.locals[task]:
                if _accepts(problem, order, index, chosen, local):
                    nxt = dict(chosen)
                    nxt[task] = local
                    expanded.append((score + local.utility, nxt))
        expanded.sort(key=lambda item: (-item[0], _signature(item[1], order)))
        beams = []
        seen = set()
        for score, chosen in expanded:
            sig = _signature(chosen, order)
            if sig in seen:
                continue
            seen.add(sig)
            beams.append((score, chosen))
            if len(beams) >= beam_size:
                break
        if not beams:
            return None
    score, chosen = beams[0]
    return Solution(
        assignments=dict(chosen),
        score=score,
        violations=problem.violations_of(chosen),
        exact=False,
        decoder="beam",
    )


def _independent(problem) -> Solution:
    """Pick each task's best local. Only that task is treated as decided."""
    chosen: dict = {}
    for task in problem.task_order:
        picked = None
        for local in problem.locals[task]:
            if _accepts(problem, [task], 0, {}, local):
                picked = local
                break
        if picked is None:
            picked = problem.locals[task][0]
        chosen[task] = picked
    return Solution(
        assignments=dict(chosen),
        score=sum(local.utility for local in chosen.values()),
        violations=problem.violations_of(chosen),
        exact=True,
        decoder="independent",
    )


def _select_decoder(problem, requested: str) -> str:
    """Use exact search when any constraint crosses tasks."""
    if requested != "auto":
        return requested
    cross_task = any(len(_walk(constraint)[0]) > 1 for constraint in problem.constraints)
    return "exact" if cross_task else "independent"


def _primary(problem, config):
    decoder = _select_decoder(problem, config.decoder)
    if decoder == "independent":
        solution = _independent(problem)
        return solution if solution.feasible else None
    if decoder == "beam":
        solution = _beam_search(problem, config.beam_size)
        return solution if (solution is not None and solution.feasible) else None
    try:
        solution = _recursive_search(problem, mode="exact", budget=config.exact_node_budget)
    except _BudgetExceeded:
        solution = _beam_search(problem, config.beam_size)
    return solution if (solution is not None and solution.feasible) else None


def _decode_problem(problem, config, *, widen=None) -> Solution:
    """Run the decoder, then relax, minimize violations, or raise."""
    solution = _primary(problem, config)
    if solution is not None:
        return solution
    mode = config.on_infeasible
    working = problem
    if mode == "relax" and widen is not None:
        relaxed = widen()
        recovered = _primary(relaxed, config)
        if recovered is not None:
            return recovered
        working = relaxed
    if mode in ("relax", "min_violations"):
        return _recursive_search(working, mode="min_violations", budget=config.exact_node_budget)
    diagnosis = _recursive_search(working, mode="min_violations", budget=config.exact_node_budget)
    raise InfeasibleError(
        "no assignment satisfies the classification constraints",
        violations=diagnosis.violations,
    )


@dataclass(frozen=True)
class Violation:
    """One constraint the returned assignment breaks."""

    constraint: object
    tasks: tuple
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))

    def __str__(self) -> str:
        tasks = ", ".join(self.tasks)
        return f"violated {self.constraint} (tasks: {tasks}; weight {self.weight})"


def _confidence(spec, labels, probs) -> float | None:
    if not labels:
        return 1.0 if spec.default is not None else None
    if spec.is_exclusive:
        return probs[labels[0]]
    selected = set(labels)
    components = [
        probs[label_name] if label_name in selected else 1.0 - probs[label_name] for label_name in spec.label_names
    ]
    return _geometric_mean(components)


def _result_dict(compiled, scores, solution, *, active_order=None, include_confidence=True) -> dict:
    order = active_order or [task for task in compiled.task_order if task in solution.assignments]
    out: dict = {}
    feasible = not solution.violations
    for task in order:
        spec = compiled.task(task)
        selected = solution.assignments[task].labels
        labels = tuple(label_name for label_name in spec.label_names if label_name in selected)
        probs = {label_name: scores.probability(task, label_name) for label_name in spec.label_names}
        value = labels[0] if spec.is_exclusive else list(labels)
        if spec.is_exclusive and not labels:
            value = None
        confidence = _confidence(spec, labels, probs) if include_confidence else None
        if include_confidence:
            out[task] = {"value": value, "confidence": confidence, "probabilities": probs}
        else:
            out[task] = value
    violations = tuple(
        Violation(constraint, tuple(sorted(_walk(constraint)[0]))) for constraint in solution.violations
    )
    if include_confidence or not feasible:
        out["_meta"] = {
            "feasible": feasible,
            "decoder": solution.decoder,
            "exact": solution.exact,
            "objective": float(solution.score),
            "violations": [str(item) for item in violations],
        }
    return out


@dataclass(frozen=True)
class ClassificationConfig:
    """Decoder and retention controls for one call."""

    decoder: str = "auto"
    exact_node_budget: int = 200_000
    beam_size: int = 16
    candidate_threshold: float = 0.5
    max_candidates_per_task: int = 64
    include_confidence: bool = True
    on_infeasible: str = "relax"

    def __post_init__(self):
        if self.decoder not in _DECODERS:
            raise ValueError(f"decoder must be one of {_DECODERS}")
        if self.on_infeasible not in _ON_INFEASIBLE:
            raise ValueError(f"on_infeasible must be one of {_ON_INFEASIBLE}")
        if self.exact_node_budget <= 0:
            raise ValueError("exact_node_budget must be positive")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if not 0 <= self.candidate_threshold <= 1:
            raise ValueError("candidate_threshold must be in [0, 1]")
        if self.max_candidates_per_task <= 0:
            raise ValueError("max_candidates_per_task must be positive")


def _aggregate(values, mode):
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    return values[0]


def _schema_from_model_dict(data: Mapping) -> ClassificationSchema:
    schema = ClassificationSchema()
    for entry in data.get("classifications", ()):
        labels = entry["labels"]
        descriptions = entry.get("label_descriptions") or {}
        if descriptions:
            labels = {name: descriptions.get(name) for name in labels}
        multi = bool(entry.get("multi_label", False))
        kwargs = {
            "threshold": entry.get("cls_threshold", 0.5),
            "activation": entry.get("class_act", "auto"),
            "instruction": entry.get("prompt"),
            "examples": tuple(tuple(pair) for pair in entry.get("examples", ())),
        }
        if multi:
            schema.multi(entry["task"], labels, **kwargs)
        else:
            schema.single(entry["task"], labels, **kwargs)
    return schema


def _coerce_schema(schema):
    if isinstance(schema, CompiledClassificationSchema):
        return schema
    if isinstance(schema, ClassificationSchema):
        return schema
    if isinstance(schema, Mapping):
        if "tasks" in schema:
            return ClassificationSchema.from_dict(schema)
        if "classifications" in schema:
            return _schema_from_model_dict(schema)
    if hasattr(schema, "to_dict") and hasattr(schema, "task_order"):
        payload = schema.to_dict()
        if isinstance(payload, Mapping) and "tasks" in payload:
            return ClassificationSchema.from_dict(payload)
    raise SchemaError(f"expected a classification schema, got {type(schema).__name__}")


def _effective_temperature(spec: TaskSpec, temperature) -> float:
    if isinstance(temperature, Mapping):
        value = float(temperature.get(spec.name, spec.temperature))
    else:
        value = float(spec.temperature) * float(temperature)
    if not value > 0:
        raise ValueError("temperature must be positive")
    return value


def _vector(values) -> list:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "tolist") and not isinstance(values, (str, bytes, Mapping)):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise SchemaError("label logits must be a mapping or a sequence")
    return list(values)


def _label_logit_map(values, names) -> dict:
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        if missing:
            raise SchemaError(f"logits missing labels {missing}")
        unknown = [name for name in values if name not in names]
        if unknown:
            raise SchemaError(f"logits have unknown labels {unknown}")
        return {name: _number(values[name]) for name in names}
    row = _vector(values)
    finite_tail = row
    if len(row) > len(names) and all(not math.isfinite(_number(item)) for item in row[len(names) :]):
        finite_tail = row[: len(names)]
    if len(finite_tail) != len(names):
        raise SchemaError(f"expected {len(names)} label logits, got {len(row)}")
    return {name: _number(value) for name, value in zip(names, finite_tail)}


def _rows(logits, count):
    shape = getattr(logits, "shape", None)
    if shape is not None and len(tuple(shape)) == 2:
        if int(shape[0]) != count:
            raise SchemaError(f"expected {count} task rows, got {int(shape[0])}")
        return [logits[index] for index in range(count)]
    if isinstance(logits, (list, tuple)):
        if len(logits) != count:
            raise SchemaError(f"expected {count} task logit rows, got {len(logits)}")
        return list(logits)
    raise SchemaError("logits must be a task mapping or one row per task")


def label_names_from_tokens(schema_tokens: Sequence[str]) -> tuple:
    """Recover label order from the ``[L]`` tokens that were encoded."""
    return tuple(schema_tokens[i + 1] for i in range(len(schema_tokens) - 1) if schema_tokens[i] == _L)


def recover_task_name(schema_tokens: Sequence[str], known: Sequence[str]) -> str:
    """Resolve the owning task by boundary-aware longest match."""
    prompt_str = schema_tokens[2] if len(schema_tokens) > 2 else ""
    best = None
    for name in known:
        if prompt_str.startswith(name):
            rest = prompt_str[len(name) :]
            if rest == "" or rest[0] in (":", " "):
                if best is None or len(name) > len(best):
                    best = name
    if best is not None:
        return best
    return prompt_str.split(" [DESCRIPTION] ", 1)[0].split(":", 1)[0]


def _align_encoded(payload, compiled) -> dict:
    tokens = payload.get("schema_tokens_list", payload.get("schema_tokens"))
    raw = payload["logits"]
    if not tokens:
        raise SchemaError("schema_tokens is empty")
    if isinstance(tokens[0], str):
        groups = [tokens]
        rows = [raw]
    else:
        groups = list(tokens)
        rows = _rows(raw, len(groups))
    known = compiled.task_order
    found = {task: {} for task in known}
    for group, row in zip(groups, rows):
        names = label_names_from_tokens(group)
        task = recover_task_name(group, known)
        if task not in known:
            raise SchemaError(f"encoded tokens name unknown task {task!r}")
        spec = compiled.task(task)
        mapped = _label_logit_map(row, names)
        unknown = [name for name in mapped if name not in spec.label_names]
        if unknown:
            raise SchemaError(f"task {task!r} logits recovered unknown labels {unknown}")
        found[task].update(mapped)
    out = {}
    for spec in compiled.task_specs:
        if not found[spec.name]:
            raise SchemaError(f"logits missing task {spec.name!r}")
        out[spec.name] = {
            label_name: found[spec.name].get(label_name, float("-inf")) for label_name in spec.label_names
        }
    return out


def _align_logits(logits, compiled) -> dict:
    if isinstance(logits, Mapping) and ("schema_tokens" in logits or "schema_tokens_list" in logits):
        return _align_encoded(logits, compiled)
    payload = logits
    if (
        isinstance(payload, Mapping)
        and "tasks" in payload
        and not any(task in payload for task in compiled.task_order)
    ):
        payload = payload["tasks"]
    if isinstance(payload, Mapping):
        missing = [task for task in compiled.task_order if task not in payload]
        if missing:
            raise SchemaError(f"logits missing tasks {missing}")
        return {task: _label_logit_map(payload[task], compiled.task(task).label_names) for task in compiled.task_order}
    if len(compiled.task_order) == 1:
        shape = getattr(payload, "shape", None)
        if shape is None or len(tuple(shape)) == 1:
            names = compiled.task(compiled.task_order[0]).label_names
            return {compiled.task_order[0]: _label_logit_map(payload, names)}
    rows = _rows(payload, len(compiled.task_order))
    return {
        task: _label_logit_map(row, compiled.task(task).label_names) for task, row in zip(compiled.task_order, rows)
    }


def _with_temperature(compiled, temperature):
    """Scale every task temperature by the call temperature before search."""
    specs = []
    changed = False
    for spec in compiled.task_specs:
        temp = _effective_temperature(spec, temperature)
        if temp == spec.temperature:
            specs.append(spec)
        else:
            specs.append(replace(spec, temperature=temp))
            changed = True
    if not changed:
        return compiled
    return replace(compiled, task_specs=tuple(specs))


def _scores_from_logits(logits, compiled, text: str) -> ClassificationScores:
    if hasattr(logits, "probability") and hasattr(logits, "utility") and hasattr(logits, "tasks"):
        if getattr(logits, "fingerprint", compiled.fingerprint) != compiled.fingerprint:
            raise SchemaError(
                "scores were produced for a different schema (fingerprint mismatch); re-score before decoding"
            )
        return logits
    return ClassificationScores(
        text=text,
        tasks=_align_logits(logits, compiled),
        fingerprint=compiled.fingerprint,
        specs={spec.name: spec for spec in compiled.task_specs},
    )


def aggregate_classification_logits(chunk_logits, schema, mode: str = "max") -> dict:
    """Aggregate per-chunk label logits so decoding runs once."""
    if mode not in _AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {_AGGREGATIONS}")
    if not chunk_logits:
        raise ValueError("cannot aggregate an empty list of chunk scores")
    compiled = _compile_classification(_coerce_schema(schema))
    aligned = [_align_logits(item, compiled) for item in chunk_logits]
    tasks = {}
    for spec in compiled.task_specs:
        tasks[spec.name] = {
            label_name: _aggregate([row[spec.name][label_name] for row in aligned], mode)
            for label_name in spec.label_names
        }
    return tasks


def decode_constrained_classification(
    logits,
    schema,
    temperature,
    *,
    text: str = "",
    active: Sequence[str] | None = None,
    decoder: str = "auto",
    exact_node_budget: int = 200_000,
    beam_size: int = 16,
    candidate_threshold: float = 0.5,
    max_candidates_per_task: int = 64,
    include_confidence: bool = True,
    on_infeasible: str = "relax",
) -> dict:
    """Decode label logits under hard cross-task constraints.

    ``temperature`` scales logits together with each task's schema temperature.
    Pass ``1.0`` when ``logits`` are already temperature-scaled. Returns the
    classification result mapping, including ``_meta`` when confidence or an
    infeasible assignment is requested.
    """
    config = ClassificationConfig(
        decoder=decoder,
        exact_node_budget=exact_node_budget,
        beam_size=beam_size,
        candidate_threshold=candidate_threshold,
        max_candidates_per_task=max_candidates_per_task,
        include_confidence=include_confidence,
        on_infeasible=on_infeasible,
    )
    compiled = _compile_classification(_coerce_schema(schema))
    prebuilt = hasattr(logits, "probability") and hasattr(logits, "utility") and hasattr(logits, "tasks")
    if not prebuilt:
        compiled = _with_temperature(compiled, temperature)
    scores = _scores_from_logits(logits, compiled, text)
    problem = build_problem(compiled, scores, config, active=active)

    def widen():
        return build_problem(
            compiled,
            scores,
            config,
            active=active,
            full_retention_tasks=problem.task_order,
        )

    solution = _decode_problem(problem, config, widen=widen)
    return _result_dict(
        compiled,
        scores,
        solution,
        active_order=problem.task_order,
        include_confidence=config.include_confidence,
    )


def _batch_rows(metadata):
    """Normalize processor metadata into one dict per example."""
    if isinstance(metadata, Mapping):
        if "metadata" in metadata and "schema_meta" not in metadata and "groups" not in metadata:
            metadata = metadata["metadata"]
        else:
            metadata = [metadata]
    rows = []
    for item in metadata:
        if isinstance(item, Mapping) and "schema_meta" in item:
            meta = dict(item["schema_meta"])
            meta["words"] = list(item.get("words", meta.get("words", [])))
        else:
            meta = dict(item)
        rows.append(meta)
    return rows


def _sample_outputs(outputs, batch_size):
    """Split a batched model output into one mapping per row."""
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != batch_size:
            raise ValueError(f"outputs length ({len(outputs)}) != metadata length ({batch_size})")
        return list(outputs)
    boundary = getattr(outputs, "boundary", None)
    if not isinstance(outputs, Mapping):
        outputs = dict(outputs.items()) if hasattr(outputs, "items") else dict(outputs)
    keys = (
        "span_scores",
        "span_logits",
        "counts",
        "classification_logits",
        "classification_probs",
        "raw_logits",
        "record_logits",
        "pair_logits",
        "grouped_candidates",
    )
    present = [key for key in keys if key in outputs and outputs[key] is not None]
    samples = []
    for index in range(batch_size):
        sample = {key: outputs[key][index] for key in present}
        if boundary is not None and getattr(boundary, "candidates", None) is not None:
            candidates = boundary.candidates
            sample["candidates"] = type(candidates)(
                indices=candidates.indices[index : index + 1],
                proposal_logits=(
                    None if candidates.proposal_logits is None else candidates.proposal_logits[index : index + 1]
                ),
                pair_logits=candidates.pair_logits[index : index + 1],
                valid_mask=candidates.valid_mask[index : index + 1],
                query_mask=candidates.query_mask[index : index + 1],
                candidate_states=(
                    None if candidates.candidate_states is None else candidates.candidate_states[index : index + 1]
                ),
            )
            sample["pair_logits"] = candidates.pair_logits[index]
            sample["null_logits"] = None if boundary.null_logits is None else boundary.null_logits[index]
        samples.append(sample)
    return samples


def _classification_schema_from_row(meta, threshold):
    schema = ClassificationSchema()
    for entry in meta.get("classifications") or []:
        labels = entry["labels"]
        descriptions = entry.get("label_descriptions") or {}
        if descriptions:
            labels = {name: descriptions.get(name) for name in labels}
        kwargs = {
            "threshold": entry.get("cls_threshold", threshold),
            "activation": entry.get("class_act", "auto"),
            "instruction": entry.get("prompt"),
            "examples": tuple(tuple(pair) for pair in entry.get("examples", ()) or ()),
        }
        if bool(entry.get("multi_label", False)):
            schema.multi(entry["task"], labels, **kwargs)
        else:
            schema.single(entry["task"], labels, **kwargs)
    for raw in meta.get("constraints") or []:
        if isinstance(raw, Expr):
            schema.constrain(raw)
        elif isinstance(raw, Mapping) and raw.get("type") in _EXPR_FIELDS:
            schema.constrain(_expr_from_dict(raw))
    return schema


def _task_logits(sample, classifications):
    vectors = sample.get("classification_logits")
    if vectors is None:
        vectors = sample.get("classification_probs")
    if isinstance(vectors, Mapping):
        return vectors
    if torch.is_tensor(vectors) and vectors.ndim == 1:
        grouped = [vectors]
    elif torch.is_tensor(vectors) and vectors.ndim == 2:
        grouped = [vectors[index] for index in range(vectors.shape[0])]
    else:
        grouped = list(vectors or [])
    if len(grouped) != len(classifications):
        raise SchemaError(f"expected {len(classifications)} classification rows, got {len(grouped)}")
    logits = {}
    for entry, vector in zip(classifications, grouped):
        logits[entry["task"]] = _label_logit_map(vector, list(entry["labels"]))
    return logits


def _batch_classification(outputs, rows, temperature, **kwargs):
    if temperature is None:
        temperature = kwargs.pop("temperature", 1.0)
    else:
        kwargs.pop("temperature", None)
    threshold = kwargs.pop("threshold", 0.5)
    include_confidence = kwargs.pop("include_confidence", False)
    decoder = kwargs.pop("decoder", "auto")
    text = kwargs.pop("text", "")
    rows = _batch_rows(rows)
    if not rows:
        return []
    samples = _sample_outputs(outputs, len(rows))
    decoded = []
    for sample, row in zip(samples, rows):
        meta = row
        classifications = list(meta.get("classifications") or [])
        schema = _classification_schema_from_row(meta, threshold)
        decoded.append(
            decode_constrained_classification(
                _task_logits(sample, classifications),
                schema,
                temperature,
                text=meta.get("text") or text,
                decoder=decoder,
                include_confidence=include_confidence,
                candidate_threshold=threshold,
                **kwargs,
            )
        )
    return decoded


def decode_classification(logits, schema, temperature=None, **kwargs):
    """Decode classification logits, including a batch of processor rows.

    A list of metadata rows uses ``threshold``, ``include_confidence``,
    ``temperature``, and ``decoder`` from the processor. A schema uses the
    single-example constrained decoder.
    """
    if isinstance(schema, (list, tuple)):
        return _batch_classification(logits, schema, temperature, **kwargs)
    if temperature is None:
        raise TypeError("decode_classification() missing required argument: 'temperature'")
    return decode_constrained_classification(logits, schema, temperature, **kwargs)


def _joint_prob(value, field_name) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be in [0, 1]")


def _joint_name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} name must be a non-empty string")
    return value


def _joint_types(value, side: str) -> tuple:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"relation {side} must contain at least one entity type")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"relation {side} contains an invalid entity type")
    if len(set(values)) != len(values):
        raise ValueError(f"relation {side} entity types must be unique")
    return values


def _joint_constraint(kind: str, **fields) -> dict:
    """One joint constraint record. Operators share ``_joint_check``."""
    if kind not in _JOINT_FIELDS:
        raise ValueError(f"unknown constraint type {kind!r}")
    data = {"type": kind}
    for name in _JOINT_FIELDS[kind]:
        if name in fields:
            data[name] = fields[name]
        elif name == "directed":
            data[name] = True
        elif name == "slot":
            data[name] = "head"
        elif name == "policy":
            data[name] = "disallow"
        else:
            data[name] = None
    if kind == "TypedEndpoints":
        data["head_types"] = tuple(data["head_types"] or ())
        data["tail_types"] = tuple(data["tail_types"] or ())
    if kind == "EntityOverlapPolicy" and data["policy"] not in {"allow", "disallow", "nested"}:
        raise ValueError("policy must be 'allow', 'disallow', or 'nested'")
    if kind == "UniqueRelationSlot" and data["slot"] not in {"head", "tail", "slot"}:
        raise ValueError("slot must be 'head', 'tail', or 'slot'")
    if kind in {"MaxRelationsPerHead", "MaxRelationsPerTail"} and data["limit"] < 0:
        raise ValueError("limit must be non-negative")
    return data


def _joint_from_dict(data: Mapping) -> dict:
    values = dict(data)
    kind = values.pop("type", None)
    return _joint_constraint(kind, **values)


def _label_of(value: Any) -> Any:
    return _get(value, "label", "type", "name", "entity_type", "relation", "relation_type")


def _endpoint_of(relation: Any, side: str) -> Any:
    return _get(relation, side, f"{side}_entity")


def _identity_of(value: Any) -> Any:
    if value is None:
        return None
    identifier = _get(value, "id", "entity_id", "candidate_id", "index")
    if identifier is not None:
        return identifier
    start, end = _get(value, "start"), _get(value, "end")
    if start is not None and end is not None:
        return (start, end, _label_of(value))
    text = _get(value, "text", "value")
    if text is not None:
        return (text, _label_of(value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _relation_key(value: Any) -> tuple:
    return (_label_of(value), _identity_of(_endpoint_of(value, "head")), _identity_of(_endpoint_of(value, "tail")))


def _matches(relation: Any, relation_type: str | None) -> bool:
    return relation_type is None or _label_of(relation) == relation_type


def _joint_check(constraint: Mapping, candidate, relations=(), nodes=(), mode: str = "edge") -> bool:
    """True when one joint constraint allows an edge, a node, or a full graph."""
    kind = constraint["type"]
    if mode == "node":
        if kind != "EntityOverlapPolicy":
            return True
        policy = constraint["policy"]
        if policy == "allow":
            return True
        start, end = _get(candidate, "start"), _get(candidate, "end")
        if start is None or end is None:
            return True
        for old in nodes:
            if old is candidate:
                continue
            old_start, old_end = _get(old, "start"), _get(old, "end")
            if old_start is None or old_end is None or end <= old_start or old_end <= start:
                continue
            nested = (start >= old_start and end <= old_end) or (old_start >= start and old_end <= end)
            if policy == "disallow" or not nested:
                return False
        return True
    if mode == "validate" and kind == "SymmetricRelation":
        keys = {_relation_key(item) for item in relations}
        return all(
            _label_of(item) != constraint["relation"]
            or (
                constraint["relation"],
                _identity_of(_endpoint_of(item, "tail")),
                _identity_of(_endpoint_of(item, "head")),
            )
            in keys
            for item in relations
        )
    if mode == "validate" and kind == "InverseRelation":
        keys = {_relation_key(item) for item in relations}
        for item in relations:
            label_name = _label_of(item)
            reverse = (_identity_of(_endpoint_of(item, "tail")), _identity_of(_endpoint_of(item, "head")))
            if label_name == constraint["relation"] and (constraint["inverse"], *reverse) not in keys:
                return False
            if label_name == constraint["inverse"] and (constraint["relation"], *reverse) not in keys:
                return False
        return True
    if mode == "validate":
        accepted = []
        for item in relations:
            if not _joint_check(constraint, item, accepted, nodes, "edge"):
                return False
            accepted.append(item)
        return True
    if kind == "TypedEndpoints":
        if not _matches(candidate, constraint["relation"]):
            return True
        head, tail = _label_of(_endpoint_of(candidate, "head")), _label_of(_endpoint_of(candidate, "tail"))
        heads, tails = constraint["head_types"], constraint["tail_types"]
        return (not heads or head in heads) and (not tails or tail in tails)
    if kind == "NoSelfLoops":
        if not _matches(candidate, constraint["relation"]):
            return True
        return _identity_of(_endpoint_of(candidate, "head")) != _identity_of(_endpoint_of(candidate, "tail"))
    if kind == "UniqueRelationPair":
        if not _matches(candidate, constraint["relation"]):
            return True
        head, tail = _identity_of(_endpoint_of(candidate, "head")), _identity_of(_endpoint_of(candidate, "tail"))
        for existing in relations:
            if _label_of(existing) != _label_of(candidate):
                continue
            old_head = _identity_of(_endpoint_of(existing, "head"))
            old_tail = _identity_of(_endpoint_of(existing, "tail"))
            if (head, tail) == (old_head, old_tail) or (
                not constraint["directed"] and (head, tail) == (old_tail, old_head)
            ):
                return False
        return True
    if kind == "UniqueRelationSlot":
        if not _matches(candidate, constraint["relation"]):
            return True
        value = _identity_of(_endpoint_of(candidate, constraint["slot"]))
        return all(
            _label_of(old) != _label_of(candidate) or _identity_of(_endpoint_of(old, constraint["slot"])) != value
            for old in relations
        )
    if kind == "MaxRelationsPerHead":
        if not _matches(candidate, constraint["relation"]):
            return True
        key = _identity_of(_endpoint_of(candidate, "head"))
        used = sum(
            _matches(old, constraint["relation"]) and _identity_of(_endpoint_of(old, "head")) == key
            for old in relations
        )
        return used < constraint["limit"]
    if kind == "MaxRelationsPerTail":
        if not _matches(candidate, constraint["relation"]):
            return True
        key = _identity_of(_endpoint_of(candidate, "tail"))
        used = sum(
            _matches(old, constraint["relation"]) and _identity_of(_endpoint_of(old, "tail")) == key
            for old in relations
        )
        return used < constraint["limit"]
    if kind == "AcyclicRelation":
        if not _matches(candidate, constraint["relation"]):
            return True
        head, tail = _identity_of(_endpoint_of(candidate, "head")), _identity_of(_endpoint_of(candidate, "tail"))
        if head == tail:
            return False
        graph: dict = defaultdict(list)
        for old in relations:
            if _matches(old, constraint["relation"]):
                graph[_identity_of(_endpoint_of(old, "head"))].append(_identity_of(_endpoint_of(old, "tail")))
        stack, seen = [tail], set()
        while stack:
            node = stack.pop()
            if node == head:
                return False
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node])
        return True
    return True


@dataclass(frozen=True)
class EntitySpec:
    """One entity type and its candidate limits."""

    name: str
    description: str | None = None
    threshold: float | None = None
    candidate_threshold: float | None = None
    max_candidates: int | None = None
    allow_nested: bool | None = None

    def __post_init__(self):
        _joint_name(self.name, "entity")
        _joint_prob(self.threshold, "threshold")
        _joint_prob(self.candidate_threshold, "candidate_threshold")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")


@dataclass(frozen=True)
class RelationSpec:
    """One relation type and its endpoint and cardinality rules."""

    name: str
    head: tuple
    tail: tuple
    description: str | None = None
    threshold: float | None = None
    candidate_threshold: float | None = None
    directed: bool = True
    symmetric: bool = False
    inverse: str | None = None
    allow_self: bool = False
    max_per_head: int | None = None
    max_per_tail: int | None = None

    def __post_init__(self):
        _joint_name(self.name, "relation")
        object.__setattr__(self, "head", _joint_types(self.head, "head"))
        object.__setattr__(self, "tail", _joint_types(self.tail, "tail"))
        _joint_prob(self.threshold, "threshold")
        _joint_prob(self.candidate_threshold, "candidate_threshold")
        if self.inverse is not None:
            _joint_name(self.inverse, "inverse relation")
        if self.symmetric and self.inverse:
            raise ValueError("a relation cannot be both symmetric and inverse")
        if self.symmetric and set(self.head) != set(self.tail):
            raise ValueError("symmetric relations require compatible head and tail types")
        if self.symmetric and self.directed:
            object.__setattr__(self, "directed", False)
        for attr in ("max_per_head", "max_per_tail"):
            value = getattr(self, attr)
            if value is not None and value < 0:
                raise ValueError(f"{attr} must be non-negative")


class JointSchema:
    """Mutable builder for entity types, relations, and constraints."""

    def __init__(self):
        self._entities = {}
        self._relations = {}
        self._constraints = []

    @property
    def entity_specs(self):
        return tuple(self._entities.values())

    @property
    def relation_specs(self):
        return tuple(self._relations.values())

    @property
    def constraints(self):
        return tuple(self._constraints)

    def entity(
        self,
        name,
        description=None,
        *,
        threshold=None,
        candidate_threshold=None,
        max_candidates=None,
        allow_nested=None,
    ):
        if name in self._entities:
            raise ValueError(f"entity {name!r} is already defined")
        self._entities[name] = EntitySpec(
            name, description, threshold, candidate_threshold, max_candidates, allow_nested
        )
        return self

    def entities(self, entities):
        if isinstance(entities, str):
            return self.entity(entities)
        items = entities.items() if isinstance(entities, Mapping) else ((item, None) for item in entities)
        for name, value in items:
            if isinstance(value, Mapping):
                self.entity(name, **dict(value))
            else:
                self.entity(name, value)
        return self

    def relation(
        self,
        name,
        head,
        tail,
        description=None,
        *,
        threshold=None,
        candidate_threshold=None,
        directed=True,
        symmetric=False,
        inverse=None,
        allow_self=False,
        max_per_head=None,
        max_per_tail=None,
        **aliases,
    ):
        if name in self._relations:
            raise ValueError(f"relation {name!r} is already defined")
        inverse_of = aliases.pop("inverse_of", None)
        allow_self_loops = aliases.pop("allow_self_loops", None)
        no_self = aliases.pop("no_self_loops", None)
        unique_head = aliases.pop("unique_head", False)
        unique_tail = aliases.pop("unique_tail", False)
        aliases.pop("unique_pair", None)
        acyclic = aliases.pop("acyclic", False)
        if aliases:
            raise TypeError(f"unknown relation options: {sorted(aliases)}")
        if inverse is not None and inverse_of is not None and inverse != inverse_of:
            raise ValueError("inverse and inverse_of disagree")
        inverse = inverse or inverse_of
        if allow_self_loops is not None:
            allow_self = allow_self_loops
        if no_self is not None:
            allow_self = not no_self
        if unique_head and max_per_head is None:
            max_per_head = 1
        if unique_tail and max_per_tail is None:
            max_per_tail = 1
        spec = RelationSpec(
            name,
            _joint_types(head, "head"),
            _joint_types(tail, "tail"),
            description,
            threshold,
            candidate_threshold,
            directed,
            symmetric,
            inverse,
            allow_self,
            max_per_head,
            max_per_tail,
        )
        unknown = (set(spec.head) | set(spec.tail)) - set(self._entities)
        if unknown:
            raise ValueError(f"relation {name!r} references unknown entity types: {sorted(unknown)}")
        self._relations[name] = spec
        if acyclic:
            self.acyclic(name)
        return self

    def constraint(self, constraint):
        if isinstance(constraint, Mapping) and "type" in constraint:
            constraint = _joint_from_dict(constraint)
        elif not isinstance(constraint, Mapping):
            raise TypeError("constraint must implement Constraint")
        self._constraints.append(constraint)
        return self

    def _validate_relation_name(self, name):
        if name is not None and name not in self._relations:
            raise ValueError(f"unknown relation {name!r}")

    def no_self_loops(self, relation=None):
        self._validate_relation_name(relation)
        return self.constraint(_joint_constraint("NoSelfLoops", relation=relation))

    def acyclic(self, relation):
        self._validate_relation_name(relation)
        return self.constraint(_joint_constraint("AcyclicRelation", relation=relation))

    def at_most(self, relation=None, *, per_head=None, per_tail=None, per=None, limit=None):
        if isinstance(relation, int):
            old_limit = relation
            relation = limit if isinstance(limit, str) else None
            limit = old_limit
        self._validate_relation_name(relation)
        if limit is not None:
            if per == "tail":
                per_tail = limit
            else:
                per_head = limit
        if per_head is None and per_tail is None:
            raise ValueError("provide per_head and/or per_tail")
        if per_head is not None:
            self.constraint(_joint_constraint("MaxRelationsPerHead", limit=per_head, relation=relation))
        if per_tail is not None:
            self.constraint(_joint_constraint("MaxRelationsPerTail", limit=per_tail, relation=relation))
        return self

    def to_dict(self):
        return {
            "entities": {
                spec.name: {key: value for key, value in asdict(spec).items() if key != "name" and value is not None}
                for spec in self.entity_specs
            },
            "relations": {
                spec.name: {key: value for key, value in asdict(spec).items() if key != "name" and value is not None}
                for spec in self.relation_specs
            },
            "constraints": [dict(item) for item in self.constraints],
        }

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data):
        schema = cls()
        entities = data.get("entities", {})
        if isinstance(entities, Mapping):
            for name, value in entities.items():
                if isinstance(value, Mapping):
                    schema.entity(name, **dict(value))
                else:
                    schema.entity(name, value)
        else:
            for value in entities:
                schema.entity(value) if isinstance(value, str) else schema.entity(**value)
        relations = data.get("relations", {})
        if isinstance(relations, Mapping):
            for name, value in relations.items():
                schema.relation(name, **dict(value))
        else:
            for value in relations:
                schema.relation(**value)
        for value in data.get("constraints", ()):
            schema.constraint(value)
        return schema

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class CompiledJointSchema:
    """Model schema dict plus the constraints the optimizer enforces."""

    model_schema: dict
    entity_specs: dict
    relation_specs: dict
    constraints: tuple
    entity_order: tuple
    relation_order: tuple

    def build(self):
        return self.model_schema


def _add_constraint(values, constraint):
    if constraint not in values:
        values.append(constraint)


def _compile_joint(schema) -> CompiledJointSchema:
    """Lower relation flags into concrete constraints."""
    if isinstance(schema, CompiledJointSchema):
        return schema
    if not isinstance(schema, JointSchema):
        raise TypeError("schema must be a JointSchema")
    entities = {spec.name: spec for spec in schema.entity_specs}
    relations = {spec.name: spec for spec in schema.relation_specs}
    model = {
        "json_structures": [],
        "classifications": [],
        "entities": dict.fromkeys(entities, ""),
        "relations": [{name: {"head": "", "tail": ""}} for name in relations],
        "json_descriptions": {},
        "entity_descriptions": {
            name: spec.description for name, spec in entities.items() if spec.description is not None
        },
    }
    constraints = list(schema.constraints)
    if entities:
        if all(spec.allow_nested for spec in entities.values()):
            policy = "allow"
        elif any(spec.allow_nested for spec in entities.values()):
            policy = "nested"
        else:
            policy = "disallow"
        _add_constraint(constraints, _joint_constraint("EntityOverlapPolicy", policy=policy))
    for spec in relations.values():
        _add_constraint(
            constraints,
            _joint_constraint("TypedEndpoints", relation=spec.name, head_types=spec.head, tail_types=spec.tail),
        )
        if not spec.allow_self:
            _add_constraint(constraints, _joint_constraint("NoSelfLoops", relation=spec.name))
        _add_constraint(
            constraints, _joint_constraint("UniqueRelationPair", relation=spec.name, directed=spec.directed)
        )
        _add_constraint(constraints, _joint_constraint("UniqueRelationSlot", relation=spec.name, slot="slot"))
        if spec.max_per_head is not None:
            _add_constraint(
                constraints, _joint_constraint("MaxRelationsPerHead", limit=spec.max_per_head, relation=spec.name)
            )
        if spec.max_per_tail is not None:
            _add_constraint(
                constraints, _joint_constraint("MaxRelationsPerTail", limit=spec.max_per_tail, relation=spec.name)
            )
        if spec.symmetric:
            _add_constraint(constraints, _joint_constraint("SymmetricRelation", relation=spec.name))
        if spec.inverse:
            if spec.inverse not in relations:
                raise ValueError(f"relation {spec.name!r} has unknown inverse {spec.inverse!r}")
            other = relations[spec.inverse]
            if set(spec.head) != set(other.tail) or set(spec.tail) != set(other.head):
                raise ValueError(f"inverse endpoint types for {spec.name!r} and {spec.inverse!r} are incompatible")
            _add_constraint(
                constraints, _joint_constraint("InverseRelation", relation=spec.name, inverse=spec.inverse)
            )
    return CompiledJointSchema(model, entities, relations, tuple(constraints), tuple(entities), tuple(relations))


def _coerce_joint_schema(schema):
    if isinstance(schema, CompiledJointSchema):
        return schema
    if isinstance(schema, JointSchema):
        return schema
    if isinstance(schema, Mapping) and ("entities" in schema or "relations" in schema or "constraints" in schema):
        relations = schema.get("relations", {})
        if not isinstance(relations, list):
            return JointSchema.from_dict(schema)
        if "constraints" in schema or "entities" in schema:
            return JointSchema.from_dict(schema)
    if hasattr(schema, "to_dict") and hasattr(schema, "entity_specs"):
        payload = schema.to_dict()
        if isinstance(payload, Mapping):
            return JointSchema.from_dict(payload)
    raise TypeError(f"expected a joint schema, got {type(schema).__name__}")


class CandidateSource(str, Enum):
    """How a node entered the joint lattice."""

    ENTITY = "entity"
    RELATION_RESCUE = "relation_rescue"
    PROVIDED = "provided"


@dataclass(frozen=True)
class NodeCandidate:
    """A typed entity span."""

    entity_type: str
    start: int
    end: int
    score: float
    probability: float | None = None
    source: CandidateSource = CandidateSource.ENTITY
    candidate_id: Hashable | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("node spans must be non-empty and non-negative")
        if self.probability is None:
            object.__setattr__(self, "probability", sigmoid(self.score))
        if not isinstance(self.source, CandidateSource):
            object.__setattr__(self, "source", CandidateSource(self.source))
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", (self.entity_type, self.start, self.end))

    @property
    def key(self) -> tuple:
        return (self.entity_type, self.start, self.end)


@dataclass(frozen=True)
class EdgeCandidate:
    """A typed directed relation between two nodes."""

    relation_type: str
    head: Hashable
    tail: Hashable
    score: float
    head_probability: float | None = None
    tail_probability: float | None = None
    head_entity_probability: float | None = None
    tail_entity_probability: float | None = None
    count_probability: float | None = None
    derived: bool = False
    slot: Hashable | None = None
    hypothesis: Hashable | None = None
    count_alternative: Hashable | None = None
    candidate_id: Hashable | None = None

    def __post_init__(self) -> None:
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", self.key)

    @property
    def key(self) -> tuple:
        return (self.relation_type, self.head, self.tail, self.slot, self.count_alternative)

    @property
    def exclusion_keys(self) -> tuple:
        if self.slot is None:
            return ()
        return (("slot", self.hypothesis, self.count_alternative, self.slot),)

    @property
    def count_choice(self):
        if self.count_alternative is None or self.hypothesis is None:
            return None
        return (self.hypothesis, self.count_alternative)


@dataclass(frozen=True)
class RelationHypothesis:
    """One relation lattice with shape ``[count_slots, 2, L, W]``."""

    relation_type: str
    role_logits: Any
    head_types: tuple = ()
    tail_types: tuple = ()
    threshold: float | None = None
    candidate_threshold: float | None = None
    count_probability: float = 1.0
    count_utility: float = 0.0
    count_alternative: Hashable | None = None
    hypothesis_id: Hashable | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_types", tuple(self.head_types))
        object.__setattr__(self, "tail_types", tuple(self.tail_types))


@dataclass(frozen=True)
class JointProblem:
    """Nodes, edges, and hard constraints for one document."""

    nodes: tuple
    edges: tuple
    constraints: tuple = ()

    def __post_init__(self) -> None:
        ids = [node.candidate_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node candidate ids must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.head not in known or edge.tail not in known:
                raise ValueError("edge endpoints must refer to nodes in the problem")

    @property
    def node_by_id(self) -> dict:
        return {node.candidate_id: node for node in self.nodes}


@dataclass(frozen=True)
class JointSolution:
    """Selected nodes and edges."""

    nodes: tuple
    edges: tuple
    score: float
    feasible: bool = True

    @property
    def node_ids(self):
        return frozenset(node.candidate_id for node in self.nodes)


def _span_entries(lattice) -> list:
    shape = _shape(lattice)
    if len(shape) != 2:
        raise ValueError(f"span lattice must have shape [L, W], got {shape}")
    length, widths = shape
    entries = []
    for start in range(length):
        row = lattice[start]
        for width in range(widths):
            end = start + width + 1
            if end <= length:
                entries.append((_number(row[width]), start, end))
    return entries


def _node_rank(node: NodeCandidate):
    source_rank = 0 if node.source == CandidateSource.ENTITY else 1
    return (-node.score, node.start, node.end, node.entity_type, source_rank)


def _edge_rank(edge: EdgeCandidate):
    return (
        -edge.score,
        str(edge.hypothesis),
        str(edge.slot),
        str(edge.count_alternative),
        str(edge.head),
        str(edge.tail),
        edge.relation_type,
    )


def _build_span_problem(
    entity_logits,
    entity_types,
    relation_hypotheses=(),
    *,
    entity_thresholds=None,
    entity_candidate_thresholds=None,
    entity_max_candidates=None,
    constraints=(),
    candidate_threshold=0.05,
    relation_role_threshold=0.05,
    top_k_entities=32,
    top_k_roles=12,
    relation_pair_cap=128,
    max_edges_per_type=256,
    rescue_per_role=None,
    entity_weight=1.0,
    role_weight=1.0,
    count_weight=1.0,
    entity_threshold=None,
) -> JointProblem:
    """Build nodes from a dense span lattice and rescue relation endpoints."""
    entity_threshold = candidate_threshold if entity_threshold is None else entity_threshold
    rescue_per_role = top_k_roles if rescue_per_role is None else rescue_per_role
    shape = _shape(entity_logits)
    if len(shape) != 3 or shape[0] != len(entity_types):
        raise ValueError(f"entity logits must have shape [types, L, W], got {shape} for {len(entity_types)} types")
    thresholds = dict(entity_thresholds or {})
    candidate_thresholds = dict(entity_candidate_thresholds or {})
    maxima = dict(entity_max_candidates or {})
    node_map: dict = {}
    raw_entity_scores: dict = {}
    for type_index, entity_type in enumerate(entity_types):
        threshold = thresholds.get(entity_type) or 0.5
        floor = candidate_thresholds.get(entity_type)
        if floor is None:
            floor = entity_threshold
        candidates = []
        for raw, start, end in _span_entries(entity_logits[type_index]):
            if not math.isfinite(raw):
                continue
            score = center_logit(raw, threshold) * entity_weight
            probability = sigmoid(raw)
            raw_entity_scores[(entity_type, start, end)] = (score, probability)
            if probability >= floor:
                candidates.append(NodeCandidate(entity_type, start, end, score, probability))
        cap = maxima.get(entity_type, top_k_entities)
        for node in sorted(candidates, key=_node_rank)[:cap]:
            node_map[node.key] = node
    edge_groups: dict = {}
    for index, value in enumerate(relation_hypotheses):
        hypothesis = value if isinstance(value, RelationHypothesis) else RelationHypothesis(**value)
        role_shape = _shape(hypothesis.role_logits)
        if len(role_shape) != 4 or role_shape[1] != 2:
            raise ValueError(f"relation role logits must have shape [count_slots, 2, L, W], got {role_shape}")
        hypothesis_id = hypothesis.hypothesis_id
        if hypothesis_id is None:
            hypothesis_id = (hypothesis.relation_type, index)
        final_threshold = hypothesis.threshold if hypothesis.threshold is not None else 0.5
        floor = hypothesis.candidate_threshold
        if floor is None:
            floor = relation_role_threshold
        relation_offset = probability_to_logit(final_threshold)
        for count_slot in range(role_shape[0]):
            role_options = []
            for role in range(2):
                entries = [
                    (raw, start, end)
                    for raw, start, end in _span_entries(hypothesis.role_logits[count_slot][role])
                    if math.isfinite(raw) and sigmoid(raw) >= floor
                ]
                entries.sort(key=lambda item: (-item[0], item[1], item[2]))
                role_options.append(entries[:top_k_roles])
            typed_roles = [[], []]
            for role, types in enumerate((hypothesis.head_types, hypothesis.tail_types)):
                rescued = 0
                for raw_role_score, start, end in role_options[role]:
                    for entity_type in types:
                        key = (entity_type, start, end)
                        node = node_map.get(key)
                        if node is None:
                            if rescued >= rescue_per_role:
                                continue
                            node_score, node_probability = raw_entity_scores.get(key, (float("-inf"), 0.0))
                            node = NodeCandidate(
                                entity_type,
                                start,
                                end,
                                node_score,
                                node_probability,
                                source=CandidateSource.RELATION_RESCUE,
                            )
                            node_map[key] = node
                            rescued += 1
                        typed_roles[role].append(
                            ((raw_role_score - relation_offset) * role_weight, node, sigmoid(raw_role_score))
                        )
            pairs = []
            for (head_score, head, head_prob), (tail_score, tail, tail_prob) in product(*typed_roles):
                pairs.append(
                    (
                        head_score + tail_score + hypothesis.count_utility * count_weight,
                        head,
                        tail,
                        head_prob,
                        tail_prob,
                    )
                )
            pairs.sort(key=lambda item: (-item[0], str(item[1].candidate_id), str(item[2].candidate_id)))
            for score, head, tail, head_prob, tail_prob in pairs[:relation_pair_cap]:
                edge_groups.setdefault(hypothesis.relation_type, []).append(
                    EdgeCandidate(
                        hypothesis.relation_type,
                        head.candidate_id,
                        tail.candidate_id,
                        score,
                        head_probability=head_prob,
                        tail_probability=tail_prob,
                        head_entity_probability=head.probability,
                        tail_entity_probability=tail.probability,
                        count_probability=hypothesis.count_probability,
                        slot=count_slot,
                        hypothesis=hypothesis_id,
                        count_alternative=hypothesis.count_alternative,
                    )
                )
    edges = []
    for relation_type in sorted(edge_groups):
        unique = {}
        for edge in sorted(edge_groups[relation_type], key=_edge_rank):
            previous = unique.get(edge.key)
            if previous is None or edge.score > previous.score:
                unique[edge.key] = edge
        edges.extend(sorted(unique.values(), key=_edge_rank)[:max_edges_per_type])
    nodes = tuple(sorted(node_map.values(), key=lambda node: (node.entity_type, node.start, node.end, -node.score)))
    return JointProblem(nodes, tuple(sorted(edges, key=_edge_rank)), tuple(constraints))


@dataclass(frozen=True)
class MentionScore:
    """One scored mention in half-open token offsets."""

    query_id: int
    entity_type: str
    start: int
    end: int
    logit: float
    probability: float
    threshold: float = 0.5
    candidate_threshold: float | None = None

    @property
    def key(self) -> tuple:
        return (self.entity_type, self.start, self.end)


@dataclass(frozen=True)
class ScoredRelationEdge:
    """A scored (head, tail) proposal. Endpoints are mention keys."""

    relation_type: str
    head: Hashable
    tail: Hashable
    logit: float
    probability: float
    threshold: float = 0.5
    candidate_threshold: float | None = None


def _boundary_mentions(
    text,
    candidates,
    query_specs,
    *,
    sample_index=0,
    token_offset=0,
    text_length=None,
    pair_temperature=1.0,
    entity_thresholds=None,
    entity_candidate_thresholds=None,
    extra_mentions=(),
):
    """Turn one boundary candidate row into mention scores."""
    if pair_temperature <= 0:
        raise ValueError("pair_temperature must be positive")
    if text_length is None:
        text_length = 0
    thresholds = dict(entity_thresholds or {})
    candidate_thresholds = dict(entity_candidate_thresholds or {})
    best = {mention.key: mention for mention in extra_mentions}
    indices = _get(candidates, "indices")
    pair_logits = _get(candidates, "pair_logits")
    valid_mask = _get(candidates, "valid_mask")
    query_mask = _get(candidates, "query_mask")
    for query_id, spec in enumerate(query_specs):
        task_type = spec.get("task_type") if isinstance(spec, Mapping) else getattr(spec, "task_type", None)
        if task_type != "entities" or query_id >= indices.shape[1]:
            continue
        entity_type = str(
            spec.get("field_name") if isinstance(spec, Mapping) else getattr(spec, "role_name", query_id)
        )
        threshold = thresholds.get(entity_type)
        threshold = 0.5 if threshold is None else float(threshold)
        valid = valid_mask[sample_index, query_id] & query_mask[sample_index, query_id]
        for candidate_id in valid.nonzero(as_tuple=False).flatten().tolist():
            start = int(indices[sample_index, query_id, candidate_id, 0]) - token_offset
            end = int(indices[sample_index, query_id, candidate_id, 1]) - token_offset
            if not (0 <= start < end <= int(text_length)):
                continue
            logit = float(pair_logits[sample_index, query_id, candidate_id].detach().float()) / pair_temperature
            mention = MentionScore(
                query_id,
                entity_type,
                start,
                end,
                logit,
                sigmoid(logit),
                threshold,
                candidate_thresholds.get(entity_type),
            )
            previous = best.get(mention.key)
            if previous is None or mention.logit > previous.logit:
                best[mention.key] = mention
    return tuple(sorted(best.values(), key=lambda item: (item.entity_type, item.start, item.end, -item.logit)))


def _mentions_to_problem(
    mentions,
    edges,
    *,
    constraints=(),
    mention_threshold=0.5,
    decision_threshold=0.5,
    max_mentions_per_type=None,
    max_mentions_by_type=None,
    rescue_relation_endpoints=False,
    edge_candidate_threshold=0.0,
    max_edges_per_type=None,
    entity_weight=1.0,
    relation_weight=1.0,
) -> JointProblem:
    """Build a joint problem from sparse mention and edge scores."""
    edge_by_key = {}
    for edge in edges:
        threshold = edge_candidate_threshold if edge.candidate_threshold is None else edge.candidate_threshold
        if edge.probability < threshold:
            continue
        key = (edge.relation_type, edge.head, edge.tail)
        previous = edge_by_key.get(key)
        if previous is None or edge.logit > previous.logit:
            edge_by_key[key] = edge
    edge_counts: dict = {}
    retained_edges = []
    for edge in sorted(
        edge_by_key.values(), key=lambda item: (item.relation_type, -item.logit, str(item.head), str(item.tail))
    ):
        if max_edges_per_type is not None and edge_counts.get(edge.relation_type, 0) >= max_edges_per_type:
            continue
        retained_edges.append(edge)
        edge_counts[edge.relation_type] = edge_counts.get(edge.relation_type, 0) + 1
    rescue_ids = (
        {endpoint for edge in retained_edges for endpoint in (edge.head, edge.tail)}
        if rescue_relation_endpoints
        else set()
    )
    selected = []
    per_type: dict = {}
    type_limits = dict(max_mentions_by_type or {})
    for mention in sorted(mentions, key=lambda item: (item.entity_type, -item.probability, item.start, item.end)):
        floor = mention_threshold if mention.candidate_threshold is None else mention.candidate_threshold
        if mention.probability < floor and mention.key not in rescue_ids:
            continue
        type_limit = type_limits.get(mention.entity_type, max_mentions_per_type)
        if (
            type_limit is not None
            and per_type.get(mention.entity_type, 0) >= type_limit
            and mention.key not in rescue_ids
        ):
            continue
        selected.append(mention)
        per_type[mention.entity_type] = per_type.get(mention.entity_type, 0) + 1
    nodes = []
    keep_ids = set()
    for mention in selected:
        floor = mention_threshold if mention.candidate_threshold is None else mention.candidate_threshold
        node = NodeCandidate(
            mention.entity_type,
            mention.start,
            mention.end,
            entity_weight
            * center_logit(mention.logit, mention.threshold if mention.threshold is not None else decision_threshold),
            mention.probability,
            source=(
                CandidateSource.RELATION_RESCUE
                if mention.key in rescue_ids and mention.probability < floor
                else CandidateSource.ENTITY
            ),
            candidate_id=mention.key,
        )
        nodes.append(node)
        keep_ids.add(mention.key)
    edge_cands = []
    for edge_slot, edge in enumerate(retained_edges):
        if edge.head not in keep_ids or edge.tail not in keep_ids:
            continue
        edge_cands.append(
            EdgeCandidate(
                edge.relation_type,
                edge.head,
                edge.tail,
                relation_weight
                * center_logit(edge.logit, edge.threshold if edge.threshold is not None else decision_threshold),
                head_probability=edge.probability,
                tail_probability=edge.probability,
                slot=edge_slot,
                hypothesis=edge.relation_type,
            )
        )
    return JointProblem(tuple(nodes), tuple(edge_cands), tuple(constraints))


def _resolve_edge(problem: JointProblem, edge: EdgeCandidate):
    node_by_id = problem.node_by_id
    return SimpleNamespace(
        relation_type=edge.relation_type,
        type=edge.relation_type,
        head=node_by_id.get(edge.head, edge.head),
        tail=node_by_id.get(edge.tail, edge.tail),
        slot=edge.slot,
        candidate_id=edge.candidate_id,
    )


def _allow_node(problem, node, nodes, edges) -> bool:
    del edges
    return all(_joint_check(constraint, node, (), nodes, "node") for constraint in problem.constraints)


def _allow_edge(problem, edge, nodes, edges) -> bool:
    candidate = _resolve_edge(problem, edge)
    accepted = tuple(_resolve_edge(problem, value) for value in edges)
    return all(_joint_check(constraint, candidate, accepted, nodes, "edge") for constraint in problem.constraints)


def _edge_conflicts(edge: EdgeCandidate, used) -> bool:
    used_set = set(used)
    if any(key in used_set for key in edge.exclusion_keys):
        return True
    choice = edge.count_choice
    if choice is None:
        return False
    group, alternative = choice
    return any(
        isinstance(key, tuple)
        and len(key) == 3
        and key[0] == "count-choice"
        and key[1] == group
        and key[2] != alternative
        for key in used_set
    )


def _edge_usage(edge: EdgeCandidate):
    keys = set(edge.exclusion_keys)
    if edge.count_choice is not None:
        keys.add(("count-choice",) + edge.count_choice)
    return frozenset(keys)


def _try_edge(problem, selected, edges, used, score, edge):
    """Shared greedy and beam expansion. Head==tail is listed twice."""
    if _edge_conflicts(edge, used):
        return None
    node_by_id = problem.node_by_id
    new_ids = [value for value in (edge.head, edge.tail) if value not in selected]
    proposed = [node_by_id[value] for value in new_ids]
    current = [node_by_id[value] for value in selected]
    proposed_nodes = current + proposed
    if not all(_allow_node(problem, node, proposed_nodes, edges) for node in proposed):
        return None
    if not _allow_edge(problem, edge, proposed_nodes, edges):
        return None
    gain = edge.score + sum(node.score for node in proposed)
    if gain < 0.0:
        return None
    new_selected = set(selected)
    new_selected.update(new_ids)
    new_used = set(used)
    new_used.update(_edge_usage(edge))
    return new_selected, list(edges) + [edge], new_used, score + gain


def _add_positive_nodes(problem, selected, edges, score):
    node_by_id = problem.node_by_id
    selected = set(selected)
    for node in sorted(problem.nodes, key=lambda item: (-item.score, item.entity_type, item.start, item.end)):
        if node.candidate_id in selected or node.score <= 0.0:
            continue
        current = [node_by_id[value] for value in selected]
        if _allow_node(problem, node, current + [node], edges):
            selected.add(node.candidate_id)
            score += node.score
    return selected, score


def _companions(problem, edges):
    chosen = list(edges)
    keys = {(edge.relation_type, edge.head, edge.tail) for edge in chosen}
    companions = []
    for constraint in problem.constraints:
        if constraint["type"] == "SymmetricRelation":
            pairs = [
                (constraint["relation"], edge.tail, edge.head, edge)
                for edge in chosen
                if edge.relation_type == constraint["relation"]
            ]
        elif constraint["type"] == "InverseRelation":
            pairs = []
            for edge in chosen:
                if edge.relation_type == constraint["relation"]:
                    pairs.append((constraint["inverse"], edge.tail, edge.head, edge))
                elif edge.relation_type == constraint["inverse"]:
                    pairs.append((constraint["relation"], edge.tail, edge.head, edge))
        else:
            continue
        for relation, head, tail, source in pairs:
            key = (relation, head, tail)
            if key in keys:
                continue
            companions.append(
                EdgeCandidate(
                    relation,
                    head,
                    tail,
                    0.0,
                    head_probability=source.tail_probability,
                    tail_probability=source.head_probability,
                    head_entity_probability=source.tail_entity_probability,
                    tail_entity_probability=source.head_entity_probability,
                    count_probability=source.count_probability,
                    derived=True,
                    candidate_id=("derived", relation, head, tail),
                )
            )
            keys.add(key)
    return tuple(
        sorted(
            chosen + companions, key=lambda edge: (edge.relation_type, str(edge.head), str(edge.tail), edge.derived)
        )
    )


def _make_solution(problem, node_ids, edges, score) -> JointSolution:
    selected = set(node_ids)
    nodes = tuple(node for node in problem.nodes if node.candidate_id in selected)
    return JointSolution(nodes, _companions(problem, edges), float(score))


def _valid_solution(problem, solution: JointSolution) -> bool:
    accepted_nodes = []
    for node in solution.nodes:
        if not _allow_node(problem, node, accepted_nodes + [node], solution.edges):
            return False
        accepted_nodes.append(node)
    accepted_edges = []
    for edge in solution.edges:
        if not _allow_edge(problem, edge, list(solution.nodes), accepted_edges):
            return False
        accepted_edges.append(edge)
    resolved = tuple(_resolve_edge(problem, edge) for edge in solution.edges)
    for constraint in problem.constraints:
        if not _joint_check(constraint, None, resolved, list(solution.nodes), "validate"):
            return False
    return True


def _greedy_edge_key(edge, node_by_id):
    # Greedy omits the tail score when both ends are the same node.
    endpoint = node_by_id[edge.head].score
    if edge.tail != edge.head:
        endpoint += node_by_id[edge.tail].score
    return (
        -(edge.score + endpoint),
        -edge.score,
        edge.relation_type,
        str(edge.hypothesis),
        str(edge.slot),
        str(edge.head),
        str(edge.tail),
    )


def _beam_edge_key(edge, node_by_id):
    return (
        -(edge.score + node_by_id[edge.head].score + node_by_id[edge.tail].score),
        edge.relation_type,
        str(edge.hypothesis),
        str(edge.slot),
        str(edge.head),
        str(edge.tail),
    )


def _empty_solution() -> JointSolution:
    return JointSolution((), (), 0.0, feasible=False)


def _optimize_greedy(problem: JointProblem) -> JointSolution:
    node_by_id = problem.node_by_id
    selected, edges, used, score = set(), [], set(), 0.0
    for edge in sorted(problem.edges, key=lambda item: _greedy_edge_key(item, node_by_id)):
        expanded = _try_edge(problem, selected, edges, used, score, edge)
        if expanded is None:
            continue
        selected, edges, used, score = expanded
    selected, score = _add_positive_nodes(problem, selected, edges, score)
    result = _make_solution(problem, selected, edges, score)
    if _valid_solution(problem, result):
        return result
    logger.warning(
        "greedy joint-IE decoding produced a constraint-violating assignment; returning empty solution",
    )
    return _empty_solution()


def _state_signature(node_ids, edges):
    return (tuple(sorted(map(str, node_ids))), tuple(str(edge.candidate_id) for edge in edges))


def _solution_key(solution: JointSolution):
    return (
        solution.score,
        tuple(sorted(map(str, solution.node_ids))),
        tuple(str(edge.candidate_id) for edge in solution.edges),
    )


def _optimize_beam(problem: JointProblem, beam_width: int) -> JointSolution:
    """Beam over the same edge expansion greedy uses, without a suffix bound."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    node_by_id = problem.node_by_id
    ordered = sorted(problem.edges, key=lambda item: _beam_edge_key(item, node_by_id))
    beam = [(set(), [], set(), 0.0)]
    for edge in ordered:
        expanded = list(beam)
        for selected, edges, used, score in beam:
            trial = _try_edge(problem, selected, edges, used, score, edge)
            if trial is not None:
                expanded.append(trial)
        unique = {}
        for selected, edges, used, score in expanded:
            key = (frozenset(selected), frozenset(edge.candidate_id for edge in edges), frozenset(used))
            old = unique.get(key)
            if old is None or score > old[3]:
                unique[key] = (selected, edges, used, score)
        beam = sorted(unique.values(), key=lambda state: (-state[3], _state_signature(state[0], state[1])))[
            :beam_width
        ]
    candidates = []
    for selected, edges, used, score in beam:
        del used
        selected, score = _add_positive_nodes(problem, selected, edges, score)
        candidates.append(_make_solution(problem, selected, edges, score))
    candidates.append(_optimize_greedy(problem))
    feasible = [solution for solution in candidates if _valid_solution(problem, solution)]
    if feasible:
        return max(feasible, key=_solution_key)
    logger.warning(
        "joint-IE decoding found no constraint-satisfying assignment among %d candidates; returning empty solution",
        len(candidates),
    )
    return _empty_solution()


def _component_scores(value, include_count: bool) -> list:
    scores = []
    explicit = _get(value, "probabilities", "confidences", "scores", default=None)
    if explicit is not None and not isinstance(explicit, (str, bytes, Mapping)):
        scores.extend(float(item) for item in explicit)
    else:
        for name in (
            "entity_confidence",
            "head_confidence",
            "tail_confidence",
            "confidence",
            "head_probability",
            "tail_probability",
            "head_entity_probability",
            "tail_entity_probability",
            "probability",
        ):
            score = _get(value, name, default=None)
            if score is not None:
                scores.append(float(score))
    if include_count:
        count = _get(value, "count_confidence", "count_probability", default=None)
        if count is not None:
            scores.append(float(count))
        else:
            count_logit = _get(value, "count_logit", default=None)
            if count_logit is not None:
                scores.append(sigmoid(count_logit))
    return scores


def _project_span(node, text, starts, ends):
    token_start, token_end = int(node.start), int(node.end)
    if starts is not None:
        char_start = int(starts[token_start])
        char_end = int(ends[max(token_start, token_end - 1)])
    else:
        char_start, char_end = token_start, token_end
    surface = text[char_start:char_end]
    return char_start, char_end, surface


def _joint_result_dict(solution, problem, text, starts, ends, *, include_confidence, include_spans) -> dict:
    rows = []
    for node in solution.nodes:
        char_start, char_end, surface = _project_span(node, text, starts, ends)
        rows.append(
            (node, char_start, char_end, surface, _component_scores(node, True), "rescue" in str(node.source).lower())
        )
    rows.sort(key=lambda item: (item[1], item[2], item[0].entity_type, item[3]))
    id_of = {}
    entities = []
    for index, (node, char_start, char_end, surface, scores, rescued) in enumerate(rows):
        entity_id = f"e{index + 1}"
        id_of[("value", node.candidate_id)] = entity_id
        payload = {"id": entity_id, "type": node.entity_type, "text": surface}
        if include_spans:
            payload.update(start=char_start, end=char_end)
        if include_confidence:
            confidence = _geometric_mean(scores)
            if confidence is not None:
                payload["confidence"] = confidence
        if rescued:
            payload["rescued"] = True
        entities.append(payload)
    relations = []
    for edge in solution.edges:
        head = id_of.get(("value", edge.head))
        tail = id_of.get(("value", edge.tail))
        if head is None or tail is None:
            raise ValueError("relation endpoint does not identify a selected entity")
        payload = {"type": edge.relation_type, "head": head, "tail": tail}
        if include_confidence:
            confidence = _geometric_mean(_component_scores(edge, True))
            if confidence is not None:
                payload["confidence"] = confidence
        if edge.derived:
            payload["derived"] = True
        relations.append(payload)
    relations.sort(key=lambda item: (item["type"], item["head"], item["tail"]))
    return {"entities": entities, "relations": relations}


def _mask_invalid(logits, valid):
    if valid is None or not hasattr(logits, "masked_fill"):
        return logits
    mask = valid
    while getattr(mask, "ndim", 0) < logits.ndim:
        mask = mask.unsqueeze(0)
    if hasattr(mask, "bool"):
        mask = mask.bool()
    return logits.masked_fill(~mask, float("-inf"))


def _as_tuple(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(item) for item in value)


def _hypothesis_from_mapping(raw: Mapping) -> RelationHypothesis:
    payload = dict(raw)
    payload.setdefault("head_types", payload.pop("head", ()))
    payload.setdefault("tail_types", payload.pop("tail", ()))
    known = set(RelationHypothesis.__dataclass_fields__)
    return RelationHypothesis(**{key: value for key, value in payload.items() if key in known})


def _problem_from_span(scores, compiled: CompiledJointSchema, config) -> JointProblem:
    tasks = list(_get(scores, "tasks", default=()) or ())
    entity_task = next((task for task in tasks if _get(task, "task_type") == "entities"), None)
    relation_specs = compiled.relation_specs
    if entity_task is not None and _get(entity_task, "count_hypotheses"):
        hypotheses_src = list(_get(entity_task, "count_hypotheses"))
        entity_logits = hypotheses_src[0].role_logits[0]
        entity_types = tuple(_get(entity_task, "roles"))
    else:
        entity_logits = _get(scores, "entity_logits", "entity_scores", default=None)
        entity_types = tuple(_get(scores, "entity_types", default=None) or compiled.entity_order)
    if entity_logits is None:
        raise TypeError("span scores need entity_logits [types, L, W] or a task lattice")
    valid = _get(scores, "valid_span_mask", default=None)
    entity_logits = _mask_invalid(entity_logits, valid)
    if len(entity_types) == 0:
        shape = _shape(_get(scores, "valid_span_mask", default=None) or entity_logits)
        length, width = (shape[-2], shape[-1]) if len(shape) >= 2 else (0, 0)
        entity_logits = torch.empty((0, length, width))
    hypotheses = []
    if tasks:
        for task in tasks:
            if _get(task, "task_type") != "relations":
                continue
            spec = relation_specs.get(_get(task, "name"))
            head_types = getattr(spec, "head", entity_types)
            tail_types = getattr(spec, "tail", entity_types)
            for alternative, count in enumerate(_get(task, "count_hypotheses") or ()):
                if int(_get(count, "count", default=0)) <= 0:
                    continue
                hypotheses.append(
                    RelationHypothesis(
                        _get(task, "name"),
                        _mask_invalid(_get(count, "role_logits"), valid),
                        head_types,
                        tail_types,
                        getattr(spec, "threshold", None),
                        getattr(spec, "candidate_threshold", None),
                        float(_get(count, "probability", default=1.0)),
                        float(_get(count, "logit", default=0.0)),
                        alternative,
                        _get(task, "name"),
                    )
                )
    else:
        for raw in _get(scores, "relation_hypotheses", "relations", default=()) or ():
            if isinstance(raw, RelationHypothesis):
                hypotheses.append(raw)
            elif isinstance(raw, Mapping) and "role_logits" in raw:
                hypotheses.append(_hypothesis_from_mapping(raw))
    specs = compiled.entity_specs
    return _build_span_problem(
        entity_logits,
        entity_types,
        hypotheses,
        entity_thresholds={name: spec.threshold for name, spec in specs.items()},
        entity_candidate_thresholds={name: spec.candidate_threshold for name, spec in specs.items()},
        entity_max_candidates={
            name: spec.max_candidates for name, spec in specs.items() if spec.max_candidates is not None
        },
        constraints=compiled.constraints,
        candidate_threshold=config.candidate_threshold,
        relation_role_threshold=config.relation_role_threshold,
        top_k_entities=config.top_k_entities,
        top_k_roles=config.top_k_roles,
        relation_pair_cap=config.relation_pair_cap,
        max_edges_per_type=config.max_edges_per_type,
        rescue_per_role=config.rescue_per_role,
        entity_weight=config.entity_weight,
        role_weight=config.role_weight,
        count_weight=config.count_weight,
        entity_threshold=config.entity_threshold,
    )


def _mention_from_mapping(raw: Mapping, index: int) -> MentionScore:
    entity_type = str(raw.get("entity_type", raw.get("type", raw.get("label"))))
    start = int(raw["start"] if "start" in raw else raw.get("token_start"))
    end = int(raw["end"] if "end" in raw else raw.get("token_end"))
    probability = raw.get("probability")
    logit = raw.get("logit", raw.get("score"))
    if logit is None and probability is None:
        raise ValueError("mention scores need a logit or a probability")
    if logit is None:
        logit = probability_to_logit(float(probability))
    logit = float(logit)
    if probability is None:
        probability = sigmoid(logit)
    return MentionScore(
        int(raw.get("query_id", index)),
        entity_type,
        start,
        end,
        logit,
        float(probability),
        float(raw.get("threshold", 0.5)),
        raw.get("candidate_threshold"),
    )


def _endpoint_key(value: Any):
    if isinstance(value, Mapping):
        entity_type = str(value.get("entity_type", value.get("type", value.get("label"))))
        start = int(value["start"] if "start" in value else value["token_start"])
        end = int(value["end"] if "end" in value else value["token_end"])
        return (entity_type, start, end)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (str(value[0]), int(value[1]), int(value[2]))
    return value


def _edge_from_mapping(raw: Mapping) -> ScoredRelationEdge:
    probability = raw.get("probability")
    logit = raw.get("logit", raw.get("score"))
    if logit is None and probability is None:
        raise ValueError("relation edges need a logit or a probability")
    if logit is None:
        logit = probability_to_logit(float(probability))
    logit = float(logit)
    if probability is None:
        probability = sigmoid(logit)
    return ScoredRelationEdge(
        str(raw.get("relation_type", raw.get("type", raw.get("label")))),
        _endpoint_key(raw.get("head")),
        _endpoint_key(raw.get("tail")),
        logit,
        float(probability),
        float(raw.get("threshold", 0.5)),
        raw.get("candidate_threshold"),
    )


def _sparse_problem(scores, compiled: CompiledJointSchema, config) -> JointProblem:
    specs = compiled.entity_specs
    candidates = _get(scores, "candidates", default=None)
    query_specs = _get(scores, "query_specs", default=None)
    if (
        candidates is not None
        and query_specs is not None
        and _get(candidates, "pair_logits", default=None) is not None
    ):
        mentions = _boundary_mentions(
            str(_get(scores, "text", default="") or ""),
            candidates,
            query_specs,
            sample_index=int(_get(scores, "sample_index", default=0)),
            token_offset=int(_get(scores, "token_offset", default=0)),
            text_length=_get(scores, "text_length", default=None)
            or len(_get(scores, "start_mappings", default=()) or ()),
            pair_temperature=float(_get(scores, "pair_temperature", default=1.0)),
            entity_thresholds={name: spec.threshold for name, spec in specs.items()},
            entity_candidate_thresholds={name: spec.candidate_threshold for name, spec in specs.items()},
            extra_mentions=tuple(
                item if isinstance(item, MentionScore) else _mention_from_mapping(item, index)
                for index, item in enumerate(_get(scores, "extra_mentions", default=()) or ())
            ),
        )
        edges = tuple(
            item if isinstance(item, ScoredRelationEdge) else _edge_from_mapping(item)
            for item in _get(scores, "edges", default=()) or ()
        )
    else:
        mentions_raw = _get(scores, "mentions", default=None)
        if mentions_raw is None and _get(scores, "entity_scores", default=None) is not None:
            mentions = []
            block = _get(scores, "entity_scores")
            table = block.scores if hasattr(block, "scores") else block
            calibrator = _get(scores, "calibrator", default=None)
            for label, values in table.items():
                for index, span in enumerate(_get(scores, "spans")):
                    logit = _number(values[index])
                    if calibrator is not None:
                        logit = _number(calibrator.calibrate(logit))
                    mentions.append(MentionScore(0, str(label), int(span.start), int(span.end), logit, sigmoid(logit)))
            mentions_raw = mentions
            edges = ()
        elif mentions_raw is None:
            raise TypeError("boundary scores need mentions or boundary candidates")
        else:
            mentions_raw = [
                item if isinstance(item, MentionScore) else _mention_from_mapping(item, index)
                for index, item in enumerate(mentions_raw)
            ]
            edges = tuple(
                item if isinstance(item, ScoredRelationEdge) else _edge_from_mapping(item)
                for item in _get(scores, "edges", default=()) or ()
            )
        mentions = tuple(mentions_raw)
    return _mentions_to_problem(
        mentions,
        edges,
        constraints=compiled.constraints,
        mention_threshold=(
            config.entity_threshold if config.entity_threshold is not None else config.candidate_threshold
        ),
        max_mentions_per_type=config.top_k_entities,
        max_mentions_by_type={
            name: spec.max_candidates for name, spec in specs.items() if spec.max_candidates is not None
        },
        rescue_relation_endpoints=True,
        edge_candidate_threshold=config.relation_role_threshold,
        max_edges_per_type=min(config.relation_pair_cap, config.max_edges_per_type),
        entity_weight=config.entity_weight,
        relation_weight=config.role_weight,
    )


@dataclass(frozen=True)
class JointIEConfig:
    """Candidate caps and the choice of greedy or beam search."""

    optimizer: str = "beam"
    beam_size: int = 32
    candidate_threshold: float = 0.05
    relation_role_threshold: float = 0.05
    top_k_entities: int = 32
    top_k_roles: int = 12
    count_top_k: int = 2
    include_confidence: bool = True
    include_spans: bool = True
    entity_threshold: float | None = None
    relation_pair_cap: int = 128
    max_edges_per_type: int = 256
    rescue_per_role: int | None = None
    entity_weight: float = 1.0
    role_weight: float = 1.0
    count_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.optimizer.lower() not in {"beam", "greedy", "auto"}:
            raise ValueError("optimizer must be 'beam' or 'greedy'")
        for name in (
            "beam_size",
            "top_k_entities",
            "top_k_roles",
            "count_top_k",
            "relation_pair_cap",
            "max_edges_per_type",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rescue_per_role is not None and self.rescue_per_role <= 0:
            raise ValueError("rescue_per_role must be positive")


def _architecture(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("architecture must be 'span' or 'boundary'")
    key = value.strip().lower()
    if key not in {"span", "boundary"}:
        raise ValueError(f"Unknown extractor architecture {value!r}. Expected one of: 'span', 'boundary'.")
    return key


def _with_overlap(compiled: CompiledJointSchema, policy: str | None):
    if policy is None:
        return compiled, None
    canonical = normalize_overlap_policy(policy)
    constraints = [item for item in compiled.constraints if item["type"] != "EntityOverlapPolicy"]
    if canonical == "longest":
        return replace(compiled, constraints=tuple(constraints)), "longest"
    mapped = "allow" if canonical == "allow" else "nested" if canonical == "nested" else "disallow"
    constraints.append(_joint_constraint("EntityOverlapPolicy", policy=mapped))
    return replace(compiled, constraints=tuple(constraints)), None


def _filter_longest(problem, solution: JointSolution) -> JointSolution:
    kept = resolve_overlaps(
        list(solution.nodes),
        "longest",
        score=lambda node: node.score,
        start=lambda node: node.start,
        end=lambda node: node.end,
    )
    ids = {node.candidate_id for node in kept}
    edges = tuple(edge for edge in solution.edges if edge.head in ids and edge.tail in ids and not edge.derived)
    return _make_solution(problem, ids, edges, solution.score)


def _optimize(problem: JointProblem, config: JointIEConfig) -> JointSolution:
    if config.optimizer.lower() in {"auto", "greedy"}:
        return _optimize_greedy(problem)
    return _optimize_beam(problem, config.beam_size)


def _decode_joint_one(scores, schema, architecture, *, text="", config: JointIEConfig, overlap_policy=None) -> dict:
    compiled = _compile_joint(_coerce_joint_schema(schema))
    compiled, longest = _with_overlap(compiled, overlap_policy)
    kind = _architecture(architecture)
    document = text or str(_get(scores, "text", default="") or "")
    starts = _as_tuple(_get(scores, "start_mappings", default=None)) or None
    ends = _as_tuple(_get(scores, "end_mappings", default=None)) or None
    mentions = _get(scores, "mentions", default=None)
    dense = (
        _get(scores, "entity_logits", default=None) is not None
        or _get(scores, "tasks", default=None) is not None
        or _get(scores, "entity_scores", default=None) is not None
        and _get(scores, "spans", default=None) is None
    )
    use_span = kind == "span" and (
        dense
        or (
            mentions is None
            and _get(scores, "candidates", default=None) is None
            and _get(scores, "entity_scores", default=None) is None
        )
    )
    if (
        use_span
        and _get(scores, "entity_scores", default=None) is not None
        and _get(scores, "spans", default=None) is not None
    ):
        use_span = False
    if use_span:
        problem = _problem_from_span(scores, compiled, config)
    else:
        problem = _sparse_problem(scores, compiled, config)
        if not document:
            document = str(_get(scores, "text", default="") or "")
        if starts is None:
            starts = _as_tuple(_get(scores, "start_mappings", default=None)) or None
            ends = _as_tuple(_get(scores, "end_mappings", default=None)) or None
    solution = _optimize(problem, config)
    if longest == "longest":
        solution = _filter_longest(problem, solution)
    return _joint_result_dict(
        solution,
        problem,
        document,
        starts,
        ends,
        include_confidence=config.include_confidence,
        include_spans=config.include_spans,
    )


def _confidence_key(value) -> float:
    return float("-inf") if value is None else float(value)


def _view_entity(item):
    if isinstance(item, Mapping):
        return SimpleNamespace(
            id=str(item.get("id", "")),
            type=str(item.get("type", item.get("label", ""))),
            text=str(item.get("text", "")),
            start=int(item.get("start", 0)),
            end=int(item.get("end", 0)),
            confidence=item.get("confidence"),
            sentence_id=item.get("sentence_id"),
            rescued=bool(item.get("rescued", False)),
        )
    return item


def _view_relation(item):
    if isinstance(item, Mapping):
        return SimpleNamespace(
            type=str(item.get("type", item.get("label", ""))),
            head=str(item["head"]),
            tail=str(item["tail"]),
            confidence=item.get("confidence"),
            derived=bool(item.get("derived", False)),
        )
    return item


def merge_joint_chunks(text: str, fragments, *, include_confidence: bool = True, include_spans: bool = True) -> dict:
    """Merge per-chunk joint results onto document character offsets.

    Relations stay inside the chunk that produced both endpoints. Equal
    confidences keep the earlier entity.
    """
    entity_by_key = {}
    relation_rows = {}
    for start_char, raw in fragments:
        if isinstance(raw, Mapping):
            entities = raw.get("entities", [])
            relations = raw.get("relations", [])
        else:
            entities = raw.entities
            relations = raw.relations
        local_keys = {}
        for entity in entities:
            entity = _view_entity(entity)
            start = entity.start + start_char
            end = entity.end + start_char
            key = (entity.type, start, end)
            local_keys[entity.id] = key
            previous = entity_by_key.get(key)
            if previous is None or _confidence_key(entity.confidence) > _confidence_key(previous.confidence):
                entity_by_key[key] = SimpleNamespace(
                    type=entity.type,
                    text=text[start:end],
                    start=start,
                    end=end,
                    confidence=entity.confidence,
                    sentence_id=entity.sentence_id,
                    rescued=entity.rescued,
                )
        for relation in relations:
            relation = _view_relation(relation)
            if relation.head not in local_keys or relation.tail not in local_keys:
                continue
            head_key, tail_key = local_keys[relation.head], local_keys[relation.tail]
            key = (relation.type, head_key, tail_key)
            row = (relation.type, head_key, tail_key, relation.confidence, relation.derived)
            previous = relation_rows.get(key)
            if previous is None or _confidence_key(row[3]) > _confidence_key(previous[3]):
                relation_rows[key] = row
    ordered_keys = sorted(entity_by_key, key=lambda key: (key[1], key[2], key[0]))
    key_to_id = {key: f"e{index + 1}" for index, key in enumerate(ordered_keys)}
    entities = []
    for key in ordered_keys:
        item = entity_by_key[key]
        payload = {"id": key_to_id[key], "type": item.type, "text": item.text}
        if include_spans:
            payload.update(start=item.start, end=item.end)
            if item.sentence_id is not None:
                payload["sentence_id"] = item.sentence_id
        if include_confidence and item.confidence is not None:
            payload["confidence"] = item.confidence
        if item.rescued:
            payload["rescued"] = True
        entities.append(payload)
    relations = []
    for label, head, tail, confidence, derived in relation_rows.values():
        payload = {"type": label, "head": key_to_id[head], "tail": key_to_id[tail]}
        if include_confidence and confidence is not None:
            payload["confidence"] = confidence
        if derived:
            payload["derived"] = True
        relations.append(payload)
    relations.sort(key=lambda item: (item["type"], item["head"], item["tail"]))
    return {"entities": entities, "relations": relations}


def _doc_axis(tensor, doc_len):
    if doc_len <= 0:
        return tensor[..., :0, :]
    return tensor[..., -doc_len:, :]


def _schema_from_meta(meta) -> JointSchema:
    source = meta.get("schema") or {}
    schema = JointSchema()
    entities = source.get("entities") or {}
    if isinstance(entities, list):
        schema.entities(entities)
    elif isinstance(entities, Mapping):
        for name, value in entities.items():
            if isinstance(value, Mapping) and any(
                key in value
                for key in ("threshold", "candidate_threshold", "max_candidates", "allow_nested", "description")
            ):
                fields = {
                    key: value[key]
                    for key in ("description", "threshold", "candidate_threshold", "max_candidates", "allow_nested")
                    if key in value
                }
                description = fields.pop("description", None)
                schema.entity(name, description, **fields)
            elif isinstance(value, str) and value:
                schema.entity(name, value)
            else:
                schema.entity(name)
    relations = source.get("relations") or []
    if isinstance(relations, Mapping):
        for name, value in relations.items():
            schema.relation(name, **dict(value))
    else:
        for item in relations:
            if "name" in item and "head" in item:
                schema.relation(**dict(item))
                continue
            for name, fields in item.items():
                head = fields.get("head")
                tail = fields.get("tail")
                if isinstance(head, Mapping):
                    head = head.get("type") or head.get("entity")
                if isinstance(tail, Mapping):
                    tail = tail.get("type") or tail.get("entity")
                options = {
                    key: fields[key]
                    for key in (
                        "description",
                        "threshold",
                        "candidate_threshold",
                        "directed",
                        "symmetric",
                        "inverse",
                        "inverse_of",
                        "allow_self",
                        "allow_self_loops",
                        "no_self_loops",
                        "max_per_head",
                        "max_per_tail",
                        "unique_head",
                        "unique_tail",
                        "acyclic",
                    )
                    if key in fields
                }
                schema.relation(name, head, tail, **options)
    for raw in list(meta.get("constraints") or []) + list(source.get("constraints") or []):
        if isinstance(raw, Mapping) and raw.get("type") in _JOINT_FIELDS:
            schema.constraint(raw)
    return schema


def _span_scores_from_sample(sample, meta, schema: JointSchema):
    groups = [group for group in (meta.get("groups") or []) if group["task_type"] != "classifications"]
    tensors = sample.get("span_logits")
    probabilities = False
    if tensors is None:
        tensors = sample.get("span_scores")
        probabilities = True
    if tensors is None:
        raise TypeError("span scores need entity_logits [types, L, W] or a task lattice")
    if torch.is_tensor(tensors):
        tensors = [tensors]
    counts = sample.get("counts")
    if torch.is_tensor(counts):
        counts = counts.detach().cpu().tolist()
    doc_len = len(meta.get("start") or [])
    entity_logits = None
    entity_types = ()
    hypotheses = []
    specs = {spec.name: spec for spec in schema.relation_specs}
    for index, group in enumerate(groups):
        tensor = tensors[index]
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor, dtype=torch.float)
        tensor = tensor.detach().float().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        count = int(counts[index]) if counts is not None else int(tensor.shape[0])
        tensor = tensor[: max(count, 0)]
        if probabilities:
            tensor = torch.special.logit(tensor.clamp(1e-6, 1 - 1e-6))
        tensor = _doc_axis(tensor, doc_len)
        if group["task_type"] == "entities":
            entity_types = tuple(group["fields"])
            entity_logits = (
                tensor[0]
                if tensor.shape[0]
                else tensor.new_zeros((len(entity_types), doc_len, tensor.shape[-1] if tensor.ndim == 4 else 1))
            )
        elif group["task_type"] == "relations" and tensor.shape[0] > 0 and tensor.shape[1] >= 2:
            spec = specs.get(group["name"])
            hypotheses.append(
                RelationHypothesis(
                    group["name"],
                    tensor[:, :2],
                    getattr(spec, "head", ()),
                    getattr(spec, "tail", ()),
                    getattr(spec, "threshold", None),
                    getattr(spec, "candidate_threshold", None),
                    count_alternative=0,
                    hypothesis_id=group["name"],
                )
            )
    return {
        "text": meta.get("text") or "",
        "entity_logits": entity_logits if entity_logits is not None else torch.empty((0, doc_len, 1)),
        "entity_types": entity_types,
        "relation_hypotheses": hypotheses,
        "start_mappings": tuple(meta.get("start") or ()),
        "end_mappings": tuple(meta.get("end") or ()),
    }


def _boundary_scores_from_sample(sample, meta):
    return {
        "text": meta.get("text") or "",
        "candidates": sample.get("candidates"),
        "query_specs": _query_specs(meta),
        "token_offset": int(meta.get("prefix_len") or 0),
        "text_length": len(meta.get("start") or []),
        "start_mappings": tuple(meta.get("start") or ()),
        "end_mappings": tuple(meta.get("end") or ()),
        "edges": sample.get("edges") or (),
    }


def _batch_joint(outputs, metadata, *, threshold, include_confidence, include_spans, optimizer, overlap_policy):
    rows = _batch_rows(metadata)
    if not rows:
        return []
    samples = _sample_outputs(outputs, len(rows))
    if optimizer == "exact":
        optimizer = "beam"
    decoded = []
    for sample, meta in zip(samples, rows):
        schema = _schema_from_meta(meta)
        architecture = meta.get("architecture") or ("boundary" if sample.get("candidates") is not None else "span")
        if architecture == "boundary":
            scores = _boundary_scores_from_sample(sample, meta)
        else:
            scores = _span_scores_from_sample(sample, meta, schema)
        config = JointIEConfig(
            optimizer=optimizer,
            candidate_threshold=threshold,
            relation_role_threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            entity_threshold=threshold,
        )
        decoded.append(
            _decode_joint_one(
                scores, schema, architecture, text=meta.get("text") or "", config=config, overlap_policy=overlap_policy
            )
        )
    return decoded


def decode_joint(
    span_or_boundary_scores,
    schema,
    architecture=None,
    *,
    text: str = "",
    optimizer: str = "beam",
    beam_size: int = 32,
    candidate_threshold: float = 0.05,
    relation_role_threshold: float = 0.05,
    top_k_entities: int = 32,
    top_k_roles: int = 12,
    count_top_k: int = 2,
    include_confidence: bool = True,
    include_spans: bool = True,
    entity_threshold: float | None = None,
    relation_pair_cap: int = 128,
    max_edges_per_type: int = 256,
    rescue_per_role: int | None = None,
    entity_weight: float = 1.0,
    role_weight: float = 1.0,
    count_weight: float = 1.0,
    threshold: float | None = None,
    overlap_policy: str | None = None,
) -> dict | list:
    """Decode span or boundary scores into entities and relations.

    ``architecture`` is ``"span"`` or ``"boundary"``. ``optimizer`` is
    ``"beam"`` or ``"greedy"``. Processor calls omit ``architecture`` and pass
    metadata plus ``threshold`` and ``overlap_policy``; ``exact`` uses beam.
    """
    if architecture is None:
        return _batch_joint(
            span_or_boundary_scores,
            schema,
            threshold=0.5 if threshold is None else threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            optimizer=optimizer,
            overlap_policy=overlap_policy,
        )
    config = JointIEConfig(
        optimizer=optimizer,
        beam_size=beam_size,
        candidate_threshold=candidate_threshold,
        relation_role_threshold=relation_role_threshold,
        top_k_entities=top_k_entities,
        top_k_roles=top_k_roles,
        count_top_k=count_top_k,
        include_confidence=include_confidence,
        include_spans=include_spans,
        entity_threshold=entity_threshold,
        relation_pair_cap=relation_pair_cap,
        max_edges_per_type=max_edges_per_type,
        rescue_per_role=rescue_per_role,
        entity_weight=entity_weight,
        role_weight=role_weight,
        count_weight=count_weight,
    )
    return _decode_joint_one(
        span_or_boundary_scores,
        schema,
        architecture,
        text=text,
        config=config,
        overlap_policy=overlap_policy,
    )
