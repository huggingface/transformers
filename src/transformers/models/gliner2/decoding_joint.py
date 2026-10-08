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
"""Inference-only joint entity and relation decoder."""

from __future__ import annotations

import bisect
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from itertools import product
from types import SimpleNamespace
from typing import Any

import torch


logger = logging.getLogger(__name__)

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


def _number(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def sigmoid(logit: Any) -> float:
    """Numerically stable scalar sigmoid."""
    value = _number(logit)
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


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


def center_logits(logits: Any, threshold: float = 0.5) -> Any:
    """Center nested sequences or tensors without requiring a specific backend."""
    offset = probability_to_logit(threshold)
    try:
        return logits - offset
    except (TypeError, AttributeError):
        if isinstance(logits, (list, tuple)):
            values = [center_logits(value, threshold) for value in logits]
            return type(logits)(values)
        return float(logits) - offset


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


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


class Calibrator(ABC):
    """Transform applied to logits before sigmoid."""

    @abstractmethod
    def calibrate(self, logits: Any) -> Any:
        """Return calibrated logits, preserving scalar or container shape."""

    def __call__(self, logits: Any) -> Any:
        return self.calibrate(logits)


@dataclass(frozen=True)
class IdentityCalibrator(Calibrator):
    """Leave logits unchanged."""

    def calibrate(self, logits: Any) -> Any:
        return logits


@dataclass(frozen=True)
class TemperatureCalibrator(Calibrator):
    """Divide logits by a positive temperature."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")

    def calibrate(self, logits: Any) -> Any:
        try:
            return logits / self.temperature
        except TypeError:
            if isinstance(logits, tuple):
                return tuple(self.calibrate(value) for value in logits)
            if isinstance(logits, list):
                return [self.calibrate(value) for value in logits]
            if isinstance(logits, dict):
                return {key: self.calibrate(value) for key, value in logits.items()}
            raise


@dataclass(frozen=True, order=True)
class SpanRef:
    """Half-open token span and its character projection."""

    start: int
    end: int
    char_start: int
    char_end: int
    text: str
    sentence_id: int | None = None

    def __post_init__(self) -> None:
        if min(self.start, self.end, self.char_start, self.char_end) < 0:
            raise ValueError("span offsets must be non-negative")
        if self.end < self.start or self.char_end < self.char_start:
            raise ValueError("span end must not precede start")

    @property
    def token_start(self) -> int:
        return self.start

    @property
    def token_end(self) -> int:
        return self.end


@dataclass
class ScoreBlock:
    """Named dense logit block. ``scores[label][candidate]`` is container-agnostic."""

    scores: Mapping[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.scores)

    def logits(self, label: str) -> Any:
        return self.scores[label]


@dataclass
class JointScoreLattice:
    """Per-span entity scores and directed relation-role scores."""

    spans: Sequence[SpanRef]
    entity_scores: Any = field(default_factory=ScoreBlock)
    head_scores: Any = field(default_factory=ScoreBlock)
    tail_scores: Any = field(default_factory=ScoreBlock)
    calibrator: Calibrator = field(default_factory=IdentityCalibrator)

    def __post_init__(self) -> None:
        self.spans = tuple(self.spans)
        self.entity_scores = self._block(self.entity_scores)
        self.head_scores = self._block(self.head_scores)
        self.tail_scores = self._block(self.tail_scores)

    @staticmethod
    def _block(value: Any) -> ScoreBlock:
        if isinstance(value, ScoreBlock):
            return value
        return ScoreBlock(value or {})

    def probability(self, logit: Any) -> float:
        return sigmoid(self.calibrator.calibrate(logit))

    def top_entities(self, span: Any | None = None, k: int | None = None) -> list[tuple[Any, ...]]:
        """Rank entity scores. Ties break on span coordinates, then label."""
        if span is not None:
            index = self._span_index(span)
            rows = [
                (label, self.probability(self._at(values, index)))
                for label, values in self.entity_scores.scores.items()
            ]
            rows.sort(key=lambda row: (-row[1], row[0]))
        else:
            rows = [
                (candidate, label, self.probability(self._at(values, index)))
                for label, values in self.entity_scores.scores.items()
                for index, candidate in enumerate(self.spans)
            ]
            rows.sort(
                key=lambda row: (
                    -row[2],
                    row[0].start,
                    row[0].end,
                    row[0].char_start,
                    row[0].char_end,
                    row[1],
                )
            )
        return rows if k is None else rows[: max(0, k)]

    def top_heads(self, relation: str, tail: Any | None = None, k: int | None = None):
        return self._top_role(self.head_scores, relation, tail, k)

    def top_tails(self, relation: str, head: Any | None = None, k: int | None = None):
        return self._top_role(self.tail_scores, relation, head, k)

    def _top_role(self, block: ScoreBlock, relation: str, other: Any | None, k: int | None):
        values = block.scores[relation]
        if other is not None:
            other_index = self._span_index(other)
            rows = [
                (span, self.probability(self._matrix_at(values, i, other_index))) for i, span in enumerate(self.spans)
            ]
        else:
            rows = [(span, self.probability(self._at(values, i))) for i, span in enumerate(self.spans)]
        rows.sort(
            key=lambda row: (
                -row[1],
                row[0].start,
                row[0].end,
                row[0].char_start,
                row[0].char_end,
                row[0].text,
            )
        )
        return rows if k is None else rows[: max(0, k)]

    def _span_index(self, span: Any) -> int:
        if isinstance(span, int):
            if span < 0 or span >= len(self.spans):
                raise IndexError(span)
            return span
        return self.spans.index(span)

    @staticmethod
    def _at(values: Any, index: int) -> Any:
        return values[index]

    @staticmethod
    def _matrix_at(values: Any, row: int, column: int) -> Any:
        value = values[row]
        try:
            return value[column]
        except (IndexError, TypeError):
            return value


@dataclass
class CountHypothesis:
    """One count alternative and its role logits."""

    count: int
    logit: float
    probability: float
    role_logits: Any
    role_probabilities: Any = None


@dataclass
class TaskLattice:
    """Dense scores for one entity or relation task."""

    name: str
    task_type: str
    roles: tuple[str, ...]
    count_hypotheses: list[CountHypothesis]
    schema_tokens: tuple[str, ...] = ()


@dataclass
class ScoreLattice:
    """Dense span scores plus caller-coordinate metadata."""

    text: str
    text_tokens: tuple[str, ...] = ()
    start_mappings: tuple[int, ...] = ()
    end_mappings: tuple[int, ...] = ()
    span_starts: Any = None
    span_ends: Any = None
    valid_span_mask: Any = None
    tasks: list[TaskLattice] = field(default_factory=list)
    schema: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.start_mappings)

    def _span(self, start: int, end_inclusive: int) -> SpanRef:
        char_start = self.start_mappings[start]
        char_end = self.end_mappings[end_inclusive]
        return SpanRef(start, end_inclusive + 1, char_start, char_end, self.text[char_start:char_end])

    def _task(self, name: str, task_type: str) -> TaskLattice:
        for task in self.tasks:
            if task.name == name and task.task_type == task_type:
                return task
        raise KeyError(name)

    def top_entities(self, entity_type: str, k: int | None = None):
        task = next((item for item in self.tasks if item.task_type == "entities"), None)
        if task is None or entity_type not in task.roles:
            return []
        role = task.roles.index(entity_type)
        values = task.count_hypotheses[0].role_probabilities[0, role]
        rows = [
            (self._span(i, i + w), float(values[i, w]))
            for i in range(values.shape[0])
            for w in range(values.shape[1])
            if bool(self.valid_span_mask[i, w])
        ]
        rows.sort(key=lambda row: (-row[1], row[0].start, row[0].end))
        return rows if k is None else rows[: max(0, k)]

    def _top_role(self, relation: str, slot: int, role: int, k: int | None):
        task = self._task(relation, "relations")
        rows = []
        for hypothesis in task.count_hypotheses:
            if slot >= hypothesis.count or slot >= hypothesis.role_probabilities.shape[0]:
                continue
            values = hypothesis.role_probabilities[slot, role]
            rows.extend(
                (self._span(i, i + w), float(values[i, w]), hypothesis.count, hypothesis.probability)
                for i in range(values.shape[0])
                for w in range(values.shape[1])
                if bool(self.valid_span_mask[i, w])
            )
        rows.sort(key=lambda row: (-row[1], -row[3], row[0].start, row[0].end, row[2]))
        return rows if k is None else rows[: max(0, k)]

    def top_heads(self, relation: str, slot: int = 0, k: int | None = None):
        return self._top_role(relation, slot, 0, k)

    def top_tails(self, relation: str, slot: int = 0, k: int | None = None):
        return self._top_role(relation, slot, 1, k)


def field_names_from_tokens(schema_tokens: Sequence[str]) -> tuple[str, ...]:
    """Recover ``[E]``, ``[C]``, and ``[R]`` field names from encoded tokens."""
    markers = {"[E]", "[C]", "[R]"}
    return tuple(
        schema_tokens[index + 1] for index in range(len(schema_tokens) - 1) if schema_tokens[index] in markers
    )


def task_name_from_tokens(schema_tokens: Sequence[str], fallback: str) -> str:
    """Read the task name that precedes an optional description marker."""
    if len(schema_tokens) > 2:
        return schema_tokens[2].split(" [DESCRIPTION] ", 1)[0]
    return fallback


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
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("node spans must be non-empty and non-negative")
        if self.probability is None:
            object.__setattr__(self, "probability", sigmoid(self.score))
        if not isinstance(self.source, CandidateSource):
            object.__setattr__(self, "source", CandidateSource(self.source))
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", self.key)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.entity_type, self.start, self.end)

    @property
    def utility(self) -> float:
        return self.score


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
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", self.key)

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.relation_type, self.head, self.tail, self.slot, self.count_alternative)

    @property
    def utility(self) -> float:
        return self.score

    @property
    def exclusion_keys(self) -> tuple[Hashable, ...]:
        if self.slot is None:
            return ()
        return (("slot", self.hypothesis, self.count_alternative, self.slot),)

    @property
    def count_choice(self) -> tuple[Hashable, Hashable] | None:
        if self.count_alternative is None or self.hypothesis is None:
            return None
        return (self.hypothesis, self.count_alternative)


@dataclass(frozen=True)
class RelationHypothesis:
    """One relation lattice with shape ``[count_slots, 2, L, W]``."""

    relation_type: str
    role_logits: Any
    head_types: Sequence[str]
    tail_types: Sequence[str]
    threshold: float | None = None
    candidate_threshold: float | None = None
    count_probability: float = 1.0
    count_utility: float = 0.0
    count_alternative: Hashable | None = None
    hypothesis_id: Hashable | None = None


@dataclass(frozen=True)
class JointProblem:
    """Nodes, edges, and hard constraints for one document."""

    nodes: tuple[NodeCandidate, ...]
    edges: tuple[EdgeCandidate, ...]
    constraints: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        ids = [node.candidate_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node candidate ids must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.head not in known or edge.tail not in known:
                raise ValueError("edge endpoints must refer to nodes in the problem")

    @property
    def node_by_id(self) -> dict[Hashable, NodeCandidate]:
        return {node.candidate_id: node for node in self.nodes}


class CandidateBuilder:
    """Build a bounded joint problem. Overlap is left to the optimizer."""

    def __init__(
        self,
        *,
        candidate_threshold: float = 0.05,
        relation_role_threshold: float = 0.05,
        top_k_entities: int = 32,
        top_k_roles: int = 12,
        count_top_k: int = 2,
        entity_weight: float = 1.0,
        role_weight: float = 1.0,
        count_weight: float = 1.0,
        entity_threshold: float | None = None,
        max_nodes_per_type: int | None = None,
        relation_role_cap: int | None = None,
        relation_pair_cap: int = 128,
        max_edges_per_type: int = 256,
        rescue_per_role: int | None = None,
    ) -> None:
        max_nodes_per_type = top_k_entities if max_nodes_per_type is None else max_nodes_per_type
        relation_role_cap = top_k_roles if relation_role_cap is None else relation_role_cap
        entity_threshold = candidate_threshold if entity_threshold is None else entity_threshold
        for name, value in (
            ("max_nodes_per_type", max_nodes_per_type),
            ("relation_role_cap", relation_role_cap),
            ("relation_pair_cap", relation_pair_cap),
            ("max_edges_per_type", max_edges_per_type),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.entity_threshold = entity_threshold
        self.relation_role_threshold = relation_role_threshold
        self.count_top_k = count_top_k
        self.entity_weight, self.role_weight, self.count_weight = entity_weight, role_weight, count_weight
        self.max_nodes_per_type = max_nodes_per_type
        self.relation_role_cap = relation_role_cap
        self.relation_pair_cap = relation_pair_cap
        self.max_edges_per_type = max_edges_per_type
        self.rescue_per_role = relation_role_cap if rescue_per_role is None else rescue_per_role

    @staticmethod
    def _span_entries(lattice: Any) -> list[tuple[float, int, int]]:
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

    @staticmethod
    def _node_rank(node: NodeCandidate) -> tuple[Any, ...]:
        source_rank = 0 if node.source == CandidateSource.ENTITY else 1
        return (-node.score, node.start, node.end, node.entity_type, source_rank)

    @staticmethod
    def _edge_rank(edge: EdgeCandidate) -> tuple[Any, ...]:
        return (
            -edge.score,
            str(edge.hypothesis),
            str(edge.slot),
            str(edge.count_alternative),
            str(edge.head),
            str(edge.tail),
            edge.relation_type,
        )

    def build(
        self,
        entity_logits: Any,
        entity_types: Sequence[str],
        relation_hypotheses: Sequence[Any] = (),
        *,
        entity_thresholds: Mapping[str, float | None] | None = None,
        entity_candidate_thresholds: Mapping[str, float | None] | None = None,
        entity_max_candidates: Mapping[str, int] | None = None,
        constraints: Iterable[Any] = (),
    ) -> JointProblem:
        shape = _shape(entity_logits)
        if len(shape) != 3 or shape[0] != len(entity_types):
            raise ValueError(f"entity logits must have shape [types, L, W], got {shape} for {len(entity_types)} types")
        thresholds = dict(entity_thresholds or {})
        candidate_thresholds = dict(entity_candidate_thresholds or {})
        maxima = dict(entity_max_candidates or {})
        node_map: dict[tuple[str, int, int], NodeCandidate] = {}
        raw_entity_scores: dict[tuple[str, int, int], tuple[float, float]] = {}
        for type_index, entity_type in enumerate(entity_types):
            threshold = thresholds.get(entity_type) or 0.5
            candidate_threshold = candidate_thresholds.get(entity_type)
            if candidate_threshold is None:
                candidate_threshold = self.entity_threshold
            candidates: list[NodeCandidate] = []
            for raw, start, end in self._span_entries(entity_logits[type_index]):
                if not math.isfinite(raw):
                    continue
                score = center_logit(raw, threshold) * self.entity_weight
                probability = sigmoid(raw)
                raw_entity_scores[(entity_type, start, end)] = (score, probability)
                if probability >= candidate_threshold:
                    candidates.append(NodeCandidate(entity_type, start, end, score, probability))
            cap = maxima.get(entity_type, self.max_nodes_per_type)
            for node in sorted(candidates, key=self._node_rank)[:cap]:
                node_map[node.key] = node
        edge_groups: dict[str, list[EdgeCandidate]] = {}
        for index, value in enumerate(relation_hypotheses):
            hypothesis = value if isinstance(value, RelationHypothesis) else RelationHypothesis(**value)
            role_shape = _shape(hypothesis.role_logits)
            if len(role_shape) != 4 or role_shape[1] != 2:
                raise ValueError(f"relation role logits must have shape [count_slots, 2, L, W], got {role_shape}")
            hypothesis_id = hypothesis.hypothesis_id
            if hypothesis_id is None:
                hypothesis_id = (hypothesis.relation_type, index)
            final_threshold = hypothesis.threshold if hypothesis.threshold is not None else 0.5
            candidate_threshold = hypothesis.candidate_threshold
            if candidate_threshold is None:
                candidate_threshold = self.relation_role_threshold
            relation_offset = probability_to_logit(final_threshold)
            for count_slot in range(role_shape[0]):
                role_options: list[list[tuple[float, int, int]]] = []
                for role in range(2):
                    entries = [
                        (raw, start, end)
                        for raw, start, end in self._span_entries(hypothesis.role_logits[count_slot][role])
                        if math.isfinite(raw) and sigmoid(raw) >= candidate_threshold
                    ]
                    entries.sort(key=lambda item: (-item[0], item[1], item[2]))
                    role_options.append(entries[: self.relation_role_cap])
                typed_roles: list[list] = [[], []]
                for role, types in enumerate((hypothesis.head_types, hypothesis.tail_types)):
                    rescued = 0
                    for raw_role_score, start, end in role_options[role]:
                        for entity_type in types:
                            key = (entity_type, start, end)
                            node = node_map.get(key)
                            if node is None:
                                if rescued >= self.rescue_per_role:
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
                                (
                                    (raw_role_score - relation_offset) * self.role_weight,
                                    node,
                                    sigmoid(raw_role_score),
                                )
                            )
                pairs = []
                for (head_score, head, head_prob), (tail_score, tail, tail_prob) in product(*typed_roles):
                    pairs.append(
                        (
                            head_score + tail_score + hypothesis.count_utility * self.count_weight,
                            head,
                            tail,
                            head_prob,
                            tail_prob,
                        )
                    )
                pairs.sort(key=lambda item: (-item[0], str(item[1].candidate_id), str(item[2].candidate_id)))
                for score, head, tail, head_prob, tail_prob in pairs[: self.relation_pair_cap]:
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
        edges: list[EdgeCandidate] = []
        for relation_type in sorted(edge_groups):
            unique: dict[tuple[Any, ...], EdgeCandidate] = {}
            for edge in sorted(edge_groups[relation_type], key=self._edge_rank):
                previous = unique.get(edge.key)
                if previous is None or edge.score > previous.score:
                    unique[edge.key] = edge
            edges.extend(sorted(unique.values(), key=self._edge_rank)[: self.max_edges_per_type])
        nodes = tuple(
            sorted(
                node_map.values(),
                key=lambda node: (
                    node.entity_type,
                    node.start,
                    node.end,
                    -node.score,
                ),
            )
        )
        return JointProblem(nodes, tuple(sorted(edges, key=self._edge_rank)), tuple(constraints))


class ProblemBuilder(CandidateBuilder):
    """Alias for :class:`CandidateBuilder`."""


def _label(value: Any) -> Any:
    return _get(value, "label", "type", "name", "entity_type", "relation", "relation_type")


def _endpoint(relation: Any, side: str) -> Any:
    return _get(relation, side, f"{side}_entity")


def _identity(value: Any) -> Any:
    if value is None:
        return None
    identifier = _get(value, "id", "entity_id", "candidate_id", "index")
    if identifier is not None:
        return identifier
    start, end = _get(value, "start"), _get(value, "end")
    if start is not None and end is not None:
        return (start, end, _label(value))
    text = _get(value, "text", "value")
    if text is not None:
        return (text, _label(value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _relation_key(value: Any) -> tuple:
    return (_label(value), _identity(_endpoint(value, "head")), _identity(_endpoint(value, "tail")))


def _matches(relation: Any, relation_type: str | None) -> bool:
    return relation_type is None or _label(relation) == relation_type


class Constraint(ABC):
    """Incremental constraint checked against already accepted output."""

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def allow_node(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        return True

    def allow_edge(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        return self.allows(candidate, edges, nodes)

    def penalty_edge(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> float:
        return 0.0

    def __call__(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return self.allows(candidate, relations, entities)

    def apply(self, relations: Iterable[Any], entities: Sequence[Any] = ()) -> list:
        accepted: list = []
        for candidate in relations:
            if self.allows(candidate, accepted, entities):
                accepted.append(candidate)
        return accepted

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        accepted: list = []
        for candidate in relations:
            if not self.allows(candidate, accepted, entities):
                return False
            accepted.append(candidate)
        return True

    def to_dict(self) -> dict:
        data = {"type": type(self).__name__}
        data.update(self.__dict__)
        return data


@dataclass(frozen=True)
class TypedEndpoints(Constraint):
    """Head and tail types must belong to the declared sets."""

    relation: str | None = None
    head_types: tuple = ()
    tail_types: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_types", tuple(self.head_types))
        object.__setattr__(self, "tail_types", tuple(self.tail_types))

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _label(_endpoint(candidate, "head")), _label(_endpoint(candidate, "tail"))
        return (not self.head_types or head in self.head_types) and (not self.tail_types or tail in self.tail_types)


@dataclass(frozen=True)
class NoSelfLoops(Constraint):
    """Reject an edge whose endpoints are the same entity."""

    relation: str | None = None

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        return _identity(_endpoint(candidate, "head")) != _identity(_endpoint(candidate, "tail"))


@dataclass(frozen=True)
class UniqueRelationPair(Constraint):
    """One edge per endpoint pair. Undirected relations also block the reverse."""

    relation: str | None = None
    directed: bool = True

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _identity(_endpoint(candidate, "head")), _identity(_endpoint(candidate, "tail"))
        for existing in relations:
            if _label(existing) != _label(candidate):
                continue
            old_head = _identity(_endpoint(existing, "head"))
            old_tail = _identity(_endpoint(existing, "tail"))
            if (head, tail) == (old_head, old_tail) or (not self.directed and (head, tail) == (old_tail, old_head)):
                return False
        return True


@dataclass(frozen=True)
class UniqueRelationSlot(Constraint):
    """One edge per head, tail, or count slot."""

    relation: str | None = None
    slot: str = "head"

    def __post_init__(self) -> None:
        if self.slot not in {"head", "tail", "slot"}:
            raise ValueError("slot must be 'head', 'tail', or 'slot'")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        value = _identity(_endpoint(candidate, self.slot))
        return all(
            _label(old) != _label(candidate) or _identity(_endpoint(old, self.slot)) != value for old in relations
        )


@dataclass(frozen=True)
class EntityOverlapPolicy(Constraint):
    """Greedy span overlap policy: allow, disallow, or nested."""

    policy: str = "disallow"

    def __post_init__(self) -> None:
        if self.policy not in {"allow", "disallow", "nested"}:
            raise ValueError("policy must be 'allow', 'disallow', or 'nested'")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def allow_node(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        if self.policy == "allow":
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
            if self.policy == "disallow" or not nested:
                return False
        return True

    def apply_entities(self, entities: Iterable[Any]) -> list:
        if self.policy == "allow":
            return list(entities)
        accepted: list = []
        for candidate in entities:
            start, end = _get(candidate, "start"), _get(candidate, "end")
            if start is None or end is None:
                accepted.append(candidate)
                continue
            valid = True
            for old in accepted:
                old_start, old_end = _get(old, "start"), _get(old, "end")
                if old_start is None or old_end is None or end <= old_start or old_end <= start:
                    continue
                nested = (start >= old_start and end <= old_end) or (old_start >= start and old_end <= end)
                if self.policy == "disallow" or not nested:
                    valid = False
                    break
            if valid:
                accepted.append(candidate)
        return accepted


@dataclass(frozen=True)
class MaxRelationsPerHead(Constraint):
    """Cap edges leaving one head."""

    limit: int
    relation: str | None = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        key = _identity(_endpoint(candidate, "head"))
        used = sum(_matches(old, self.relation) and _identity(_endpoint(old, "head")) == key for old in relations)
        return used < self.limit


@dataclass(frozen=True)
class MaxRelationsPerTail(Constraint):
    """Cap edges entering one tail."""

    limit: int
    relation: str | None = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        key = _identity(_endpoint(candidate, "tail"))
        used = sum(_matches(old, self.relation) and _identity(_endpoint(old, "tail")) == key for old in relations)
        return used < self.limit


@dataclass(frozen=True)
class SymmetricRelation(Constraint):
    """A relation is present in both directions."""

    relation: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        keys = {_relation_key(item) for item in relations}
        return all(
            _label(item) != self.relation
            or (self.relation, _identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head"))) in keys
            for item in relations
        )


@dataclass(frozen=True)
class InverseRelation(Constraint):
    """Each direction requires its inverse."""

    relation: str
    inverse: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        keys = {_relation_key(item) for item in relations}
        for item in relations:
            label = _label(item)
            reverse = (_identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head")))
            if label == self.relation and (self.inverse, *reverse) not in keys:
                return False
            if label == self.inverse and (self.relation, *reverse) not in keys:
                return False
        return True


@dataclass(frozen=True)
class AcyclicRelation(Constraint):
    """The relation graph has no directed cycle."""

    relation: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _identity(_endpoint(candidate, "head")), _identity(_endpoint(candidate, "tail"))
        if head == tail:
            return False
        graph: dict = defaultdict(list)
        for old in relations:
            if _matches(old, self.relation):
                graph[_identity(_endpoint(old, "head"))].append(_identity(_endpoint(old, "tail")))
        stack, seen = [tail], set()
        while stack:
            node = stack.pop()
            if node == head:
                return False
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node])
        return True


_CONSTRAINT_TYPES = {
    cls.__name__: cls
    for cls in (
        TypedEndpoints,
        NoSelfLoops,
        UniqueRelationPair,
        UniqueRelationSlot,
        EntityOverlapPolicy,
        MaxRelationsPerHead,
        MaxRelationsPerTail,
        SymmetricRelation,
        InverseRelation,
        AcyclicRelation,
    )
}


def constraint_from_dict(data: Mapping[str, Any]) -> Constraint:
    """Rebuild a joint constraint from its ``to_dict`` payload."""
    values = dict(data)
    kind = values.pop("type", None)
    try:
        cls = _CONSTRAINT_TYPES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown constraint type {kind!r}") from exc
    return cls(**values)


def _name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} name must be a non-empty string")
    return value


def _types(value, side: str) -> tuple:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"relation {side} must contain at least one entity type")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"relation {side} contains an invalid entity type")
    if len(set(values)) != len(values):
        raise ValueError(f"relation {side} entity types must be unique")
    return values


def _prob(value: float | None, field_name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be in [0, 1]")


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
        _name(self.name, "entity")
        _prob(self.threshold, "threshold")
        _prob(self.candidate_threshold, "candidate_threshold")
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
        _name(self.name, "relation")
        object.__setattr__(self, "head", _types(self.head, "head"))
        object.__setattr__(self, "tail", _types(self.tail, "tail"))
        _prob(self.threshold, "threshold")
        _prob(self.candidate_threshold, "candidate_threshold")
        if self.inverse is not None:
            _name(self.inverse, "inverse relation")
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
            name,
            description,
            threshold,
            candidate_threshold,
            max_candidates,
            allow_nested,
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
            _types(head, "head"),
            _types(tail, "tail"),
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
        if not isinstance(constraint, Constraint):
            raise TypeError("constraint must implement Constraint")
        self._constraints.append(constraint)
        return self

    def _validate_relation_name(self, name):
        if name is not None and name not in self._relations:
            raise ValueError(f"unknown relation {name!r}")

    def no_self_loops(self, relation=None):
        self._validate_relation_name(relation)
        return self.constraint(NoSelfLoops(relation))

    def acyclic(self, relation):
        self._validate_relation_name(relation)
        return self.constraint(AcyclicRelation(relation))

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
            self.constraint(MaxRelationsPerHead(per_head, relation))
        if per_tail is not None:
            self.constraint(MaxRelationsPerTail(per_tail, relation))
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
            "constraints": [item.to_dict() for item in self.constraints],
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
            schema.constraint(constraint_from_dict(value))
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


def _add(values, constraint):
    if constraint not in values:
        values.append(constraint)


def compile_schema(schema) -> CompiledJointSchema:
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
        _add(constraints, EntityOverlapPolicy(policy))
    for spec in relations.values():
        _add(constraints, TypedEndpoints(spec.name, spec.head, spec.tail))
        if not spec.allow_self:
            _add(constraints, NoSelfLoops(spec.name))
        _add(constraints, UniqueRelationPair(spec.name, directed=spec.directed))
        _add(constraints, UniqueRelationSlot(spec.name, "slot"))
        if spec.max_per_head is not None:
            _add(constraints, MaxRelationsPerHead(spec.max_per_head, spec.name))
        if spec.max_per_tail is not None:
            _add(constraints, MaxRelationsPerTail(spec.max_per_tail, spec.name))
        if spec.symmetric:
            _add(constraints, SymmetricRelation(spec.name))
        if spec.inverse:
            if spec.inverse not in relations:
                raise ValueError(f"relation {spec.name!r} has unknown inverse {spec.inverse!r}")
            other = relations[spec.inverse]
            if set(spec.head) != set(other.tail) or set(spec.tail) != set(other.head):
                raise ValueError(f"inverse endpoint types for {spec.name!r} and {spec.inverse!r} are incompatible")
            _add(constraints, InverseRelation(spec.name, spec.inverse))
    return CompiledJointSchema(
        model,
        entities,
        relations,
        tuple(constraints),
        tuple(entities),
        tuple(relations),
    )


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
    def key(self) -> tuple[str, int, int]:
        return (self.entity_type, self.start, self.end)


@dataclass(frozen=True)
class RelationRoleScore:
    """A mention's compatibility with one role of one relation."""

    relation_type: str
    role: str
    mention_id: Hashable
    logit: float
    probability: float


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


@dataclass
class CandidateScoreSet:
    """Sparse mention and edge scores for one text."""

    text: str
    mentions: tuple[MentionScore, ...]
    relation_roles: tuple[RelationRoleScore, ...] = ()
    edges: tuple[ScoredRelationEdge, ...] = ()
    classifications: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    text_tokens: tuple[str, ...] = ()
    start_mappings: tuple[int, ...] = ()
    end_mappings: tuple[int, ...] = ()


def score_lattice_to_candidate_score_set(lattice: Any) -> CandidateScoreSet:
    """Convert a dense span lattice into sparse mention scores."""
    span_starts = lattice.span_starts
    span_ends = lattice.span_ends
    valid = lattice.valid_span_mask
    mentions: list[MentionScore] = []
    query_id = 0
    for task in lattice.tasks:
        if task.task_type != "entities" or not task.count_hypotheses:
            continue
        hyp = task.count_hypotheses[0]
        role_logits = hyp.role_logits[0]
        role_probs = hyp.role_probabilities[0]
        num_types = role_logits.shape[0]
        for type_index in range(num_types):
            entity_type = task.roles[type_index] if type_index < len(task.roles) else str(type_index)
            length = role_logits.shape[1]
            width = role_logits.shape[2]
            for row in range(length):
                for col in range(width):
                    if not bool(valid[row, col]):
                        continue
                    mentions.append(
                        MentionScore(
                            query_id=query_id,
                            entity_type=entity_type,
                            start=int(span_starts[row, col]),
                            end=int(span_ends[row, col]) + 1,
                            logit=float(role_logits[type_index, row, col]),
                            probability=float(role_probs[type_index, row, col]),
                        )
                    )
            query_id += 1
    return CandidateScoreSet(
        text=lattice.text,
        mentions=tuple(mentions),
        text_tokens=tuple(getattr(lattice, "text_tokens", ())),
        start_mappings=tuple(getattr(lattice, "start_mappings", ())),
        end_mappings=tuple(getattr(lattice, "end_mappings", ())),
    )


def boundary_candidates_to_candidate_score_set(
    text: str,
    candidates: Any,
    query_specs: Sequence[Any],
    *,
    sample_index: int = 0,
    token_offset: int = 0,
    text_length: int | None = None,
    pair_temperature: float = 1.0,
    entity_thresholds: Mapping[str, float | None] | None = None,
    entity_candidate_thresholds: Mapping[str, float | None] | None = None,
    extra_mentions: Sequence[MentionScore] = (),
    edges: Sequence[ScoredRelationEdge] = (),
    text_tokens: Sequence[str] = (),
    start_mappings: Sequence[int] = (),
    end_mappings: Sequence[int] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CandidateScoreSet:
    """Convert one boundary candidate row into sparse joint scores."""
    if pair_temperature <= 0:
        raise ValueError("pair_temperature must be positive")
    if text_length is None:
        text_length = len(start_mappings)
    thresholds = dict(entity_thresholds or {})
    candidate_thresholds = dict(entity_candidate_thresholds or {})
    best: dict = {mention.key: mention for mention in extra_mentions}
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
        candidate_ids = valid.nonzero(as_tuple=False).flatten().tolist()
        for candidate_id in candidate_ids:
            start = int(indices[sample_index, query_id, candidate_id, 0]) - token_offset
            end = int(indices[sample_index, query_id, candidate_id, 1]) - token_offset
            if not (0 <= start < end <= int(text_length)):
                continue
            logit = float(pair_logits[sample_index, query_id, candidate_id].detach().float()) / pair_temperature
            mention = MentionScore(
                query_id=query_id,
                entity_type=entity_type,
                start=start,
                end=end,
                logit=logit,
                probability=sigmoid(logit),
                threshold=threshold,
                candidate_threshold=candidate_thresholds.get(entity_type),
            )
            previous = best.get(mention.key)
            if previous is None or mention.logit > previous.logit:
                best[mention.key] = mention
    mentions = tuple(
        sorted(
            best.values(),
            key=lambda item: (
                item.entity_type,
                item.start,
                item.end,
                -item.logit,
            ),
        )
    )
    return CandidateScoreSet(
        text=text,
        mentions=mentions,
        edges=tuple(edges),
        metadata=dict(metadata or {}),
        text_tokens=tuple(text_tokens),
        start_mappings=tuple(start_mappings),
        end_mappings=tuple(end_mappings),
    )


def candidate_score_set_to_problem(
    score_set: CandidateScoreSet,
    edges: Sequence[ScoredRelationEdge] | None = None,
    *,
    mention_threshold: float = 0.5,
    constraints: Sequence[Any] = (),
    decision_threshold: float = 0.5,
    max_mentions_per_type: int | None = None,
    max_mentions_by_type: Mapping[str, int] | None = None,
    rescue_relation_endpoints: bool = False,
    edge_candidate_threshold: float = 0.0,
    max_edges_per_type: int | None = None,
    entity_weight: float = 1.0,
    relation_weight: float = 1.0,
) -> JointProblem:
    """Build a joint problem from sparse mention and edge scores."""
    raw_edges = tuple(score_set.edges if edges is None else edges)
    edge_by_key: dict = {}
    for edge in raw_edges:
        threshold = edge_candidate_threshold if edge.candidate_threshold is None else edge.candidate_threshold
        if edge.probability < threshold:
            continue
        key = (edge.relation_type, edge.head, edge.tail)
        previous = edge_by_key.get(key)
        if previous is None or edge.logit > previous.logit:
            edge_by_key[key] = edge
    edge_counts: dict = {}
    retained_edges: list[ScoredRelationEdge] = []
    for edge in sorted(
        edge_by_key.values(),
        key=lambda item: (
            item.relation_type,
            -item.logit,
            str(item.head),
            str(item.tail),
        ),
    ):
        if max_edges_per_type is not None and edge_counts.get(edge.relation_type, 0) >= max_edges_per_type:
            continue
        retained_edges.append(edge)
        edge_counts[edge.relation_type] = edge_counts.get(edge.relation_type, 0) + 1
    relation_edges = tuple(retained_edges)
    rescue_ids = (
        {endpoint for edge in relation_edges for endpoint in (edge.head, edge.tail)}
        if rescue_relation_endpoints
        else set()
    )
    selected_mentions: list[MentionScore] = []
    per_type: dict = {}
    type_limits = dict(max_mentions_by_type or {})
    for mention in sorted(
        score_set.mentions,
        key=lambda item: (
            item.entity_type,
            -item.probability,
            item.start,
            item.end,
        ),
    ):
        candidate_threshold = mention_threshold if mention.candidate_threshold is None else mention.candidate_threshold
        if mention.probability < candidate_threshold and mention.key not in rescue_ids:
            continue
        type_limit = type_limits.get(mention.entity_type, max_mentions_per_type)
        if (
            type_limit is not None
            and per_type.get(mention.entity_type, 0) >= type_limit
            and mention.key not in rescue_ids
        ):
            continue
        selected_mentions.append(mention)
        per_type[mention.entity_type] = per_type.get(mention.entity_type, 0) + 1
    nodes: list[NodeCandidate] = []
    keep_ids = set()
    for mention in selected_mentions:
        candidate_threshold = mention_threshold if mention.candidate_threshold is None else mention.candidate_threshold
        node = NodeCandidate(
            entity_type=mention.entity_type,
            start=mention.start,
            end=mention.end,
            score=entity_weight
            * center_logit(
                mention.logit,
                mention.threshold if mention.threshold is not None else decision_threshold,
            ),
            probability=mention.probability,
            source=(
                CandidateSource.RELATION_RESCUE
                if mention.key in rescue_ids and mention.probability < candidate_threshold
                else CandidateSource.ENTITY
            ),
            candidate_id=mention.key,
        )
        nodes.append(node)
        keep_ids.add(mention.key)
    edge_cands: list[EdgeCandidate] = []
    for edge_slot, edge in enumerate(relation_edges):
        if edge.head not in keep_ids or edge.tail not in keep_ids:
            continue
        edge_cands.append(
            EdgeCandidate(
                relation_type=edge.relation_type,
                head=edge.head,
                tail=edge.tail,
                score=relation_weight
                * center_logit(
                    edge.logit,
                    edge.threshold if edge.threshold is not None else decision_threshold,
                ),
                head_probability=edge.probability,
                tail_probability=edge.probability,
                slot=edge_slot,
                hypothesis=edge.relation_type,
            )
        )
    return JointProblem(nodes=tuple(nodes), edges=tuple(edge_cands), constraints=tuple(constraints))


@dataclass(frozen=True)
class JointSolution:
    """Selected nodes and edges."""

    nodes: tuple[NodeCandidate, ...]
    edges: tuple[EdgeCandidate, ...]
    score: float
    feasible: bool = True

    @property
    def node_ids(self) -> frozenset[Hashable]:
        return frozenset(node.candidate_id for node in self.nodes)


class BaseOptimizer:
    """Shared constraint dispatch for greedy and beam search."""

    def optimize(self, problem: JointProblem) -> JointSolution:
        raise NotImplementedError

    @staticmethod
    def _invoke(
        method: Any, item: Any, nodes: Sequence[NodeCandidate], edges: Sequence[EdgeCandidate], default: Any
    ) -> Any:
        if method is None:
            return default
        for args in ((item, nodes, edges), (item, nodes), (item,)):
            try:
                return method(*args)
            except TypeError:
                continue
        return method(item, nodes, edges)

    def allow_node(self, problem, node, nodes, edges) -> bool:
        return all(
            bool(self._invoke(getattr(constraint, "allow_node", None), node, nodes, edges, True))
            for constraint in problem.constraints
        )

    @staticmethod
    def _resolve_edge(problem: JointProblem, edge: EdgeCandidate) -> Any:
        node_by_id = problem.node_by_id
        return SimpleNamespace(
            relation_type=edge.relation_type,
            type=edge.relation_type,
            head=node_by_id.get(edge.head, edge.head),
            tail=node_by_id.get(edge.tail, edge.tail),
            slot=edge.slot,
            candidate_id=edge.candidate_id,
        )

    def allow_edge(self, problem, edge, nodes, edges) -> bool:
        candidate = self._resolve_edge(problem, edge)
        accepted = tuple(self._resolve_edge(problem, value) for value in edges)
        return all(
            bool(self._invoke(getattr(constraint, "allow_edge", None), candidate, nodes, accepted, True))
            for constraint in problem.constraints
        )

    def validate_solution(self, problem: JointProblem, solution: JointSolution) -> bool:
        """Recheck hard constraints after derived companion edges are injected."""
        accepted_nodes: list[NodeCandidate] = []
        for node in solution.nodes:
            if not self.allow_node(problem, node, accepted_nodes + [node], solution.edges):
                return False
            accepted_nodes.append(node)
        accepted_edges: list[EdgeCandidate] = []
        for edge in solution.edges:
            if not self.allow_edge(problem, edge, list(solution.nodes), accepted_edges):
                return False
            accepted_edges.append(edge)
        resolved = tuple(self._resolve_edge(problem, edge) for edge in solution.edges)
        for constraint in problem.constraints:
            validate = getattr(constraint, "validate", None)
            if validate is None:
                continue
            try:
                ok = validate(resolved, list(solution.nodes))
            except TypeError:
                ok = validate(resolved)
            if not ok:
                return False
        return True

    def edge_penalty(self, problem, edge, nodes, edges) -> float:
        return sum(
            float(self._invoke(getattr(constraint, "penalty_edge", None), edge, nodes, edges, 0.0))
            for constraint in problem.constraints
        )

    @staticmethod
    def edge_conflicts(edge: EdgeCandidate, used: Iterable[Hashable]) -> bool:
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

    @staticmethod
    def edge_usage(edge: EdgeCandidate) -> frozenset[Hashable]:
        keys = set(edge.exclusion_keys)
        if edge.count_choice is not None:
            keys.add(("count-choice",) + edge.count_choice)
        return frozenset(keys)

    @staticmethod
    def solution(problem, node_ids, edges, score: float) -> JointSolution:
        selected = set(node_ids)
        nodes = tuple(node for node in problem.nodes if node.candidate_id in selected)
        chosen = list(edges)
        keys = {(edge.relation_type, edge.head, edge.tail) for edge in chosen}
        companions = []
        for constraint in problem.constraints:
            if isinstance(constraint, SymmetricRelation):
                pairs = [
                    (constraint.relation, edge.tail, edge.head, edge)
                    for edge in chosen
                    if edge.relation_type == constraint.relation
                ]
            elif isinstance(constraint, InverseRelation):
                pairs = []
                for edge in chosen:
                    if edge.relation_type == constraint.relation:
                        pairs.append((constraint.inverse, edge.tail, edge.head, edge))
                    elif edge.relation_type == constraint.inverse:
                        pairs.append((constraint.relation, edge.tail, edge.head, edge))
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
        chosen_edges = tuple(
            sorted(
                chosen + companions,
                key=lambda edge: (edge.relation_type, str(edge.head), str(edge.tail), edge.derived),
            )
        )
        return JointSolution(nodes, chosen_edges, float(score))


class GreedyOptimizer(BaseOptimizer):
    """Select profitable edges atomically, then independent positive nodes."""

    @staticmethod
    def _rank(edge: EdgeCandidate):
        return (
            -edge.score,
            edge.relation_type,
            str(edge.hypothesis),
            str(edge.slot),
            str(edge.head),
            str(edge.tail),
        )

    def optimize(self, problem: JointProblem) -> JointSolution:
        node_by_id = problem.node_by_id
        selected = set()
        edges: list[EdgeCandidate] = []
        used = set()
        score = 0.0
        ranked = sorted(
            problem.edges,
            key=lambda edge: (
                -(
                    edge.score
                    + node_by_id[edge.head].score
                    + (0.0 if edge.tail == edge.head else node_by_id[edge.tail].score)
                ),
                self._rank(edge),
            ),
        )
        for edge in ranked:
            if self.edge_conflicts(edge, used):
                continue
            new_ids = [value for value in (edge.head, edge.tail) if value not in selected]
            proposed_nodes = [node_by_id[value] for value in new_ids]
            current_nodes = [node_by_id[value] for value in selected]
            if not all(
                self.allow_node(problem, node, current_nodes + proposed_nodes, edges) for node in proposed_nodes
            ):
                continue
            if not self.allow_edge(problem, edge, current_nodes + proposed_nodes, edges):
                continue
            gain = edge.score + sum(node.score for node in proposed_nodes)
            gain -= self.edge_penalty(problem, edge, current_nodes + proposed_nodes, edges)
            if gain < 0.0:
                continue
            selected.update(new_ids)
            edges.append(edge)
            used.update(self.edge_usage(edge))
            score += gain
        for node in sorted(problem.nodes, key=lambda item: (-item.score, item.entity_type, item.start, item.end)):
            if node.candidate_id in selected or node.score <= 0.0:
                continue
            current_nodes = [node_by_id[value] for value in selected]
            if self.allow_node(problem, node, current_nodes + [node], edges):
                selected.add(node.candidate_id)
                score += node.score
        result = self.solution(problem, selected, edges, score)
        if self.validate_solution(problem, result):
            return result
        logger.warning(
            "greedy joint-IE decoding produced a constraint-violating assignment; returning empty solution",
        )
        return replace(self.solution(problem, frozenset(), (), 0.0), feasible=False)


@dataclass(frozen=True)
class _State:
    node_ids: frozenset[Hashable] = frozenset()
    edges: tuple[EdgeCandidate, ...] = ()
    used: frozenset[Hashable] = frozenset()
    score: float = 0.0


class BeamOptimizer(BaseOptimizer):
    """Beam search over edges, then the greedy completion of free nodes."""

    def __init__(self, beam_width: int = 16) -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        self.beam_width = beam_width

    @staticmethod
    def _signature(state: _State):
        return (
            tuple(sorted(map(str, state.node_ids))),
            tuple(str(edge.candidate_id) for edge in state.edges),
        )

    def _finish_nodes(self, problem: JointProblem, state: _State) -> _State:
        node_by_id = problem.node_by_id
        selected = set(state.node_ids)
        score = state.score
        for node in sorted(problem.nodes, key=lambda item: (-item.score, item.entity_type, item.start, item.end)):
            if node.candidate_id in selected or node.score <= 0.0:
                continue
            nodes = [node_by_id[value] for value in selected]
            if self.allow_node(problem, node, nodes + [node], state.edges):
                selected.add(node.candidate_id)
                score += node.score
        return _State(frozenset(selected), state.edges, state.used, score)

    def optimize(self, problem: JointProblem) -> JointSolution:
        node_by_id = problem.node_by_id
        ordered = sorted(
            problem.edges,
            key=lambda edge: (
                -(edge.score + node_by_id[edge.head].score + node_by_id[edge.tail].score),
                edge.relation_type,
                str(edge.hypothesis),
                str(edge.slot),
                str(edge.head),
                str(edge.tail),
            ),
        )
        beam = [_State()]
        for edge in ordered:
            expanded = list(beam)
            for state in beam:
                if self.edge_conflicts(edge, state.used):
                    continue
                new_ids = [value for value in (edge.head, edge.tail) if value not in state.node_ids]
                added_nodes = [node_by_id[value] for value in new_ids]
                current_nodes = [node_by_id[value] for value in state.node_ids]
                proposed_nodes = current_nodes + added_nodes
                if not all(self.allow_node(problem, node, proposed_nodes, state.edges) for node in added_nodes):
                    continue
                if not self.allow_edge(problem, edge, proposed_nodes, state.edges):
                    continue
                gain = edge.score + sum(node.score for node in added_nodes)
                gain -= self.edge_penalty(problem, edge, proposed_nodes, state.edges)
                if gain < 0.0:
                    continue
                expanded.append(
                    _State(
                        state.node_ids.union(new_ids),
                        state.edges + (edge,),
                        state.used.union(self.edge_usage(edge)),
                        state.score + gain,
                    )
                )
            unique = {}
            for state in expanded:
                key = (state.node_ids, frozenset(edge.candidate_id for edge in state.edges), state.used)
                old = unique.get(key)
                if old is None or state.score > old.score:
                    unique[key] = state
            beam = sorted(unique.values(), key=lambda state: (-state.score, self._signature(state)))[: self.beam_width]
        candidates: list[JointSolution] = [
            self.solution(problem, state.node_ids, state.edges, state.score)
            for state in (self._finish_nodes(problem, state) for state in beam)
        ]
        candidates.append(GreedyOptimizer().optimize(problem))
        feasible = [solution for solution in candidates if self.validate_solution(problem, solution)]
        if feasible:
            return max(feasible, key=self._solution_key)
        logger.warning(
            "joint-IE decoding found no constraint-satisfying assignment among %d candidates; "
            "returning empty solution",
            len(candidates),
        )
        return replace(self.solution(problem, frozenset(), (), 0.0), feasible=False)

    def _solution_key(self, solution: JointSolution):
        return (
            solution.score,
            tuple(sorted(map(str, solution.node_ids))),
            tuple(str(edge.candidate_id) for edge in solution.edges),
        )


def _geometric_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("confidence components must be probabilities in [0, 1]")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


@dataclass(frozen=True)
class JointEntity:
    """One selected entity mention."""

    id: str
    type: str
    text: str
    start: int
    end: int
    confidence: float | None = None
    sentence_id: int | None = None
    rescued: bool = False

    @property
    def label(self) -> str:
        return self.type

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self, include_confidence: bool = True, include_spans: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {"id": self.id, "type": self.type, "text": self.text}
        if include_spans:
            value.update(start=self.start, end=self.end)
            if self.sentence_id is not None:
                value["sentence_id"] = self.sentence_id
        if include_confidence and self.confidence is not None:
            value["confidence"] = self.confidence
        if self.rescued:
            value["rescued"] = True
        return value


@dataclass(frozen=True)
class JointRelation:
    """One selected relation."""

    type: str
    head: str
    tail: str
    confidence: float | None = None
    derived: bool = False

    @property
    def label(self) -> str:
        return self.type

    def to_dict(self, include_confidence: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {"type": self.type, "head": self.head, "tail": self.tail}
        if include_confidence and self.confidence is not None:
            value["confidence"] = self.confidence
        if self.derived:
            value["derived"] = True
        return value


@dataclass
class JointResult:
    """Entities and relations for one document."""

    text: str
    entities: list[JointEntity] = field(default_factory=list)
    relations: list[JointRelation] = field(default_factory=list)
    default_include_confidence: bool = field(default=True, repr=False, compare=False)
    default_include_spans: bool = field(default=True, repr=False, compare=False)
    feasible: bool = field(default=True, compare=False)

    def __post_init__(self) -> None:
        self.entities = list(self.entities)
        self.relations = list(self.relations)
        ids = [entity.id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity IDs must be unique")
        known = set(ids)
        if any(rel.head not in known or rel.tail not in known for rel in self.relations):
            raise ValueError("relation endpoints must reference entities in this result")

    def entity(self, entity_id: str) -> JointEntity:
        for value in self.entities:
            if value.id == entity_id:
                return value
        raise KeyError(entity_id)

    get_entity = entity

    def entities_by_type(self, entity_type: str) -> list[JointEntity]:
        return [entity for entity in self.entities if entity.type == entity_type]

    def relations_by_type(self, relation_type: str) -> list[JointRelation]:
        return [relation for relation in self.relations if relation.type == relation_type]

    def outgoing(self, entity: Any, relation_type: str | None = None) -> list[JointRelation]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        return [
            relation
            for relation in self.relations
            if relation.head == entity_id and (relation_type is None or relation.type == relation_type)
        ]

    def incoming(self, entity: Any, relation_type: str | None = None) -> list[JointRelation]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        return [
            relation
            for relation in self.relations
            if relation.tail == entity_id and (relation_type is None or relation.type == relation_type)
        ]

    def neighbors(self, entity: Any, relation_type: str | None = None) -> list[JointEntity]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        ids: list[str] = []
        for relation in self.relations:
            if relation_type is not None and relation.type != relation_type:
                continue
            if relation.head == entity_id:
                ids.append(relation.tail)
            elif relation.tail == entity_id:
                ids.append(relation.head)
        seen = set()
        return [self.entity(value) for value in ids if not (value in seen or seen.add(value))]

    def relations_of(self, entity: Any, relation_type: str | None = None) -> list[JointRelation]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        return [
            relation
            for relation in self.relations
            if (relation.head == entity_id or relation.tail == entity_id)
            and (relation_type is None or relation.type == relation_type)
        ]

    def to_dict(
        self, include_confidence: bool | None = None, include_spans: bool | None = None, include_text: bool = False
    ) -> dict[str, Any]:
        if include_confidence is None:
            include_confidence = self.default_include_confidence
        if include_spans is None:
            include_spans = self.default_include_spans
        value = {
            "entities": [entity.to_dict(include_confidence, include_spans) for entity in self.entities],
            "relations": [relation.to_dict(include_confidence) for relation in self.relations],
        }
        if include_text:
            value = {"text": self.text, **value}
        return value


def _result_identity(value: Any) -> Any:
    try:
        hash(value)
        return ("value", value)
    except TypeError:
        return ("object", id(value))


def _scores(value: Any, include_count: bool) -> list[float]:
    scores: list[float] = []
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


class ResultBuilder:
    """Build a stable result from an optimizer solution."""

    def __init__(
        self,
        include_confidence: bool = True,
        include_spans: bool = True,
        include_count: bool = True,
        config: Any = None,
    ):
        self.include_confidence = bool(_get(config, "include_confidence", default=include_confidence))
        self.include_spans = bool(_get(config, "include_spans", default=include_spans))
        self.include_count = include_count

    def build(
        self,
        solution: Any,
        problem: Any = None,
        text: str | None = None,
        include_confidence: bool | None = None,
        include_spans: bool | None = None,
        candidates: Any = None,
        candidate_set: Any = None,
        lattice: Any = None,
        **_: Any,
    ) -> JointResult:
        confidence_flag = self.include_confidence if include_confidence is None else include_confidence
        spans_flag = self.include_spans if include_spans is None else include_spans
        problem = problem or candidates or candidate_set
        if problem is None:
            raise TypeError("ResultBuilder requires a candidate problem")
        document = text if text is not None else str(_get(lattice, "text", default=_get(problem, "text", default="")))
        entity_records = self._selected(solution, problem, "entities")
        normalized = [self._entity_parts(item, problem, document, lattice) for item in entity_records]
        normalized.sort(key=lambda item: (item[1].char_start, item[1].char_end, item[0], item[1].text))
        entities: list[JointEntity] = []
        record_to_id: dict[Any, str] = {}
        span_label_to_id: dict[tuple[str, int, int], str] = {}
        for index, (label, span, scores, rescued, source) in enumerate(normalized):
            entity_id = f"e{index + 1}"
            confidence = _geometric_mean(scores) if confidence_flag else None
            entity = JointEntity(
                entity_id,
                label,
                span.text,
                span.char_start,
                span.char_end,
                confidence,
                span.sentence_id,
                rescued,
            )
            entities.append(entity)
            record_to_id[_result_identity(source)] = entity_id
            candidate_id = _get(source, "candidate_id", "id", default=None)
            if candidate_id is not None:
                record_to_id[_result_identity(candidate_id)] = entity_id
            span_label_to_id[(label, span.char_start, span.char_end)] = entity_id
        relations: list[JointRelation] = []
        for item in self._selected(solution, problem, "relations"):
            label = str(_get(item, "type", "label", "relation", "relation_type"))
            head = self._endpoint_id(
                _get(item, "head", "source", "head_entity"), record_to_id, span_label_to_id, normalized
            )
            tail = self._endpoint_id(
                _get(item, "tail", "target", "tail_entity"), record_to_id, span_label_to_id, normalized
            )
            scores = _scores(item, self.include_count)
            derived = bool(_get(item, "derived", "is_derived", default=False))
            relations.append(
                JointRelation(
                    label,
                    head,
                    tail,
                    _geometric_mean(scores) if confidence_flag else None,
                    derived,
                )
            )
        relations.sort(key=lambda rel: (rel.type, rel.head, rel.tail))
        feasible = bool(_get(solution, "feasible", default=True))
        return JointResult(document, entities, relations, confidence_flag, spans_flag, feasible)

    __call__ = build

    @staticmethod
    def _selected(solution: Any, problem: Any, kind: str) -> list[Any]:
        aliases = (kind, "nodes") if kind == "entities" else (kind, "edges")
        direct = _get(solution, *aliases, default=None)
        if direct is not None:
            return list(direct)
        selected = _get(solution, f"selected_{kind}", default=[])
        problem_aliases = (
            f"{kind}_candidates",
            f"candidate_{kind}",
            kind,
            "nodes" if kind == "entities" else "edges",
        )
        candidates = list(_get(problem, *problem_aliases, default=[]))
        result = []
        for value in selected:
            if isinstance(value, int):
                result.append(candidates[value])
            elif isinstance(value, bool):
                continue
            else:
                result.append(value)
        return result

    @staticmethod
    def _entity_parts(item: Any, problem: Any, text: str, lattice: Any = None):
        label = str(_get(item, "type", "label", "entity_type"))
        span = _get(item, "span", "span_ref", default=None)
        if isinstance(span, int):
            span = list(_get(problem, "spans", "span_candidates"))[span]
        if span is None:
            span = item
        if not isinstance(span, SpanRef):
            token_start = int(_get(span, "token_start", "start_token", "start", default=0))
            token_end = int(_get(span, "token_end", "end_token", "end", default=token_start))
            starts = _get(lattice, "start_mappings", default=None)
            ends = _get(lattice, "end_mappings", default=None)
            char_start_value = _get(span, "char_start", default=None)
            char_end_value = _get(span, "char_end", default=None)
            char_start = int(
                starts[token_start]
                if char_start_value is None and starts is not None
                else token_start
                if char_start_value is None
                else char_start_value
            )
            last_token = max(token_start, token_end - 1)
            char_end = int(
                ends[last_token]
                if char_end_value is None and ends is not None
                else token_end
                if char_end_value is None
                else char_end_value
            )
            value = _get(span, "text", default=text[char_start:char_end])
            sentence_id = _get(span, "sentence_id", "sentence", default=None)
            span = SpanRef(token_start, token_end, char_start, char_end, str(value), sentence_id)
        source = _get(item, "source", default="")
        rescued = bool(_get(item, "rescued", "is_rescued", default=False)) or "rescue" in str(source).lower()
        return label, span, _scores(item, True), rescued, item

    @staticmethod
    def _endpoint_id(
        endpoint: Any,
        records: Mapping[Any, str],
        spans: Mapping[tuple[str, int, int], str],
        normalized: Sequence[tuple[Any, ...]],
    ) -> str:
        if isinstance(endpoint, str) and endpoint.startswith("e"):
            return endpoint
        identity = _result_identity(endpoint)
        if identity in records:
            return records[identity]
        if isinstance(endpoint, int) and 0 <= endpoint < len(normalized):
            return records[_result_identity(normalized[endpoint][4])]
        label = str(_get(endpoint, "type", "label", "entity_type", default=""))
        span = _get(endpoint, "span", "span_ref", default=endpoint)
        start = int(_get(span, "char_start", "start", default=-1))
        end = int(_get(span, "char_end", "end", default=-1))
        try:
            return spans[(label, start, end)]
        except KeyError as exc:
            raise ValueError("relation endpoint does not identify a selected entity") from exc


def normalize_overlap_policy(policy: str | None, *, default: str | None = None) -> str:
    """Return a canonical overlap policy."""
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
    """Resolve half-open spans for ``flat``, ``nested``, or ``longest``.

    ``flat`` is the maximum-total-score non-overlapping set. ``nested`` keeps
    containment and rejects crossings. ``longest`` drops spans strictly inside
    another candidate.
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


def _confidence_key(value: float | None) -> float:
    return float("-inf") if value is None else value


def merge_joint_chunks(
    text: str, fragments: Sequence[tuple[int, Any]], *, include_confidence: bool = True, include_spans: bool = True
) -> dict:
    """Merge per-chunk joint results onto document character offsets.

    ``fragments`` is ``(chunk_start_char, result)``. Relations stay inside the
    chunk that produced both endpoints.
    """
    entity_by_key: dict[tuple[str, int, int], JointEntity] = {}
    relation_rows: dict[tuple[Any, ...], tuple] = {}
    for start_char, raw in fragments:
        result = raw if isinstance(raw, JointResult) else _coerce_joint_result(raw, text)
        local_keys: dict[str, tuple[str, int, int]] = {}
        for entity in result.entities:
            start = entity.start + start_char
            end = entity.end + start_char
            key = (entity.type, start, end)
            local_keys[entity.id] = key
            remapped = JointEntity(
                "",
                entity.type,
                text[start:end],
                start,
                end,
                entity.confidence,
                entity.sentence_id,
                entity.rescued,
            )
            previous = entity_by_key.get(key)
            if previous is None or _confidence_key(remapped.confidence) > _confidence_key(previous.confidence):
                entity_by_key[key] = remapped
        for relation in result.relations:
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
        entities.append(
            JointEntity(
                key_to_id[key],
                item.type,
                item.text,
                item.start,
                item.end,
                item.confidence,
                item.sentence_id,
                item.rescued,
            )
        )
    relations = [
        JointRelation(label, key_to_id[head], key_to_id[tail], confidence, derived)
        for label, head, tail, confidence, derived in relation_rows.values()
    ]
    relations.sort(key=lambda value: (value.type, value.head, value.tail))
    return JointResult(text, entities, relations, include_confidence, include_spans).to_dict()


def _coerce_joint_result(value: Any, text: str) -> JointResult:
    if isinstance(value, JointResult):
        return value
    if not isinstance(value, dict):
        raise TypeError("chunk extraction must return JointResult or a result dictionary")
    entities = [
        JointEntity(
            str(item.get("id", f"e{index}")),
            str(item.get("type", item.get("label", ""))),
            str(item.get("text", "")),
            int(item.get("start", 0)),
            int(item.get("end", 0)),
            item.get("confidence"),
            item.get("sentence_id"),
            bool(item.get("rescued", False)),
        )
        for index, item in enumerate(value.get("entities", []))
    ]
    relations = [
        JointRelation(
            str(item.get("type", item.get("label", ""))),
            str(item["head"]),
            str(item["tail"]),
            item.get("confidence"),
            bool(item.get("derived", False)),
        )
        for item in value.get("relations", [])
    ]
    return JointResult(str(value.get("text", text)), entities, relations)


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


def _coerce_joint_schema(schema):
    if isinstance(schema, CompiledJointSchema):
        return schema
    if isinstance(schema, JointSchema):
        return schema
    if isinstance(schema, Mapping) and ("entities" in schema or "relations" in schema or "constraints" in schema):
        if isinstance(schema.get("relations", {}), Mapping) or "constraints" in schema or "entities" in schema:
            relations = schema.get("relations", {})
            if not isinstance(relations, list):
                return JointSchema.from_dict(schema)
    if hasattr(schema, "to_dict") and hasattr(schema, "entity_specs"):
        payload = schema.to_dict()
        if isinstance(payload, Mapping):
            return JointSchema.from_dict(payload)
    raise TypeError(f"expected a joint schema, got {type(schema).__name__}")


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


def _lattice_view(text: str, scores: Any):
    starts = _as_tuple(_get(scores, "start_mappings", default=None))
    ends = _as_tuple(_get(scores, "end_mappings", default=None))
    return SimpleNamespace(
        text=text or str(_get(scores, "text", default="") or ""),
        start_mappings=starts or None,
        end_mappings=ends or None,
    )


def _hypothesis_from_mapping(raw: Mapping) -> RelationHypothesis:
    payload = dict(raw)
    payload.setdefault("head_types", payload.pop("head", ()))
    payload.setdefault("tail_types", payload.pop("tail", ()))
    known = set(RelationHypothesis.__dataclass_fields__)
    return RelationHypothesis(**{key: value for key, value in payload.items() if key in known})


def _problem_from_span(scores, compiled: CompiledJointSchema, config: JointIEConfig) -> JointProblem:
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
                role_logits = _mask_invalid(_get(count, "role_logits"), valid)
                hypotheses.append(
                    RelationHypothesis(
                        relation_type=_get(task, "name"),
                        role_logits=role_logits,
                        head_types=head_types,
                        tail_types=tail_types,
                        threshold=getattr(spec, "threshold", None),
                        candidate_threshold=getattr(spec, "candidate_threshold", None),
                        count_probability=float(_get(count, "probability", default=1.0)),
                        count_utility=float(_get(count, "logit", default=0.0)),
                        count_alternative=alternative,
                        hypothesis_id=_get(task, "name"),
                    )
                )
    else:
        for raw in _get(scores, "relation_hypotheses", "relations", default=()) or ():
            if isinstance(raw, RelationHypothesis):
                hypotheses.append(raw)
            elif isinstance(raw, Mapping) and "role_logits" in raw:
                hypotheses.append(_hypothesis_from_mapping(raw))
    specs = compiled.entity_specs
    builder = CandidateBuilder(
        candidate_threshold=config.candidate_threshold,
        relation_role_threshold=config.relation_role_threshold,
        top_k_entities=config.top_k_entities,
        top_k_roles=config.top_k_roles,
        count_top_k=config.count_top_k,
        entity_threshold=config.entity_threshold,
        relation_pair_cap=config.relation_pair_cap,
        max_edges_per_type=config.max_edges_per_type,
        rescue_per_role=config.rescue_per_role,
        entity_weight=config.entity_weight,
        role_weight=config.role_weight,
        count_weight=config.count_weight,
    )
    return builder.build(
        entity_logits,
        entity_types,
        hypotheses,
        entity_thresholds={name: spec.threshold for name, spec in specs.items()},
        entity_candidate_thresholds={name: spec.candidate_threshold for name, spec in specs.items()},
        entity_max_candidates={
            name: spec.max_candidates for name, spec in specs.items() if spec.max_candidates is not None
        },
        constraints=compiled.constraints,
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
        query_id=int(raw.get("query_id", index)),
        entity_type=entity_type,
        start=start,
        end=end,
        logit=logit,
        probability=float(probability),
        threshold=float(raw.get("threshold", 0.5)),
        candidate_threshold=raw.get("candidate_threshold"),
    )


def _endpoint_key(value: Any) -> Hashable:
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
        relation_type=str(raw.get("relation_type", raw.get("type", raw.get("label")))),
        head=_endpoint_key(raw.get("head")),
        tail=_endpoint_key(raw.get("tail")),
        logit=logit,
        probability=float(probability),
        threshold=float(raw.get("threshold", 0.5)),
        candidate_threshold=raw.get("candidate_threshold"),
    )


def _as_candidate_score_set(scores, compiled: CompiledJointSchema) -> CandidateScoreSet:
    if isinstance(scores, CandidateScoreSet):
        return scores
    if isinstance(scores, JointScoreLattice):
        mentions = []
        for label, values in scores.entity_scores.scores.items():
            for index, span in enumerate(scores.spans):
                logit = float(scores.calibrator.calibrate(_number(values[index])))
                mentions.append(
                    MentionScore(
                        query_id=0,
                        entity_type=str(label),
                        start=span.start,
                        end=span.end,
                        logit=logit,
                        probability=sigmoid(logit),
                    )
                )
        return CandidateScoreSet(
            text=_get(scores, "text", default=""),
            mentions=tuple(mentions),
        )
    candidates = _get(scores, "candidates", default=None)
    query_specs = _get(scores, "query_specs", default=None)
    if (
        candidates is not None
        and query_specs is not None
        and _get(candidates, "pair_logits", default=None) is not None
    ):
        specs = compiled.entity_specs
        return boundary_candidates_to_candidate_score_set(
            str(_get(scores, "text", default="") or ""),
            candidates,
            query_specs,
            sample_index=int(_get(scores, "sample_index", default=0)),
            token_offset=int(_get(scores, "token_offset", default=0)),
            text_length=_get(scores, "text_length", default=None),
            pair_temperature=float(_get(scores, "pair_temperature", default=1.0)),
            entity_thresholds={name: spec.threshold for name, spec in specs.items()},
            entity_candidate_thresholds={name: spec.candidate_threshold for name, spec in specs.items()},
            extra_mentions=tuple(
                item if isinstance(item, MentionScore) else _mention_from_mapping(item, index)
                for index, item in enumerate(_get(scores, "extra_mentions", default=()) or ())
            ),
            edges=tuple(
                item if isinstance(item, ScoredRelationEdge) else _edge_from_mapping(item)
                for item in _get(scores, "edges", default=()) or ()
            ),
            text_tokens=tuple(_get(scores, "text_tokens", default=()) or ()),
            start_mappings=tuple(_as_tuple(_get(scores, "start_mappings", default=())) or ()),
            end_mappings=tuple(_as_tuple(_get(scores, "end_mappings", default=())) or ()),
            metadata=_get(scores, "metadata", default=None),
        )
    mentions_raw = _get(scores, "mentions", default=None)
    if mentions_raw is None:
        raise TypeError("boundary scores need mentions or boundary candidates")
    mentions = tuple(
        item if isinstance(item, MentionScore) else _mention_from_mapping(item, index)
        for index, item in enumerate(mentions_raw)
    )
    edges = tuple(
        item if isinstance(item, ScoredRelationEdge) else _edge_from_mapping(item)
        for item in _get(scores, "edges", default=()) or ()
    )
    return CandidateScoreSet(
        text=str(_get(scores, "text", default="") or ""),
        mentions=mentions,
        edges=edges,
        text_tokens=tuple(_get(scores, "text_tokens", default=()) or ()),
        start_mappings=tuple(_as_tuple(_get(scores, "start_mappings", default=())) or ()),
        end_mappings=tuple(_as_tuple(_get(scores, "end_mappings", default=())) or ()),
        metadata=dict(_get(scores, "metadata", default={}) or {}),
    )


def _problem_from_scores(scores, compiled: CompiledJointSchema, config: JointIEConfig) -> JointProblem:
    specs = compiled.entity_specs
    return candidate_score_set_to_problem(
        scores,
        mention_threshold=(
            config.entity_threshold if config.entity_threshold is not None else config.candidate_threshold
        ),
        constraints=compiled.constraints,
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


def _optimize(problem: JointProblem, config: JointIEConfig) -> JointSolution:
    if config.optimizer.lower() in {"auto", "greedy"}:
        return GreedyOptimizer().optimize(problem)
    return BeamOptimizer(beam_width=config.beam_size).optimize(problem)


def decode_joint(
    span_or_boundary_scores,
    schema,
    architecture,
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
) -> dict:
    """Decode span or boundary scores into entities and relations.

    ``architecture`` is ``"span"`` or ``"boundary"``. ``optimizer`` is
    ``"beam"`` or ``"greedy"``. Returns the ``JointResult.to_dict()`` mapping.
    """
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
    compiled = compile_schema(_coerce_joint_schema(schema))
    kind = _architecture(architecture)
    document = text or str(_get(span_or_boundary_scores, "text", default="") or "")
    if kind == "span" and not isinstance(span_or_boundary_scores, CandidateScoreSet):
        mentions = _get(span_or_boundary_scores, "mentions", default=None)
        dense = (
            _get(span_or_boundary_scores, "entity_logits", default=None) is not None
            or _get(span_or_boundary_scores, "tasks", default=None)
            or isinstance(span_or_boundary_scores, ScoreLattice)
        )
        if dense or (mentions is None and not isinstance(span_or_boundary_scores, JointScoreLattice)):
            problem = _problem_from_span(span_or_boundary_scores, compiled, config)
            lattice = _lattice_view(document, span_or_boundary_scores)
            solution = _optimize(problem, config)
            result = ResultBuilder(
                include_confidence=config.include_confidence,
                include_spans=config.include_spans,
            ).build(solution, problem, text=document, lattice=lattice)
            return result.to_dict()
    score_set = _as_candidate_score_set(span_or_boundary_scores, compiled)
    if not document:
        document = score_set.text
    problem = _problem_from_scores(score_set, compiled, config)
    lattice = _lattice_view(document, score_set)
    solution = _optimize(problem, config)
    result = ResultBuilder(
        include_confidence=config.include_confidence,
        include_spans=config.include_spans,
    ).build(solution, problem, text=document, lattice=lattice)
    return result.to_dict()
