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
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Any

import torch

from ...utils.import_utils import requires_backends


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
_ON_INFEASIBLE = ("relax", "min_violations", "raise")
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
_DECODERS = ("auto", "independent", "exact", "beam", "min_violations")
_RELATION_OPTIONS = (
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
    "unique_pair",
    "acyclic",
)


def linear_sum_assignment(cost_matrix: torch.Tensor):
    """Minimum-cost assignment using gliner2's SciPy cost tie-break."""
    requires_backends(linear_sum_assignment, ["scipy"])
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2-D")
    cost = cost_matrix.detach().cpu().to(torch.float64)
    if torch.isnan(cost).any():
        raise ValueError("cost_matrix contains NaN")
    if torch.isinf(cost).any():
        finite = cost[torch.isfinite(cost)]
        scale = float(finite.abs().max()) if finite.numel() else 1.0
        big = 1e6 * (scale + 1.0)
        cost = torch.nan_to_num(cost, posinf=big, neginf=-big)
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    from scipy.optimize import linear_sum_assignment as scipy_assignment

    array = cost.numpy()
    scale = max(float(abs(array).max()), 1.0)
    epsilon = torch.finfo(torch.float64).eps * scale
    tie = torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols).numpy()
    row_ind, col_ind = scipy_assignment(array + epsilon * tie)
    pairs = sorted(zip(row_ind.tolist(), col_ind.tolist()))
    return (
        torch.tensor([row for row, _ in pairs], dtype=torch.long),
        torch.tensor([col for _, col in pairs], dtype=torch.long),
    )


@dataclass(frozen=True)
class RecordField:
    """One record field and its boundary query id."""

    query_id: int
    name: str = ""
    cardinality: str = "zero_or_more"
    is_anchor: bool = False
    exclusive: bool = False

    @property
    def is_scalar(self) -> bool:
        return self.cardinality in ("optional_one", "required_one")

    @property
    def allows_absent(self) -> bool:
        return self.cardinality in ("optional_one", "zero_or_more")


@dataclass(frozen=True)
class GroupRecord:
    """Record head configuration for one structure group."""

    mode: str
    fields: tuple[RecordField, ...]
    anchor_query_id: int | None = None
    anchor: str | None = None
    occurrence_policy: str = "latent_all"
    task_index: int = 0


@dataclass(frozen=True)
class DecodeOptions:
    """Cutoffs and payload switches shared by the span, record, and label decoders."""

    threshold: float = 0.5
    include_confidence: bool = False
    include_spans: bool = False
    overlap: str | None = None
    temperature: float = 1.0


@dataclass
class DecodedRecord:
    """One decoded record."""

    fields: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    field_scores: dict[int, list[float]] = field(default_factory=dict)
    anchor_span: tuple[int, int] | None = None
    score: float = 0.0


def _dedup_key(record: DecodedRecord) -> tuple:
    return tuple((qid, tuple(sorted(spans))) for qid, spans in sorted(record.fields.items()))


def _select_record_instances(group, anchor_threshold, object_threshold, temperature):
    """Instance indexes above the object or anchor threshold, best first, and all probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    obj_prob = torch.sigmoid(group.object_logits.detach() / temperature)
    select_thr = object_threshold if group.spec.mode == "anchorless" else anchor_threshold
    order = sorted(range(int(group.num_instances)), key=lambda index: (-float(obj_prob[index]), index))
    return [inst for inst in order if float(obj_prob[inst]) >= select_thr], obj_prob


def _exclusive_field_choices(group, selected_instances, field_threshold, temperature):
    """Hungarian scalar choices and list-candidate owners for exclusive fields."""
    scalar_choices: dict[tuple[int, int], tuple[int, float] | None] = {}
    list_owners: dict[tuple[int, int], tuple[int, float]] = {}
    for f_idx, fspec in enumerate(group.field_specs):
        if not fspec.exclusive or not selected_instances:
            continue
        logits = torch.stack([group.assign_logits[f_idx][inst].detach() / temperature for inst in selected_instances])
        candidate_count = max(int(logits.shape[-1]) - 1, 0)
        if fspec.is_scalar:
            _assign_exclusive_scalar(
                scalar_choices, f_idx, fspec, logits, selected_instances, candidate_count, field_threshold
            )
        elif candidate_count:
            probabilities = torch.sigmoid(logits[:, 1:])
            for cand_idx in range(candidate_count):
                probability, row = probabilities[:, cand_idx].max(dim=0)
                if float(probability) >= field_threshold:
                    list_owners[(f_idx, cand_idx)] = (selected_instances[int(row)], float(probability))
    return scalar_choices, list_owners


def _assign_exclusive_scalar(
    scalar_choices, f_idx, fspec, logits, selected_instances, candidate_count, field_threshold
):
    """Assign one exclusive scalar field with an absent column per instance."""
    if candidate_count == 0:
        for inst in selected_instances:
            scalar_choices[(inst, f_idx)] = None
        return
    probs = torch.softmax(logits, dim=-1)
    candidate_probs = probs[:, 1:]
    eps = torch.finfo(candidate_probs.dtype).eps
    candidate_cost = -torch.log(candidate_probs.clamp_min(eps))
    row_count = len(selected_instances)
    diagonal = -torch.log(probs[:, 0].clamp_min(eps))
    if not fspec.allows_absent:
        diagonal = candidate_cost.max().detach() + 50.0
        diagonal = diagonal.expand(row_count)
    invalid_cost = max(float(candidate_cost.max()), float(diagonal.max())) + 1_000.0
    absent_cost = candidate_cost.new_full((row_count, row_count), invalid_cost)
    absent_cost[torch.arange(row_count), torch.arange(row_count)] = diagonal
    cost = torch.cat((candidate_cost, absent_cost), dim=-1)
    rows, cols = linear_sum_assignment(cost)
    assignments = {int(row): int(col) for row, col in zip(rows, cols)}
    for row, inst in enumerate(selected_instances):
        col = assignments.get(row, candidate_count + row)
        probability = float(candidate_probs[row, col]) if col < candidate_count else 0.0
        if col >= candidate_count or (probability < field_threshold and fspec.allows_absent):
            scalar_choices[(inst, f_idx)] = None
        else:
            scalar_choices[(inst, f_idx)] = (col, probability)


def _append_span(rec, qid, spans_tensor, cand_idx, probability):
    rec.fields.setdefault(qid, []).append((int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1])))
    rec.field_scores.setdefault(qid, []).append(probability)


def _append_scalar_span(rec, fspec, spans_tensor, logits_row, choice, field_threshold):
    """Add the exclusive choice, or the best non-absent column above the threshold."""
    if fspec.exclusive:
        if choice is not None:
            _append_span(rec, fspec.query_id, spans_tensor, choice[0], choice[1])
        return
    probs = torch.softmax(logits_row, dim=-1)
    chosen = None
    for col in torch.argsort(probs, descending=True).tolist():
        if col == 0 and not fspec.allows_absent:
            continue
        chosen = col
        break
    if not chosen or (float(probs[chosen]) < field_threshold and fspec.allows_absent):
        return
    _append_span(rec, fspec.query_id, spans_tensor, chosen - 1, float(probs[chosen]))


def _append_list_spans(rec, fspec, f_idx, spans_tensor, logits_row, list_owners, inst, field_threshold):
    """Add the list spans this instance owns, or that clear the threshold when not exclusive."""
    probs = torch.sigmoid(logits_row[1:])
    for cand_idx in range(probs.shape[0]):
        if fspec.exclusive:
            owner = list_owners.get((f_idx, cand_idx))
            if owner is None or owner[0] != inst:
                continue
            probability = owner[1]
        else:
            probability = float(probs[cand_idx])
            if probability < field_threshold:
                continue
        _append_span(rec, fspec.query_id, spans_tensor, cand_idx, probability)


def _record_from_instance(group, inst, obj_prob, scalar_choices, list_owners, field_threshold, temperature):
    """One record for a selected instance, or `None` when no field was filled."""
    rec = DecodedRecord(score=float(obj_prob[inst]))
    anchor_field_idx = None
    if group.spec.mode == "natural":
        anchor_field_idx = group.field_query_ids.index(group.spec.anchor_query_id)
        if group.instance_seed[inst] is not None:
            rec.anchor_span = group.instance_spans[inst]
    for f_idx, fspec in enumerate(group.field_specs):
        spans_tensor = group.field_spans[f_idx]
        logits_row = group.assign_logits[f_idx][inst].detach() / temperature
        if f_idx == anchor_field_idx:
            if rec.anchor_span is not None:
                rec.fields.setdefault(fspec.query_id, []).append(rec.anchor_span)
                rec.field_scores.setdefault(fspec.query_id, []).append(rec.score)
        elif fspec.is_scalar:
            choice = scalar_choices.get((inst, f_idx))
            _append_scalar_span(rec, fspec, spans_tensor, logits_row, choice, field_threshold)
        else:
            _append_list_spans(rec, fspec, f_idx, spans_tensor, logits_row, list_owners, inst, field_threshold)
    return rec if rec.fields else None


def decode_group(
    group,
    *,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
    temperature: float = 1.0,
) -> list[DecodedRecord]:
    """Decode one record-head group into field spans.

    Latent and anchorless records keep the best copy of identical field spans.
    Natural records are sorted by anchor span.

    Args:
        group:
            `RecordGroupOutput` whose `spec` and `field_specs` are a `GroupRecord` and its fields.
        anchor_threshold (`float`, *optional*, defaults to 0.5):
            Object probability needed by natural and latent instances.
        field_threshold (`float`, *optional*, defaults to 0.5):
            Probability needed to keep a field span.
        object_threshold (`float`, *optional*, defaults to 0.5):
            Object probability needed by anchorless instances.
        temperature (`float`, *optional*, defaults to 1.0):
            Divisor applied to object and assignment logits.
    """
    if int(group.num_instances) == 0:
        return []
    selected, obj_prob = _select_record_instances(group, anchor_threshold, object_threshold, temperature)
    scalar_choices, list_owners = _exclusive_field_choices(group, selected, field_threshold, temperature)
    records = []
    for inst in selected:
        rec = _record_from_instance(group, inst, obj_prob, scalar_choices, list_owners, field_threshold, temperature)
        if rec is not None:
            records.append(rec)
    if group.spec.mode in ("latent", "anchorless"):
        best: dict[tuple, DecodedRecord] = {}
        for rec in records:
            key = _dedup_key(rec)
            if key not in best or rec.score > best[key].score:
                best[key] = rec
        return list(best.values())
    if group.spec.mode == "natural":
        records.sort(key=lambda record: (record.anchor_span is None, record.anchor_span or (0, 0)))
    return records


class SchemaError(ValueError):
    """Invalid schema, task, label, or constraint."""


class InfeasibleError(RuntimeError):
    """No assignment satisfies the constraints."""

    def __init__(self, message: str, violations=()):
        super().__init__(message)
        self.violations = tuple(violations)


def probability_to_logit(probability: float) -> float:
    """Return log-odds, accepting exact zero and one.

    Args:
        probability: Value in ``[0, 1]``. Exact endpoints become infinities.

    Returns:
        The log-odds as a Python float taken from one float64 tensor.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf
    prob = torch.tensor(probability, dtype=torch.float64)
    return (torch.log(prob) - torch.log1p(-prob)).item()


def center_logit(logit: float, threshold: float = 0.5) -> float:
    """Center a logit so positive utility means passing ``threshold``."""
    return float(logit) - probability_to_logit(threshold)


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid.

    Args:
        value: Logit to squash.

    Returns:
        The sigmoid as a Python float taken from one float64 tensor.
    """
    return torch.sigmoid(torch.tensor(float(value), dtype=torch.float64)).item()


def _softmax(values: Sequence[float]) -> list:
    """Max-subtracted softmax in float64, materialized once."""
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64)
    return torch.softmax(tensor, dim=0).tolist()


def _geometric_mean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("confidence components must be probabilities in [0, 1]")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


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


def _greedy_overlaps(spans, score_fn, start_fn, end_fn):
    """Keep the highest-scoring span that does not overlap an earlier pick."""
    ranked = sorted(enumerate(spans), key=lambda row: (-float(score_fn(row[1])), row[0]))
    kept = []
    for _, item in ranked:
        item_start, item_end = int(start_fn(item)), int(end_fn(item))
        if any(item_start < int(end_fn(other)) and int(start_fn(other)) < item_end for other in kept):
            continue
        kept.append(item)
    return kept


def _distinct_ranked(spans, score_fn, start_fn, end_fn):
    """Rank spans and drop later copies of the same boundaries."""

    def rank_key(row):
        index, item = row
        return (-float(score_fn(item)), int(start_fn(item)), int(end_fn(item)), index)

    ranked = sorted(enumerate(spans), key=rank_key)
    distinct = []
    seen_boundaries = set()
    for row in ranked:
        boundaries = (int(start_fn(row[1])), int(end_fn(row[1])))
        if boundaries in seen_boundaries:
            continue
        seen_boundaries.add(boundaries)
        distinct.append(row)
    return distinct, rank_key


def _keep_nested(distinct, start_fn, end_fn):
    """Keep spans that nest inside earlier picks instead of crossing them."""
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


def _keep_longest(distinct, start_fn, end_fn):
    """Drop spans strictly contained in any other distinct span."""
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


def _keep_flat(distinct, score_fn, start_fn, end_fn, rank_key):
    """Weighted interval selection. Equal scores prefer more spans, then rank."""
    by_end = sorted(
        distinct,
        key=lambda row: (int(end_fn(row[1])), int(start_fn(row[1])), -float(score_fn(row[1])), row[0]),
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


def resolve_overlaps(
    spans,
    policy,
    *,
    score: Callable[[Any], float] | None = None,
    start: Callable[[Any], int] | None = None,
    end: Callable[[Any], int] | None = None,
    default: str | None = None,
):
    """Resolve half-open spans. None is the confidence-sorted greedy.

    ``policy=None`` without ``default`` keeps ``_greedy_overlaps``. That is the
    span and record default, so a missing policy does not raise.

    Args:
        spans: Half-open spans to filter.
        policy: Overlap name, or ``None`` for the greedy default.
        score: Optional score accessor.
        start: Optional inclusive start accessor.
        end: Optional exclusive end accessor.
        default: Policy used when ``policy`` is ``None``.

    Returns:
        The spans kept under the selected policy.
    """
    score_fn = score or _item_score
    start_fn = start or (lambda item: _item_bound(item, "start", ("char_start", "token_start")))
    end_fn = end or (lambda item: _item_bound(item, "end", ("char_end", "token_end")))
    if not spans:
        return []
    if policy is None and default is None:
        return _greedy_overlaps(spans, score_fn, start_fn, end_fn)
    canonical = normalize_overlap_policy(policy, default=default)
    distinct, rank_key = _distinct_ranked(spans, score_fn, start_fn, end_fn)
    if canonical == "allow":
        return [item for _, item in distinct]
    if canonical == "nested":
        return _keep_nested(distinct, start_fn, end_fn)
    if canonical == "longest":
        return _keep_longest(distinct, start_fn, end_fn)
    return _keep_flat(distinct, score_fn, start_fn, end_fn, rank_key)


def _char_span(start, end, offset, start_map, end_map, text):
    """Map a half-open word span onto `(surface, char_start, char_end)` in the text."""
    token_start, token_end = start - offset, end - offset
    if not (0 <= token_start < token_end <= len(end_map)):
        return None
    char_start = int(start_map[token_start])
    char_end = int(end_map[token_end - 1])
    surface = text[char_start:char_end].strip()
    if not surface:
        return None
    return surface, char_start, char_end


def _query_specs(groups):
    """Extractive fields in marker order, which is the boundary candidate query axis."""
    return [
        {"task_type": group.task, "task_name": group.name, "field_name": item.name, "field": item}
        for group in groups
        if group.task != "classifications"
        for item in group.fields
    ]


def _group_candidates(candidates, threshold, temperature):
    """Threshold pair logits into per-query `(score, start, end)` lists."""
    probs = torch.sigmoid(candidates.pair_logits / temperature)
    keep = candidates.valid_mask & candidates.query_mask.unsqueeze(-1) & (probs >= threshold)
    grouped = [[[] for _ in range(candidates.indices.shape[1])] for _ in range(candidates.indices.shape[0])]
    for b, q, c in keep.nonzero().tolist():
        start, end = candidates.indices[b, q, c].tolist()
        grouped[b][q].append((float(probs[b, q, c]), int(start), int(end)))
    for sample in grouped:
        for scored in sample:
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return grouped


def format_span(surface, score, char_start, char_end, include_confidence, include_spans):
    """Public span payload: the text alone, or a dict with confidence and offsets."""
    if include_spans and include_confidence:
        return {"text": surface, "confidence": score, "start": char_start, "end": char_end}
    if include_spans:
        return {"text": surface, "start": char_start, "end": char_end}
    if include_confidence:
        return {"text": surface, "confidence": score}
    return surface


def _passes_text(surface: str, validators) -> bool:
    return all(not hasattr(validator, "validate") or validator.validate(surface) for validator in validators)


def _matrix_spans(scores: torch.Tensor, threshold: float, text: str, start_map, end_map) -> list:
    """Spans at or above `threshold` as `(text, score, start, end)`."""
    doc_len = len(start_map)
    found = []
    starts, widths = torch.where(scores >= threshold)
    for start, width in zip(starts.tolist(), widths.tolist()):
        end = start + int(width) + 1
        if not (0 <= start < doc_len and end <= doc_len):
            continue
        char_start, char_end = start_map[start], end_map[end - 1]
        span_text = text[char_start:char_end].strip()
        if span_text:
            found.append((span_text, float(scores[start, width].item()), int(char_start), int(char_end)))
    return found


def decode_spans(scores, row, options, *, dtype="list", threshold=None, validators=(), blank=False):
    """Threshold, resolve overlaps, and format one field.

    Args:
        scores:
            `(words, width)` probabilities on the document axis, or scored
            `(score, start, end)` word spans that include the choice prefix.
        row (`dict`):
            Processor metadata with `text`, `start`, `end`, and `prefix_len`.
        options (`DecodeOptions`):
            Shared cutoff, overlap policy, and payload switches.
        dtype (`str`, *optional*, defaults to `"list"`):
            `"list"` keeps every span. Any other value keeps the best one.
        threshold (`float`, *optional*):
            Field cutoff for probability matrices. Defaults to `options.threshold`.
        validators (`tuple`, *optional*):
            Objects whose `validate(text)` must accept the span text.
        blank (`bool`, *optional*, defaults to `False`):
            Return `""` instead of `None` for an empty scalar without payload options.
    """
    text, start_map, end_map = row["text"], row["start"], row["end"]
    if torch.is_tensor(scores):
        cutoff = options.threshold if threshold is None else threshold
        found = _matrix_spans(scores, float(cutoff), text, start_map, end_map)
        found = [span for span in found if _passes_text(span[0], validators)]
        selected = resolve_overlaps(
            found, options.overlap, score=lambda span: span[1], start=lambda span: span[2], end=lambda span: span[3]
        )
    else:
        kept = resolve_overlaps(
            list(scores),
            options.overlap,
            score=lambda span: span[0],
            start=lambda span: span[1],
            end=lambda span: span[2],
        )
        selected = []
        for score, start, end in kept:
            mapped = _char_span(start, end, row["prefix_len"], start_map, end_map, text)
            if mapped is not None and _passes_text(mapped[0], validators):
                selected.append((mapped[0], score, mapped[1], mapped[2]))
    formatted = [
        format_span(surface, score, char_start, char_end, options.include_confidence, options.include_spans)
        for surface, score, char_start, char_end in selected
    ]
    if dtype == "list":
        return formatted
    if formatted:
        return formatted[0]
    return "" if blank and not (options.include_confidence or options.include_spans) else None


def _deduplicate_relation_edges(edges):
    """Collapse contained mentions and repeated head/tail text."""
    if len(edges) < 2:
        return edges

    def canonical_mentions(side):
        mentions = {(edge[side][1], edge[side][2]): edge[side] for edge in edges}
        canonical = {}
        for start, end in mentions:
            containing = [
                candidate for candidate in mentions.values() if candidate[1] <= start and candidate[2] >= end
            ]
            canonical[(start, end)] = max(
                containing, key=lambda candidate: (candidate[2] - candidate[1], -candidate[1])
            )
        return canonical

    head_canonical = canonical_mentions("head")
    tail_canonical = canonical_mentions("tail")
    exact = {}
    for edge in edges:
        head = head_canonical[(edge["head"][1], edge["head"][2])]
        tail = tail_canonical[(edge["tail"][1], edge["tail"][2])]
        key = (head[1], head[2], tail[1], tail[2])
        if key not in exact or edge["score"] > exact[key]["score"]:
            exact[key] = {**edge, "head": head, "tail": tail}

    def words(value):
        return " ".join(value.casefold().split())

    def rank(candidate):
        _, head_start, head_end = candidate["head"]
        _, tail_start, tail_end = candidate["tail"]
        distance = max(head_start - tail_end, tail_start - head_end, 0)
        return (distance, -candidate["score"], head_start, tail_start)

    semantic = {}
    for edge in exact.values():
        key = (words(edge["head"][0]), words(edge["tail"][0]))
        if key not in semantic or rank(edge) < rank(semantic[key]):
            semantic[key] = edge
    values = list(semantic.values())
    tokens = [(set(words(edge["head"][0]).split()), set(words(edge["tail"][0]).split())) for edge in values]
    kept = [
        edge
        for (head, tail), edge in zip(tokens, values)
        if not any(
            (head < other_head and tail == other_tail) or (tail < other_tail and head == other_head)
            for other_head, other_tail in tokens
        )
    ]
    return sorted(kept, key=lambda edge: (edge["head"][1], edge["tail"][1], -edge["score"]))


def format_relation(edge, include_confidence, include_spans):
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
        return {"head": {"text": head, "confidence": score}, "tail": {"text": tail, "confidence": score}}
    return (head, tail)


def decode_relations(sample, groups, row, options):
    """Map forward's typed relation pairs onto character offsets, keyed by relation name.

    Args:
        sample (`dict`):
            One row of model output with `relation_pairs` `[pairs, 6]` and `relation_logits`.
        groups (`Sequence[FieldGroup]`):
            Compiled schema groups. Relation groups are indexed in order.
        row (`dict`):
            Processor metadata with `text`, `start`, `end`, and `prefix_len`.
        options (`DecodeOptions`):
            Cutoff and payload switches. Relation groups may override the cutoff.
    """
    pairs, logits = sample.get("relation_pairs"), sample.get("relation_logits")
    if pairs is None or logits is None or pairs.numel() == 0:
        return {}
    temperature = float(sample.get("relation_temperature") or 1.0)
    probabilities = torch.sigmoid(logits.detach().float().cpu() / temperature).tolist()
    relations = [group for group in groups if group.task == "relations"]
    edges = {}
    for pair, score in zip(pairs.tolist(), probabilities):
        if not 0 <= pair[1] < len(relations):
            continue
        group = relations[pair[1]]
        if score < (options.threshold if group.threshold is None else group.threshold):
            continue
        head = _char_span(pair[2], pair[3], row["prefix_len"], row["start"], row["end"], row["text"])
        tail = _char_span(pair[4], pair[5], row["prefix_len"], row["start"], row["end"], row["text"])
        if head is not None and tail is not None:
            edges.setdefault(group.name, []).append({"score": score, "head": head, "tail": tail})
    return {
        name: [
            format_relation(edge, options.include_confidence, options.include_spans)
            for edge in _deduplicate_relation_edges(found)
        ]
        for name, found in edges.items()
    }


def decode_boundary(sample, groups, row, options):
    """Decode one boundary row into entities, structures, relations, and records.

    Args:
        sample (`dict`):
            One row of model output with `candidates`.
        groups (`Sequence[FieldGroup]`):
            Compiled schema groups.
        row (`dict`):
            Processor metadata with `text`, `start`, `end`, and `prefix_len`.
        options (`DecodeOptions`):
            Cutoff, overlap policy, pair temperature, and payload switches.
    """
    grouped = _group_candidates(sample["candidates"], options.threshold, options.temperature)[0]
    null_logits = sample.get("null_logits")
    entities, structures = {}, {}
    for query_id, spec in enumerate(_query_specs(groups)[: len(grouped)]):
        abstained = null_logits is not None and float(torch.sigmoid(null_logits[query_id])) > 0.5
        scored = [] if abstained else grouped[query_id]
        validators = spec["field"].validators
        if spec["task_type"] == "entities":
            entities[spec["field_name"]] = decode_spans(scored, row, options, validators=validators, blank=True)
        elif spec["task_type"] == "json_structures":
            decoded = decode_spans(scored, row, options, dtype="str", validators=validators)
            structures.setdefault(spec["task_name"], {})[spec["field_name"]] = decoded
    result = {"entities": [entities]} if entities else {}
    for name, instance in structures.items():
        if any(value is not None and value != [] for value in instance.values()):
            result[name] = [instance]
    result.update(decode_relations(sample, groups, row, options))
    for group in groups:
        if group.task == "relations":
            result.setdefault(group.name, [])
    result.update(decode_records(sample, groups, row, options))
    return result


def _record_field_spans(record, item, field, candidates, row, options):
    """Score a record field's spans by the lower of candidate and assignment probability."""
    kept = []
    spans = record.fields.get(item.query_id, [])
    for (start, end), assigned in zip(spans, record.field_scores.get(item.query_id, [])):
        candidate = 0.0
        if item.query_id < candidates.indices.shape[1]:
            target = candidates.indices.new_tensor([start, end])
            exact = candidates.valid_mask[0, item.query_id] & (candidates.indices[0, item.query_id] == target).all(-1)
            if bool(exact.any()):
                logits = candidates.pair_logits[0, item.query_id, exact] / options.temperature
                candidate = float(torch.sigmoid(logits).max())
        probability = min(candidate, float(assigned))
        mapped = _char_span(start, end, row["prefix_len"], row["start"], row["end"], row["text"])
        if (
            mapped is None
            or (item.allows_absent and candidate < options.threshold)
            or (field.threshold is not None and probability < float(field.threshold))
            or not _passes_text(mapped[0], field.validators)
        ):
            continue
        kept.append((probability, start, end))
    return kept


def decode_records(sample, groups, row, options):
    """Assign forward's `record_logits` groups to field spans, keyed by structure name.

    Args:
        sample (`dict`):
            One row of model output. `record_logits` holds one `RecordGroupOutput`
            per group with a `record`, in group order.
        groups (`Sequence[FieldGroup]`):
            Compiled schema groups.
        row (`dict`):
            Processor metadata with `text`, `start`, `end`, and `prefix_len`.
        options (`DecodeOptions`):
            Cutoff, overlap policy, and payload switches.
    """
    scored_groups = sample.get("record_logits")
    records = [group for group in groups if group.record is not None]
    if not scored_groups or not records:
        return {}
    temperature = float(sample.get("record_temperature") or 1.0)
    result = {}
    for group, scored in zip(records, scored_groups):
        scored = replace(scored, spec=group.record, field_specs=list(group.record.fields))
        fields = {item.name: item for item in group.fields}
        instances = []
        for record in decode_group(
            scored,
            anchor_threshold=options.threshold,
            field_threshold=options.threshold,
            object_threshold=options.threshold,
            temperature=temperature,
        ):
            instance = {}
            for item in group.record.fields:
                field = fields[item.name]
                raw = _record_field_spans(record, item, field, sample["candidates"], row, options)
                scalar = item.is_scalar or field.dtype == "str"
                instance[item.name] = decode_spans(raw, row, options, dtype="str" if scalar else "list")
            if any(value is not None and value != [] for value in instance.values()):
                instances.append(instance)
        if instances:
            result[group.name] = instances
    return result


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

    def __post_init__(self) -> None:
        _clean(self.name, "task name")
        labels = tuple(_clean(label, "label name") for label in self.labels)
        object.__setattr__(self, "labels", labels)
        if not labels or len(set(labels)) != len(labels):
            raise SchemaError(f"task {self.name!r} needs at least one label and no duplicates")
        if not isinstance(self.min_labels, int) or self.min_labels < 0:
            raise SchemaError(f"task {self.name!r}: min_labels must be a non-negative int")
        if self.max_labels is not None and not (
            isinstance(self.max_labels, int) and 0 <= self.max_labels <= len(labels)
        ):
            raise SchemaError(f"task {self.name!r}: max_labels must be an int in [0, {len(labels)}]")
        if self.default is not None:
            if self.default not in labels:
                raise SchemaError(f"task {self.name!r}: default {self.default!r} is not one of its labels")
            object.__setattr__(self, "min_labels", max(self.min_labels, 1))
        if self.max_labels is not None and self.min_labels > self.max_labels:
            raise SchemaError(f"task {self.name!r}: min_labels exceeds max_labels")
        if self.ordered and len(labels) < 2:
            raise SchemaError(f"ordered task {self.name!r} requires at least two labels")
        if not isinstance(self.threshold, (int, float)) or not 0 < self.threshold < 1:
            raise SchemaError(f"task {self.name!r}: threshold must be in (0, 1)")
        _prob(self.candidate_threshold, f"task {self.name!r}: candidate_threshold")
        if self.activation not in _ACTIVATIONS:
            raise SchemaError(f"task {self.name!r}: activation must be one of {_ACTIVATIONS}")
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise SchemaError(f"task {self.name!r}: temperature must be positive")

    @property
    def label_names(self) -> tuple:
        return self.labels

    @property
    def is_exclusive(self) -> bool:
        return self.min_labels == 1 and self.max_labels == 1

    def effective_max_labels(self) -> int:
        return len(self.labels) if self.max_labels is None else self.max_labels


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
    """Task specs plus the constraint set the decoder searches."""

    task_specs: tuple
    constraints: tuple
    task_order: tuple

    def task(self, name) -> TaskSpec:
        for spec in self.task_specs:
            if spec.name == name:
                return spec
        raise SchemaError(f"unknown task {name!r}")


def _static_feasibility(constraints, task_specs) -> None:
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


def _lower_defaults(constraints, task_specs) -> tuple:
    constraints = list(constraints)
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


def _constraint_list(raw) -> tuple:
    constraints = []
    for item in raw or ():
        if isinstance(item, Expr):
            constraints.append(item)
        elif isinstance(item, Mapping) and item.get("type") in _EXPR_FIELDS:
            constraints.append(_expr_from_dict(item))
    return tuple(constraints)


def _task_spec_from_body(name, body) -> TaskSpec:
    """Build one task record from a tasks-dict or classifications entry."""
    body = dict(body or {})
    labels = body.pop("labels", None)
    if isinstance(labels, str) or labels is None:
        raise SchemaError("labels must be a list or a {label: description} mapping")
    for key in ("label_descriptions", "examples", "prompt", "instruction", "task"):
        body.pop(key, None)
    if "multi_label" in body or "cls_threshold" in body or "class_act" in body:
        multi = bool(body.pop("multi_label", False))
        body.setdefault("min_labels", 0 if multi else 1)
        body.setdefault("max_labels", None if multi else 1)
        body["threshold"] = body.pop("cls_threshold", body.get("threshold", 0.5))
        body["activation"] = body.pop("class_act", body.get("activation", "auto"))
    return TaskSpec(name=name, labels=tuple(labels), **body)


def _specs_from_schema(schema: Mapping) -> tuple:
    constraints = _constraint_list(schema.get("constraints"))
    tasks = schema.get("tasks")
    if isinstance(tasks, Mapping):
        return tuple(_task_spec_from_body(name, spec) for name, spec in tasks.items()), constraints
    entries = []
    for entry in schema.get("classifications") or []:
        entries.append(_task_spec_from_body(entry["task"], entry))
    return tuple(entries), constraints


def _compile_classification(schema) -> CompiledClassificationSchema:
    task_specs, constraints = _specs_from_schema(schema)
    if not task_specs:
        raise SchemaError("cannot compile a schema with no tasks")
    compiled = CompiledClassificationSchema(
        task_specs=tuple(task_specs),
        constraints=_lower_defaults(constraints, task_specs),
        task_order=tuple(spec.name for spec in task_specs),
    )
    for constraint in constraints:
        _walk(constraint, compiled)
    _static_feasibility(constraints, task_specs)
    return compiled


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
    """Raw per-label logits for one compiled schema."""

    def __init__(self, text, tasks, specs):
        self.text = text
        self.tasks = {task: dict(values) for task, values in tasks.items()}
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


def _signature(chosen, order):
    return tuple((task, tuple(sorted(chosen[task].labels))) for task in order if task in chosen)


def _search_beam(problem, order, beam_size: int) -> Solution | None:
    """Keep the highest-utility accepting locals within the beam."""
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


def _independent_solution(problem) -> Solution:
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


def _search(problem, *, mode: str, budget: int, beam_size: int = 16) -> tuple[Solution | None, bool]:
    """Exact, beam, independent, and min-violations over the same locals."""
    if mode == "independent":
        return _independent_solution(problem), False
    order = _search_order(problem)
    if mode == "beam":
        return _search_beam(problem, order, beam_size), False
    suffix = _suffix_max(order, problem) if mode == "exact" else None
    best = {"assign": None, "score": -math.inf, "weight": math.inf}
    nodes = {"n": 0}

    def dfs(index, chosen, score) -> bool:
        nodes["n"] += 1
        if nodes["n"] > budget:
            return True
        if mode == "exact" and score + suffix[index] <= best["score"]:
            return False
        if index == len(order):
            if mode == "exact":
                best["assign"] = dict(chosen)
                best["score"] = score
                return False
            violated = problem.violations_of(chosen)
            weight = float(len(violated))
            key = (weight, -score)
            current = (best["weight"], -best["score"])
            if key < current:
                best["assign"] = dict(chosen)
                best["weight"] = weight
                best["score"] = score
            return False
        task = order[index]
        for local in problem.locals[task]:
            if mode == "exact" and score + local.utility + suffix[index + 1] <= best["score"]:
                break
            if mode == "exact" and not _accepts(problem, order, index, chosen, local):
                continue
            chosen[task] = local
            exceeded = dfs(index + 1, chosen, score + local.utility)
            del chosen[task]
            if exceeded:
                return True
        return False

    if mode == "min_violations":
        exhausted = dfs(0, {}, 0.0)
        assign = best["assign"] or {task: problem.locals[task][0] for task in problem.task_order}
        return (
            Solution(
                assignments=dict(assign),
                score=sum(local.utility for local in assign.values()),
                violations=problem.violations_of(assign),
                exact=not exhausted,
                decoder="min_violations",
            ),
            False,
        )
    exceeded = dfs(0, {}, 0.0)
    if exceeded:
        return None, True
    if best["assign"] is None:
        return None, False
    return (
        Solution(
            assignments=best["assign"],
            score=best["score"],
            violations=(),
            exact=True,
            decoder="exact",
        ),
        False,
    )


def _select_decoder(problem, requested: str) -> str:
    """Use exact search when any constraint crosses tasks."""
    if requested != "auto":
        return requested
    cross_task = any(len(_walk(constraint)[0]) > 1 for constraint in problem.constraints)
    return "exact" if cross_task else "independent"


def _primary(problem, config):
    """Run the requested decoder. Exact search falls back to beam on budget."""
    decoder = _select_decoder(problem, config.decoder)
    if decoder == "min_violations":
        solution, _ = _search(problem, mode="min_violations", budget=config.exact_node_budget)
        return solution
    if decoder == "independent":
        solution, _ = _search(problem, mode="independent", budget=config.exact_node_budget)
        return solution if solution.feasible else None
    if decoder == "beam":
        solution, _ = _search(problem, mode="beam", budget=config.exact_node_budget, beam_size=config.beam_size)
        return solution if (solution is not None and solution.feasible) else None
    solution, exceeded = _search(problem, mode="exact", budget=config.exact_node_budget)
    if exceeded:
        solution, _ = _search(problem, mode="beam", budget=config.exact_node_budget, beam_size=config.beam_size)
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
        solution, _ = _search(working, mode="min_violations", budget=config.exact_node_budget)
        return solution
    diagnosis, _ = _search(working, mode="min_violations", budget=config.exact_node_budget)
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


def _align_logits(logits, compiled) -> dict:
    """Read `{task: {label: logit}}` into schema order."""
    aligned = {}
    for task in compiled.task_order:
        names = compiled.task(task).label_names
        values = logits.get(task)
        if values is None or set(values) != set(names):
            raise SchemaError(f"logits for task {task!r} must cover exactly the labels {list(names)}")
        aligned[task] = {name: float(values[name]) for name in names}
    return aligned


def _with_temperature(compiled, temperature):
    """Scale task temperatures by the call temperature, or read them from a per-task mapping."""
    specs = []
    for spec in compiled.task_specs:
        if isinstance(temperature, Mapping):
            value = float(temperature.get(spec.name, spec.temperature))
        else:
            value = float(spec.temperature) * float(temperature)
        if not value > 0:
            raise ValueError("temperature must be positive")
        specs.append(spec if value == spec.temperature else replace(spec, temperature=value))
    return replace(compiled, task_specs=tuple(specs))


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
    compiled = _with_temperature(_compile_classification(schema), temperature)
    scores = ClassificationScores(
        text=text, tasks=_align_logits(logits, compiled), specs={spec.name: spec for spec in compiled.task_specs}
    )
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


_JOINT_DEFAULTS = {"directed": True, "slot": "head", "policy": "disallow"}
_BEAM_SIZE, _TOP_K_ENTITIES, _TOP_K_ROLES, _PAIR_CAP, _MAX_EDGES_PER_TYPE = 32, 32, 12, 128, 256


def _joint_constraint(kind: str, **fields) -> dict:
    """One joint constraint record with defaults filled in."""
    if kind not in _JOINT_FIELDS:
        raise ValueError(f"unknown constraint type {kind!r}")
    data = {"type": kind, **{name: fields.get(name, _JOINT_DEFAULTS.get(name)) for name in _JOINT_FIELDS[kind]}}
    if kind == "TypedEndpoints":
        data["head_types"], data["tail_types"] = tuple(data["head_types"] or ()), tuple(data["tail_types"] or ())
    if data.get("policy", "allow") not in ("allow", "disallow", "nested"):
        raise ValueError("policy must be 'allow', 'disallow', or 'nested'")
    if data.get("slot", "head") not in ("head", "tail", "slot") or data.get("limit", 0) < 0:
        raise ValueError(f"invalid {kind} constraint")
    return data


def _node_allowed(problem, node, nodes) -> bool:
    """Apply the entity overlap policies to `node` against `nodes`."""
    for constraint in problem.constraints:
        if constraint["type"] != "EntityOverlapPolicy" or constraint["policy"] == "allow":
            continue
        for old in nodes:
            if old is node or node.end <= old.start or old.end <= node.start:
                continue
            nested = (node.start >= old.start and node.end <= old.end) or (
                old.start >= node.start and old.end <= node.end
            )
            if constraint["policy"] == "disallow" or not nested:
                return False
    return True


def _edge_allowed(problem, edge, accepted) -> bool:
    """True when every constraint admits `edge` next to the accepted edges."""
    for constraint in problem.constraints:
        kind, relation = constraint["type"], constraint.get("relation")
        if relation is not None and edge.relation_type != relation:
            continue
        same = [old for old in accepted if old.relation_type == edge.relation_type]
        matching = [old for old in accepted if relation is None or old.relation_type == relation]
        if kind == "TypedEndpoints":
            nodes, heads, tails = problem.node_by_id, constraint["head_types"], constraint["tail_types"]
            allowed = (not heads or nodes[edge.head].entity_type in heads) and (
                not tails or nodes[edge.tail].entity_type in tails
            )
        elif kind == "NoSelfLoops":
            allowed = edge.head != edge.tail
        elif kind == "UniqueRelationPair":
            allowed = all(
                (edge.head, edge.tail) != (old.head, old.tail)
                and (constraint["directed"] or (edge.head, edge.tail) != (old.tail, old.head))
                for old in same
            )
        elif kind == "UniqueRelationSlot":
            slot = constraint["slot"]
            allowed = all(getattr(old, slot) != getattr(edge, slot) for old in same)
        elif kind in ("MaxRelationsPerHead", "MaxRelationsPerTail"):
            side = "head" if kind == "MaxRelationsPerHead" else "tail"
            allowed = sum(getattr(old, side) == getattr(edge, side) for old in matching) < constraint["limit"]
        elif kind == "AcyclicRelation":
            graph = defaultdict(list)
            for old in matching:
                graph[old.head].append(old.tail)
            stack, seen, allowed = [edge.tail], set(), edge.head != edge.tail
            while stack and allowed:
                node = stack.pop()
                allowed = node != edge.head
                if node not in seen:
                    seen.add(node)
                    stack.extend(graph[node])
        else:
            allowed = True
        if not allowed:
            return False
    return True


@dataclass(frozen=True)
class EntitySpec:
    """One entity type and its candidate limits."""

    name: str
    threshold: float | None = None
    candidate_threshold: float | None = None
    max_candidates: int | None = None
    allow_nested: bool | None = None


@dataclass(frozen=True)
class RelationSpec:
    """One relation type and its endpoint and cardinality rules."""

    name: str
    head: tuple
    tail: tuple
    threshold: float | None = None
    candidate_threshold: float | None = None
    directed: bool = True
    symmetric: bool = False
    inverse: str | None = None
    allow_self: bool = False
    max_per_head: int | None = None
    max_per_tail: int | None = None


def _joint_types(value, side: str) -> tuple:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values or len(set(values)) != len(values) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"relation {side} needs unique, non-empty entity type names")
    return values


def _entities_from_raw(raw) -> dict:
    keys = ("threshold", "candidate_threshold", "max_candidates", "allow_nested")
    if not isinstance(raw, Mapping):
        raw = {value if isinstance(value, str) else value["name"]: value for value in raw or ()}
    return {
        name: EntitySpec(name, **{key: value[key] for key in keys if key in value})
        if isinstance(value, Mapping)
        else EntitySpec(name)
        for name, value in raw.items()
    }


def _relation_spec(name, head, tail, options) -> RelationSpec:
    """One relation record from the schema's relation options and their aliases."""
    unknown = set(options) - set(_RELATION_OPTIONS)
    if unknown:
        raise TypeError(f"unknown relation options: {sorted(unknown)}")
    inverse, inverse_of = options.get("inverse"), options.get("inverse_of")
    if inverse is not None and inverse_of is not None and inverse != inverse_of:
        raise ValueError("inverse and inverse_of disagree")
    allow_self = options.get("allow_self", False)
    if options.get("allow_self_loops") is not None:
        allow_self = options["allow_self_loops"]
    if options.get("no_self_loops") is not None:
        allow_self = not options["no_self_loops"]
    limits = {}
    for side in ("head", "tail"):
        limit = options.get(f"max_per_{side}")
        limits[side] = 1 if limit is None and options.get(f"unique_{side}") else limit
    head, tail = _joint_types(head, "head"), _joint_types(tail, "tail")
    symmetric, inverse = options.get("symmetric", False), inverse or inverse_of
    if symmetric and (inverse or set(head) != set(tail)):
        raise ValueError("symmetric relations need matching endpoint types and no inverse")
    if any(value is not None and value < 0 for value in limits.values()):
        raise ValueError("relation limits must be non-negative")
    directed = options.get("directed", True) and not symmetric
    return RelationSpec(
        name,
        head,
        tail,
        options.get("threshold"),
        options.get("candidate_threshold"),
        directed,
        symmetric,
        inverse,
        allow_self,
        limits["head"],
        limits["tail"],
    )


def _endpoint(value):
    return value.get("type") or value.get("entity") if isinstance(value, Mapping) else value


def _relations_from_raw(raw, entities) -> tuple:
    if isinstance(raw, Mapping):
        items = [(name, dict(body)) for name, body in raw.items()]
    else:
        items = []
        for item in raw or ():
            if "name" in item and "head" in item:
                body = dict(item)
                items.append((body.pop("name"), body))
                continue
            keep = (*_RELATION_OPTIONS, "head", "tail")
            items += [
                (name, {key: value for key, value in fields.items() if key in keep}) for name, fields in item.items()
            ]
    relations, extras = {}, []
    for name, body in items:
        head, tail = _endpoint(body.pop("head", None)), _endpoint(body.pop("tail", None))
        spec = _relation_spec(name, head, tail, body)
        unknown = (set(spec.head) | set(spec.tail)) - set(entities)
        if unknown or spec.name in relations:
            raise ValueError(f"relation {spec.name!r} is duplicated or references unknown types {sorted(unknown)}")
        relations[spec.name] = spec
        if body.get("acyclic"):
            extras.append(_joint_constraint("AcyclicRelation", relation=name))
    return relations, extras


@dataclass(frozen=True)
class CompiledJointSchema:
    """Entity and relation specs plus the constraints the optimizer enforces."""

    entity_specs: dict
    relation_specs: dict
    constraints: tuple
    entity_order: tuple


def _compile_joint(schema) -> CompiledJointSchema:
    """Lower relation flags into concrete constraints."""
    entities = _entities_from_raw(schema.get("entities") or {})
    relations, extras = _relations_from_raw(schema.get("relations") or {}, entities)
    constraints = []

    def add(kind, **fields):
        constraint = _joint_constraint(kind, **fields)
        if constraint not in constraints:
            constraints.append(constraint)

    for item in list(schema.get("constraints") or []) + extras:
        add(item["type"], **{key: value for key, value in item.items() if key != "type"})
    if entities:
        nested = [bool(spec.allow_nested) for spec in entities.values()]
        add("EntityOverlapPolicy", policy="allow" if all(nested) else "nested" if any(nested) else "disallow")
    for spec in relations.values():
        add("TypedEndpoints", relation=spec.name, head_types=spec.head, tail_types=spec.tail)
        if not spec.allow_self:
            add("NoSelfLoops", relation=spec.name)
        add("UniqueRelationPair", relation=spec.name, directed=spec.directed)
        add("UniqueRelationSlot", relation=spec.name, slot="slot")
        if spec.max_per_head is not None:
            add("MaxRelationsPerHead", limit=spec.max_per_head, relation=spec.name)
        if spec.max_per_tail is not None:
            add("MaxRelationsPerTail", limit=spec.max_per_tail, relation=spec.name)
        if spec.symmetric:
            add("SymmetricRelation", relation=spec.name)
        if spec.inverse:
            other = relations.get(spec.inverse)
            if other is None or set(spec.head) != set(other.tail) or set(spec.tail) != set(other.head):
                raise ValueError(f"relation {spec.name!r} has an unknown or incompatible inverse {spec.inverse!r}")
            add("InverseRelation", relation=spec.name, inverse=spec.inverse)
    return CompiledJointSchema(entities, relations, tuple(constraints), tuple(entities))


@dataclass(frozen=True)
class NodeCandidate:
    """A typed entity span."""

    entity_type: str
    start: int
    end: int
    score: float
    probability: float
    rescued: bool = False

    @property
    def key(self) -> tuple:
        return (self.entity_type, self.start, self.end)

    candidate_id = key


@dataclass(frozen=True)
class EdgeCandidate:
    """A typed directed relation between two node ids."""

    relation_type: str
    head: tuple
    tail: tuple
    score: float
    head_probability: float | None = None
    tail_probability: float | None = None
    head_entity_probability: float | None = None
    tail_entity_probability: float | None = None
    count_probability: float | None = None
    derived: bool = False
    slot: int | None = None
    hypothesis: str | None = None
    count_alternative: int | None = None
    candidate_id: tuple | None = None

    def __post_init__(self) -> None:
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", self.key)

    @property
    def key(self) -> tuple:
        return (self.relation_type, self.head, self.tail, self.slot, self.count_alternative)


@dataclass(frozen=True)
class RelationHypothesis:
    """One relation lattice with shape `[count_slots, 2, L, W]`."""

    relation_type: str
    role_logits: Any
    head_types: tuple = ()
    tail_types: tuple = ()
    threshold: float | None = None
    candidate_threshold: float | None = None


@dataclass(frozen=True)
class JointProblem:
    """Nodes, edges, and hard constraints for one document."""

    nodes: tuple
    edges: tuple
    constraints: tuple = ()

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


def _span_entries(lattice) -> list:
    """Cells of one `[L, W]` logit lattice as `(logit, start, end)`."""
    rows = lattice.tolist()
    return [
        (row[width], start, start + width + 1)
        for start, row in enumerate(rows)
        for width in range(len(row))
        if start + width < len(rows)
    ]


def _node_rank(node: NodeCandidate):
    return (-node.score, node.start, node.end, node.entity_type, int(node.rescued))


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


def _bind_role(node_map, raw_scores, entries, types, offset) -> list:
    """Bind one role's spans to nodes, rescuing missing endpoints in rank order."""
    bound, rescued = [], 0
    for raw, start, end in entries:
        for entity_type in types:
            key = (entity_type, start, end)
            if key not in node_map:
                if rescued >= _TOP_K_ROLES:
                    continue
                score, probability = raw_scores.get(key, (float("-inf"), 0.0))
                node_map[key] = NodeCandidate(entity_type, start, end, score, probability, rescued=True)
                rescued += 1
            bound.append((raw - offset, node_map[key], sigmoid(raw)))
    return bound


def _span_problem(scores, schema: CompiledJointSchema, threshold: float) -> JointProblem:
    """Build nodes from the dense span lattice and rescue relation endpoints."""
    entity_types = tuple(scores["entity_types"] or schema.entity_order)
    if scores["entity_logits"].shape[0] != len(entity_types):
        raise ValueError(f"entity logits must have shape [types, L, W] for {len(entity_types)} types")
    node_map, raw_scores = {}, {}
    for type_index, entity_type in enumerate(entity_types):
        spec = schema.entity_specs.get(entity_type) or EntitySpec(entity_type)
        floor = threshold if spec.candidate_threshold is None else spec.candidate_threshold
        candidates = []
        for raw, start, end in _span_entries(scores["entity_logits"][type_index]):
            if math.isfinite(raw):
                score, probability = center_logit(raw, spec.threshold or 0.5), sigmoid(raw)
                raw_scores[(entity_type, start, end)] = (score, probability)
                if probability >= floor:
                    candidates.append(NodeCandidate(entity_type, start, end, score, probability))
        cap = _TOP_K_ENTITIES if spec.max_candidates is None else spec.max_candidates
        node_map.update((node.key, node) for node in sorted(candidates, key=_node_rank)[:cap])
    edge_groups: dict = {}
    for hypothesis in scores["relation_hypotheses"]:
        floor = threshold if hypothesis.candidate_threshold is None else hypothesis.candidate_threshold
        offset = probability_to_logit(0.5 if hypothesis.threshold is None else hypothesis.threshold)
        for count_slot, roles in enumerate(hypothesis.role_logits):
            typed_roles = []
            for role, types in enumerate((hypothesis.head_types, hypothesis.tail_types)):
                entries = [
                    entry
                    for entry in _span_entries(roles[role])
                    if math.isfinite(entry[0]) and sigmoid(entry[0]) >= floor
                ]
                entries.sort(key=lambda item: (-item[0], item[1], item[2]))
                typed_roles.append(_bind_role(node_map, raw_scores, entries[:_TOP_K_ROLES], types, offset))
            pairs = [(hs + ts, head, tail, hp, tp) for (hs, head, hp), (ts, tail, tp) in product(*typed_roles)]
            pairs.sort(key=lambda item: (-item[0], str(item[1].candidate_id), str(item[2].candidate_id)))
            edge_groups.setdefault(hypothesis.relation_type, []).extend(
                EdgeCandidate(
                    hypothesis.relation_type,
                    head.candidate_id,
                    tail.candidate_id,
                    score,
                    head_probability=head_prob,
                    tail_probability=tail_prob,
                    head_entity_probability=head.probability,
                    tail_entity_probability=tail.probability,
                    count_probability=1.0,
                    slot=count_slot,
                    hypothesis=hypothesis.relation_type,
                    count_alternative=0,
                )
                for score, head, tail, head_prob, tail_prob in pairs[:_PAIR_CAP]
            )
    edges = []
    for relation_type in sorted(edge_groups):
        unique = {}
        for edge in sorted(edge_groups[relation_type], key=_edge_rank):
            if edge.key not in unique or edge.score > unique[edge.key].score:
                unique[edge.key] = edge
        edges.extend(sorted(unique.values(), key=_edge_rank)[:_MAX_EDGES_PER_TYPE])
    nodes = tuple(sorted(node_map.values(), key=lambda node: (node.entity_type, node.start, node.end, -node.score)))
    return JointProblem(nodes, tuple(sorted(edges, key=_edge_rank)), schema.constraints)


def _boundary_problem(sample, row, schema: CompiledJointSchema, threshold: float) -> JointProblem:
    """Build entity nodes from one boundary row's candidates."""
    candidates, best = sample["candidates"], {}
    for query_id, spec in enumerate(_query_specs(row["groups"])):
        if spec["task_type"] != "entities" or query_id >= candidates.indices.shape[1]:
            continue
        valid = candidates.valid_mask[0, query_id] & candidates.query_mask[0, query_id]
        for candidate_id in valid.nonzero(as_tuple=False).flatten().tolist():
            start, end = (int(value) - row["prefix_len"] for value in candidates.indices[0, query_id, candidate_id])
            logit = float(candidates.pair_logits[0, query_id, candidate_id].detach().float())
            key = (str(spec["field_name"]), start, end)
            if 0 <= start < end <= len(row["start"]) and (key not in best or logit > best[key]):
                best[key] = logit
    mentions = sorted(
        ((key, logit, sigmoid(logit)) for key, logit in best.items()), key=lambda m: (m[0][0], -m[2], *m[0][1:])
    )
    nodes, per_type = [], defaultdict(int)
    for (entity_type, start, end), logit, probability in mentions:
        spec = schema.entity_specs.get(entity_type) or EntitySpec(entity_type)
        floor = threshold if spec.candidate_threshold is None else spec.candidate_threshold
        limit = _TOP_K_ENTITIES if spec.max_candidates is None else spec.max_candidates
        if probability >= floor and per_type[entity_type] < limit:
            per_type[entity_type] += 1
            decision = 0.5 if spec.threshold is None else float(spec.threshold)
            nodes.append(NodeCandidate(entity_type, start, end, center_logit(logit, decision), probability))
    return JointProblem(tuple(nodes), (), schema.constraints)


def _edge_usage(edge: EdgeCandidate) -> set:
    return set() if edge.slot is None else {("slot", edge.hypothesis, edge.count_alternative, edge.slot)}


def _try_edge(problem, selected, edges, used, score, edge):
    """Shared greedy and beam expansion. Head==tail is listed twice."""
    if _edge_usage(edge) & used:
        return None
    node_by_id = problem.node_by_id
    new_ids = [value for value in (edge.head, edge.tail) if value not in selected]
    proposed = [node_by_id[value] for value in new_ids]
    proposed_nodes = [node_by_id[value] for value in selected] + proposed
    if not all(_node_allowed(problem, node, proposed_nodes) for node in proposed):
        return None
    if not _edge_allowed(problem, edge, edges):
        return None
    gain = edge.score + sum(node.score for node in proposed)
    if gain < 0.0:
        return None
    return set(selected) | set(new_ids), list(edges) + [edge], used | _edge_usage(edge), score + gain


def _add_positive_nodes(problem, selected, edges, score):
    node_by_id, selected = problem.node_by_id, set(selected)
    for node in sorted(problem.nodes, key=lambda item: (-item.score, item.entity_type, item.start, item.end)):
        if node.candidate_id in selected or node.score <= 0.0:
            continue
        if _node_allowed(problem, node, [node_by_id[value] for value in selected] + [node]):
            selected.add(node.candidate_id)
            score += node.score
    return selected, score


def _make_solution(problem, node_ids, edges, score) -> JointSolution:
    """Selected nodes plus edges, completed with symmetric and inverse companions."""
    chosen = list(edges)
    keys = {(edge.relation_type, edge.head, edge.tail) for edge in chosen}
    for constraint in problem.constraints:
        if constraint["type"] not in ("SymmetricRelation", "InverseRelation"):
            continue
        relation = constraint["relation"]
        inverse = constraint.get("inverse", relation)
        for source in list(chosen):
            if source.derived or source.relation_type not in (relation, inverse):
                continue
            label = inverse if source.relation_type == relation else relation
            if (label, source.tail, source.head) in keys:
                continue
            keys.add((label, source.tail, source.head))
            chosen.append(
                EdgeCandidate(
                    label,
                    source.tail,
                    source.head,
                    0.0,
                    head_probability=source.tail_probability,
                    tail_probability=source.head_probability,
                    head_entity_probability=source.tail_entity_probability,
                    tail_entity_probability=source.head_entity_probability,
                    count_probability=source.count_probability,
                    derived=True,
                    candidate_id=("derived", label, source.tail, source.head),
                )
            )
    chosen.sort(key=lambda edge: (edge.relation_type, str(edge.head), str(edge.tail), edge.derived))
    node_ids = set(node_ids)
    nodes = tuple(node for node in problem.nodes if node.candidate_id in node_ids)
    return JointSolution(nodes, tuple(chosen), float(score))


def _valid_solution(problem, solution: JointSolution) -> bool:
    nodes, edges = [], []
    for node in solution.nodes:
        if not _node_allowed(problem, node, nodes + [node]):
            return False
        nodes.append(node)
    for edge in solution.edges:
        if not _edge_allowed(problem, edge, edges):
            return False
        edges.append(edge)
    keys = {(edge.relation_type, edge.head, edge.tail) for edge in solution.edges}
    for constraint in problem.constraints:
        if constraint["type"] in ("SymmetricRelation", "InverseRelation"):
            relation = constraint["relation"]
            inverse = constraint.get("inverse", relation)
            for edge in solution.edges:
                if edge.relation_type == relation and (inverse, edge.tail, edge.head) not in keys:
                    return False
                if edge.relation_type == inverse and (relation, edge.tail, edge.head) not in keys:
                    return False
    return True


def _expansion_order(edge: EdgeCandidate, nodes: Mapping, greedy: bool) -> tuple:
    """Greedy counts a self-loop endpoint once and breaks ties on `-edge.score`."""
    head, tail = nodes[edge.head].score, nodes[edge.tail].score
    ties = (edge.relation_type, str(edge.hypothesis), str(edge.slot), str(edge.head), str(edge.tail))
    if greedy:
        return (-(edge.score + (head if edge.tail == edge.head else head + tail)), -edge.score, *ties)
    return (-(edge.score + head + tail), *ties)


def _optimize_greedy(problem: JointProblem) -> JointSolution:
    node_by_id = problem.node_by_id
    selected, edges, used, score = set(), [], set(), 0.0
    for edge in sorted(problem.edges, key=lambda item: _expansion_order(item, node_by_id, True)):
        expanded = _try_edge(problem, selected, edges, used, score, edge)
        if expanded is not None:
            selected, edges, used, score = expanded
    selected, score = _add_positive_nodes(problem, selected, edges, score)
    result = _make_solution(problem, selected, edges, score)
    if _valid_solution(problem, result):
        return result
    logger.warning("greedy joint-IE decoding produced a constraint-violating assignment; returning empty solution")
    return JointSolution((), (), 0.0, feasible=False)


def _solution_key(solution: JointSolution):
    ids = tuple(sorted(str(node.candidate_id) for node in solution.nodes))
    return (solution.score, ids, tuple(str(edge.candidate_id) for edge in solution.edges))


def _state_order(state) -> tuple:
    selected, edges, _, score = state
    return (-score, tuple(sorted(map(str, selected))), tuple(str(edge.candidate_id) for edge in edges))


def _optimize_beam(problem: JointProblem) -> JointSolution:
    """Beam over the same edge expansion greedy uses, without a suffix bound."""
    node_by_id = problem.node_by_id
    beam = [(set(), [], set(), 0.0)]
    for edge in sorted(problem.edges, key=lambda item: _expansion_order(item, node_by_id, False)):
        unique = {}
        for state in beam + [_try_edge(problem, *state, edge) for state in beam]:
            if state is None:
                continue
            selected, edges, used, score = state
            key = (frozenset(selected), frozenset(edge.candidate_id for edge in edges), frozenset(used))
            if key not in unique or score > unique[key][3]:
                unique[key] = state
        beam = sorted(unique.values(), key=_state_order)[:_BEAM_SIZE]
    candidates = []
    for selected, edges, _, score in beam:
        selected, score = _add_positive_nodes(problem, selected, edges, score)
        candidates.append(_make_solution(problem, selected, edges, score))
    candidates.append(_optimize_greedy(problem))
    feasible = [solution for solution in candidates if _valid_solution(problem, solution)]
    if feasible:
        return max(feasible, key=_solution_key)
    logger.warning("joint-IE decoding found no constraint-satisfying assignment among %d candidates", len(candidates))
    return JointSolution((), (), 0.0, feasible=False)


def _component_scores(value) -> list:
    """Probabilities that make up one entity or relation confidence."""
    if isinstance(value, NodeCandidate):
        return [] if value.probability is None else [float(value.probability)]
    parts = (
        value.head_probability,
        value.tail_probability,
        value.head_entity_probability,
        value.tail_entity_probability,
        value.count_probability,
    )
    return [float(score) for score in parts if score is not None]


def _joint_result_dict(solution, text, starts, ends, include_confidence, include_spans) -> dict:
    rows = []
    for node in solution.nodes:
        char_start, char_end = node.start, node.end
        if starts is not None:
            char_start, char_end = int(starts[node.start]), int(ends[max(node.start, node.end - 1)])
        rows.append((node, char_start, char_end, text[char_start:char_end]))
    rows.sort(key=lambda item: (item[1], item[2], item[0].entity_type, item[3]))
    id_of, entities = {}, []
    for index, (node, char_start, char_end, surface) in enumerate(rows):
        id_of[node.candidate_id] = f"e{index + 1}"
        payload = {"id": id_of[node.candidate_id], "type": node.entity_type, "text": surface}
        if include_spans:
            payload.update(start=char_start, end=char_end)
        confidence = _geometric_mean(_component_scores(node)) if include_confidence else None
        if confidence is not None:
            payload["confidence"] = confidence
        if node.rescued:
            payload["rescued"] = True
        entities.append(payload)
    relations = []
    for edge in solution.edges:
        if edge.head not in id_of or edge.tail not in id_of:
            raise ValueError("relation endpoint does not identify a selected entity")
        payload = {"type": edge.relation_type, "head": id_of[edge.head], "tail": id_of[edge.tail]}
        confidence = _geometric_mean(_component_scores(edge)) if include_confidence else None
        if confidence is not None:
            payload["confidence"] = confidence
        if edge.derived:
            payload["derived"] = True
        relations.append(payload)
    relations.sort(key=lambda item: (item["type"], item["head"], item["tail"]))
    return {"entities": entities, "relations": relations}


def _doc_axis(tensor, doc_len):
    """Drop the choice prefix from the word axis of `(..., words, width)` scores."""
    if doc_len <= 0:
        return tensor[..., :0, :]
    return tensor[..., -doc_len:, :]


def _span_scores_from_sample(sample, row, schema):
    """Entity matrix and relation hypotheses for the span-architecture joint decoder."""
    groups = [group for group in row["groups"] if group.task != "classifications"]
    tensors = sample.get("span_logits")
    if tensors is None:
        raise TypeError("span joint decoding needs span_logits")
    if torch.is_tensor(tensors):
        tensors = [tensors]
    counts = sample.get("counts")
    if torch.is_tensor(counts):
        counts = counts.detach().cpu().tolist()
    doc_len = len(row["start"])
    entity_logits, entity_types, hypotheses = None, (), []
    for index, group in enumerate(groups):
        tensor = torch.as_tensor(tensors[index], dtype=torch.float).detach().cpu()
        if tensor.ndim != 4:
            raise ValueError("span logits must have shape (count, fields, words, width)")
        count = int(counts[index]) if counts is not None else int(tensor.shape[0])
        tensor = _doc_axis(tensor[: max(count, 0)], doc_len)
        if group.task == "entities":
            entity_types = tuple(item.name for item in group.fields)
            empty = tensor.new_zeros((len(entity_types), doc_len, tensor.shape[-1]))
            entity_logits = tensor[0] if tensor.shape[0] else empty
        elif group.task == "relations" and tensor.shape[0] > 0 and tensor.shape[1] >= 2:
            spec = schema.relation_specs.get(group.name)
            hypotheses.append(
                RelationHypothesis(
                    group.name,
                    tensor[:, :2],
                    getattr(spec, "head", ()),
                    getattr(spec, "tail", ()),
                    getattr(spec, "threshold", None),
                    getattr(spec, "candidate_threshold", None),
                )
            )
    return {
        "entity_logits": entity_logits if entity_logits is not None else torch.empty((0, doc_len, 1)),
        "entity_types": entity_types,
        "relation_hypotheses": hypotheses,
    }


def decode_joint_sample(
    sample,
    row,
    *,
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
    optimizer: str = "beam",
    overlap_policy: str | None = None,
) -> dict:
    """Decode one processor row with the joint entity and relation optimizer.

    Args:
        sample (`dict`):
            One row of model output: `span_logits` for span checkpoints, or
            boundary `candidates`.
        row (`dict`):
            Processor metadata with `groups`, `schema`, `text`, and word offsets.
        threshold (`float`, *optional*, defaults to 0.5):
            Candidate, role, and entity cutoff.
        include_confidence (`bool`, *optional*, defaults to `False`):
            Keep scores.
        include_spans (`bool`, *optional*, defaults to `False`):
            Keep character offsets.
        optimizer (`str`, *optional*, defaults to `"beam"`):
            `beam`, `greedy`, `auto`, or `exact`, which runs beam search.
        overlap_policy (`str`, *optional*):
            Entity overlap rule added as a constraint.
    """
    source = row.get("schema") or {}
    constraints = [
        raw for raw in source.get("constraints") or [] if isinstance(raw, Mapping) and raw.get("type") in _JOINT_FIELDS
    ]
    schema = _compile_joint(
        {
            "entities": source.get("entities") or {},
            "relations": source.get("relations") or [],
            "constraints": constraints,
        }
    )
    policy = None if overlap_policy is None else normalize_overlap_policy(overlap_policy)
    if policy is not None:
        kept = [item for item in schema.constraints if item["type"] != "EntityOverlapPolicy"]
        if policy != "longest":
            kept.append(_joint_constraint("EntityOverlapPolicy", policy=policy))
        schema = replace(schema, constraints=tuple(kept))
    if row["architecture"] == "boundary":
        problem = _boundary_problem(sample, row, schema, threshold)
    else:
        problem = _span_problem(_span_scores_from_sample(sample, row, schema), schema, threshold)
    solution = _optimize_greedy(problem) if optimizer.lower() in ("auto", "greedy") else _optimize_beam(problem)
    if policy == "longest":
        kept = resolve_overlaps(
            list(solution.nodes),
            "longest",
            score=lambda node: node.score,
            start=lambda node: node.start,
            end=lambda node: node.end,
        )
        ids = {node.candidate_id for node in kept}
        edges = [edge for edge in solution.edges if edge.head in ids and edge.tail in ids and not edge.derived]
        solution = _make_solution(problem, ids, edges, solution.score)
    starts = tuple(row["start"]) or None
    return _joint_result_dict(solution, row["text"], starts, tuple(row["end"]), include_confidence, include_spans)
