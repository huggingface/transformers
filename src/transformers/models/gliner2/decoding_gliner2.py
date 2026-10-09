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
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations, product
from types import SimpleNamespace
from typing import Any, Literal

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
_DECODERS = ("auto", "independent", "exact", "beam", "min_violations")


class _LatticeTag(str, Enum):
    """Labels that still serialize as count-choice, derived, and rescue."""

    COUNT_CHOICE = "count-choice"
    DERIVED = "derived"
    RESCUE = "rescue"


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


@dataclass
class DecodedRecord:
    """One decoded record."""

    fields: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    field_scores: dict[int, list[float]] = field(default_factory=dict)
    anchor_span: tuple[int, int] | None = None
    score: float = 0.0


def _dedup_key(record: DecodedRecord) -> tuple:
    """Return a span key that identifies duplicate decoded records."""
    return tuple((qid, tuple(sorted(spans))) for qid, spans in sorted(record.fields.items()))


def _select_record_instances(group, anchor_threshold, object_threshold, temperature):
    """Instances that clear the object or anchor threshold, best score first.

    Args:
        group: Record group with object logits and a spec mode.
        anchor_threshold: Minimum probability for anchored modes.
        object_threshold: Minimum probability for anchorless groups.
        temperature: Positive divisor applied to object logits.

    Returns:
        Selected instance indexes and their object probabilities.
    """
    count = int(group.num_instances)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    obj_prob = torch.sigmoid(group.object_logits.detach() / temperature)
    select_thr = object_threshold if group.spec.mode == "anchorless" else anchor_threshold
    order = sorted(range(count), key=lambda index: (-float(obj_prob[index]), index))
    selected = [inst for inst in order if float(obj_prob[inst]) >= select_thr]
    return selected, obj_prob


def _exclusive_field_choices(group, selected_instances, field_threshold, temperature):
    """Exclusive scalar assignments and list owners for the selected instances.

    Args:
        group: Record group whose exclusive fields are assigned jointly.
        selected_instances: Instance indexes already above the object threshold.
        field_threshold: Minimum probability kept for an assigned span.
        temperature: Positive divisor applied to assignment logits.

    Returns:
        Scalar ``(candidate, probability)`` choices and list-field owners.
    """
    scalar_choices: dict[tuple[int, int], tuple[int, float] | None] = {}
    list_owners: dict[tuple[int, int], tuple[int, float]] = {}
    for f_idx, fspec in enumerate(group.field_specs):
        if not fspec.exclusive or not selected_instances:
            continue
        logits = torch.stack([group.assign_logits[f_idx][inst].detach() / temperature for inst in selected_instances])
        candidate_count = max(int(logits.shape[-1]) - 1, 0)
        if fspec.cardinality.is_scalar:
            _assign_exclusive_scalar(
                scalar_choices, f_idx, fspec, logits, selected_instances, candidate_count, field_threshold
            )
        elif candidate_count:
            _assign_exclusive_list(list_owners, f_idx, logits, selected_instances, candidate_count, field_threshold)
    return scalar_choices, list_owners


def _assign_exclusive_scalar(
    scalar_choices, f_idx, fspec, logits, selected_instances, candidate_count, field_threshold
):
    """Hungarian assignment of one exclusive scalar field.

    Args:
        scalar_choices: Destination map keyed by instance and field index.
        f_idx: Field index inside the record group.
        fspec: Field spec, including whether absence is allowed.
        logits: Assignment logits for the selected instances.
        selected_instances: Instance indexes corresponding to logit rows.
        candidate_count: Number of real spans, excluding the absent column.
        field_threshold: Minimum probability kept when absence is allowed.
    """
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
        if col >= candidate_count:
            scalar_choices[(inst, f_idx)] = None
            continue
        probability = float(candidate_probs[row, col])
        if probability < field_threshold and fspec.allows_absent:
            scalar_choices[(inst, f_idx)] = None
            continue
        scalar_choices[(inst, f_idx)] = (col, probability)


def _assign_exclusive_list(list_owners, f_idx, logits, selected_instances, candidate_count, field_threshold):
    """Give each list candidate to the selected instance with the highest score.

    Args:
        list_owners: Destination map keyed by field and candidate index.
        f_idx: Field index inside the record group.
        logits: Assignment logits for the selected instances.
        selected_instances: Instance indexes corresponding to logit rows.
        candidate_count: Number of real spans, excluding the absent column.
        field_threshold: Minimum probability required to claim a candidate.
    """
    probabilities = torch.sigmoid(logits[:, 1:])
    for cand_idx in range(candidate_count):
        probability, row = probabilities[:, cand_idx].max(dim=0)
        if float(probability) >= field_threshold:
            list_owners[(f_idx, cand_idx)] = (selected_instances[int(row)], float(probability))


def _append_scalar_span(rec, fspec, spans_tensor, logits_row, choice, field_threshold):
    """Append one scalar span chosen independently or by the exclusive assignment.

    Args:
        rec: Record receiving the span.
        fspec: Field spec for this column.
        spans_tensor: Candidate ``(start, end)`` rows.
        logits_row: Assignment logits for this instance, used when not exclusive.
        choice: Exclusive ``(candidate, probability)``, or ``None`` when absent.
        field_threshold: Minimum probability kept when absence is allowed.

    Returns:
        Nothing. The record is updated in place.
    """
    qid = fspec.query_id
    if fspec.exclusive:
        if choice is None:
            return
        cand_idx, probability = choice
        span = (int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1]))
        rec.fields.setdefault(qid, []).append(span)
        rec.field_scores.setdefault(qid, []).append(probability)
        return
    probs = torch.softmax(logits_row, dim=-1)
    chosen = None
    for col in torch.argsort(probs, descending=True).tolist():
        if col == 0:
            if fspec.allows_absent:
                chosen = 0
                break
            continue
        chosen = col
        break
    if chosen is None or chosen == 0:
        return
    if float(probs[chosen]) < field_threshold and fspec.allows_absent:
        return
    cand_idx = chosen - 1
    span = (int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1]))
    rec.fields.setdefault(qid, []).append(span)
    rec.field_scores.setdefault(qid, []).append(float(probs[chosen]))


def _append_list_spans(rec, fspec, f_idx, spans_tensor, logits_row, list_owners, inst, field_threshold):
    """Append list spans this instance owns or that clear the field threshold.

    Args:
        rec: Record receiving the spans.
        fspec: Field spec for this column.
        f_idx: Field index used to look up exclusive owners.
        spans_tensor: Candidate ``(start, end)`` rows.
        logits_row: Assignment logits for this instance.
        list_owners: Exclusive owners keyed by field and candidate.
        inst: Instance index being filled.
        field_threshold: Minimum probability for a non-exclusive span.
    """
    cand_logits = logits_row[1:]
    if cand_logits.numel() == 0:
        return
    probs = torch.sigmoid(cand_logits)
    selected: list[tuple[int, int]] = []
    selected_scores: list[float] = []
    for cand_idx in range(cand_logits.shape[0]):
        if fspec.exclusive:
            owner = list_owners.get((f_idx, cand_idx))
            if owner is None or owner[0] != inst:
                continue
            probability = owner[1]
        else:
            probability = float(probs[cand_idx])
            if probability < field_threshold:
                continue
        span = (int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1]))
        selected.append(span)
        selected_scores.append(probability)
    if selected:
        qid = fspec.query_id
        rec.fields.setdefault(qid, []).extend(selected)
        rec.field_scores.setdefault(qid, []).extend(selected_scores)


def _record_from_instance(group, inst, obj_prob, scalar_choices, list_owners, field_threshold, temperature):
    """Build one record from exclusive choices and per-instance logits.

    Args:
        group: Record group supplying field specs and spans.
        inst: Selected instance index.
        obj_prob: Object probabilities indexed by instance.
        scalar_choices: Exclusive scalar assignments.
        list_owners: Exclusive list-candidate owners.
        field_threshold: Minimum probability for a kept span.
        temperature: Positive divisor applied to assignment logits.

    Returns:
        The record when it has at least one field, otherwise ``None``.
    """
    rec = DecodedRecord(score=float(obj_prob[inst]))
    anchor_field_idx = None
    if group.spec.mode == "natural":
        anchor_field_idx = group.field_query_ids.index(group.spec.anchor_query_id)
        if group.instance_seed[inst] is not None:
            rec.anchor_span = group.instance_spans[inst]
    for f_idx, fspec in enumerate(group.field_specs):
        spans_tensor = group.field_spans[f_idx]
        logits_row = group.assign_logits[f_idx][inst].detach() / temperature
        if anchor_field_idx is not None and f_idx == anchor_field_idx:
            if rec.anchor_span is not None:
                rec.fields.setdefault(fspec.query_id, []).append(rec.anchor_span)
                rec.field_scores.setdefault(fspec.query_id, []).append(rec.score)
            continue
        if fspec.cardinality.is_scalar:
            _append_scalar_span(
                rec, fspec, spans_tensor, logits_row, scalar_choices.get((inst, f_idx)), field_threshold
            )
        else:
            _append_list_spans(rec, fspec, f_idx, spans_tensor, logits_row, list_owners, inst, field_threshold)
    return rec if rec.fields else None


def _collapse_records(group, records):
    """Deduplicate latent records or sort natural records by anchor span.

    Args:
        group: Record group whose spec mode selects the collapse.
        records: Records that already contain at least one field.

    Returns:
        The collapsed record list.
    """
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


def _decode_group(
    group,
    *,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
    temperature: float = 1.0,
) -> list[DecodedRecord]:
    """Assign field spans for one record group."""
    if int(group.num_instances) == 0:
        return []
    selected_instances, obj_prob = _select_record_instances(group, anchor_threshold, object_threshold, temperature)
    scalar_choices, list_owners = _exclusive_field_choices(group, selected_instances, field_threshold, temperature)
    records = []
    for inst in selected_instances:
        rec = _record_from_instance(group, inst, obj_prob, scalar_choices, list_owners, field_threshold, temperature)
        if rec is not None:
            records.append(rec)
    return _collapse_records(group, records)


def decode_group(
    group,
    *,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
    temperature: float = 1.0,
) -> list[DecodedRecord]:
    """Decode one record group into selected field spans."""
    return _decode_group(
        group,
        anchor_threshold=anchor_threshold,
        field_threshold=field_threshold,
        object_threshold=object_threshold,
        temperature=temperature,
    )


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


def _coerce_logits(value: Any, *, n_rows: int | None = None, as_shape: bool = False):
    """Coerce logits to a float, a vector, row slices, or a shape.

    Args:
        value: Scalar, sequence, matrix, or nested logits.
        n_rows: Split a matrix into this many rows.
        as_shape: Return the shape instead of the values.

    Returns:
        A Python float, a list of values, a list of rows, or a shape tuple.
        Callers trim a longer all-non-finite tail themselves.

    Raises:
        SchemaError: If a label vector or row block has the wrong kind.
    """
    if as_shape:
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
    if n_rows is not None:
        shape = getattr(value, "shape", None)
        if shape is not None and len(tuple(shape)) == 2:
            if int(shape[0]) != n_rows:
                raise SchemaError(f"expected {n_rows} task rows, got {int(shape[0])}")
            return [value[index] for index in range(n_rows)]
        if isinstance(value, (list, tuple)):
            if len(value) != n_rows:
                raise SchemaError(f"expected {n_rows} task logit rows, got {len(value)}")
            return list(value)
        raise SchemaError("logits must be a task mapping or one row per task")
    if isinstance(value, (str, bytes)):
        raise SchemaError("label logits must be a mapping or a sequence")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "ndim") and int(value.ndim) == 0:
        return float(value.item())
    if hasattr(value, "shape"):
        shape = tuple(int(dim) for dim in value.shape)
        if len(shape) == 0:
            return float(value.item())
        listed = value.tolist()
        return listed if isinstance(listed, list) else float(listed)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, Mapping)):
        listed = value.tolist()
        return listed if isinstance(listed, list) else float(listed)
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


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
    """Rank spans and drop later copies of the same boundaries.

    Args:
        spans: Half-open spans in input order.
        score_fn: Score used as the primary rank, higher first.
        start_fn: Inclusive start offset.
        end_fn: Exclusive end offset.

    Returns:
        Distinct ``(index, span)`` rows and the rank key used to order them.
    """

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
    """Keep spans that nest inside earlier picks instead of crossing them.

    Args:
        distinct: Ranked ``(index, span)`` rows with unique boundaries.
        start_fn: Inclusive start offset.
        end_fn: Exclusive end offset.

    Returns:
        Spans whose overlaps are proper containment.
    """
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
    """Drop spans strictly contained in any other distinct span.

    Args:
        distinct: Ranked ``(index, span)`` rows with unique boundaries.
        start_fn: Inclusive start offset.
        end_fn: Exclusive end offset.

    Returns:
        Spans that are not strictly inside another candidate.
    """
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
    """Weighted interval selection. Equal scores prefer more spans, then rank.

    Args:
        distinct: Ranked ``(index, span)`` rows with unique boundaries.
        score_fn: Additive span score.
        start_fn: Inclusive start offset.
        end_fn: Exclusive end offset.
        rank_key: Tie-break used when score and span count match.

    Returns:
        A non-overlapping subset ordered by ``rank_key``.
    """
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
        policy,
        score=lambda item: item[0],
        start=lambda item: item[1],
        end=lambda item: item[2],
    )


def format_span(surface, score, char_start, char_end, include_confidence, include_spans):
    """Public span payload."""
    if include_spans and include_confidence:
        return {"text": surface, "confidence": score, "start": char_start, "end": char_end}
    if include_spans:
        return {"text": surface, "start": char_start, "end": char_end}
    if include_confidence:
        return {"text": surface, "confidence": score}
    return surface


def _passes_text(surface: str, validators) -> bool:
    for validator in validators or []:
        if hasattr(validator, "validate") and not validator.validate(surface):
            return False
    return True


def _matrix_spans(scores: torch.Tensor, threshold: float, text: str, start_map, end_map) -> list:
    """Spans at or above `threshold` as `(text, score, start, end)`."""
    doc_len = len(start_map)
    found = []
    starts, widths = torch.where(scores >= threshold)
    for start, width in zip(starts.tolist(), widths.tolist()):
        end = start + int(width) + 1
        if not (0 <= start < doc_len and end <= doc_len):
            continue
        try:
            char_start, char_end = start_map[start], end_map[end - 1]
            span_text = text[char_start:char_end].strip()
        except (IndexError, KeyError):
            continue
        if span_text:
            found.append((span_text, float(scores[start, width].item()), int(char_start), int(char_end)))
    return found


def decode_spans(scores, group, text, maps, threshold, overlap):
    """Threshold, resolve overlaps, and format one field."""
    include_confidence = bool(group.get("include_confidence", False))
    include_spans = bool(group.get("include_spans", False))
    dtype = group.get("dtype") or "list"
    validators = group.get("validators") or []
    field_threshold = group.get("threshold", threshold)
    if field_threshold is None:
        field_threshold = threshold
    if isinstance(maps, Mapping):
        start_map, end_map = maps.get("start") or [], maps.get("end") or []
        offset = int(maps.get("offset") or 0)
    else:
        start_map, end_map = maps[0], maps[1]
        offset = int(maps[2]) if len(maps) > 2 else 0
    raw = []
    extras = dict(group.get("extras") or {})
    if torch.is_tensor(scores):
        for surface, score, char_start, char_end in _matrix_spans(
            scores, float(field_threshold), text, start_map, end_map
        ):
            if _passes_text(surface, validators):
                raw.append((surface, score, char_start, char_end))
        selected = resolve_overlaps(
            raw,
            overlap,
            score=lambda span: span[1],
            start=lambda span: span[2],
            end=lambda span: span[3],
        )
    else:
        token_rows = []
        char_rows = []
        for item in scores or []:
            if len(item) >= 4 and isinstance(item[0], str):
                surface, score, char_start, char_end = item[0], item[1], item[2], item[3]
                if surface and _passes_text(surface, validators):
                    char_rows.append((surface, float(score), int(char_start), int(char_end)))
            else:
                token_rows.append((float(item[0]), int(item[1]), int(item[2])))
        if token_rows:
            kept = resolve_overlaps(
                token_rows,
                overlap,
                score=lambda span: span[0],
                start=lambda span: span[1],
                end=lambda span: span[2],
            )
            for score, start, end in kept:
                mapped = _char_span(start, end, offset, start_map, end_map, text)
                if mapped is not None and _passes_text(mapped[0], validators):
                    char_rows.append((mapped[0], score, mapped[1], mapped[2]))
        elif char_rows:
            char_rows = resolve_overlaps(
                char_rows,
                overlap,
                score=lambda span: span[1],
                start=lambda span: span[2],
                end=lambda span: span[3],
            )
        selected = char_rows
    if dtype != "list":
        selected = selected[:1]
    formatted = []
    for surface, score, char_start, char_end in selected:
        payload = format_span(surface, score, char_start, char_end, include_confidence, include_spans)
        extra = extras.get((char_start, char_end))
        if extra:
            payload = {"text": surface, **extra} if not isinstance(payload, dict) else {**payload, **extra}
        formatted.append(payload)
    if dtype == "list":
        return formatted
    if formatted:
        return formatted[0]
    if group.get("empty") == "blank":
        return "" if not include_spans and not include_confidence else None
    return None


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
            canonical[coordinates] = max(
                containing, key=lambda candidate: (candidate[2] - candidate[1], -candidate[1])
            )
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
        return {
            "head": {"text": head, "confidence": score},
            "tail": {"text": tail, "confidence": score},
        }
    return (head, tail)


def _decode_relations(sample_out, meta, threshold, include_confidence, include_spans):
    """Map forward relation pairs and logits onto character offsets."""
    pairs = sample_out.get("relation_pairs")
    logits = sample_out.get("relation_logits")
    if pairs is None or logits is None or pairs.numel() == 0:
        return {}
    temperature = float(sample_out.get("relation_temperature") or 1.0)
    probabilities = torch.sigmoid(logits.detach().float().cpu() / temperature)
    relation_groups = [group for group in (meta.get("groups") or []) if group.get("task_type") == "relations"]
    aliases = {
        f"{name}: {description}": name for name, description in (meta.get("relation_descriptions") or {}).items()
    }
    relation_metadata = meta.get("relation_metadata") or {}
    offset = int(meta.get("prefix_len") or 0)
    start_map = list(meta.get("start") or [])
    end_map = list(meta.get("end") or [])
    text = meta.get("text") or ""
    edges = {}
    for pair_index, probability in enumerate(probabilities):
        relation_index = int(pairs[pair_index, 1])
        if relation_index < 0 or relation_index >= len(relation_groups):
            continue
        relation_name = relation_groups[relation_index]["name"]
        relation_type = aliases.get(relation_name, relation_name)
        relation_threshold = relation_metadata.get(relation_type, {}).get("threshold", threshold)
        if relation_threshold is None:
            relation_threshold = threshold
        score = float(probability.detach())
        if score < relation_threshold:
            continue
        head = _char_span(
            int(pairs[pair_index, 2]),
            int(pairs[pair_index, 3]),
            offset,
            start_map,
            end_map,
            text,
        )
        tail = _char_span(
            int(pairs[pair_index, 4]),
            int(pairs[pair_index, 5]),
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
            format_relation(edge, include_confidence, include_spans)
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
    """Decode one boundary sample into the pre-format result dict."""
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
    maps = {"start": start_map, "end": end_map, "offset": offset}
    entities = OrderedDict()
    structures = OrderedDict()
    field_options = {}
    for group in meta.get("groups") or []:
        stored = (group.get("options") or {}).get("fields") or {}
        for field_name, options in stored.items():
            field_options[(group.get("name"), field_name)] = options
    for query_id, spec in enumerate(specs):
        if query_id >= len(grouped):
            break
        if null_logits is not None and float(torch.sigmoid(null_logits[query_id])) > abstain:
            scored = []
        else:
            scored = grouped[query_id]
        options = field_options.get((spec["task_name"], spec["field_name"]), {})
        dtype = "list" if spec["task_type"] == "entities" else "str"
        decoded = decode_spans(
            scored,
            {
                "dtype": dtype,
                "threshold": options.get("threshold", threshold),
                "validators": options.get("validators") or [],
                "include_confidence": include_confidence,
                "include_spans": include_spans,
                "empty": "blank" if spec["task_type"] == "entities" else None,
            },
            text,
            maps,
            threshold,
            overlap_policy,
        )
        if spec["task_type"] == "entities":
            entities[spec["field_name"]] = decoded
        elif spec["task_type"] == "json_structures":
            structures.setdefault(spec["task_name"], OrderedDict())[spec["field_name"]] = decoded
    result = {}
    if entities:
        result["entities"] = [entities]
    for name, instance in structures.items():
        if any(value is not None and value != [] for value in instance.values()):
            result[name] = [instance]
    result.update(_decode_relations(sample_out, meta, threshold, include_confidence, include_spans))
    for group in meta.get("groups") or []:
        if group["task_type"] == "relations" and group["name"] not in result:
            result[group["name"]] = []
    recorded = decode_records(
        sample_out,
        meta,
        threshold=threshold,
        include_confidence=include_confidence,
        include_spans=include_spans,
        overlap_policy=overlap_policy,
    )
    result.update(recorded)
    return result


def _records_enabled(sample_out, meta) -> bool:
    """False when records are off or the checkpoint predates config version 3."""
    version = sample_out.get("config_version", meta.get("config_version"))
    if version is not None and int(version) < 3:
        return False
    enabled = sample_out.get("enable_records", meta.get("enable_records"))
    if enabled is None:
        return sample_out.get("record_logits") is not None
    return bool(enabled)


def _query_spans(candidates, query_id: int) -> torch.Tensor:
    """Valid `(start, end)` rows for one boundary query."""
    if candidates is None or query_id < 0 or query_id >= int(candidates.indices.shape[1]):
        return torch.zeros((0, 2), dtype=torch.long)
    mask = candidates.valid_mask[0, query_id]
    keep = torch.nonzero(mask, as_tuple=False).flatten()
    return candidates.indices[0, query_id][keep].to(torch.long)


@dataclass
class RecordGroupView:
    """Record group whose spans line up with forward's `record_logits`."""

    spec: object
    object_logits: torch.Tensor
    assign_logits: list
    field_query_ids: list[int]
    field_specs: list
    field_spans: list
    instance_seed: list
    instance_spans: list
    num_instances: int


def _record_group(spec, logits, candidates):
    """Record group whose spans line up with forward's `record_logits`."""
    from .processing_gliner2 import FieldCardinality, RecordFieldSpec, RecordSpec

    object_logits, assign_logits = logits
    if not torch.is_tensor(object_logits):
        object_logits = torch.tensor(object_logits, dtype=torch.float)
    fields = []
    spans = []
    for field_spec in spec.get("fields") or []:
        card = str(field_spec.get("cardinality") or "zero_or_more")
        fields.append(
            RecordFieldSpec(
                query_id=int(field_spec["query_id"]),
                name=field_spec.get("name"),
                exclusive=bool(field_spec.get("exclusive", False)),
                allows_absent=card in ("optional_one", "zero_or_more"),
                cardinality=FieldCardinality(is_scalar=card in ("optional_one", "required_one")),
            )
        )
        spans.append(_query_spans(candidates, int(field_spec["query_id"])))
    mode = spec.get("mode")
    seeds: list = []
    instance_spans: list = []
    if mode == "natural":
        anchor_id = spec.get("anchor_query_id")
        anchor_index = next(index for index, field in enumerate(fields) if field.query_id == anchor_id)
        anchor_spans = spans[anchor_index]
        for cand_idx in range(int(anchor_spans.shape[0])):
            seeds.append((anchor_index, cand_idx))
            instance_spans.append((int(anchor_spans[cand_idx, 0]), int(anchor_spans[cand_idx, 1])))
    elif mode == "latent":
        for field_index, field_spans in enumerate(spans):
            for cand_idx in range(int(field_spans.shape[0])):
                seeds.append((field_index, cand_idx))
                instance_spans.append((int(field_spans[cand_idx, 0]), int(field_spans[cand_idx, 1])))
    else:
        count = int(object_logits.shape[0])
        seeds = [None] * count
        instance_spans = [None] * count
    return RecordGroupView(
        spec=RecordSpec(
            mode=mode,
            fields=tuple(fields),
            anchor_query_id=spec.get("anchor_query_id"),
            task_index=int(spec.get("task_index", 0)),
        ),
        object_logits=object_logits,
        assign_logits=list(assign_logits),
        field_query_ids=[field.query_id for field in fields],
        field_specs=fields,
        field_spans=spans,
        instance_seed=seeds,
        instance_spans=instance_spans,
        num_instances=int(object_logits.shape[0]),
    )


def decode_records(
    sample_out,
    meta,
    *,
    threshold=0.5,
    include_confidence=False,
    include_spans=False,
    overlap_policy=None,
):
    """Assign `record_logits` with `decode_group` when records are enabled."""
    if not _records_enabled(sample_out, meta):
        return {}
    groups = sample_out.get("record_logits")
    specs = list(meta.get("record_specs") or [])
    if not groups or not specs:
        return dict(sample_out.get("records") or {})
    candidates = sample_out.get("candidates")
    text = meta.get("text") or ""
    maps = {
        "start": list(meta.get("start") or []),
        "end": list(meta.get("end") or []),
        "offset": int(meta.get("prefix_len") or 0),
    }
    temperature = float(sample_out.get("record_temperature") or meta.get("record_temperature") or 1.0)
    out = OrderedDict()
    for spec, logits in zip(specs, groups):
        group = _record_group(spec, logits, candidates)
        if group.num_instances != len(group.instance_seed):
            continue
        decoded = decode_group(
            group,
            anchor_threshold=threshold,
            field_threshold=threshold,
            object_threshold=threshold,
            temperature=temperature,
        )
        instances = []
        for record in decoded:
            instance = OrderedDict()
            for field_spec in group.field_specs:
                spans = record.fields.get(field_spec.query_id, [])
                scores = record.field_scores.get(field_spec.query_id, [])
                raw = [(score, start, end) for (start, end), score in zip(spans, scores)]
                instance[field_spec.name] = decode_spans(
                    raw,
                    {
                        "dtype": "str" if field_spec.cardinality.is_scalar else "list",
                        "include_confidence": include_confidence,
                        "include_spans": include_spans,
                    },
                    text,
                    maps,
                    threshold,
                    overlap_policy,
                )
            if any(value is not None and value != [] for value in instance.values()):
                instances.append(instance)
        if instances:
            out[spec.get("task_name") or spec.get("name")] = instances
    return dict(out)


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


class _TaskIndex:
    """Task lookup used while checking constraint references."""

    def __init__(self, specs):
        self._specs = {spec.name: spec for spec in specs}

    def task_spec(self, name):
        try:
            return self._specs[name]
        except KeyError:
            raise SchemaError(f"unknown task {name!r}") from None

    def task(self, name):
        return self.task_spec(name)


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
    descriptions = body.pop("label_descriptions", None) or {}
    if descriptions and not isinstance(labels, Mapping):
        labels = {label: descriptions.get(label) for label in labels}
    examples = tuple(tuple(pair) for pair in (body.pop("examples", ()) or ()))
    if "multi_label" in body or "cls_threshold" in body or "class_act" in body or "task" in body:
        multi = bool(body.pop("multi_label", False))
        body.pop("task", None)
        return TaskSpec(
            name=name,
            labels=_coerce_labels(labels),
            min_labels=int(body.pop("min_labels", 0 if multi else 1)),
            max_labels=body.pop("max_labels", None if multi else 1),
            ordered=bool(body.pop("ordered", False)),
            threshold=body.pop("cls_threshold", body.pop("threshold", 0.5)),
            candidate_threshold=body.pop("candidate_threshold", None),
            activation=body.pop("class_act", body.pop("activation", "auto")),
            temperature=body.pop("temperature", 1.0),
            default=body.pop("default", None),
            instruction=body.pop("prompt", body.pop("instruction", None)),
            examples=examples,
        )
    return TaskSpec(name=name, labels=_coerce_labels(labels), examples=examples, **body)


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
    if isinstance(schema, CompiledClassificationSchema):
        return schema
    if not isinstance(schema, Mapping):
        if hasattr(schema, "to_dict"):
            schema = schema.to_dict()
        else:
            raise SchemaError(f"expected a classification schema dict, got {type(schema).__name__}")
    task_specs, constraints = _specs_from_schema(schema)
    if not task_specs:
        raise SchemaError("cannot compile a schema with no tasks")
    index = _TaskIndex(task_specs)
    for constraint in constraints:
        _walk(constraint, index)
    order = tuple(spec.name for spec in task_specs)
    _check_prefix_collisions(order)
    _static_feasibility(constraints, task_specs)
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
        constraints=_lower_defaults(constraints, task_specs),
        task_order=order,
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
    """Keep the highest-utility accepting locals within the beam.

    Args:
        problem: Decode problem whose locals are expanded one task at a time.
        order: Task order already sorted for search.
        beam_size: Maximum distinct partial assignments retained.

    Returns:
        The best beam assignment, or ``None`` when every branch is rejected.
    """
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
    """Pick each task's best local. Only that task is treated as decided.

    Args:
        problem: Locals in task order. Each task falls back to ``locals[0]``.

    Returns:
        An independent solution. Acceptance never sees other tasks as decided.
    """
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
    """Exact, beam, independent, and min-violations over the same locals.

    Independent mode keeps the per-task ``locals[0]`` fallback and decides one
    task at a time. Exact search returns a budget flag instead of raising;
    callers fall back to beam when that flag is set.

    Args:
        problem: Compiled locals and the constraints that touch them.
        mode: ``independent``, ``exact``, ``beam``, or ``min_violations``.
        budget: Maximum nodes expanded by exact and min-violations search.
        beam_size: Width used when ``mode`` is ``beam``.

    Returns:
        The solution, if search finished, and whether the node budget stopped
        exact search before a complete assignment.
    """
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


def _coerce_schema(schema):
    if isinstance(schema, CompiledClassificationSchema):
        return schema
    if isinstance(schema, Mapping) and ("tasks" in schema or "classifications" in schema):
        return schema
    if hasattr(schema, "to_dict"):
        payload = schema.to_dict()
        if isinstance(payload, Mapping) and ("tasks" in payload or "classifications" in payload):
            return payload
    raise SchemaError(f"expected a classification schema, got {type(schema).__name__}")


def _effective_temperature(spec: TaskSpec, temperature) -> float:
    if isinstance(temperature, Mapping):
        value = float(temperature.get(spec.name, spec.temperature))
    else:
        value = float(spec.temperature) * float(temperature)
    if not value > 0:
        raise ValueError("temperature must be positive")
    return value


def _label_logit_map(values, names) -> dict:
    """Align one logit row to label names, trimming a non-finite tail.

    Args:
        values: Mapping or sequence of label logits.
        names: Label names in schema order.

    Returns:
        Logits keyed by ``names``. A tail longer than ``names`` is dropped
        only when every extra value is non-finite.
    """
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        if missing:
            raise SchemaError(f"logits missing labels {missing}")
        unknown = [name for name in values if name not in names]
        if unknown:
            raise SchemaError(f"logits have unknown labels {unknown}")
        return {name: _coerce_logits(values[name]) for name in names}
    row = _coerce_logits(values)
    finite_tail = row
    if len(row) > len(names) and all(not math.isfinite(_coerce_logits(item)) for item in row[len(names) :]):
        finite_tail = row[: len(names)]
    if len(finite_tail) != len(names):
        raise SchemaError(f"expected {len(names)} label logits, got {len(row)}")
    return {name: _coerce_logits(value) for name, value in zip(names, finite_tail)}


def label_names_from_tokens(schema_tokens: Sequence[str]) -> tuple:
    """Recover label order from the ``[L]`` tokens that were encoded."""
    return tuple(schema_tokens[i + 1] for i in range(len(schema_tokens) - 1) if schema_tokens[i] == _L)


def _task_name(schema_tokens: Sequence[str], known: Sequence[str]) -> str:
    """Resolve a task with the processor's boundary-aware prompt match.

    Args:
        schema_tokens: Encoded prompt tokens. The task text is token 2.
        known: Task names in schema order.

    Returns:
        The longest task name that matches the prompt boundary, else the bare
        name before a description or colon.
    """
    prompt_str = str(schema_tokens[2] if len(schema_tokens) > 2 else "")
    best = None
    for name in known:
        if not name or not prompt_str.startswith(name):
            continue
        rest = prompt_str[len(name) :]
        if rest == "" or rest[0] in (":", " "):
            if best is None or len(name) > len(best):
                best = name
    if best is not None:
        return best
    bare = prompt_str.split(" [DESCRIPTION] ", 1)[0].split(":", 1)[0]
    if bare in known:
        return bare
    for name in known:
        if name and prompt_str.startswith(name):
            return name
    return bare


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
        rows = _coerce_logits(raw, n_rows=len(groups))
    known = compiled.task_order
    found = {task: {} for task in known}
    for group, row in zip(groups, rows):
        names = label_names_from_tokens(group)
        task = _task_name(group, known)
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
    rows = _coerce_logits(payload, n_rows=len(compiled.task_order))
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
        return logits
    return ClassificationScores(
        text=text,
        tasks=_align_logits(logits, compiled),
        specs={spec.name: spec for spec in compiled.task_specs},
    )


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
            meta["words"] = list(meta.get("words", []))
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
    if boundary is not None and getattr(boundary, "candidates", None) is not None:
        samples = []
        candidates = boundary.candidates
        for index in range(batch_size):
            sample = {
                "candidates": type(candidates)(
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
                ),
                "pair_logits": candidates.pair_logits[index],
                "null_logits": None if boundary.null_logits is None else boundary.null_logits[index],
                "count_log_rates": None if boundary.count_log_rates is None else boundary.count_log_rates[index],
            }
            if outputs.get("classification_logits") is not None:
                sample["classification_logits"] = outputs["classification_logits"][index]
            text_states = outputs.get("text_states")
            query_states = outputs.get("query_states")
            if text_states is not None:
                sample["text_states"] = text_states[index : index + 1]
            if query_states is not None:
                sample["query_states"] = query_states[index : index + 1]
            relation_pairs = outputs.get("relation_pairs")
            relation_logits = outputs.get("relation_logits")
            if relation_pairs is not None and relation_logits is not None and relation_pairs.numel():
                keep = relation_pairs[:, 0] == index
                sample["relation_pairs"] = relation_pairs[keep]
                sample["relation_logits"] = relation_logits[keep]
                sample["relation_temperature"] = outputs.get("relation_temperature")
            if outputs.get("record_logits") is not None:
                sample["record_logits"] = outputs["record_logits"][index]
            samples.append(sample)
        return samples
    keys = (
        "span_logits",
        "counts",
        "classification_logits",
        "record_logits",
        "relation_pairs",
        "relation_logits",
        "pair_logits",
        "grouped_candidates",
    )
    present = [key for key in keys if key in outputs and outputs[key] is not None]
    if not present:
        raise ValueError(
            "outputs need span_logits (count, fields, words, width) "
            "and/or classification_logits (num_labels,). "
            "Boundary pair_logits require candidates on the forward output."
        )
    return [{key: outputs[key][index] for key in present} for index in range(batch_size)]


def _classification_schema_from_row(meta, threshold):
    entries = []
    for entry in meta.get("classifications") or []:
        row = dict(entry)
        row.setdefault("cls_threshold", threshold)
        entries.append(row)
    return {"classifications": entries, "constraints": list(meta.get("constraints") or [])}


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
    """Entity type of a node, or relation type of a resolved edge."""
    if isinstance(value, NodeCandidate):
        return value.entity_type
    relation_type = getattr(value, "relation_type", None)
    if relation_type is not None:
        return relation_type
    return getattr(value, "type", None)


def _endpoint_of(relation: Any, side: str) -> Any:
    """Return the head or tail node stored on a resolved edge."""
    return getattr(relation, side)


def _identity_of(value: Any) -> Any:
    """Stable id for a node. Raw endpoint ids are returned unchanged."""
    if value is None:
        return None
    if isinstance(value, NodeCandidate):
        return value.candidate_id
    candidate_id = getattr(value, "candidate_id", None)
    if candidate_id is not None:
        return candidate_id
    start = getattr(value, "start", None)
    end = getattr(value, "end", None)
    if start is not None and end is not None:
        return (start, end, _label_of(value))
    return value


def _relation_key(value: Any) -> tuple:
    return (_label_of(value), _identity_of(_endpoint_of(value, "head")), _identity_of(_endpoint_of(value, "tail")))


def _matches(relation: Any, relation_type: str | None) -> bool:
    return relation_type is None or _label_of(relation) == relation_type


def _node_overlap_ok(constraint, candidate, nodes) -> bool:
    """Apply an entity overlap policy to one node against those already kept.

    Args:
        constraint: Joint constraint record. Non-overlap kinds allow the node.
        candidate: Node being considered. Uses ``start`` and ``end``.
        nodes: Nodes already accepted, including ``candidate`` when checking it.

    Returns:
        Whether the node satisfies the overlap policy.
    """
    if constraint["type"] != "EntityOverlapPolicy":
        return True
    policy = constraint["policy"]
    if policy == "allow":
        return True
    start, end = candidate.start, candidate.end
    for old in nodes:
        if old is candidate:
            continue
        old_start, old_end = old.start, old.end
        if end <= old_start or old_end <= start:
            continue
        nested = (start >= old_start and end <= old_end) or (old_start >= start and old_end <= end)
        if policy == "disallow" or not nested:
            return False
    return True


def _graph_constraints_ok(constraint, relations) -> bool:
    """Check symmetric and inverse relations on a finished edge list.

    Args:
        constraint: ``SymmetricRelation`` or ``InverseRelation`` record.
        relations: Resolved edges already selected.

    Returns:
        Whether every required reverse edge is present.
    """
    kind = constraint["type"]
    keys = {_relation_key(item) for item in relations}
    if kind == "SymmetricRelation":
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
    for item in relations:
        label_name = _label_of(item)
        reverse = (_identity_of(_endpoint_of(item, "tail")), _identity_of(_endpoint_of(item, "head")))
        if label_name == constraint["relation"] and (constraint["inverse"], *reverse) not in keys:
            return False
        if label_name == constraint["inverse"] and (constraint["relation"], *reverse) not in keys:
            return False
    return True


def _pair_constraints_ok(constraint, candidate, relations) -> bool:
    """Typed endpoints, self-loops, and unique pair or slot rules.

    Args:
        constraint: One edge constraint record.
        candidate: Resolved edge being added.
        relations: Resolved edges already accepted.

    Returns:
        Whether this constraint allows ``candidate``, or ``None`` if another
        checker owns the constraint kind.
    """
    kind = constraint["type"]
    if kind == "TypedEndpoints":
        if not _matches(candidate, constraint["relation"]):
            return True
        head = _label_of(_endpoint_of(candidate, "head"))
        tail = _label_of(_endpoint_of(candidate, "tail"))
        heads, tails = constraint["head_types"], constraint["tail_types"]
        return (not heads or head in heads) and (not tails or tail in tails)
    if kind == "NoSelfLoops":
        if not _matches(candidate, constraint["relation"]):
            return True
        return _identity_of(_endpoint_of(candidate, "head")) != _identity_of(_endpoint_of(candidate, "tail"))
    if kind == "UniqueRelationPair":
        if not _matches(candidate, constraint["relation"]):
            return True
        head = _identity_of(_endpoint_of(candidate, "head"))
        tail = _identity_of(_endpoint_of(candidate, "tail"))
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
    return None


def _count_constraints_ok(constraint, candidate, relations) -> bool:
    """Per-head, per-tail, and acyclic limits for one candidate edge.

    Args:
        constraint: Cardinality or acyclic constraint record.
        candidate: Resolved edge being added.
        relations: Resolved edges already accepted.

    Returns:
        Whether this constraint allows ``candidate``. Unrelated kinds allow it.
    """
    kind = constraint["type"]
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
        head = _identity_of(_endpoint_of(candidate, "head"))
        tail = _identity_of(_endpoint_of(candidate, "tail"))
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


def _joint_check(constraint: Mapping, candidate, relations=(), nodes=(), mode: str = "edge") -> bool:
    """True when one joint constraint allows an edge, a node, or a full graph."""
    kind = constraint["type"]
    if mode == "node":
        return _node_overlap_ok(constraint, candidate, nodes)
    if mode == "validate" and kind in {"SymmetricRelation", "InverseRelation"}:
        return _graph_constraints_ok(constraint, relations)
    if mode == "validate":
        accepted = []
        for item in relations:
            if not _joint_check(constraint, item, accepted, nodes, "edge"):
                return False
            accepted.append(item)
        return True
    pair = _pair_constraints_ok(constraint, candidate, relations)
    if pair is not None:
        return pair
    return _count_constraints_ok(constraint, candidate, relations)


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


def _entity_spec(name, value=None) -> EntitySpec:
    """One entity record from a string, a field dict, or a bare name."""
    if isinstance(value, Mapping):
        fields = {
            key: value[key]
            for key in ("description", "threshold", "candidate_threshold", "max_candidates", "allow_nested")
            if key in value
        }
        description = fields.pop("description", None)
        return EntitySpec(
            name,
            description,
            fields.get("threshold"),
            fields.get("candidate_threshold"),
            fields.get("max_candidates"),
            fields.get("allow_nested"),
        )
    if isinstance(value, str) and value:
        return EntitySpec(name, value)
    return EntitySpec(name)


def _entities_from_raw(raw) -> dict:
    entities = {}
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            entities[name] = _entity_spec(name, value)
        return entities
    for value in raw or ():
        if isinstance(value, str):
            entities[value] = EntitySpec(value)
        elif isinstance(value, Mapping) and "name" in value:
            entities[value["name"]] = _entity_spec(value["name"], value)
        else:
            raise ValueError(f"invalid entity entry {value!r}")
    return entities


def _relation_spec(name, head, tail, description=None, **aliases):
    """One relation record plus constraints implied by aliases such as acyclic."""
    inverse = aliases.pop("inverse", None)
    inverse_of = aliases.pop("inverse_of", None)
    allow_self = aliases.pop("allow_self", False)
    allow_self_loops = aliases.pop("allow_self_loops", None)
    no_self = aliases.pop("no_self_loops", None)
    unique_head = aliases.pop("unique_head", False)
    unique_tail = aliases.pop("unique_tail", False)
    aliases.pop("unique_pair", None)
    acyclic = aliases.pop("acyclic", False)
    threshold = aliases.pop("threshold", None)
    candidate_threshold = aliases.pop("candidate_threshold", None)
    directed = aliases.pop("directed", True)
    symmetric = aliases.pop("symmetric", False)
    max_per_head = aliases.pop("max_per_head", None)
    max_per_tail = aliases.pop("max_per_tail", None)
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
    extras = [_joint_constraint("AcyclicRelation", relation=name)] if acyclic else []
    return spec, extras


def _unwrap_endpoint(value):
    if isinstance(value, Mapping):
        return value.get("type") or value.get("entity")
    return value


def _relations_from_raw(raw, entities) -> tuple:
    relations = {}
    extras = []

    def add(spec, more):
        unknown = (set(spec.head) | set(spec.tail)) - set(entities)
        if unknown:
            raise ValueError(f"relation {spec.name!r} references unknown entity types: {sorted(unknown)}")
        if spec.name in relations:
            raise ValueError(f"relation {spec.name!r} is already defined")
        relations[spec.name] = spec
        extras.extend(more)

    if isinstance(raw, Mapping):
        for name, value in raw.items():
            body = dict(value)
            add(*_relation_spec(name, body.pop("head"), body.pop("tail"), body.pop("description", None), **body))
        return relations, extras
    for item in raw or ():
        if isinstance(item, Mapping) and "name" in item and "head" in item:
            body = dict(item)
            add(
                *_relation_spec(
                    body.pop("name"), body.pop("head"), body.pop("tail"), body.pop("description", None), **body
                )
            )
            continue
        for name, fields in item.items():
            head = _unwrap_endpoint(fields.get("head"))
            tail = _unwrap_endpoint(fields.get("tail"))
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
            add(*_relation_spec(name, head, tail, **options))
    return relations, extras


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
    if not isinstance(schema, Mapping):
        if hasattr(schema, "to_dict"):
            schema = schema.to_dict()
        else:
            raise TypeError(f"expected a joint schema dict, got {type(schema).__name__}")
    entities = _entities_from_raw(schema.get("entities") or {})
    relations, extras = _relations_from_raw(schema.get("relations") or {}, entities)
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
    constraints = [
        _joint_from_dict(item) if isinstance(item, Mapping) and "type" in item else item
        for item in list(schema.get("constraints") or []) + extras
    ]
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
    if isinstance(schema, Mapping) and ("entities" in schema or "relations" in schema or "constraints" in schema):
        return schema
    if hasattr(schema, "to_dict"):
        payload = schema.to_dict()
        if isinstance(payload, Mapping):
            return payload
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
    """Cells of one ``[L, W]`` logit lattice as ``(logit, start, end)``.

    Args:
        lattice: Span scores indexed by start and width.

    Returns:
        Entries whose end still falls inside the sequence.
    """
    shape = _coerce_logits(lattice, as_shape=True)
    if len(shape) != 2:
        raise ValueError(f"span lattice must have shape [L, W], got {shape}")
    length, widths = shape
    entries = []
    for start in range(length):
        row = lattice[start]
        for width in range(widths):
            end = start + width + 1
            if end <= length:
                entries.append((_coerce_logits(row[width]), start, end))
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


def _collect_span_nodes(
    entity_logits,
    entity_types,
    thresholds,
    candidate_thresholds,
    maxima,
    entity_threshold,
    top_k_entities,
    entity_weight,
):
    """Keep the top entity spans and the raw score of every finite cell.

    Args:
        entity_logits: Dense scores shaped ``[types, L, W]``.
        entity_types: Type name per leading axis entry.
        thresholds: Decision threshold by entity type.
        candidate_thresholds: Retention floor by entity type.
        maxima: Optional cap by entity type.
        entity_threshold: Floor used when a type has no candidate threshold.
        top_k_entities: Default cap per type.
        entity_weight: Multiplier on the centered entity logit.

    Returns:
        Selected nodes keyed by ``(type, start, end)``, and raw scores for
        every finite cell so relation rescue can still see dropped spans.
    """
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
    return node_map, raw_entity_scores


def _rescue_role_spans(
    node_map, raw_entity_scores, role_options, types, rescue_per_role, relation_offset, role_weight
):
    """Bind one role's spans to nodes, rescuing missing endpoints in rank order.

    Args:
        node_map: Nodes already kept, updated with rescued endpoints.
        raw_entity_scores: Centered score and probability for finite entity cells.
        role_options: ``(logit, start, end)`` rows, highest logit first.
        types: Entity types allowed for this role.
        rescue_per_role: Maximum new nodes added for this role.
        relation_offset: Log-odds of the relation decision threshold.
        role_weight: Multiplier on the centered role logit.

    Returns:
        ``(centered role score, node, role probability)`` rows.
    """
    bound = []
    rescued = 0
    for raw_role_score, start, end in role_options:
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
            bound.append(((raw_role_score - relation_offset) * role_weight, node, sigmoid(raw_role_score)))
    return bound


def _edges_for_hypothesis(
    hypothesis,
    index,
    node_map,
    raw_entity_scores,
    *,
    relation_role_threshold,
    top_k_roles,
    rescue_per_role,
    relation_pair_cap,
    role_weight,
    count_weight,
):
    """Edges for one relation hypothesis, including rescued endpoints.

    Args:
        hypothesis: Relation lattice with role logits shaped ``[count, 2, L, W]``.
        index: Position used when the hypothesis has no id.
        node_map: Entity nodes, mutated when an endpoint is rescued.
        raw_entity_scores: Scores for cells that missed the entity cap.
        relation_role_threshold: Default role retention floor.
        top_k_roles: Role spans kept before pairing.
        rescue_per_role: New nodes allowed per role.
        relation_pair_cap: Pairs kept after sorting by score.
        role_weight: Multiplier on centered role logits.
        count_weight: Multiplier on the hypothesis count utility.

    Returns:
        Edge candidates for this hypothesis, not yet deduplicated across slots.
    """
    if not isinstance(hypothesis, RelationHypothesis):
        raise TypeError("relation hypotheses must be RelationHypothesis values")
    role_shape = _coerce_logits(hypothesis.role_logits, as_shape=True)
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
    edges = []
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
        typed_roles = [
            _rescue_role_spans(
                node_map, raw_entity_scores, role_options[role], types, rescue_per_role, relation_offset, role_weight
            )
            for role, types in enumerate((hypothesis.head_types, hypothesis.tail_types))
        ]
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
            edges.append(
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
    return edges


def _cap_span_edges(edge_groups, max_edges_per_type):
    """Dedup edges by key, keeping the higher score, then apply the type cap.

    Args:
        edge_groups: Edges grouped by relation type.
        max_edges_per_type: Maximum edges retained for each relation.

    Returns:
        Edges ordered by ``_edge_rank``.
    """
    edges = []
    for relation_type in sorted(edge_groups):
        unique = {}
        for edge in sorted(edge_groups[relation_type], key=_edge_rank):
            previous = unique.get(edge.key)
            if previous is None or edge.score > previous.score:
                unique[edge.key] = edge
        edges.extend(sorted(unique.values(), key=_edge_rank)[:max_edges_per_type])
    return edges


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
    shape = _coerce_logits(entity_logits, as_shape=True)
    if len(shape) != 3 or shape[0] != len(entity_types):
        raise ValueError(f"entity logits must have shape [types, L, W], got {shape} for {len(entity_types)} types")
    node_map, raw_entity_scores = _collect_span_nodes(
        entity_logits,
        entity_types,
        dict(entity_thresholds or {}),
        dict(entity_candidate_thresholds or {}),
        dict(entity_max_candidates or {}),
        entity_threshold,
        top_k_entities,
        entity_weight,
    )
    edge_groups: dict = {}
    for index, hypothesis in enumerate(relation_hypotheses):
        for edge in _edges_for_hypothesis(
            hypothesis,
            index,
            node_map,
            raw_entity_scores,
            relation_role_threshold=relation_role_threshold,
            top_k_roles=top_k_roles,
            rescue_per_role=rescue_per_role,
            relation_pair_cap=relation_pair_cap,
            role_weight=role_weight,
            count_weight=count_weight,
        ):
            edge_groups.setdefault(hypothesis.relation_type, []).append(edge)
    edges = _cap_span_edges(edge_groups, max_edges_per_type)
    nodes = tuple(sorted(node_map.values(), key=lambda node: (node.entity_type, node.start, node.end, -node.score)))
    return JointProblem(nodes, tuple(sorted(edges, key=_edge_rank)), tuple(constraints))


def _boundary_mentions(
    text,
    candidates,
    query_specs,
    *,
    sample_index=0,
    token_offset=0,
    text_length=None,
    pair_temperature=1.0,
):
    """Turn one boundary candidate row into nodes whose scores are logits.

    Args:
        text: Document text. Offsets are checked against ``text_length``.
        candidates: Candidate batch with indices, pair logits, and masks.
        query_specs: Extractive fields in marker order.
        sample_index: Row of the candidate batch to read.
        token_offset: Prefix tokens subtracted from candidate indexes.
        text_length: Number of document tokens. Missing lengths keep nothing.
        pair_temperature: Positive divisor applied to pair logits.

    Returns:
        One ``NodeCandidate`` per mention key. ``score`` is the logit and the
        higher logit wins a duplicate span. Order is type, start, end, logit.
    """
    del text
    if pair_temperature <= 0:
        raise ValueError("pair_temperature must be positive")
    if text_length is None:
        text_length = 0
    indices = candidates.indices
    pair_logits = candidates.pair_logits
    valid_mask = candidates.valid_mask
    query_mask = candidates.query_mask
    best = {}
    for query_id, spec in enumerate(query_specs):
        if spec["task_type"] != "entities" or query_id >= indices.shape[1]:
            continue
        entity_type = str(spec["field_name"])
        valid = valid_mask[sample_index, query_id] & query_mask[sample_index, query_id]
        for candidate_id in valid.nonzero(as_tuple=False).flatten().tolist():
            start = int(indices[sample_index, query_id, candidate_id, 0]) - token_offset
            end = int(indices[sample_index, query_id, candidate_id, 1]) - token_offset
            if not (0 <= start < end <= int(text_length)):
                continue
            logit = float(pair_logits[sample_index, query_id, candidate_id].detach().float()) / pair_temperature
            node = NodeCandidate(entity_type, start, end, logit, sigmoid(logit))
            previous = best.get(node.key)
            if previous is None or node.score > previous.score:
                best[node.key] = node
    return tuple(sorted(best.values(), key=lambda item: (item.entity_type, item.start, item.end, -item.score)))


def _retain_logit_edges(edges, edge_candidate_threshold, max_edges_per_type):
    """Dedup logit edges and cap each relation type.

    Args:
        edges: Edges whose ``score`` is a logit.
        edge_candidate_threshold: Minimum sigmoid probability.
        max_edges_per_type: Cap after sorting by relation, logit, and endpoints.

    Returns:
        ``(edge, probability)`` rows in rescue order.
    """
    edge_by_key = {}
    for edge in edges:
        if not isinstance(edge, EdgeCandidate):
            raise TypeError("boundary edges must be EdgeCandidate values")
        probability = sigmoid(edge.score)
        if probability < edge_candidate_threshold:
            continue
        key = (edge.relation_type, edge.head, edge.tail)
        previous = edge_by_key.get(key)
        if previous is None or edge.score > previous.score:
            edge_by_key[key] = (edge, probability)
    retained = []
    counts: dict = {}
    ranked = sorted(
        edge_by_key.values(),
        key=lambda item: (item[0].relation_type, -item[0].score, str(item[0].head), str(item[0].tail)),
    )
    for edge, probability in ranked:
        if max_edges_per_type is not None and counts.get(edge.relation_type, 0) >= max_edges_per_type:
            continue
        retained.append((edge, probability))
        counts[edge.relation_type] = counts.get(edge.relation_type, 0) + 1
    return retained


def _select_rescued_mentions(mentions, rescue_ids, floors, mention_threshold, max_mentions_per_type, type_limits):
    """Keep mentions above the floor, plus rescued endpoints within the cap.

    Args:
        mentions: Nodes whose ``score`` is still a logit.
        rescue_ids: Endpoint keys that bypass the probability floor and the cap.
        floors: Candidate threshold by entity type. ``None`` uses the default.
        mention_threshold: Default retention floor.
        max_mentions_per_type: Cap used when a type has no specific limit.
        type_limits: Cap by entity type.

    Returns:
        ``(mention, floor)`` rows in type, probability, start, end order.
    """
    selected = []
    per_type: dict = {}
    ordered = sorted(mentions, key=lambda item: (item.entity_type, -item.probability, item.start, item.end))
    for mention in ordered:
        floor = floors.get(mention.entity_type, mention_threshold)
        if floor is None:
            floor = mention_threshold
        if mention.probability < floor and mention.key not in rescue_ids:
            continue
        type_limit = type_limits.get(mention.entity_type, max_mentions_per_type)
        if (
            type_limit is not None
            and per_type.get(mention.entity_type, 0) >= type_limit
            and mention.key not in rescue_ids
        ):
            continue
        selected.append((mention, floor))
        per_type[mention.entity_type] = per_type.get(mention.entity_type, 0) + 1
    return selected


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
    entity_thresholds=None,
    entity_candidate_thresholds=None,
) -> JointProblem:
    """Center boundary logits into nodes and edges, rescuing relation endpoints.

    Args:
        mentions: Nodes whose ``score`` is still a logit.
        edges: Edges whose ``score`` is still a logit and whose ends are node keys.
        constraints: Joint constraints copied onto the problem.
        mention_threshold: Retention floor when a type has no candidate threshold.
        decision_threshold: Centering threshold when a type has no decision threshold.
        max_mentions_per_type: Default cap applied after the rescue set is known.
        max_mentions_by_type: Per-type cap.
        rescue_relation_endpoints: Keep edge endpoints that miss the mention floor.
        edge_candidate_threshold: Minimum edge probability.
        max_edges_per_type: Cap applied after edges are sorted by logit.
        entity_weight: Multiplier on the centered entity logit.
        relation_weight: Multiplier on the centered edge logit.
        entity_thresholds: Decision threshold by entity type.
        entity_candidate_thresholds: Retention floor by entity type.

    Returns:
        A joint problem. Mention order is type, descending probability, start, end.
        Edges keep that same rescue order, then higher logit, then endpoint ids.
    """
    retained_edges = _retain_logit_edges(edges, edge_candidate_threshold, max_edges_per_type)
    rescue_ids = (
        {endpoint for edge, _ in retained_edges for endpoint in (edge.head, edge.tail)}
        if rescue_relation_endpoints
        else set()
    )
    selected = _select_rescued_mentions(
        mentions,
        rescue_ids,
        dict(entity_candidate_thresholds or {}),
        mention_threshold,
        max_mentions_per_type,
        dict(max_mentions_by_type or {}),
    )
    thresholds = dict(entity_thresholds or {})
    nodes = []
    keep_ids = set()
    for mention, floor in selected:
        decision = thresholds.get(mention.entity_type)
        decision = decision_threshold if decision is None else float(decision)
        node = NodeCandidate(
            mention.entity_type,
            mention.start,
            mention.end,
            entity_weight * center_logit(mention.score, decision),
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
    for edge_slot, (edge, probability) in enumerate(retained_edges):
        if edge.head not in keep_ids or edge.tail not in keep_ids:
            continue
        edge_cands.append(
            EdgeCandidate(
                edge.relation_type,
                edge.head,
                edge.tail,
                relation_weight * center_logit(edge.score, decision_threshold),
                head_probability=probability,
                tail_probability=probability,
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
        and key[0] == _LatticeTag.COUNT_CHOICE.value
        and key[1] == group
        and key[2] != alternative
        for key in used_set
    )


def _edge_usage(edge: EdgeCandidate):
    keys = set(edge.exclusion_keys)
    if edge.count_choice is not None:
        keys.add((_LatticeTag.COUNT_CHOICE.value,) + edge.count_choice)
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
                    candidate_id=(_LatticeTag.DERIVED.value, relation, head, tail),
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


def edge_rank(edge: EdgeCandidate, nodes: Mapping, *, mode: Literal["greedy", "beam"]) -> tuple:
    """Rank an edge for greedy or beam expansion.

    Args:
        edge: Relation edge to order.
        nodes: Candidate id mapped to the node that supplies the endpoint score.
        mode: ``greedy`` omits the tail score when ``head == tail``, then breaks
            ties with ``-edge.score``. ``beam`` adds both endpoint scores and
            has no second score key.

    Returns:
        A sort key. Smaller values are expanded first.
    """
    head_score = nodes[edge.head].score
    if mode == "greedy":
        endpoint = head_score
        if edge.tail != edge.head:
            endpoint += nodes[edge.tail].score
        return (
            -(edge.score + endpoint),
            -edge.score,
            edge.relation_type,
            str(edge.hypothesis),
            str(edge.slot),
            str(edge.head),
            str(edge.tail),
        )
    if mode != "beam":
        raise ValueError("mode must be 'greedy' or 'beam'")
    return (
        -(edge.score + head_score + nodes[edge.tail].score),
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
    for edge in sorted(problem.edges, key=lambda item: edge_rank(item, node_by_id, mode="greedy")):
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
    ordered = sorted(problem.edges, key=lambda item: edge_rank(item, node_by_id, mode="beam"))
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
    """Probabilities that make up one entity or relation confidence.

    Args:
        value: Selected node or edge.
        include_count: Include the edge count probability when it is present.

    Returns:
        Component probabilities in the historical field order.
    """
    if isinstance(value, NodeCandidate):
        if value.probability is None:
            return []
        return [float(value.probability)]
    scores = []
    for score in (
        value.head_probability,
        value.tail_probability,
        value.head_entity_probability,
        value.tail_entity_probability,
    ):
        if score is not None:
            scores.append(float(score))
    if include_count and value.count_probability is not None:
        scores.append(float(value.count_probability))
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
            (
                node,
                char_start,
                char_end,
                surface,
                _component_scores(node, True),
                _LatticeTag.RESCUE.value in str(node.source).lower(),
            )
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
            payload[_LatticeTag.DERIVED.value] = True
        relations.append(payload)
    relations.sort(key=lambda item: (item["type"], item["head"], item["tail"]))
    return {"entities": entities, "relations": relations}


def _problem_from_span(scores, compiled: CompiledJointSchema, config) -> JointProblem:
    """Build a span problem from entity logits and relation hypotheses.

    Args:
        scores: Mapping with ``entity_logits`` shaped ``[types, L, W]``.
            Optional ``entity_types`` and ``relation_hypotheses`` use the
            schema order and compiled relation specs when omitted.
        compiled: Joint schema whose entity specs set thresholds and caps.
        config: Candidate caps and loss weights.

    Returns:
        Nodes and edges for greedy or beam search.

    Raises:
        TypeError: If ``entity_logits`` is missing or a hypothesis is not a
            ``RelationHypothesis``.
    """
    if not isinstance(scores, Mapping) or scores.get("entity_logits") is None:
        raise TypeError("span scores need entity_logits [types, L, W]")
    entity_logits = scores["entity_logits"]
    entity_types = tuple(scores.get("entity_types") or compiled.entity_order)
    hypotheses = []
    for raw in scores.get("relation_hypotheses") or ():
        if not isinstance(raw, RelationHypothesis):
            raise TypeError("relation_hypotheses entries must be RelationHypothesis")
        hypotheses.append(raw)
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


def _sparse_problem(scores, compiled: CompiledJointSchema, config) -> JointProblem:
    """Build a boundary problem from candidate tensors.

    Args:
        scores: Mapping with ``candidates`` and ``query_specs``. Optional
            ``edges`` are ``EdgeCandidate`` values whose scores are logits.
        compiled: Joint schema supplying thresholds and constraints.
        config: Candidate caps and loss weights.

    Returns:
        Centered nodes and edges. Relation endpoints below the floor are rescued.

    Raises:
        TypeError: If the candidate tensors or query specs are missing.
    """
    if not isinstance(scores, Mapping):
        raise TypeError("boundary scores need candidates with pair_logits")
    candidates = scores.get("candidates")
    query_specs = scores.get("query_specs")
    if candidates is None or query_specs is None or getattr(candidates, "pair_logits", None) is None:
        raise TypeError("boundary scores need candidates with pair_logits")
    text_length = scores.get("text_length")
    if text_length is None:
        text_length = len(scores.get("start_mappings") or ())
    specs = compiled.entity_specs
    mentions = _boundary_mentions(
        str(scores.get("text") or ""),
        candidates,
        query_specs,
        sample_index=int(scores.get("sample_index") or 0),
        token_offset=int(scores.get("token_offset") or 0),
        text_length=text_length,
        pair_temperature=float(scores.get("pair_temperature") or 1.0),
    )
    return _mentions_to_problem(
        mentions,
        tuple(scores.get("edges") or ()),
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
        entity_thresholds={name: spec.threshold for name, spec in specs.items()},
        entity_candidate_thresholds={name: spec.candidate_threshold for name, spec in specs.items()},
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


def _char_maps(scores):
    """Return start and end maps, or ``None`` when the document has no offsets.

    Args:
        scores: Span or boundary score mapping.

    Returns:
        ``(starts, ends)``. Empty maps become ``None``.
    """
    starts = scores.get("start_mappings") if isinstance(scores, Mapping) else None
    ends = scores.get("end_mappings") if isinstance(scores, Mapping) else None
    if starts is None or len(starts) == 0:
        return None, None
    return starts, ends


def _decode_joint_one(scores, schema, architecture, *, text="", config: JointIEConfig, overlap_policy=None) -> dict:
    """Decode one span matrix or one boundary candidate list."""
    compiled = _compile_joint(_coerce_joint_schema(schema))
    compiled, longest = _with_overlap(compiled, overlap_policy)
    kind = _architecture(architecture)
    if not isinstance(scores, Mapping):
        raise TypeError("span scores need entity_logits [types, L, W] or boundary candidates")
    document = text or str(scores.get("text") or "")
    starts, ends = _char_maps(scores)
    if kind == "span":
        problem = _problem_from_span(scores, compiled, config)
    else:
        problem = _sparse_problem(scores, compiled, config)
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


def _doc_axis(tensor, doc_len):
    if doc_len <= 0:
        return tensor[..., :0, :]
    return tensor[..., -doc_len:, :]


def _schema_from_meta(meta) -> dict:
    """Joint dict taken from processor metadata."""
    source = meta.get("schema") or {}
    constraints = [
        raw
        for raw in list(meta.get("constraints") or []) + list(source.get("constraints") or [])
        if isinstance(raw, Mapping) and raw.get("type") in _JOINT_FIELDS
    ]
    return {
        "entities": source.get("entities") or {},
        "relations": source.get("relations") or [],
        "constraints": constraints,
    }


def _span_scores_from_sample(sample, meta, schema):
    groups = [group for group in (meta.get("groups") or []) if group["task_type"] != "classifications"]
    tensors = sample.get("span_logits")
    probabilities = False
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
    specs = schema.relation_specs if isinstance(schema.relation_specs, Mapping) else {}
    for index, group in enumerate(groups):
        tensor = tensors[index]
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor, dtype=torch.float)
        tensor = tensor.detach().float().cpu()
        if tensor.ndim != 4:
            raise ValueError("span logits must have shape (count, fields, words, width)")
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
        schema = _compile_joint(_schema_from_meta(meta))
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

    Span scores provide ``entity_logits`` shaped ``[types, L, W]``. Boundary
    scores provide candidate tensors and ``query_specs``. ``architecture`` is
    ``"span"`` or ``"boundary"``. ``optimizer`` is ``"beam"`` or ``"greedy"``.
    Processor calls omit ``architecture`` and pass metadata plus ``threshold``
    and ``overlap_policy``; ``exact`` uses beam.

    Args:
        span_or_boundary_scores: One example's scores, or a model output when
            ``architecture`` is omitted and ``schema`` is processor metadata.
        schema: Joint schema, or processor metadata for a batch.
        architecture: ``"span"`` or ``"boundary"``. Omitted for processor batches.

    Returns:
        An entity/relation dict, or a list of those dicts for a batch.
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
