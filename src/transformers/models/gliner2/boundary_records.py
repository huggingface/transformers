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

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from ...utils.import_utils import requires_backends


@dataclass
class RecordGroupOutput:
    """One record group's instance and field-assignment scores.

    Attributes:
        spec: Record schema that produced this group.
        object_logits: Instance object scores `[Ni]`.
        assign_logits: Per-field logits `[Ni, 1 + Cf]`, column 0 is absent.
        field_query_ids: Boundary query id of each field.
        field_specs: Field schema objects, in assignment order.
        field_spans: Half-open candidate spans per field.
        field_cand_mask: Real candidates per field.
        field_cand_logits: Pair logits of those candidates.
        instance_seed: `(field_index, candidate_index)` seed, if any.
        instance_spans: Span of each seeded instance.
    """

    spec: object
    object_logits: torch.Tensor
    assign_logits: list[torch.Tensor]
    field_query_ids: list[int]
    field_specs: list
    field_spans: list[torch.Tensor]
    field_cand_mask: list[torch.Tensor]
    field_cand_logits: list[torch.Tensor]
    instance_seed: list[tuple[int, int] | None]
    instance_spans: list[tuple[int, int] | None]

    @property
    def num_instances(self) -> int:
        """Number of instance rows in this group."""
        return int(self.object_logits.shape[0])


@dataclass
class DecodedRecord:
    """One decoded record.

    Attributes:
        fields: Field query id to selected half-open spans.
        field_scores: Field query id to the score of each selected span.
        anchor_span: Anchor span for natural-mode instances.
        score: Object probability of the instance.
    """

    fields: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    field_scores: dict[int, list[float]] = field(default_factory=dict)
    anchor_span: tuple[int, int] | None = None
    score: float = 0.0


def _dedup_key(record: DecodedRecord) -> tuple:
    """Return a span key that identifies duplicate decoded records."""
    return tuple((qid, tuple(sorted(spans))) for qid, spans in sorted(record.fields.items()))


class _ScipyAssignment:
    """Minimum-cost assignment through SciPy."""

    def prepare(self) -> None:
        """Require SciPy before a decode that may assign exclusive fields."""
        requires_backends(self, ["scipy"])

    def __call__(self, cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve a rectangular assignment problem.

        Args:
            cost_matrix: Costs `[R, C]`.

        Returns:
            Matched row and column indices, with rows ascending.
        """
        requires_backends(self, ["scipy"])
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


def _decode_group(
    group: RecordGroupOutput,
    assign: _ScipyAssignment,
    *,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
    temperature: float = 1.0,
) -> list[DecodedRecord]:
    """Decode one group after SciPy has been required by the caller."""
    ni = group.num_instances
    if ni == 0:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    obj_prob = torch.sigmoid(group.object_logits.detach() / temperature)
    select_thr = object_threshold if group.spec.mode == "anchorless" else anchor_threshold
    order = sorted(range(ni), key=lambda i: (-float(obj_prob[i]), i))
    selected_instances = [inst for inst in order if float(obj_prob[inst]) >= select_thr]

    scalar_choices: dict[tuple[int, int], tuple[int, float] | None] = {}
    list_owners: dict[tuple[int, int], tuple[int, float]] = {}
    for f_idx, fspec in enumerate(group.field_specs):
        if not fspec.exclusive or not selected_instances:
            continue
        logits = torch.stack([group.assign_logits[f_idx][inst].detach() / temperature for inst in selected_instances])
        candidate_count = max(int(logits.shape[-1]) - 1, 0)
        if fspec.cardinality.is_scalar:
            if candidate_count == 0:
                for inst in selected_instances:
                    scalar_choices[(inst, f_idx)] = None
                continue
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
            rows, cols = assign(cost)
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
        else:
            if candidate_count == 0:
                continue
            probabilities = torch.sigmoid(logits[:, 1:])
            for cand_idx in range(candidate_count):
                probability, row = probabilities[:, cand_idx].max(dim=0)
                if float(probability) >= field_threshold:
                    list_owners[(f_idx, cand_idx)] = (
                        selected_instances[int(row)],
                        float(probability),
                    )

    records: list[DecodedRecord] = []
    for inst in selected_instances:
        rec = DecodedRecord(score=float(obj_prob[inst]))
        anchor_field_idx = None
        if group.spec.mode == "natural":
            anchor_field_idx = group.field_query_ids.index(group.spec.anchor_query_id)
            seed = group.instance_seed[inst]
            if seed is not None:
                rec.anchor_span = group.instance_spans[inst]
        for f_idx, fspec in enumerate(group.field_specs):
            qid = fspec.query_id
            spans_tensor = group.field_spans[f_idx]
            logits_row = group.assign_logits[f_idx][inst].detach() / temperature
            if anchor_field_idx is not None and f_idx == anchor_field_idx:
                if rec.anchor_span is not None:
                    rec.fields.setdefault(qid, []).append(rec.anchor_span)
                    rec.field_scores.setdefault(qid, []).append(rec.score)
                continue
            if fspec.cardinality.is_scalar:
                if fspec.exclusive:
                    choice = scalar_choices.get((inst, f_idx))
                    if choice is None:
                        continue
                    cand_idx, probability = choice
                    span = (int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1]))
                    rec.fields.setdefault(qid, []).append(span)
                    rec.field_scores.setdefault(qid, []).append(probability)
                    continue
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
                    continue
                if float(probs[chosen]) < field_threshold and fspec.allows_absent:
                    continue
                cand_idx = chosen - 1
                span = (int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1]))
                rec.fields.setdefault(qid, []).append(span)
                rec.field_scores.setdefault(qid, []).append(float(probs[chosen]))
            else:
                cand_logits = logits_row[1:]
                if cand_logits.numel() == 0:
                    continue
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
                    rec.fields.setdefault(qid, []).extend(selected)
                    rec.field_scores.setdefault(qid, []).extend(selected_scores)
        if rec.fields:
            records.append(rec)

    if group.spec.mode in ("latent", "anchorless"):
        best: dict[tuple, DecodedRecord] = {}
        for rec in records:
            key = _dedup_key(rec)
            if key not in best or rec.score > best[key].score:
                best[key] = rec
        records = list(best.values())
    elif group.spec.mode == "natural":
        records.sort(key=lambda record: (record.anchor_span is None, record.anchor_span or (0, 0)))
    return records


def decode_group(
    group: RecordGroupOutput,
    *,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
    temperature: float = 1.0,
) -> list[DecodedRecord]:
    """Decode one record group into selected field spans.

    Args:
        group: Scores from `RecordHead.forward_group`.
        anchor_threshold: Minimum object probability for anchored instances.
        field_threshold: Minimum field probability.
        object_threshold: Minimum object probability for anchorless instances.
        temperature: Positive divisor applied to logits before probabilities.

    Returns:
        Decoded records that contain at least one field span.
    """
    assign = _ScipyAssignment()
    assign.prepare()
    return _decode_group(
        group,
        assign,
        anchor_threshold=anchor_threshold,
        field_threshold=field_threshold,
        object_threshold=object_threshold,
        temperature=temperature,
    )


class RecordHead(nn.Module):
    """Form record instances and score field-to-instance assignment.

    Args:
        hidden_size: Candidate and query width.
        record_dim: Width of the assignment space.
        instance_queries: Learned queries used when the schema is anchorless.
    """

    def __init__(self, hidden_size: int, record_dim: int, instance_queries: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.record_dim = record_dim
        self.instance_queries = instance_queries
        self.inst_proj = nn.Linear(hidden_size, record_dim)
        self.field_proj = nn.Linear(hidden_size, record_dim)
        self.cand_proj = nn.Linear(hidden_size, record_dim)
        self.null_embed = nn.Parameter(torch.randn(record_dim) * 0.02)
        self.object_head = nn.Linear(hidden_size, 1)
        self.latent_seed_head = nn.Linear(hidden_size, 1)
        self.instance_embed = nn.Parameter(torch.randn(instance_queries, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, record_dim)
        self.k_proj = nn.Linear(hidden_size, record_dim)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def _assign_logits(
        self,
        inst_states: torch.Tensor,
        field_query_states: torch.Tensor,
        field_cand_states: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Score absent-aware assignment of each field candidate.

        Args:
            inst_states: Instance states `[Ni, H]`.
            field_query_states: Field queries `[F, H]`.
            field_cand_states: Candidate states per field, each `[Cf, H]`.

        Returns:
            Logits per field, each `[Ni, 1 + Cf]`.
        """
        inst_q = self.inst_proj(inst_states)
        field_q = self.field_proj(field_query_states)
        out: list[torch.Tensor] = []
        for f_idx, cand in enumerate(field_cand_states):
            query = inst_q + field_q[f_idx].unsqueeze(0)
            null_col = query @ self.null_embed
            if cand.shape[0] == 0:
                out.append(null_col.unsqueeze(-1))
                continue
            cand_p = self.cand_proj(cand)
            cand_scores = query @ cand_p.t()
            out.append(torch.cat([null_col.unsqueeze(-1), cand_scores], dim=-1))
        return out

    def _anchorless_states(self, field_cand_states: list[torch.Tensor]) -> torch.Tensor:
        """Cross-attend learned instance queries over field candidates.

        Args:
            field_cand_states: Non-empty candidate states are concatenated.

        Returns:
            Conditioned instance states `[I, H]`.
        """
        inst = self.instance_embed
        ctx = [cand for cand in field_cand_states if cand.shape[0] > 0]
        if not ctx:
            return inst
        ctx = torch.cat(ctx, dim=0)
        q = self.q_proj(inst)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        attn = (q @ k.t()) / math.sqrt(self.record_dim)
        weights = torch.softmax(attn, dim=-1)
        return inst + weights @ v

    def forward_group(
        self,
        spec,
        query_states: torch.Tensor,
        candidates,
        sample_index: int,
    ) -> RecordGroupOutput:
        """Score one record schema for one sample.

        Args:
            spec: Schema with `mode`, `fields`, and optional `anchor_query_id`.
            query_states: This sample's query states `[Q, H]`.
            candidates: Batch candidates containing states, spans, and logits.
            sample_index: Batch row to read.

        Returns:
            Instance object logits and per-field assignment logits.
        """
        device = query_states.device
        field_specs = list(spec.fields)
        field_query_ids = [fspec.query_id for fspec in field_specs]
        query_count = min(
            query_states.shape[0],
            candidates.valid_mask.shape[1],
            candidates.pair_logits.shape[1],
            candidates.candidate_states.shape[1],
        )
        if query_count <= 0:
            raise ValueError("record routing requires at least one boundary query")
        valid_query_ids = [0 <= query_id < query_count for query_id in field_query_ids]
        safe_field_query_ids = [min(max(query_id, 0), query_count - 1) for query_id in field_query_ids]

        field_cand_states: list[torch.Tensor] = []
        field_spans: list[torch.Tensor] = []
        field_cand_mask: list[torch.Tensor] = []
        field_cand_logits: list[torch.Tensor] = []
        cand_states_all = candidates.candidate_states
        for qid, query_valid in zip(safe_field_query_ids, valid_query_ids):
            mask = candidates.valid_mask[sample_index, qid] & query_valid
            keep = torch.nonzero(mask, as_tuple=False).flatten()
            spans = candidates.indices[sample_index, qid][keep]
            logits = candidates.pair_logits[sample_index, qid][keep]
            states = cand_states_all[sample_index, qid][keep]
            field_cand_states.append(states)
            field_spans.append(spans.to(torch.long))
            field_cand_mask.append(torch.ones(keep.shape[0], dtype=torch.bool, device=device))
            field_cand_logits.append(logits)

        fq = query_states[safe_field_query_ids]
        instance_seed: list[tuple[int, int] | None] = []
        instance_spans: list[tuple[int, int] | None] = []

        if spec.mode == "natural":
            anchor_field_idx = field_query_ids.index(spec.anchor_query_id)
            anchor_states = field_cand_states[anchor_field_idx]
            anchor_spans = field_spans[anchor_field_idx]
            anchor_logits = field_cand_logits[anchor_field_idx]
            ni = anchor_states.shape[0]
            inst_states = anchor_states
            object_logits = anchor_logits
            for cand_idx in range(ni):
                instance_seed.append((anchor_field_idx, cand_idx))
                instance_spans.append((int(anchor_spans[cand_idx, 0]), int(anchor_spans[cand_idx, 1])))
        elif spec.mode == "latent":
            seed_states: list[torch.Tensor] = []
            seed_scores: list[torch.Tensor] = []
            for f_idx, states in enumerate(field_cand_states):
                if states.shape[0] == 0:
                    continue
                scores = self.latent_seed_head(states).squeeze(-1)
                for cand_idx in range(states.shape[0]):
                    seed_states.append(states[cand_idx])
                    seed_scores.append(scores[cand_idx])
                    instance_seed.append((f_idx, cand_idx))
                    span = field_spans[f_idx][cand_idx]
                    instance_spans.append((int(span[0]), int(span[1])))
            if seed_states:
                inst_states = torch.stack(seed_states, dim=0)
                object_logits = torch.stack(seed_scores, dim=0)
            else:
                inst_states = query_states.new_zeros((0, self.hidden_size))
                object_logits = query_states.new_zeros((0,))
        else:
            inst_states = self._anchorless_states(field_cand_states)
            object_logits = self.object_head(inst_states).squeeze(-1)
            for _ in range(inst_states.shape[0]):
                instance_seed.append(None)
                instance_spans.append(None)

        assign_logits = self._assign_logits(inst_states, fq, field_cand_states)
        return RecordGroupOutput(
            spec=spec,
            object_logits=object_logits,
            assign_logits=assign_logits,
            field_query_ids=field_query_ids,
            field_specs=field_specs,
            field_spans=field_spans,
            field_cand_mask=field_cand_mask,
            field_cand_logits=field_cand_logits,
            instance_seed=instance_seed,
            instance_spans=instance_spans,
        )

    def decode_group(
        self,
        group: RecordGroupOutput,
        *,
        anchor_threshold: float = 0.5,
        field_threshold: float = 0.5,
        object_threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> list[DecodedRecord]:
        """Decode `group` with the same assignment used by `decode_group`.

        Args:
            group: Scores from `forward_group`.
            anchor_threshold: Minimum object probability for anchored instances.
            field_threshold: Minimum field probability.
            object_threshold: Minimum object probability for anchorless instances.
            temperature: Positive divisor applied before probabilities.

        Returns:
            Decoded records that contain at least one field span.
        """
        requires_backends(self, ["scipy"])
        return _decode_group(
            group,
            _ScipyAssignment(),
            anchor_threshold=anchor_threshold,
            field_threshold=field_threshold,
            object_threshold=object_threshold,
            temperature=temperature,
        )
