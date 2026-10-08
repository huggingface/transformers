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

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RelationProposalSettings:
    """Caps for typed relation-pair proposal.

    Attributes:
        heads_per_relation: Head mentions retained per relation.
        tails_per_relation: Tail mentions retained per relation.
        pair_cap: Pairs retained after the capped cross product.
        argument_threshold: Minimum mention probability.
    """

    heads_per_relation: int = 32
    tails_per_relation: int = 32
    pair_cap: int = 128
    argument_threshold: float = 0.0


@dataclass(frozen=True)
class RelationTypeSpec:
    """One relation type and the entity queries allowed at each argument.

    Attributes:
        relation_type: Relation name.
        head_query_ids: Entity queries that may be the head.
        tail_query_ids: Entity queries that may be the tail.
        allow_self: Whether a span may fill both arguments.
    """

    relation_type: str
    head_query_ids: tuple[int, ...]
    tail_query_ids: tuple[int, ...]
    allow_self: bool = False


@dataclass
class RelationPairBatch:
    """Flattened typed relation pairs.

    Index tensors have shape `[P]`. Ends are exclusive.

    Attributes:
        batch_index: Sample index of each pair.
        relation_index: Relation index of each pair.
        head_start: Inclusive head start.
        head_end: Exclusive head end.
        tail_start: Inclusive tail start.
        tail_end: Exclusive tail end.
        head_prob: Head mention probability.
        tail_prob: Tail mention probability.
        pair_mask: True for pairs that should be scored.
        head_keys: Optional decode metadata `(type, start, end)`.
        tail_keys: Optional decode metadata `(type, start, end)`.
        relation_types: Optional relation name per pair.
    """

    batch_index: torch.Tensor
    relation_index: torch.Tensor
    head_start: torch.Tensor
    head_end: torch.Tensor
    tail_start: torch.Tensor
    tail_end: torch.Tensor
    head_prob: torch.Tensor
    tail_prob: torch.Tensor
    pair_mask: torch.Tensor | None = None
    head_keys: list[tuple[str, int, int]] = field(default_factory=list)
    tail_keys: list[tuple[str, int, int]] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.batch_index.shape[0])


def _query_type(layout, query_id: int) -> str:
    """Return a query role name, or the id when the layout has no such query."""
    try:
        return layout.query(query_id).role_name
    except KeyError:
        return str(query_id)


def safe_relation_indices(relation_indices: torch.Tensor, relation_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp relation indices and report which originals were in range.

    Args:
        relation_indices: Proposed relation ids.
        relation_count: Number of relation queries.

    Returns:
        Gather-safe indices and a boolean validity mask.
    """
    if relation_count <= 0:
        return (
            torch.zeros_like(relation_indices),
            torch.zeros_like(relation_indices, dtype=torch.bool),
        )
    valid = (relation_indices >= 0) & (relation_indices < relation_count)
    return relation_indices.clamp(min=0, max=relation_count - 1), valid


class TypedRelationPairGenerator:
    """Propose a capped cross product of typed head and tail mentions.

    Args:
        settings: Retention caps. Defaults to the standard inference caps.
    """

    def __init__(self, settings: RelationProposalSettings | None = None) -> None:
        self.settings = settings or RelationProposalSettings()

    def generate(
        self,
        candidates,
        query_layouts,
        relation_schema,
        *,
        compact: bool = True,
    ) -> RelationPairBatch:
        """Propose pairs for one schema shared by every sample.

        Args:
            candidates: Per-query mention candidates.
            query_layouts: One query layout per sample, used for decode names.
            relation_schema: Relation types applied to every sample.
            compact: Drop padded pairs and attach Python metadata.

        Returns:
            Flattened relation pairs.
        """
        schemas = [relation_schema for _ in range(candidates.indices.shape[0])]
        return self.generate_batched(candidates, query_layouts, schemas, compact=compact)

    def generate_batched(
        self,
        candidates,
        query_layouts,
        relation_schemas,
        *,
        compact: bool = False,
        routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> RelationPairBatch:
        """Propose `[B, R, pair_cap]` typed pairs.

        Args:
            candidates: Per-query mention candidates.
            query_layouts: Layouts used only when `compact` attaches names.
            relation_schemas: Relation types for each sample.
            compact: Drop invalid pads and materialize presentation keys.
            routing: Optional `(head_member, tail_member, relation_valid, allow_self)`.

        Returns:
            Flattened relation pairs. Invalid pads remain when `compact` is false.
        """
        settings = self.settings
        device = candidates.indices.device
        batch_size, queries, cand_count = candidates.valid_mask.shape
        rel_count = (
            routing[0].shape[1] if routing is not None else max((len(item) for item in relation_schemas), default=0)
        )
        if rel_count == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return RelationPairBatch(
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                candidates.pair_logits.new_zeros(0),
                candidates.pair_logits.new_zeros(0),
                pair_mask=torch.zeros(0, dtype=torch.bool, device=device),
            )

        if routing is not None:
            head_member, tail_member, relation_valid, allow_self = routing
            head_member = head_member.to(device=device)
            tail_member = tail_member.to(device=device)
            relation_valid = relation_valid.to(device=device)
            allow_self = allow_self.to(device=device)
        else:
            head_member = torch.zeros(batch_size, rel_count, queries, dtype=torch.bool, device=device)
            tail_member = torch.zeros_like(head_member)
            relation_valid = torch.zeros(batch_size, rel_count, dtype=torch.bool, device=device)
            allow_self = torch.zeros_like(relation_valid)
            for batch_index, schemas in enumerate(relation_schemas):
                for relation_index, spec in enumerate(schemas):
                    relation_valid[batch_index, relation_index] = True
                    allow_self[batch_index, relation_index] = spec.allow_self
                    valid_h = [q for q in spec.head_query_ids if 0 <= q < queries]
                    valid_t = [q for q in spec.tail_query_ids if 0 <= q < queries]
                    if valid_h:
                        head_member[batch_index, relation_index, valid_h] = True
                    if valid_t:
                        tail_member[batch_index, relation_index, valid_t] = True

        probs = torch.sigmoid(candidates.pair_logits)
        base_valid = candidates.valid_mask & candidates.query_mask.unsqueeze(-1)
        flat_prob = probs.reshape(batch_size, 1, queries * cand_count).expand(-1, rel_count, -1)
        flat_valid = base_valid.reshape(batch_size, 1, -1).expand(-1, rel_count, -1)
        head_valid = flat_valid & head_member.unsqueeze(-1).expand(-1, -1, -1, cand_count).reshape(
            batch_size, rel_count, -1
        )
        tail_valid = flat_valid & tail_member.unsqueeze(-1).expand(-1, -1, -1, cand_count).reshape(
            batch_size, rel_count, -1
        )
        threshold = flat_prob >= settings.argument_threshold
        head_valid = head_valid & threshold
        tail_valid = tail_valid & threshold
        floor = torch.finfo(flat_prob.dtype).min
        flat_spans = candidates.indices.reshape(batch_size, queries * cand_count, 2)

        def select(valid: torch.Tensor, requested: int):
            take = min(requested, queries * cand_count)
            secondary = (
                torch.arange(queries * cand_count, device=device).view(1, 1, -1).expand(batch_size, rel_count, -1)
            )
            end_key = flat_spans[..., 1].unsqueeze(1).expand(-1, rel_count, -1)
            end_order = torch.argsort(end_key.gather(-1, secondary), dim=-1, stable=True)
            secondary = secondary.gather(-1, end_order)
            start_key = flat_spans[..., 0].unsqueeze(1).expand(-1, rel_count, -1)
            start_order = torch.argsort(start_key.gather(-1, secondary), dim=-1, stable=True)
            secondary = secondary.gather(-1, start_order)
            ordered_score = flat_prob.gather(-1, secondary)
            ordered_valid = valid.gather(-1, secondary)
            rank_in_secondary = torch.argsort(
                ordered_score.masked_fill(~ordered_valid, floor),
                dim=-1,
                descending=True,
                stable=True,
            )[..., :take]
            ranked = secondary.gather(-1, rank_in_secondary)
            selected_valid = valid.gather(-1, ranked)
            selected_prob = flat_prob.gather(-1, ranked)
            if take < requested:
                pad = requested - take
                ranked = F.pad(ranked, (0, pad))
                selected_valid = F.pad(selected_valid, (0, pad), value=False)
                selected_prob = F.pad(selected_prob, (0, pad))
            qslot = torch.div(ranked, cand_count, rounding_mode="floor")
            cslot = ranked - qslot * cand_count
            qslot = qslot.clamp(0, queries - 1)
            cslot = cslot.clamp(0, cand_count - 1)
            batch = torch.arange(batch_size, device=device)[:, None, None]
            spans = candidates.indices[batch, qslot, cslot]
            return selected_prob, qslot, spans, selected_valid

        hp, hq, hspan, hvalid = select(head_valid, settings.heads_per_relation)
        tp, tq, tspan, tvalid = select(tail_valid, settings.tails_per_relation)
        pair_score = hp.unsqueeze(-1) * tp.unsqueeze(-2)
        pair_valid = hvalid.unsqueeze(-1) & tvalid.unsqueeze(-2)
        same_span = (hspan.unsqueeze(-2) == tspan.unsqueeze(-3)).all(-1)
        pair_valid = pair_valid & (allow_self[..., None, None] | ~same_span)
        pair_valid = pair_valid & relation_valid[..., None, None]
        flat_pair_score = pair_score.flatten(2)
        flat_pair_valid = pair_valid.flatten(2)
        take = min(settings.pair_cap, flat_pair_score.shape[-1])
        keep = torch.argsort(
            flat_pair_score.masked_fill(~flat_pair_valid, floor),
            dim=-1,
            descending=True,
            stable=True,
        )[..., :take]
        kept_valid = flat_pair_valid.gather(-1, keep)
        if take < settings.pair_cap:
            keep = F.pad(keep, (0, settings.pair_cap - take))
            kept_valid = F.pad(kept_valid, (0, settings.pair_cap - take), value=False)
        hi = torch.div(keep, settings.tails_per_relation, rounding_mode="floor")
        ti = keep - hi * settings.tails_per_relation
        hi = hi.clamp(0, settings.heads_per_relation - 1)
        ti = ti.clamp(0, settings.tails_per_relation - 1)

        def gather_selected(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return values.gather(
                2,
                index.clamp(0, values.shape[2] - 1).unsqueeze(-1).expand(*index.shape, values.shape[-1]),
            )

        hs = gather_selected(hspan, hi)
        ts = gather_selected(tspan, ti)
        hp_out = hp.gather(2, hi.clamp(0, hp.shape[2] - 1))
        tp_out = tp.gather(2, ti.clamp(0, tp.shape[2] - 1))
        hq_out = hq.gather(2, hi.clamp(0, hq.shape[2] - 1))
        tq_out = tq.gather(2, ti.clamp(0, tq.shape[2] - 1))
        bi = torch.arange(batch_size, device=device)[:, None, None].expand_as(keep)
        ri = torch.arange(rel_count, device=device)[None, :, None].expand_as(keep)
        flat_mask = kept_valid.reshape(-1)
        tensors = [
            bi.reshape(-1),
            ri.reshape(-1),
            hs[..., 0].reshape(-1),
            hs[..., 1].reshape(-1),
            ts[..., 0].reshape(-1),
            ts[..., 1].reshape(-1),
            hp_out.reshape(-1),
            tp_out.reshape(-1),
            hq_out.reshape(-1),
            tq_out.reshape(-1),
        ]
        if compact:
            tensors = [value[flat_mask] for value in tensors]
            flat_mask = torch.ones_like(tensors[0], dtype=torch.bool)
        out = RelationPairBatch(
            batch_index=tensors[0],
            relation_index=tensors[1],
            head_start=tensors[2],
            head_end=tensors[3],
            tail_start=tensors[4],
            tail_end=tensors[5],
            head_prob=tensors[6],
            tail_prob=tensors[7],
            pair_mask=flat_mask,
        )
        if compact:
            for index in range(len(out)):
                batch_index = int(out.batch_index[index])
                relation_index = int(out.relation_index[index])
                spec = relation_schemas[batch_index][relation_index]
                layout = query_layouts[batch_index] if batch_index < len(query_layouts) else None
                out.relation_types.append(spec.relation_type)
                hquery = int(tensors[8][index])
                tquery = int(tensors[9][index])
                out.head_keys.append(
                    (
                        _query_type(layout, hquery) if layout is not None else str(hquery),
                        int(out.head_start[index]),
                        int(out.head_end[index]),
                    )
                )
                out.tail_keys.append(
                    (
                        _query_type(layout, tquery) if layout is not None else str(tquery),
                        int(out.tail_start[index]),
                        int(out.tail_end[index]),
                    )
                )
        return out


class SparseRelationScorer(nn.Module):
    """Score proposed relation pairs from the four endpoint states.

    Args:
        hidden_size: Boundary-state width.
        dropout: MLP dropout probability.
        relation_query_dim: Relation-query width. Defaults to `hidden_size`.
        use_biaffine_content: Add pooled head/tail content features.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        relation_query_dim: int | None = None,
        use_biaffine_content: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.relation_query_dim = relation_query_dim or hidden_size
        self.use_biaffine_content = use_biaffine_content
        in_dim = 4 * hidden_size + self.relation_query_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        if use_biaffine_content:
            self.head_content_projection = nn.Linear(hidden_size, hidden_size)
            self.tail_content_projection = nn.Linear(hidden_size, hidden_size)
            self.relation_content_gate = nn.Linear(self.relation_query_dim, hidden_size)
            self.content_linear = nn.Linear(2 * hidden_size + self.relation_query_dim, 1)

    def forward(
        self,
        boundary_states: torch.Tensor,
        relation_query_states: torch.Tensor,
        entity_candidates,
        relation_pairs: RelationPairBatch,
    ) -> torch.Tensor:
        """Score each proposed pair.

        Args:
            boundary_states: Boundary or token states `[B, L, H]`.
            relation_query_states: Relation queries `[B, R, H]`.
            entity_candidates: Candidate carrier kept for call compatibility.
            relation_pairs: Pairs from `TypedRelationPairGenerator`.

        Returns:
            Pair logits `[P]`, zero where the pair is invalid.
        """
        del entity_candidates  # spans are read from relation_pairs
        if len(relation_pairs) == 0:
            return boundary_states.new_zeros(0)

        batch_count = min(boundary_states.shape[0], relation_query_states.shape[0])
        if batch_count <= 0 or relation_query_states.shape[1] <= 0:
            return boundary_states.new_zeros(len(relation_pairs))
        batch_valid = (relation_pairs.batch_index >= 0) & (relation_pairs.batch_index < batch_count)
        batch = relation_pairs.batch_index.clamp(0, batch_count - 1)
        relation_index, relation_valid = safe_relation_indices(
            relation_pairs.relation_index, relation_query_states.shape[1]
        )
        pair_valid = batch_valid & relation_valid
        if relation_pairs.pair_mask is not None:
            pair_valid = pair_valid & relation_pairs.pair_mask
        length = boundary_states.shape[1]

        def gather(pos: torch.Tensor) -> torch.Tensor:
            pos = pos.clamp(0, max(length - 1, 0))
            return boundary_states[batch, pos]

        h_start = gather(relation_pairs.head_start)
        h_end = gather(relation_pairs.head_end - 1)
        t_start = gather(relation_pairs.tail_start)
        t_end = gather(relation_pairs.tail_end - 1)
        rel = relation_query_states[batch, relation_index]
        delta = (relation_pairs.tail_start - relation_pairs.head_start).to(boundary_states.dtype)
        order = torch.sign(delta).unsqueeze(-1)
        dist = (delta.abs() / float(max(length, 1))).unsqueeze(-1)
        feats = torch.cat([h_start, h_end, t_start, t_end, rel, order, dist], dim=-1)
        score = self.mlp(feats).squeeze(-1)
        if self.use_biaffine_content:
            prefix = torch.cat(
                (
                    boundary_states.new_zeros(boundary_states.shape[0], 1, self.hidden_size),
                    boundary_states.float().cumsum(1).to(boundary_states.dtype),
                ),
                dim=1,
            )

            def pool(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
                span_sum = prefix[batch, end.clamp(0, length)] - prefix[batch, start.clamp(0, length)]
                width = (end - start).clamp_min(1).unsqueeze(-1).to(span_sum.dtype)
                return span_sum / width

            head_content = self.head_content_projection(pool(relation_pairs.head_start, relation_pairs.head_end))
            tail_content = self.tail_content_projection(pool(relation_pairs.tail_start, relation_pairs.tail_end))
            gate = torch.sigmoid(self.relation_content_gate(rel))
            biaffine = (head_content * gate * tail_content).sum(-1) / (self.hidden_size**0.5)
            linear = self.content_linear(torch.cat((head_content, tail_content, rel), dim=-1)).squeeze(-1)
            score = score + biaffine + linear
        return score.masked_fill(~pair_valid, 0.0)
