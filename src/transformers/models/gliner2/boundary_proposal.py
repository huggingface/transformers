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
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .boundary_content import gather_states
from .boundary_rotary import RotaryBoundaryEmbedding


MASK_LOGIT = -1.0e4


@dataclass(frozen=True)
class ProposalSettings:
    """Budgets and flags for sparse boundary proposal.

    Attributes:
        start_top_k: Starts retained before conditional end scoring.
        end_top_k: Ends retained before conditional start scoring.
        ends_per_start: Ends kept for each selected start.
        starts_per_end: Starts kept for each selected end.
        candidate_budget: Candidates kept at evaluation.
        training_candidate_budget: Candidates kept while training.
        max_gold_per_query: Reserved gold capacity from the head settings.
        end_block_size: Streaming block width.
        bidirectional: Also propose starts conditioned on selected ends.
        export_mode: `"auto"`, `"streaming"`, or `"vectorized"`.
        vectorized_pair_elements: Auto-mode cutoff for one full block.
        enable_rotary_endpoints: Rotate proposal endpoints.
        rotary_base: Rotary frequency base.
        boundary_top_k_alpha: Length-adaptive top-k slope. Zero keeps `base_k`.
        boundary_top_k_max: Cap for the adaptive top-k.
        boundary_top_k_bucket: Rounding bucket for the adaptive top-k.
    """

    start_top_k: int
    end_top_k: int
    ends_per_start: int
    starts_per_end: int
    candidate_budget: int
    training_candidate_budget: int
    max_gold_per_query: int
    end_block_size: int
    bidirectional: bool = True
    export_mode: str = "auto"
    vectorized_pair_elements: int = 16_777_216
    enable_rotary_endpoints: bool = False
    rotary_base: float = 10000.0
    boundary_top_k_alpha: float = 0.0
    boundary_top_k_max: int = 128
    boundary_top_k_bucket: int = 8


@dataclass(frozen=True)
class ProposalStats:
    """Element counts and optional gold-recall diagnostics."""

    boundary_score_elements: int
    conditional_pair_score_elements: int
    max_materialized_pair_elements: int
    retained_candidate_count: torch.Tensor
    gold_hit_without_injection: torch.Tensor | None = None
    gold_total: torch.Tensor | None = None
    start_hit: torch.Tensor | None = None
    end_hit: torch.Tensor | None = None
    boundary_total: torch.Tensor | None = None
    unique_candidates: torch.Tensor | None = None


@dataclass
class BoundaryProposals:
    """Padded start/end candidates for one query axis.

    Attributes:
        indices: Half-open `[start, end)` pairs `[B, Q, C, 2]`.
        logits: Full prior, or None when proposal logits were not requested.
        valid_mask: True for real candidates `[B, Q, C]`.
        gold_mask: True where a training candidate was injected.
        stats: Optional proposal diagnostics.
        compat_logits: Marginal-free endpoint compatibility `[B, Q, C]`.
        score_start_states: Gathered reranker start states.
        score_end_states: Gathered reranker end states.
    """

    indices: torch.Tensor
    logits: torch.Tensor | None
    valid_mask: torch.Tensor
    gold_mask: torch.Tensor | None = None
    stats: ProposalStats | None = None
    compat_logits: torch.Tensor | None = None
    score_start_states: torch.Tensor | None = None
    score_end_states: torch.Tensor | None = None


@dataclass
class CandidateTensorBatch:
    """Per-query candidates consumed by decoding.

    Attributes:
        indices: Half-open spans `[B, Q, C, 2]`.
        proposal_logits: Proposal prior `[B, Q, C]`, if retained.
        pair_logits: Reranked pair logits `[B, Q, C]`.
        valid_mask: Real-candidate mask `[B, Q, C]`.
        query_mask: Real-query mask `[B, Q]`.
        candidate_states: Optional endpoint states `[B, Q, C, H]`.
    """

    indices: torch.Tensor
    proposal_logits: torch.Tensor | None
    pair_logits: torch.Tensor
    valid_mask: torch.Tensor
    query_mask: torch.Tensor
    candidate_states: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.indices.dim() != 4 or self.indices.shape[-1] != 2:
            raise ValueError(f"indices must be [B, Q, C, 2], got {tuple(self.indices.shape)}")
        batch, queries, count, _ = self.indices.shape
        if self.proposal_logits is not None and tuple(self.proposal_logits.shape) != (
            batch,
            queries,
            count,
        ):
            raise ValueError("proposal_logits shape must be [B, Q, C]")
        if tuple(self.pair_logits.shape) != (batch, queries, count):
            raise ValueError("pair_logits shape must be [B, Q, C]")
        if tuple(self.valid_mask.shape) != (batch, queries, count):
            raise ValueError("valid_mask shape must be [B, Q, C]")
        if tuple(self.query_mask.shape) != (batch, queries):
            raise ValueError("query_mask shape must be [B, Q]")


def select_top_boundaries(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the top-k boundaries with a stable index tie-break.

    Args:
        logits: Boundary scores `[B, Q, N]`.
        valid_mask: Positions that may be selected.
        k: Maximum boundaries to keep.

    Returns:
        Scores, indices, and a validity mask, each `[B, Q, k]`.
    """
    length = logits.shape[-1]
    k = min(k, length)
    masked = logits.masked_fill(~valid_mask, MASK_LOGIT)
    scores, idx = torch.sort(masked, dim=-1, descending=True, stable=True)
    scores = scores[..., :k]
    idx = idx[..., :k]
    valid = torch.gather(valid_mask, -1, idx)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    return scores, idx, valid


def resolve_boundary_budget(
    n_boundaries: int,
    *,
    base_k: int,
    alpha: float,
    k_max: int,
    bucket: int,
) -> int:
    """Resolve a length-adaptive boundary top-k from the sequence shape.

    Args:
        n_boundaries: Number of boundary positions.
        base_k: Budget used when `alpha` is zero.
        alpha: Fraction of the token length requested as extra budget.
        k_max: Hard cap.
        bucket: Rounding multiple.

    Returns:
        The resolved top-k.
    """
    if alpha <= 0.0:
        return base_k
    requested = int(math.ceil(alpha * max(n_boundaries - 1, 0)))
    requested = max(base_k, min(requested, k_max))
    return min(k_max, int(math.ceil(requested / bucket) * bucket))


def merge_running_topk(
    current_scores: torch.Tensor,
    current_indices: torch.Tensor,
    block_scores: torch.Tensor,
    block_indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a running top-k with the next block, breaking ties stably.

    Args:
        current_scores: Scores retained so far.
        current_indices: Indices retained so far.
        block_scores: Scores from the new block.
        block_indices: Indices from the new block.
        k: Maximum entries to keep.

    Returns:
        Merged scores and indices.
    """
    scores = torch.cat([current_scores, block_scores], dim=-1)
    indices = torch.cat([current_indices, block_indices], dim=-1)
    top_scores, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    take = min(k, scores.shape[-1])
    top_scores = top_scores[..., :take]
    order = order[..., :take]
    top_indices = torch.gather(indices, -1, order)
    return top_scores, top_indices


def _score_ends_blockwise(
    sq: torch.Tensor,
    end_proj_all: torch.Tensor,
    start_indices: torch.Tensor,
    start_scores: torch.Tensor,
    start_valid: torch.Tensor,
    boundary_mask: torch.Tensor,
    query_mask: torch.Tensor,
    end_marginals: torch.Tensor,
    block_size: int,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Stream end blocks and keep the top ends after each selected start.

    Args:
        sq: Gated start states `[B, Q, K, D]`.
        end_proj_all: Projected end states `[B, N, D]`.
        start_indices: Selected start positions `[B, Q, K]`.
        start_scores: Selected start scores `[B, Q, K]`.
        start_valid: Real start slots `[B, Q, K]`.
        boundary_mask: Valid boundaries `[B, N]`.
        query_mask: Valid queries `[B, Q]`.
        end_marginals: End marginals `[B, Q, N]`.
        block_size: End positions scored together.
        top_k: Ends retained per start.
        scale: Compatibility scale.

    Returns:
        Top scores, end indices, scored elements, and the largest block.
    """
    batch, queries, _k, _dim = sq.shape
    n_boundaries = end_proj_all.shape[1]
    device = sq.device
    top_scores = torch.full((batch, queries, _k, top_k), MASK_LOGIT, device=device, dtype=sq.dtype)
    top_idx = torch.zeros((batch, queries, _k, top_k), device=device, dtype=torch.long)
    conditional_elems = 0
    max_block_elems = 0
    for start in range(0, n_boundaries, block_size):
        stop = min(start + block_size, n_boundaries)
        width = stop - start
        block_compat = torch.einsum("bqkd,bed->bqke", sq, end_proj_all[:, start:stop]) * scale
        block = block_compat + end_marginals[:, :, start:stop].unsqueeze(2) + start_scores.unsqueeze(-1)
        end_index = torch.arange(start, stop, device=device)
        keep = (
            boundary_mask[:, start:stop].view(batch, 1, 1, width)
            & query_mask.view(batch, queries, 1, 1)
            & (end_index.view(1, 1, 1, width) > start_indices.unsqueeze(-1))
            & start_valid.unsqueeze(-1)
        )
        block = block.masked_fill(~keep, MASK_LOGIT)
        block_idx = end_index.view(1, 1, 1, width).expand(batch, queries, _k, width)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)
        conditional_elems += batch * queries * _k * width
        max_block_elems = max(max_block_elems, batch * queries * _k * width)
    return top_scores, top_idx, conditional_elems, max_block_elems


def _score_starts_blockwise(
    eq: torch.Tensor,
    start_proj_all: torch.Tensor,
    end_indices: torch.Tensor,
    end_scores: torch.Tensor,
    end_valid: torch.Tensor,
    boundary_mask: torch.Tensor,
    query_mask: torch.Tensor,
    start_marginals: torch.Tensor,
    block_size: int,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Stream start blocks and keep the top starts before each selected end.

    Args:
        eq: Gated end states `[B, Q, K, D]`.
        start_proj_all: Projected start states `[B, N, D]`.
        end_indices: Selected end positions `[B, Q, K]`.
        end_scores: Selected end scores `[B, Q, K]`.
        end_valid: Real end slots `[B, Q, K]`.
        boundary_mask: Valid boundaries `[B, N]`.
        query_mask: Valid queries `[B, Q]`.
        start_marginals: Start marginals `[B, Q, N]`.
        block_size: Start positions scored together.
        top_k: Starts retained per end.
        scale: Compatibility scale.

    Returns:
        Top scores, start indices, scored elements, and the largest block.
    """
    batch, queries, _k, _dim = eq.shape
    n_boundaries = start_proj_all.shape[1]
    device = eq.device
    top_scores = torch.full((batch, queries, _k, top_k), MASK_LOGIT, device=device, dtype=eq.dtype)
    top_idx = torch.zeros((batch, queries, _k, top_k), device=device, dtype=torch.long)
    conditional_elems = 0
    max_block_elems = 0
    for start in range(0, n_boundaries, block_size):
        stop = min(start + block_size, n_boundaries)
        width = stop - start
        block_compat = torch.einsum("bqkd,bed->bqke", eq, start_proj_all[:, start:stop]) * scale
        block = block_compat + start_marginals[:, :, start:stop].unsqueeze(2) + end_scores.unsqueeze(-1)
        start_index = torch.arange(start, stop, device=device)
        keep = (
            boundary_mask[:, start:stop].view(batch, 1, 1, width)
            & query_mask.view(batch, queries, 1, 1)
            & (start_index.view(1, 1, 1, width) < end_indices.unsqueeze(-1))
            & end_valid.unsqueeze(-1)
        )
        block = block.masked_fill(~keep, MASK_LOGIT)
        block_idx = start_index.view(1, 1, 1, width).expand(batch, queries, _k, width)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)
        conditional_elems += batch * queries * _k * width
        max_block_elems = max(max_block_elems, batch * queries * _k * width)
    return top_scores, top_idx, conditional_elems, max_block_elems


def assemble_candidates(
    pair_starts: torch.Tensor,
    pair_ends: torch.Tensor,
    pair_scores: torch.Tensor,
    pair_valid: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    capacity: int,
    n_boundaries: int,
    gold_pairs: torch.Tensor | None = None,
    gold_mask: torch.Tensor | None = None,
    gold_injection_prob: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate pairs, optionally inject gold, and pad to capacity.

    Args:
        pair_starts: Proposed starts `[B, Q, P]`.
        pair_ends: Proposed ends `[B, Q, P]`.
        pair_scores: Selection scores `[B, Q, P]`.
        pair_valid: Real proposals `[B, Q, P]`.
        query_mask: Real queries `[B, Q]`.
        capacity: Output candidate count.
        n_boundaries: Boundary axis length used to pack pair keys.
        gold_pairs: Optional gold spans `[B, Q, G, 2]`.
        gold_mask: Optional gold mask `[B, Q, G]`.
        gold_injection_prob: Probability of keeping each gold span.
        generator: Generator for partial gold injection.

    Returns:
        Indices, validity, gold mask, pre-injection keys, and pre-injection mask.
    """
    dtype = pair_scores.dtype
    floor = MASK_LOGIT
    ceiling = -floor
    invalid_key = n_boundaries * n_boundaries
    pre_valid = pair_valid & query_mask.unsqueeze(-1)
    pre_keys = pair_starts * n_boundaries + pair_ends
    pre_keys = torch.where(pre_valid, pre_keys, torch.full_like(pre_keys, invalid_key))
    keys = pre_keys
    scores = torch.where(pre_valid, pair_scores, torch.full_like(pair_scores, floor))
    valid = pre_valid
    is_gold = torch.zeros_like(valid)

    if gold_pairs is not None and gold_mask is not None:
        gvalid = gold_mask & query_mask.unsqueeze(-1)
        oob = (gold_pairs[..., 0] >= n_boundaries) | (gold_pairs[..., 1] >= n_boundaries) | (gold_pairs < 0).any(-1)
        gvalid = gvalid & ~oob
        if gold_injection_prob <= 0.0:
            gvalid = torch.zeros_like(gvalid)
        elif gold_injection_prob < 1.0:
            sampled = torch.rand(gvalid.shape, device=gvalid.device, generator=generator)
            gvalid = gvalid & (sampled < gold_injection_prob)
        safe_gold = gold_pairs.clamp(0, n_boundaries - 1)
        gkeys = safe_gold[..., 0] * n_boundaries + safe_gold[..., 1]
        gkeys = torch.where(gvalid, gkeys, torch.full_like(gkeys, invalid_key))
        gscores = torch.where(
            gvalid,
            torch.full(gvalid.shape, ceiling, dtype=dtype, device=pair_scores.device),
            torch.full(gvalid.shape, floor, dtype=dtype, device=pair_scores.device),
        )
        keys = torch.cat((keys, gkeys), dim=-1)
        scores = torch.cat((scores, gscores), dim=-1)
        valid = torch.cat((valid, gvalid), dim=-1)
        is_gold = torch.cat((is_gold, gvalid), dim=-1)

    by_score = torch.argsort(scores, dim=-1, descending=True, stable=True)
    keys = torch.gather(keys, -1, by_score)
    scores = torch.gather(scores, -1, by_score)
    valid = torch.gather(valid, -1, by_score)
    is_gold = torch.gather(is_gold, -1, by_score)
    by_key = torch.argsort(keys, dim=-1, stable=True)
    keys = torch.gather(keys, -1, by_key)
    scores = torch.gather(scores, -1, by_key)
    valid = torch.gather(valid, -1, by_key)
    is_gold = torch.gather(is_gold, -1, by_key)

    first = torch.ones_like(valid)
    first[..., 1:] = keys[..., 1:] != keys[..., :-1]
    keep = valid & first
    scores = torch.where(keep, scores, torch.full_like(scores, floor))
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    take = min(capacity, order.shape[-1])
    order = order[..., :take]
    selected_keys = torch.gather(keys, -1, order)
    selected_valid = torch.gather(keep, -1, order)
    selected_gold = torch.gather(is_gold, -1, order) & selected_valid

    starts = torch.div(selected_keys, n_boundaries, rounding_mode="floor")
    ends = selected_keys - starts * n_boundaries
    indices = torch.stack((starts, ends), dim=-1)
    indices = torch.where(selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices))
    if take < capacity:
        pad = capacity - take
        indices = F.pad(indices, (0, 0, 0, pad))
        selected_valid = F.pad(selected_valid, (0, pad), value=False)
        selected_gold = F.pad(selected_gold, (0, pad), value=False)
    return indices, selected_valid, selected_gold, pre_keys, pre_valid


class SparseBoundaryProposer(nn.Module):
    """Select a capped set of start/end pairs for each query.

    Args:
        boundary_dim: Boundary state width.
        query_dim: Query state width.
        settings: Proposal budgets and rotary flags.
    """

    def __init__(self, boundary_dim: int, query_dim: int, settings: ProposalSettings):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.settings = settings
        self.start_pair_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_key_projection = nn.Linear(boundary_dim, boundary_dim)
        self.start_query_projection = nn.Linear(
            query_dim,
            boundary_dim // 2 if settings.enable_rotary_endpoints else boundary_dim,
        )
        self.rotary = (
            RotaryBoundaryEmbedding(boundary_dim, settings.rotary_base) if settings.enable_rotary_endpoints else None
        )

    def _project(
        self, boundary_states: torch.Tensor, query_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project endpoints and the query gate.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.

        Returns:
            Start states, end states, and the query gate.
        """
        start_all = self.start_pair_projection(boundary_states)
        end_all = self.end_key_projection(boundary_states)
        if self.rotary is not None:
            positions = torch.arange(boundary_states.shape[1], device=boundary_states.device).view(1, -1)
            start_all = self.rotary(start_all, positions)
            end_all = self.rotary(end_all, positions)
        gate = torch.sigmoid(self.start_query_projection(query_states))
        if self.rotary is not None:
            gate = gate.repeat_interleave(2, dim=-1)
        return start_all, end_all, gate

    def score_explicit_pairs(
        self,
        boundary_states: torch.Tensor,
        query_states: torch.Tensor,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score the marginal-free prior at caller-provided spans.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.
            indices: Half-open spans `[B, Q, C, 2]`.
            valid_mask: Spans that receive a finite prior `[B, Q, C]`.

        Returns:
            Compatibility logits `[B, Q, C]`, zero where invalid.
        """
        if indices.dim() != 4 or indices.shape[-1] != 2:
            raise ValueError(f"indices must be [B, Q, C, 2], got {tuple(indices.shape)}")
        if valid_mask.shape != indices.shape[:-1]:
            raise ValueError(
                f"valid_mask must match indices [B, Q, C], got {tuple(valid_mask.shape)} and {tuple(indices.shape)}"
            )
        if (
            indices.shape[0] != boundary_states.shape[0]
            or indices.shape[0] != query_states.shape[0]
            or indices.shape[1] != query_states.shape[1]
        ):
            raise ValueError("explicit pair batch/query dimensions do not match states")

        start_all, end_all, gate = self._project(boundary_states, query_states)
        limit = boundary_states.shape[1] - 1
        starts = indices[..., 0].clamp(0, limit)
        ends = indices[..., 1].clamp(0, limit)
        start_states = gather_states(start_all, starts) * gate.unsqueeze(2)
        end_states = gather_states(end_all, ends)
        compatibility = (start_states * end_states).sum(-1) / math.sqrt(self.boundary_dim)
        return torch.where(valid_mask, compatibility, torch.zeros_like(compatibility))

    def forward(
        self,
        boundary_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        *,
        gold_pairs: torch.Tensor | None = None,
        gold_mask: torch.Tensor | None = None,
        return_stats: bool = False,
        return_proposal_logits: bool = True,
        gold_injection_prob: float = 1.0,
        generator: torch.Generator | None = None,
        scorer_start_states: torch.Tensor | None = None,
        scorer_end_states: torch.Tensor | None = None,
    ) -> BoundaryProposals:
        """Propose capped start/end pairs and their compatibility prior.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            boundary_mask: Valid boundaries `[B, N]`.
            query_states: Query states `[B, Q, H]`.
            query_mask: Valid queries `[B, Q]`.
            start_logits: Start marginals `[B, Q, N]`.
            end_logits: End marginals `[B, Q, N]`.
            gold_pairs: Optional supervised spans. Injected only while training.
            gold_mask: Mask for `gold_pairs`.
            return_stats: Populate proposal diagnostics.
            return_proposal_logits: Materialize the full prior.
            gold_injection_prob: Fraction of gold spans injected in training.
            generator: Generator for partial gold injection.
            scorer_start_states: Optional reranker start states `[B, N, P]`.
            scorer_end_states: Optional reranker end states `[B, N, P]`.

        Returns:
            Padded proposals. Invalid slots have a zero compatibility prior.
        """
        settings = self.settings
        batch, n_boundaries, _dim = boundary_states.shape
        queries = query_states.shape[1]
        scale = 1.0 / math.sqrt(self.boundary_dim)
        training = self.training and gold_pairs is not None
        capacity = settings.training_candidate_budget if training else settings.candidate_budget
        start_k = resolve_boundary_budget(
            n_boundaries,
            base_k=settings.start_top_k,
            alpha=settings.boundary_top_k_alpha,
            k_max=settings.boundary_top_k_max,
            bucket=settings.boundary_top_k_bucket,
        )
        end_k = resolve_boundary_budget(
            n_boundaries,
            base_k=settings.end_top_k,
            alpha=settings.boundary_top_k_alpha,
            k_max=settings.boundary_top_k_max,
            bucket=settings.boundary_top_k_bucket,
        )
        pair_elements = batch * queries * start_k * n_boundaries
        use_vectorized = settings.export_mode == "vectorized" or (
            settings.export_mode == "auto" and pair_elements <= settings.vectorized_pair_elements
        )
        end_block = n_boundaries if use_vectorized else settings.end_block_size
        b_valid = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)

        start_proj_all, end_proj_all, gate = self._project(boundary_states, query_states)
        select_start = start_proj_all.detach()
        select_end = end_proj_all.detach()
        select_gate = gate.detach()
        select_start_logits = start_logits.detach()
        select_end_logits = end_logits.detach()

        with torch.no_grad():
            st_scores, st_idx, st_valid = select_top_boundaries(select_start_logits, b_valid, start_k)
            sq = gather_states(select_start, st_idx) * select_gate.unsqueeze(2)
            fwd_scores, fwd_end_idx, cond_e1, maxe1 = _score_ends_blockwise(
                sq,
                select_end,
                st_idx,
                st_scores,
                st_valid,
                boundary_mask,
                query_mask,
                select_end_logits,
                end_block,
                settings.ends_per_start,
                scale,
            )
        fwd_start = st_idx.unsqueeze(-1).expand(-1, -1, -1, settings.ends_per_start)
        fwd_pairs_s = fwd_start.reshape(batch, queries, -1)
        fwd_pairs_e = fwd_end_idx.reshape(batch, queries, -1)
        fwd_pairs_sc = fwd_scores.reshape(batch, queries, -1)
        fwd_pairs_valid = (
            st_valid.unsqueeze(-1)
            & query_mask.view(batch, queries, 1, 1)
            & boundary_mask.gather(1, fwd_end_idx.reshape(batch, -1).clamp(0, boundary_mask.shape[1] - 1)).view_as(
                fwd_end_idx
            )
            & (fwd_end_idx > fwd_start)
        ).reshape(batch, queries, -1)

        pair_starts = [fwd_pairs_s]
        pair_ends = [fwd_pairs_e]
        pair_scores = [fwd_pairs_sc]
        pair_valid = [fwd_pairs_valid]
        cond_e2 = 0
        maxe2 = 0
        en_idx = en_valid = None
        if settings.bidirectional:
            with torch.no_grad():
                en_scores, en_idx, en_valid = select_top_boundaries(select_end_logits, b_valid, end_k)
                eq = gather_states(select_end, en_idx) * select_gate.unsqueeze(2)
                bwd_scores, bwd_start_idx, cond_e2, maxe2 = _score_starts_blockwise(
                    eq,
                    select_start,
                    en_idx,
                    en_scores,
                    en_valid,
                    boundary_mask,
                    query_mask,
                    select_start_logits,
                    end_block,
                    settings.starts_per_end,
                    scale,
                )
            bwd_end = en_idx.unsqueeze(-1).expand(-1, -1, -1, settings.starts_per_end)
            pair_starts.append(bwd_start_idx.reshape(batch, queries, -1))
            pair_ends.append(bwd_end.reshape(batch, queries, -1))
            pair_scores.append(bwd_scores.reshape(batch, queries, -1))
            pair_valid.append(
                (
                    en_valid.unsqueeze(-1)
                    & query_mask.view(batch, queries, 1, 1)
                    & boundary_mask.gather(
                        1,
                        bwd_start_idx.reshape(batch, -1).clamp(0, boundary_mask.shape[1] - 1),
                    ).view_as(bwd_start_idx)
                    & (bwd_end > bwd_start_idx)
                ).reshape(batch, queries, -1)
            )

        all_s = torch.cat(pair_starts, dim=-1)
        all_e = torch.cat(pair_ends, dim=-1)
        all_sc = torch.cat(pair_scores, dim=-1).detach()
        all_valid = torch.cat(pair_valid, dim=-1)
        with torch.no_grad():
            out_idx, out_valid, out_gold, pre_keys, pre_valid = assemble_candidates(
                all_s,
                all_e,
                all_sc,
                all_valid,
                query_mask,
                capacity=capacity,
                n_boundaries=n_boundaries,
                gold_pairs=gold_pairs if training else None,
                gold_mask=gold_mask if training else None,
                gold_injection_prob=gold_injection_prob,
                generator=generator,
            )

        si = out_idx[..., 0]
        ej = out_idx[..., 1]
        score_start_selected = score_end_selected = None
        if scorer_start_states is not None and scorer_end_states is not None:
            prop_dim = start_proj_all.shape[-1]
            score_dim = scorer_start_states.shape[-1]
            start_all = torch.cat((start_proj_all, scorer_start_states), dim=-1)
            end_all = torch.cat((end_proj_all, scorer_end_states), dim=-1)
            g_s, score_start_selected = gather_states(start_all, si).split((prop_dim, score_dim), dim=-1)
            g_e, score_end_selected = gather_states(end_all, ej).split((prop_dim, score_dim), dim=-1)
        else:
            g_s = gather_states(start_proj_all, si)
            g_e = gather_states(end_proj_all, ej)
        g_s = g_s * gate.unsqueeze(2)
        compat = (g_s * g_e).sum(-1) * scale
        out_logits = None
        if self.training or return_proposal_logits:
            sm = torch.gather(start_logits, 2, si.clamp(0, start_logits.shape[2] - 1))
            em = torch.gather(end_logits, 2, ej.clamp(0, end_logits.shape[2] - 1))
            logits_diff = compat + sm + em
            out_logits = torch.where(out_valid, logits_diff, torch.full_like(logits_diff, MASK_LOGIT))
        out_compat = torch.where(out_valid, compat, torch.zeros_like(compat))

        stats = None
        if return_stats:
            retained = out_valid.sum()
            gold_hit = gold_total = start_hit = end_hit = boundary_total = None
            if gold_pairs is not None and gold_mask is not None:
                diagnostic_gold = gold_mask & query_mask.unsqueeze(-1)
                gold_keys = gold_pairs[..., 0] * n_boundaries + gold_pairs[..., 1]
                pair_hit = ((gold_keys.unsqueeze(-1) == pre_keys.unsqueeze(-2)) & pre_valid.unsqueeze(-2)).any(
                    -1
                ) & diagnostic_gold
                gold_hit = pair_hit.sum()
                gold_total = diagnostic_gold.sum()
                selected_starts = st_idx
                selected_starts_valid = st_valid
                if settings.bidirectional:
                    selected_ends = en_idx
                    selected_ends_valid = en_valid
                else:
                    selected_ends = fwd_end_idx.reshape(batch, queries, -1)
                    selected_ends_valid = fwd_pairs_valid
                start_hit = (
                    (gold_pairs[..., 0].unsqueeze(-1) == selected_starts.unsqueeze(-2))
                    & selected_starts_valid.unsqueeze(-2)
                ).any(-1) & diagnostic_gold
                end_hit = (
                    (gold_pairs[..., 1].unsqueeze(-1) == selected_ends.unsqueeze(-2))
                    & selected_ends_valid.unsqueeze(-2)
                ).any(-1) & diagnostic_gold
                start_hit = start_hit.sum()
                end_hit = end_hit.sum()
                boundary_total = diagnostic_gold.sum()
            stats = ProposalStats(
                boundary_score_elements=batch * queries * n_boundaries * 2,
                conditional_pair_score_elements=cond_e1 + cond_e2,
                max_materialized_pair_elements=max(maxe1, maxe2),
                retained_candidate_count=retained,
                gold_hit_without_injection=gold_hit,
                gold_total=gold_total,
                start_hit=start_hit,
                end_hit=end_hit,
                boundary_total=boundary_total,
                unique_candidates=retained,
            )
        return BoundaryProposals(
            indices=out_idx,
            logits=out_logits,
            valid_mask=out_valid,
            gold_mask=out_gold if training else None,
            stats=stats,
            compat_logits=out_compat,
            score_start_states=score_start_selected,
            score_end_states=score_end_selected,
        )
