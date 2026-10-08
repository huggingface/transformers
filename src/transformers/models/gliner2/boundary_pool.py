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

from .boundary_content import SpanContentPooler, gather_rows
from .boundary_proposal import CandidateTensorBatch, ProposalStats, select_top_boundaries


MASK_LOGIT = -1.0e4

OVERLAP_IDENTICAL = 0
OVERLAP_NESTED_INSIDE = 1
OVERLAP_NESTED_OUTSIDE = 2
OVERLAP_CROSSING = 3
OVERLAP_SAME_START = 4
OVERLAP_SAME_END = 5
OVERLAP_DISJOINT_LEFT = 6
OVERLAP_DISJOINT_RIGHT = 7
NUM_OVERLAP_BUCKETS = 8


@dataclass
class PooledCandidates:
    """One document-level span pool.

    Attributes:
        indices: Half-open spans `[B, C, 2]`.
        mask: Real pool entries `[B, C]`.
        proposal_logits: Query-agnostic proposal scores `[B, C]`.
        gold_mask: Gold membership `[B, C, Q]`.
        compat_logits: Marginal-free compatibility `[B, C]`.
        stats: Optional pool diagnostics.
    """

    indices: torch.Tensor
    mask: torch.Tensor
    proposal_logits: torch.Tensor | None
    gold_mask: torch.Tensor | None
    compat_logits: torch.Tensor | None = None
    stats: ProposalStats | None = None

    def to_candidate_batch(
        self,
        pair_logits: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_states: torch.Tensor | None = None,
    ) -> CandidateTensorBatch:
        """Broadcast the document pool onto the per-query candidate layout.

        Args:
            pair_logits: Query scores in candidate-major order `[B, C, Q]`.
            query_mask: Valid queries `[B, Q]`.
            candidate_states: Optional pool states `[B, C, H]`.

        Returns:
            Candidates with query-major pair logits `[B, Q, C]`.
        """
        batch, count, _ = self.indices.shape
        queries = query_mask.shape[1]
        indices = self.indices.unsqueeze(1).expand(batch, queries, count, 2)
        valid = self.mask.unsqueeze(1).expand(batch, queries, count)
        proposal = (
            self.proposal_logits.unsqueeze(1).expand(batch, queries, count)
            if self.proposal_logits is not None
            else None
        )
        states = (
            candidate_states.unsqueeze(1).expand(batch, queries, count, candidate_states.shape[-1])
            if candidate_states is not None
            else None
        )
        return CandidateTensorBatch(
            indices=indices,
            proposal_logits=proposal,
            pair_logits=pair_logits.transpose(1, 2),
            valid_mask=valid,
            query_mask=query_mask,
            candidate_states=states,
        )


def _deduplicate_pool(
    keys: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    capacity: int,
    n_boundaries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the highest-priority copy of each document span key.

    Args:
        keys: Packed start/end keys `[B, P]`.
        scores: Priority scores `[B, P]`.
        valid: Real keys `[B, P]`.
        capacity: Output width.
        n_boundaries: Boundary count used to mark invalid keys.

    Returns:
        Selected keys and their validity mask, padded to `capacity`.
    """
    invalid_key = n_boundaries * n_boundaries
    keys = torch.where(valid, keys, torch.full_like(keys, invalid_key))
    scores = torch.where(valid, scores, torch.full_like(scores, MASK_LOGIT))
    by_score = torch.argsort(scores, dim=-1, descending=True, stable=True)
    keys = keys.gather(-1, by_score)
    scores = scores.gather(-1, by_score)
    valid = valid.gather(-1, by_score)
    by_key = torch.argsort(keys, dim=-1, stable=True)
    keys = keys.gather(-1, by_key)
    scores = scores.gather(-1, by_key)
    valid = valid.gather(-1, by_key)
    first = torch.ones_like(valid)
    first[..., 1:] = keys[..., 1:] != keys[..., :-1]
    keep = valid & first
    order = torch.argsort(
        torch.where(keep, scores, torch.full_like(scores, MASK_LOGIT)),
        dim=-1,
        descending=True,
        stable=True,
    )[..., :capacity]
    selected_keys = keys.gather(-1, order)
    selected_valid = keep.gather(-1, order)
    if selected_keys.shape[-1] < capacity:
        pad = capacity - selected_keys.shape[-1]
        selected_keys = F.pad(selected_keys, (0, pad))
        selected_valid = F.pad(selected_valid, (0, pad), value=False)
    return selected_keys, selected_valid


class DocumentCandidatePool(nn.Module):
    """Build one deduplicated span pool per document.

    Args:
        boundary_dim: Boundary state width.
        pool_boundary_top_k: Endpoints kept from the query-union marginals.
        pool_size: Maximum spans retained per document.
        min_pool_per_query: Spans reserved for each active query.
    """

    def __init__(
        self,
        boundary_dim: int,
        *,
        pool_boundary_top_k: int,
        pool_size: int,
        min_pool_per_query: int,
    ) -> None:
        super().__init__()
        self.pool_boundary_top_k = pool_boundary_top_k
        self.pool_size = pool_size
        self.min_pool_per_query = min_pool_per_query
        self.start_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_projection = nn.Linear(boundary_dim, boundary_dim)

    def forward(
        self,
        boundary_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        query_mask: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        *,
        gold_pairs: torch.Tensor | None = None,
        gold_mask: torch.Tensor | None = None,
        gold_injection_prob: float = 1.0,
        return_stats: bool = False,
        generator: torch.Generator | None = None,
    ) -> PooledCandidates:
        """Pair union endpoints once and keep a capped document pool.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            boundary_mask: Valid boundaries `[B, N]`.
            query_mask: Valid queries `[B, Q]`.
            start_logits: Start marginals `[B, Q, N]`.
            end_logits: End marginals `[B, Q, N]`.
            gold_pairs: Optional gold spans `[B, Q, G, 2]`.
            gold_mask: Mask for `gold_pairs`.
            gold_injection_prob: Fraction of gold spans forced into the pool.
            return_stats: Populate recall diagnostics.
            generator: Generator for partial gold injection.

        Returns:
            A padded document pool.
        """
        batch, n_boundaries, dim = boundary_states.shape
        queries = query_mask.shape[1]
        floor = torch.full_like(start_logits, MASK_LOGIT)
        q_boundary = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
        union_start = torch.where(q_boundary, start_logits, floor).amax(1)
        union_end = torch.where(q_boundary, end_logits, floor).amax(1)
        union_valid = boundary_mask & query_mask.any(-1, keepdim=True)

        _, starts, starts_valid = select_top_boundaries(
            union_start.unsqueeze(1),
            union_valid.unsqueeze(1),
            self.pool_boundary_top_k,
        )
        _, ends, ends_valid = select_top_boundaries(
            union_end.unsqueeze(1),
            union_valid.unsqueeze(1),
            self.pool_boundary_top_k,
        )
        starts = starts[:, 0]
        ends = ends[:, 0]
        starts_valid = starts_valid[:, 0]
        ends_valid = ends_valid[:, 0]
        ks, ke = starts.shape[1], ends.shape[1]
        pair_s = starts.unsqueeze(-1).expand(batch, ks, ke).reshape(batch, -1)
        pair_e = ends.unsqueeze(1).expand(batch, ks, ke).reshape(batch, -1)
        pair_valid = (
            starts_valid.unsqueeze(-1) & ends_valid.unsqueeze(1) & (ends.unsqueeze(1) > starts.unsqueeze(-1))
        ).reshape(batch, -1)

        start_all = self.start_projection(boundary_states)
        end_all = self.end_projection(boundary_states)
        selected_start = gather_rows(start_all, pair_s)
        selected_end = gather_rows(end_all, pair_e)
        compat = (selected_start * selected_end).sum(-1) / math.sqrt(dim)
        union_pair_score = (
            compat
            + union_start.gather(1, pair_s.clamp(0, n_boundaries - 1))
            + union_end.gather(1, pair_e.clamp(0, n_boundaries - 1))
        )

        quota = min(self.min_pool_per_query, pair_s.shape[-1])
        quota_keys = pair_s.new_zeros((batch, 0))
        quota_scores = union_pair_score.new_zeros((batch, 0))
        quota_valid = pair_valid.new_zeros((batch, 0))
        if quota:
            s_idx = pair_s.clamp(0, start_logits.shape[2] - 1).unsqueeze(1).expand(batch, queries, -1)
            e_idx = pair_e.clamp(0, end_logits.shape[2] - 1).unsqueeze(1).expand(batch, queries, -1)
            per_query = start_logits.gather(2, s_idx) + end_logits.gather(2, e_idx) + compat.unsqueeze(1)
            per_query_valid = pair_valid.unsqueeze(1) & query_mask.unsqueeze(-1)
            ranked = torch.argsort(
                per_query.masked_fill(~per_query_valid, MASK_LOGIT),
                dim=-1,
                descending=True,
                stable=True,
            )[..., :quota]
            quota_s = s_idx.gather(-1, ranked)
            quota_e = e_idx.gather(-1, ranked)
            quota_valid = per_query_valid.gather(-1, ranked).reshape(batch, -1)
            quota_keys = (quota_s * n_boundaries + quota_e).reshape(batch, -1)
            rank_bonus = torch.arange(
                quota,
                0,
                -1,
                device=boundary_states.device,
                dtype=union_pair_score.dtype,
            )
            quota_scores = (
                union_pair_score.new_full((batch, queries, quota), -MASK_LOGIT * 0.5) + rank_bonus.view(1, 1, quota)
            ).reshape(batch, -1)

        global_keys = pair_s * n_boundaries + pair_e
        all_keys = torch.cat((quota_keys, global_keys), -1)
        all_scores = torch.cat((quota_scores, union_pair_score.detach()), -1)
        all_valid = torch.cat((quota_valid, pair_valid), -1)
        diagnostic_keys = diagnostic_valid = None
        if return_stats:
            with torch.no_grad():
                diagnostic_keys, diagnostic_valid = _deduplicate_pool(
                    all_keys, all_scores, all_valid, self.pool_size, n_boundaries
                )

        if gold_pairs is not None and gold_mask is not None:
            gvalid = gold_mask & query_mask.unsqueeze(-1)
            oob = (
                (gold_pairs[..., 0] >= n_boundaries) | (gold_pairs[..., 1] >= n_boundaries) | (gold_pairs < 0).any(-1)
            )
            gvalid = gvalid & ~oob
            if gold_injection_prob <= 0.0:
                gvalid = torch.zeros_like(gvalid)
            elif gold_injection_prob < 1.0:
                sampled = torch.rand(gvalid.shape, device=gvalid.device, generator=generator)
                gvalid = gvalid & (sampled < gold_injection_prob)
            safe_gold = gold_pairs.clamp(0, n_boundaries - 1)
            gkeys = safe_gold[..., 0] * n_boundaries + safe_gold[..., 1]
            all_keys = torch.cat((all_keys, gkeys.reshape(batch, -1)), -1)
            all_valid = torch.cat((all_valid, gvalid.reshape(batch, -1)), -1)
            gold_priority = union_pair_score.new_full((batch, gkeys.shape[1] * gkeys.shape[2]), -MASK_LOGIT)
            all_scores = torch.cat((all_scores, gold_priority), -1)

        with torch.no_grad():
            selected_keys, selected_valid = _deduplicate_pool(
                all_keys, all_scores, all_valid, self.pool_size, n_boundaries
            )
        selected_keys = torch.where(selected_valid, selected_keys, torch.zeros_like(selected_keys))
        selected_s = torch.div(selected_keys, n_boundaries, rounding_mode="floor")
        selected_e = selected_keys - selected_s * n_boundaries
        indices = torch.stack((selected_s, selected_e), -1)
        indices = torch.where(selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices))

        gs = gather_rows(start_all, selected_s)
        ge = gather_rows(end_all, selected_e)
        selected_compat = (gs * ge).sum(-1) / math.sqrt(dim)
        selected_score = (
            selected_compat
            + union_start.gather(1, selected_s.clamp(0, n_boundaries - 1))
            + union_end.gather(1, selected_e.clamp(0, n_boundaries - 1))
        )
        selected_score = selected_score.masked_fill(~selected_valid, MASK_LOGIT)
        selected_compat = torch.where(selected_valid, selected_compat, torch.zeros_like(selected_compat))

        selected_gold = None
        if gold_pairs is not None and gold_mask is not None:
            selected_gold = (indices.unsqueeze(2).unsqueeze(3) == gold_pairs.unsqueeze(1)).all(-1)
            selected_gold = (selected_gold & gold_mask.unsqueeze(1) & selected_valid.unsqueeze(-1).unsqueeze(-1)).any(
                -1
            )

        stats = None
        if return_stats:
            gold_hit = gold_total = start_hit = end_hit = boundary_total = None
            if gold_pairs is not None and gold_mask is not None:
                diagnostic_gold = gold_mask & query_mask.unsqueeze(-1)
                gold_keys = gold_pairs[..., 0] * n_boundaries + gold_pairs[..., 1]
                pair_hit = (
                    (gold_keys.unsqueeze(-1) == diagnostic_keys.unsqueeze(1).unsqueeze(1))
                    & diagnostic_valid.unsqueeze(1).unsqueeze(1)
                ).any(-1) & diagnostic_gold
                gold_hit = pair_hit.sum()
                gold_total = diagnostic_gold.sum()
                start_hit = (
                    (gold_pairs[..., 0].unsqueeze(-1) == starts.unsqueeze(1).unsqueeze(1))
                    & starts_valid.unsqueeze(1).unsqueeze(1)
                ).any(-1) & diagnostic_gold
                end_hit = (
                    (gold_pairs[..., 1].unsqueeze(-1) == ends.unsqueeze(1).unsqueeze(1))
                    & ends_valid.unsqueeze(1).unsqueeze(1)
                ).any(-1) & diagnostic_gold
                start_hit, end_hit = start_hit.sum(), end_hit.sum()
                boundary_total = diagnostic_gold.sum()
            stats = ProposalStats(
                boundary_score_elements=batch * queries * n_boundaries * 2,
                conditional_pair_score_elements=batch * ks * ke,
                max_materialized_pair_elements=batch * ks * ke,
                retained_candidate_count=selected_valid.sum(),
                gold_hit_without_injection=gold_hit,
                gold_total=gold_total,
                start_hit=start_hit,
                end_hit=end_hit,
                boundary_total=boundary_total,
                unique_candidates=selected_valid.sum(),
            )
        return PooledCandidates(
            indices=indices,
            mask=selected_valid,
            proposal_logits=selected_score,
            gold_mask=selected_gold,
            compat_logits=selected_compat,
            stats=stats,
        )


def classify_overlap_buckets(indices: torch.Tensor) -> torch.Tensor:
    """Classify every ordered span pair into one of eight geometry buckets.

    Args:
        indices: Half-open spans `[B, C, 2]`.

    Returns:
        Bucket ids `[B, C, C]`.
    """
    s1 = indices[..., :, None, 0]
    e1 = indices[..., :, None, 1]
    s2 = indices[..., None, :, 0]
    e2 = indices[..., None, :, 1]
    out = torch.full_like(s1 + s2, OVERLAP_CROSSING)
    disjoint_left = e1 <= s2
    disjoint_right = e2 <= s1
    identical = (s1 == s2) & (e1 == e2)
    same_start = (s1 == s2) & (e1 != e2)
    same_end = (e1 == e2) & (s1 != s2)
    nested_inside = (s1 > s2) & (e1 < e2)
    nested_outside = (s1 < s2) & (e1 > e2)
    out = torch.where(disjoint_left, OVERLAP_DISJOINT_LEFT, out)
    out = torch.where(disjoint_right, OVERLAP_DISJOINT_RIGHT, out)
    out = torch.where(nested_inside, OVERLAP_NESTED_INSIDE, out)
    out = torch.where(nested_outside, OVERLAP_NESTED_OUTSIDE, out)
    out = torch.where(same_start, OVERLAP_SAME_START, out)
    out = torch.where(same_end, OVERLAP_SAME_END, out)
    return torch.where(identical, OVERLAP_IDENTICAL, out)


class OverlapBiasedCandidateAttention(nn.Module):
    """Candidate self-attention biased by span overlap geometry.

    Args:
        dim: Candidate feature width.
        heads: Attention heads.
        dropout: Residual dropout probability.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(NUM_OVERLAP_BUCKETS, heads))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attend across pool spans and clear padding rows.

        Args:
            states: Candidate states `[B, C, D]`.
            indices: Half-open spans `[B, C, 2]`.
            mask: Real candidates `[B, C]`.

        Returns:
            Updated candidate states.
        """
        batch, count, dim = states.shape
        qkv = self.qkv(self.norm1(states)).reshape(batch, count, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        buckets = classify_overlap_buckets(indices)
        bias = self.relative_bias[buckets].permute(0, 3, 1, 2)
        logits = logits + bias
        logits = logits.masked_fill(~mask[:, None, None, :], MASK_LOGIT)
        weights = torch.softmax(logits, -1)
        weights = weights * mask[:, None, :, None].to(weights.dtype)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, count, dim)
        states = states + self.dropout(self.output(attended))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states * mask.unsqueeze(-1).to(states.dtype)


class EvidenceConditionedQueryAttention(nn.Module):
    """Update queries after adding a pool-evidence residual.

    Args:
        query_dim: Incoming query width.
        model_dim: Attention width.
        heads: Attention heads.
        dropout: Feed-forward and attention dropout.
    """

    def __init__(self, query_dim: int, model_dim: int, heads: int, dropout: float):
        super().__init__()
        self.query_in = nn.Linear(query_dim, model_dim)
        self.evidence_in = nn.Linear(model_dim, model_dim)
        self.attention = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(
        self,
        query_states: torch.Tensor,
        evidence: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run query self-attention conditioned on pooled evidence.

        Args:
            query_states: Projected queries `[B, Q, D]`.
            evidence: Evidence summarized from the pool `[B, Q, D]`.
            query_mask: Valid queries `[B, Q]`.

        Returns:
            Updated queries, with padding rows cleared.
        """
        states = self.query_in(query_states) + self.evidence_in(evidence)
        safe_query_mask = query_mask.clone()
        safe_query_mask[:, 0] |= ~query_mask.any(-1)
        attended, _ = self.attention(
            self.norm1(states),
            self.norm1(states),
            self.norm1(states),
            key_padding_mask=~safe_query_mask,
            need_weights=False,
        )
        states = states + attended
        states = states + self.ffn(self.norm2(states))
        return states * query_mask.unsqueeze(-1).to(states.dtype)


class SharedPoolScorer(nn.Module):
    """Score every query against one shared candidate pool.

    Args:
        boundary_dim: Boundary state width.
        query_dim: Query state width.
        pair_dim: Candidate feature width.
        dropout: Dropout probability.
        candidate_attention_layers: Overlap-biased candidate blocks.
        candidate_attention_heads: Heads in candidate and query attention.
        query_attention_layers: Evidence-conditioned query blocks.
        enable_span_content: Add pooled token content.
        content_dim: Content projection width.
        content_soft_max_pool: Also pool a smooth maximum.
        text_hidden_size: Token hidden size read by the content pooler.
    """

    def __init__(
        self,
        boundary_dim: int,
        query_dim: int,
        pair_dim: int,
        *,
        dropout: float,
        candidate_attention_layers: int,
        candidate_attention_heads: int,
        query_attention_layers: int,
        enable_span_content: bool,
        content_dim: int,
        content_soft_max_pool: bool,
        text_hidden_size: int,
    ) -> None:
        super().__init__()
        self.start_projection = nn.Linear(boundary_dim, pair_dim)
        self.end_projection = nn.Linear(boundary_dim, pair_dim)
        self.length_projection = nn.Linear(3, pair_dim)
        self.prior_projection = nn.Linear(1, pair_dim)
        self.content_pooler = (
            SpanContentPooler(text_hidden_size, content_dim, dropout, content_soft_max_pool)
            if enable_span_content
            else None
        )
        content_output = self.content_pooler.output_dim if self.content_pooler else 0
        self.content_projection = nn.Linear(content_output, pair_dim) if content_output else None
        self.candidate_norm = nn.LayerNorm(pair_dim)
        self.candidate_layers = nn.ModuleList(
            [
                OverlapBiasedCandidateAttention(pair_dim, candidate_attention_heads, dropout)
                for _ in range(candidate_attention_layers)
            ]
        )
        self.query_projection = nn.Linear(query_dim, pair_dim)
        self.query_layers = nn.ModuleList(
            [
                EvidenceConditionedQueryAttention(
                    pair_dim,
                    pair_dim,
                    candidate_attention_heads,
                    dropout,
                )
                for _ in range(query_attention_layers)
            ]
        )
        self.film = nn.Linear(pair_dim, 2 * pair_dim)
        self.film_output = nn.Sequential(
            nn.Linear(pair_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        boundary_states: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        pooled: PooledCandidates,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        inside_prefix: torch.Tensor | None,
        text_lengths: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        inside_prefix_mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query scores in candidate-major order plus candidate states.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.
            query_mask: Valid queries `[B, Q]`.
            pooled: Document pool.
            start_logits: Start marginals `[B, Q, N]`.
            end_logits: End marginals `[B, Q, N]`.
            inside_prefix: Centered inside prefix `[B, Q, L + 1]`, or None.
            text_lengths: Token counts `[B]`.
            text_states: Token states `[B, L, H]`.
            text_mask: Valid tokens `[B, L]`.
            inside_prefix_mean: Mean restored onto inside intervals.

        Returns:
            Scores `[B, C, Q]` and contextual candidate states `[B, C, P]`.
        """
        starts, ends = pooled.indices[..., 0], pooled.indices[..., 1]
        start_rep = gather_rows(self.start_projection(boundary_states), starts)
        end_rep = gather_rows(self.end_projection(boundary_states), ends)
        feature_dtype = start_rep.dtype
        length = (ends - starts).clamp_min(1).to(feature_dtype)
        text_length = text_lengths[:, None].to(feature_dtype).clamp_min(1)
        length_features = torch.stack((torch.log1p(length), length / text_length, torch.rsqrt(length)), -1)
        prior = pooled.compat_logits if pooled.compat_logits is not None else pooled.proposal_logits
        if prior is None:
            prior = start_rep.new_zeros(starts.shape)
        candidate = (
            start_rep
            + end_rep
            + self.length_projection(length_features)
            + self.prior_projection(prior.unsqueeze(-1).to(feature_dtype))
        )
        if self.content_pooler is not None:
            mean_prefix, lse_prefix = self.content_pooler.build_prefix(text_states, text_mask)
            content = self.content_pooler.pool(mean_prefix, lse_prefix, starts, ends, candidate.dtype)
            candidate = candidate + self.content_projection(content)
        candidate = self.candidate_norm(candidate)
        candidate = candidate * pooled.mask.unsqueeze(-1).to(candidate.dtype)
        for layer in self.candidate_layers:
            candidate = layer(candidate, pooled.indices, pooled.mask)

        query = self.query_projection(query_states)
        if self.query_layers:
            for layer in self.query_layers:
                preliminary = torch.einsum("bcd,bqd->bcq", candidate, query)
                evidence_weights = torch.softmax(
                    preliminary.masked_fill(~pooled.mask.unsqueeze(-1), MASK_LOGIT),
                    dim=1,
                )
                evidence = torch.einsum("bcq,bcd->bqd", evidence_weights, candidate)
                query = layer(query, evidence, query_mask)

        score = torch.einsum("bcd,bqd->bcq", candidate, query) / math.sqrt(candidate.shape[-1])
        gamma, beta = self.film(query).chunk(2, -1)
        conditioned = candidate.unsqueeze(2) * (1.0 + gamma.unsqueeze(1))
        conditioned = conditioned + beta.unsqueeze(1)
        score = score + self.film_output(conditioned).squeeze(-1)

        count = starts.shape[1]
        s_idx = starts.clamp(0, start_logits.shape[2] - 1).unsqueeze(1).expand(-1, query_states.shape[1], count)
        e_idx = ends.clamp(0, end_logits.shape[2] - 1).unsqueeze(1).expand_as(s_idx)
        score = score + start_logits.gather(2, s_idx).transpose(1, 2)
        score = score + end_logits.gather(2, e_idx).transpose(1, 2)
        if inside_prefix is not None:
            interval = inside_prefix.gather(2, e_idx.clamp(max=inside_prefix.shape[2] - 1)) - inside_prefix.gather(
                2, s_idx.clamp(max=inside_prefix.shape[2] - 1)
            )
            if inside_prefix_mean is not None:
                interval = interval + inside_prefix_mean * (e_idx - s_idx).to(interval.dtype)
            score = score + (interval / torch.sqrt((e_idx - s_idx).clamp_min(1).float())).transpose(1, 2).to(
                score.dtype
            )
        score = score.masked_fill(~pooled.mask.unsqueeze(-1), MASK_LOGIT)
        score = score.masked_fill(~query_mask.unsqueeze(1), MASK_LOGIT)
        return score, candidate
