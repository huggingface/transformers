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

import torch
from torch import nn

from .boundary_content import SpanContentPooler, gather_states
from .boundary_proposal import BoundaryProposals
from .boundary_rotary import RotaryBoundaryEmbedding


MASK_LOGIT = -1.0e4


def gather_boundary_states(boundary_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather boundary states at per-query candidate indices.

    Args:
        boundary_states: States `[B, N, D]`.
        indices: Positions `[B, Q, C]`.

    Returns:
        Gathered states `[B, Q, C, D]`.
    """
    return gather_states(boundary_states, indices)


def interval_prefix_score(
    prefix: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score `[start, end)` from a length-`L + 1` prefix.

    Args:
        prefix: Cumulative scores `[B, Q, L + 1]`.
        starts: Start indices `[B, Q, C]`.
        ends: End indices `[B, Q, C]`.
        mean: Optional per-query offset restored onto the interval.

    Returns:
        Interval scores `[B, Q, C]`.
    """
    max_idx = prefix.shape[2] - 1
    starts = starts.clamp(0, max_idx)
    ends = ends.clamp(0, max_idx)
    interval = torch.gather(prefix, 2, ends) - torch.gather(prefix, 2, starts)
    if mean is not None:
        interval = interval + mean * (ends - starts).to(interval.dtype)
    return interval


def continuous_length_features(
    starts: torch.Tensor,
    ends: torch.Tensor,
    text_lengths: torch.Tensor,
) -> torch.Tensor:
    """Build length features that do not depend on a maximum span width.

    Args:
        starts: Start indices `[B, Q, C]`.
        ends: End indices `[B, Q, C]`.
        text_lengths: Token counts `[B]`.

    Returns:
        Features `[B, Q, C, 3]`.
    """
    length = (ends - starts).clamp(min=1).float()
    batch = starts.shape[0]
    text_length = text_lengths.view(batch, 1, 1).float().clamp(min=1)
    return torch.stack(
        (torch.log1p(length), length / text_length, torch.rsqrt(length)),
        dim=-1,
    )


def mask_invalid_candidate_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Replace invalid candidate logits with a finite sentinel.

    Args:
        logits: Candidate scores.
        valid_mask: True for real candidates.

    Returns:
        Masked logits.
    """
    return logits.masked_fill(~valid_mask, MASK_LOGIT)


class SparseBoundaryPairScorer(nn.Module):
    """Rerank each proposed span with marginals, content, and length.

    The added prior is marginal-free endpoint compatibility, so start and end
    marginals enter the logit once.

    Args:
        boundary_dim: Boundary state width.
        query_dim: Query state width.
        pair_dim: Endpoint compatibility width.
        use_inside_evidence: Add normalized inside-prefix evidence.
        dropout: Dropout probability.
        enable_span_content: Pool token content into the score.
        content_dim: Content projection width.
        content_soft_max_pool: Also pool a smooth maximum.
        enable_rotary_endpoints: Rotate reranker endpoints.
        rotary_base: Rotary frequency base.
        query_conditioned_inside_weight: Predict the inside coefficient.
        endpoint_difference_features: Add a learned endpoint-difference term.
        reranker_endpoint_compat: Add multi-head endpoint compatibility.
        multihead_pair_compat_heads: Heads mixed into one compatibility logit.
        content_hidden_size: Token width used by the content pooler.
    """

    def __init__(
        self,
        boundary_dim: int,
        query_dim: int,
        pair_dim: int,
        use_inside_evidence: bool = True,
        dropout: float = 0.1,
        enable_span_content: bool = False,
        content_dim: int = 64,
        content_soft_max_pool: bool = False,
        enable_rotary_endpoints: bool = False,
        rotary_base: float = 10000.0,
        query_conditioned_inside_weight: bool = False,
        endpoint_difference_features: bool = False,
        reranker_endpoint_compat: bool = True,
        multihead_pair_compat_heads: int = 1,
        content_hidden_size: int | None = None,
    ):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.pair_dim = pair_dim
        self.use_inside_evidence = use_inside_evidence
        self.enable_span_content = enable_span_content
        self.enable_rotary_endpoints = enable_rotary_endpoints
        self.query_conditioned_inside_weight = query_conditioned_inside_weight
        self.endpoint_difference_features = endpoint_difference_features
        self.reranker_endpoint_compat = reranker_endpoint_compat
        if multihead_pair_compat_heads <= 0 or pair_dim % multihead_pair_compat_heads:
            raise ValueError(
                "pair_dim must be divisible by multihead_pair_compat_heads, got "
                f"{pair_dim} and {multihead_pair_compat_heads}"
            )
        self.multihead_pair_compat_heads = multihead_pair_compat_heads
        self.start_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.end_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.query_gate = nn.Linear(query_dim, pair_dim // 2 if enable_rotary_endpoints else pair_dim)
        self.length_query_projection = nn.Linear(query_dim, 3)
        self.inside_weight = (
            nn.Linear(query_dim, 1) if query_conditioned_inside_weight else nn.Parameter(torch.tensor(1.0))
        )
        self.endpoint_difference_projection = nn.Linear(2 * pair_dim, 1) if endpoint_difference_features else None
        self.rotary = RotaryBoundaryEmbedding(pair_dim, rotary_base) if enable_rotary_endpoints else None
        self.content_pooler = (
            SpanContentPooler(
                content_hidden_size if content_hidden_size is not None else query_dim,
                content_dim,
                dropout=dropout,
                use_soft_max_pool=content_soft_max_pool,
            )
            if enable_span_content
            else None
        )
        content_output_dim = self.content_pooler.output_dim if self.content_pooler is not None else 0
        self.content_query_projection = nn.Linear(query_dim, content_output_dim) if enable_span_content else None
        self.content_bias = nn.Linear(content_output_dim, 1) if enable_span_content else None
        self.dropout = nn.Dropout(dropout)
        # Keep this layer last so its init does not shift the caller RNG stream.
        rng_state = torch.random.get_rng_state()
        self.compat_mix = nn.Linear(multihead_pair_compat_heads, 1)
        nn.init.constant_(self.compat_mix.weight, 1.0 / multihead_pair_compat_heads)
        nn.init.zeros_(self.compat_mix.bias)
        torch.random.set_rng_state(rng_state)

    def project_endpoints(self, boundary_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project every boundary before candidate gathering.

        Args:
            boundary_states: Boundary states `[B, N, D]`.

        Returns:
            Start and end projections `[B, N, P]`.
        """
        start = self.start_endpoint_projection(boundary_states)
        end = self.end_endpoint_projection(boundary_states)
        if self.rotary is not None:
            n_boundaries = boundary_states.shape[1]
            positions = torch.arange(n_boundaries, device=boundary_states.device).view(1, n_boundaries)
            start = self.rotary(start, positions)
            end = self.rotary(end, positions)
        return start, end

    def forward(
        self,
        boundary_states: torch.Tensor,
        query_states: torch.Tensor,
        proposals: BoundaryProposals,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        inside_prefix: torch.Tensor | None,
        text_lengths: torch.Tensor,
        text_states: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        inside_prefix_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one logit per proposed candidate.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.
            proposals: Sparse candidates, including a compatibility prior.
            start_logits: Start marginals `[B, Q, N]`.
            end_logits: End marginals `[B, Q, N]`.
            inside_prefix: Centered inside prefix, or None to skip it.
            text_lengths: Token counts `[B]`.
            text_states: Token states required when content pooling is enabled.
            text_mask: Token mask required when content pooling is enabled.
            inside_prefix_mean: Mean restored onto inside intervals.

        Returns:
            Masked pair logits `[B, Q, C]`.
        """
        starts = proposals.indices[..., 0]
        ends = proposals.indices[..., 1]
        valid = proposals.valid_mask
        scale = 1.0 / math.sqrt(self.pair_dim)
        if proposals.score_start_states is not None and proposals.score_end_states is not None:
            s_proj = self.dropout(proposals.score_start_states)
            e_proj = self.dropout(proposals.score_end_states)
        else:
            start_all, end_all = self.project_endpoints(boundary_states)
            s_proj = self.dropout(gather_boundary_states(start_all, starts))
            e_proj = self.dropout(gather_boundary_states(end_all, ends))
        gate = torch.sigmoid(self.query_gate(query_states))
        if self.enable_rotary_endpoints:
            gate = gate.repeat_interleave(2, dim=-1)
        gate = gate.unsqueeze(2)
        if self.reranker_endpoint_compat:
            per_head = (
                (s_proj * gate * e_proj).reshape(*s_proj.shape[:-1], self.multihead_pair_compat_heads, -1).sum(-1)
            )
            compat = self.compat_mix(per_head).squeeze(-1) * scale
        else:
            compat = torch.zeros_like(starts, dtype=s_proj.dtype)
        if self.endpoint_difference_projection is not None:
            difference = torch.cat((s_proj - e_proj, (s_proj - e_proj).abs()), dim=-1)
            compat = compat + self.endpoint_difference_projection(difference).squeeze(-1)

        a = torch.gather(start_logits, 2, starts.clamp(0, start_logits.shape[2] - 1))
        bmarg = torch.gather(end_logits, 2, ends.clamp(0, end_logits.shape[2] - 1))
        prior_source = proposals.compat_logits if proposals.compat_logits is not None else proposals.logits
        if prior_source is None:
            raise ValueError("boundary proposals must provide compat_logits or logits")
        prior = torch.where(valid, prior_source, torch.zeros_like(prior_source))
        score = compat + a + bmarg + prior

        if self.content_pooler is not None:
            if text_states is None or text_mask is None:
                raise ValueError("span content scoring requires text_states and text_mask")
            mean_prefix, lse_prefix = self.content_pooler.build_prefix(text_states, text_mask)
            span_content = self.content_pooler.pool(mean_prefix, lse_prefix, starts, ends, score.dtype)
            coefficient = self.content_query_projection(query_states).unsqueeze(2)
            content_scale = 1.0 / math.sqrt(span_content.shape[-1])
            score = score + (span_content * coefficient).sum(-1) * content_scale
            score = score + self.content_bias(span_content).squeeze(-1)

        if self.use_inside_evidence and inside_prefix is not None:
            interval = interval_prefix_score(inside_prefix, starts, ends, inside_prefix_mean).to(score.dtype)
            denom = torch.sqrt((ends - starts).clamp(min=1).to(score.dtype))
            inside_weight = (
                self.inside_weight(query_states).squeeze(-1).unsqueeze(-1)
                if self.query_conditioned_inside_weight
                else self.inside_weight
            )
            score = score + inside_weight * (interval / denom)

        feats = continuous_length_features(starts, ends, text_lengths)
        length_coeff = self.length_query_projection(query_states).unsqueeze(2)
        score = score + (feats.to(length_coeff.dtype) * length_coeff).sum(-1)
        return mask_invalid_candidate_logits(score, valid)
