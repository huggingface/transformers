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

from dataclasses import dataclass, fields

import torch
from torch import nn

from .boundary_content import gather_rows
from .boundary_encoding import BoundaryEncoder, BoundaryQueryHead
from .boundary_pool import DocumentCandidatePool, SharedPoolScorer
from .boundary_proposal import (
    BoundaryProposals,
    CandidateTensorBatch,
    ProposalSettings,
    SparseBoundaryProposer,
)
from .boundary_records import RecordHead
from .boundary_relations import SparseRelationScorer, TypedRelationPairGenerator
from .boundary_scoring import SparseBoundaryPairScorer, gather_boundary_states


@dataclass
class BoundarySettings:
    """Inference settings read by the boundary head.

    The parent model passes ``config.boundary_config``. Only fields that change
    parameters or inference math are stored here.

    Attributes:
        boundary_dim: Boundary state width.
        pair_dim: Pair-scorer width.
        boundary_refinement_layers: Residual blocks in the boundary encoder.
        boundary_ffn_multiplier: SwiGLU hidden multiplier.
        start_top_k: Starts kept before conditional end scoring.
        end_top_k: Ends kept before conditional start scoring.
        ends_per_start: Ends retained for each start.
        starts_per_end: Starts retained for each end.
        candidate_budget: Candidates kept at evaluation.
        training_candidate_budget: Candidates kept while the module is training.
        max_gold_per_query: Gold spans reserved by the proposal settings.
        end_block_size: Streaming proposal block width.
        bidirectional_proposals: Also propose starts from selected ends.
        use_inside_evidence: Add inside-prefix evidence to pair scores.
        dropout: Dropout probability.
        export_mode: `"auto"`, `"streaming"`, or `"vectorized"`.
        vectorized_pair_elements: Auto-mode cutoff for a single proposal block.
        enable_span_content: Pool token content into pair scores.
        content_dim: Content projection width.
        content_soft_max_pool: Also pool a smooth maximum.
        enable_rotary_endpoints: Rotate proposal and reranker endpoints.
        rotary_base: Rotary frequency base.
        boundary_attention_layers: Boundary self-attention blocks.
        boundary_attention_heads: Heads in those blocks.
        boundary_attention_window: Local window, or 0 for full attention.
        query_conditioned_inside_weight: Predict the inside coefficient.
        endpoint_difference_features: Add an endpoint-difference feature.
        reranker_endpoint_compat: Add multi-head endpoint compatibility.
        multihead_pair_compat_heads: Heads mixed into the compatibility logit.
        boundary_top_k_alpha: Length-adaptive top-k slope.
        boundary_top_k_max: Cap for the adaptive top-k.
        boundary_top_k_bucket: Rounding bucket for the adaptive top-k.
        candidate_pool: `"per_query"` or `"shared"`.
        pool_boundary_top_k: Endpoints kept in the shared pool.
        pool_size: Spans kept in the shared pool.
        min_pool_per_query: Spans reserved for each query.
        candidate_attention_layers: Shared-pool candidate attention blocks.
        candidate_attention_heads: Heads in shared-pool attention.
        query_attention_layers: Shared-pool query attention blocks.
        enable_abstention: Build the null-query projection.
        enable_count_head: Build the count log-rate projection.
    """

    boundary_dim: int = 128
    pair_dim: int = 128
    boundary_refinement_layers: int = 1
    boundary_ffn_multiplier: float = 2.0
    start_top_k: int = 16
    end_top_k: int = 16
    ends_per_start: int = 8
    starts_per_end: int = 8
    candidate_budget: int = 128
    training_candidate_budget: int = 160
    max_gold_per_query: int = 32
    end_block_size: int = 256
    bidirectional_proposals: bool = True
    use_inside_evidence: bool = True
    dropout: float = 0.1
    export_mode: str = "auto"
    vectorized_pair_elements: int = 16_777_216
    enable_span_content: bool = False
    content_dim: int = 64
    content_soft_max_pool: bool = False
    enable_rotary_endpoints: bool = False
    rotary_base: float = 10000.0
    boundary_attention_layers: int = 0
    boundary_attention_heads: int = 4
    boundary_attention_window: int = 0
    query_conditioned_inside_weight: bool = False
    endpoint_difference_features: bool = False
    reranker_endpoint_compat: bool = True
    multihead_pair_compat_heads: int = 8
    boundary_top_k_alpha: float = 0.0
    boundary_top_k_max: int = 128
    boundary_top_k_bucket: int = 8
    candidate_pool: str = "per_query"
    pool_boundary_top_k: int = 64
    pool_size: int = 384
    min_pool_per_query: int = 8
    candidate_attention_layers: int = 2
    candidate_attention_heads: int = 4
    query_attention_layers: int = 1
    enable_abstention: bool = True
    enable_count_head: bool = True


@dataclass
class BoundaryHeadOutput:
    """Inference outputs of `BoundaryHead`.

    Attributes:
        start_logits: Start marginals `[B, Q, L + 1]`.
        end_logits: End marginals `[B, Q, L + 1]`.
        inside_logits: Inside marginals `[B, Q, L]`.
        candidates: Sparse candidates when requested.
        null_logits: Abstention logits `[B, Q]`.
        count_log_rates: Count-head log-rates `[B, Q]`.
        loss: Unused during inference.
        total_loss: Unused during inference.
        losses: Unused during inference.
        metrics: Unused during inference.
        batch_size: Batch size.
    """

    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None
    inside_logits: torch.Tensor | None = None
    candidates: CandidateTensorBatch | None = None
    null_logits: torch.Tensor | None = None
    count_log_rates: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    total_loss: torch.Tensor | None = None
    losses: dict | None = None
    metrics: dict | None = None
    batch_size: int = 0


def _coerce_settings(settings):
    """Accept a dataclass, namespace, or dict of boundary settings."""
    if isinstance(settings, dict):
        names = {item.name for item in fields(BoundarySettings)}
        return BoundarySettings(**{key: settings[key] for key in names if key in settings})
    return settings


def _proposal_settings(settings) -> ProposalSettings:
    """Copy the proposal fields read by `SparseBoundaryProposer`."""
    return ProposalSettings(
        start_top_k=settings.start_top_k,
        end_top_k=settings.end_top_k,
        ends_per_start=settings.ends_per_start,
        starts_per_end=settings.starts_per_end,
        candidate_budget=settings.candidate_budget,
        training_candidate_budget=settings.training_candidate_budget,
        max_gold_per_query=settings.max_gold_per_query,
        end_block_size=settings.end_block_size,
        bidirectional=settings.bidirectional_proposals,
        export_mode=settings.export_mode,
        vectorized_pair_elements=settings.vectorized_pair_elements,
        enable_rotary_endpoints=settings.enable_rotary_endpoints,
        rotary_base=settings.rotary_base,
        boundary_top_k_alpha=settings.boundary_top_k_alpha,
        boundary_top_k_max=settings.boundary_top_k_max,
        boundary_top_k_bucket=settings.boundary_top_k_bucket,
    )


class BoundaryHead(nn.Module):
    """Encode boundaries, propose spans, and rerank them.

    Args:
        hidden_size: Token hidden size.
        settings: Boundary configuration namespace or dict.
        query_dim: Query width. Defaults to `hidden_size`.
        loss_weights: Accepted for caller compatibility and unused here.
        hard_negatives_per_positive: Accepted for caller compatibility.
        minimum_hard_negatives: Accepted for caller compatibility.
        build_candidate_states: Add the record-head candidate encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        settings,
        query_dim: int | None = None,
        loss_weights: dict | None = None,
        hard_negatives_per_positive: int | None = None,
        minimum_hard_negatives: int | None = None,
        build_candidate_states: bool = False,
    ):
        super().__init__()
        settings = _coerce_settings(settings)
        self.hidden_size = hidden_size
        self.settings = settings
        self.query_dim = query_dim if query_dim is not None else hidden_size
        self.loss_weights = dict(loss_weights or {})
        self.hard_negatives_per_positive = (
            hard_negatives_per_positive
            if hard_negatives_per_positive is not None
            else getattr(settings, "hard_negatives_per_positive", 5)
        )
        self.minimum_hard_negatives = (
            minimum_hard_negatives
            if minimum_hard_negatives is not None
            else getattr(settings, "minimum_hard_negatives", 8)
        )
        self.collect_diagnostics = False
        self._gold_injection_prob = 1.0

        dim = settings.boundary_dim
        self.candidate_encoder = nn.Linear(2 * dim, hidden_size) if build_candidate_states else None
        self.boundary_encoder = BoundaryEncoder(
            hidden_size,
            dim,
            settings.dropout,
            settings.boundary_refinement_layers,
            settings.boundary_ffn_multiplier,
            settings.boundary_attention_layers,
            settings.boundary_attention_heads,
            settings.boundary_attention_window,
        )
        self.boundary_query_head = BoundaryQueryHead(hidden_size, dim, self.query_dim, settings.dropout)
        self.boundary_proposer = SparseBoundaryProposer(dim, self.query_dim, _proposal_settings(settings))
        self.pair_scorer = SparseBoundaryPairScorer(
            dim,
            self.query_dim,
            settings.pair_dim,
            use_inside_evidence=settings.use_inside_evidence,
            dropout=settings.dropout,
            enable_span_content=settings.enable_span_content,
            content_dim=settings.content_dim,
            content_soft_max_pool=settings.content_soft_max_pool,
            enable_rotary_endpoints=settings.enable_rotary_endpoints,
            rotary_base=settings.rotary_base,
            query_conditioned_inside_weight=settings.query_conditioned_inside_weight,
            endpoint_difference_features=settings.endpoint_difference_features,
            reranker_endpoint_compat=settings.reranker_endpoint_compat,
            multihead_pair_compat_heads=settings.multihead_pair_compat_heads,
            content_hidden_size=hidden_size,
        )
        self.use_inside_evidence = settings.use_inside_evidence
        # Optional heads are zeroed without moving the caller RNG stream.
        auxiliary_rng_state = torch.random.get_rng_state()
        self.null_projection = nn.Linear(self.query_dim, 1) if settings.enable_abstention else None
        if self.null_projection is not None:
            nn.init.zeros_(self.null_projection.weight)
            nn.init.zeros_(self.null_projection.bias)
        self.count_head = nn.Linear(self.query_dim, 1) if settings.enable_count_head else None
        if self.count_head is not None:
            nn.init.zeros_(self.count_head.weight)
            nn.init.zeros_(self.count_head.bias)
        torch.random.set_rng_state(auxiliary_rng_state)
        shared_rng_state = torch.random.get_rng_state()
        self.shared_pool_builder = DocumentCandidatePool(
            dim,
            pool_boundary_top_k=settings.pool_boundary_top_k,
            pool_size=settings.pool_size,
            min_pool_per_query=settings.min_pool_per_query,
        )
        self.shared_pool_scorer = SharedPoolScorer(
            dim,
            self.query_dim,
            settings.pair_dim,
            dropout=settings.dropout,
            candidate_attention_layers=settings.candidate_attention_layers,
            candidate_attention_heads=settings.candidate_attention_heads,
            query_attention_layers=settings.query_attention_layers,
            enable_span_content=settings.enable_span_content,
            content_dim=settings.content_dim,
            content_soft_max_pool=settings.content_soft_max_pool,
            text_hidden_size=hidden_size,
        )
        torch.random.set_rng_state(shared_rng_state)

    def set_gold_injection_prob(self, value: float) -> None:
        """Set the training gold-injection probability.

        Args:
            value: Probability in `[0, 1]`.
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"gold_injection_prob must be in [0, 1], got {value}")
        self._gold_injection_prob = float(value)

    def score_explicit_spans(
        self,
        token_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score caller-provided half-open spans for every query.

        Args:
            token_states: Token states `[B, L, H]`.
            text_mask: Valid tokens `[B, L]`.
            query_states: Query states `[B, Q, H]`.
            query_mask: Valid queries `[B, Q]`.
            indices: Spans `[B, Q, C, 2]`.
            valid_mask: Optional extra mask `[B, Q, C]`.

        Returns:
            Pair logits `[B, Q, C]`.
        """
        if indices.dim() != 4 or indices.shape[-1] != 2:
            raise ValueError(f"indices must be [B, Q, C, 2], got {tuple(indices.shape)}")
        batch, queries, _, _ = indices.shape
        if batch != token_states.shape[0] or batch != query_states.shape[0] or queries != query_states.shape[1]:
            raise ValueError("explicit span batch/query dimensions do not match states")

        text_lengths = text_mask.sum(dim=1).long()
        starts = indices[..., 0]
        ends = indices[..., 1]
        legal = (starts >= 0) & (ends > starts) & (ends <= text_lengths.view(batch, 1, 1)) & query_mask.unsqueeze(-1)
        if valid_mask is not None:
            if valid_mask.shape != indices.shape[:-1]:
                raise ValueError(
                    "valid_mask must match indices [B, Q, C], got "
                    f"{tuple(valid_mask.shape)} and {tuple(indices.shape)}"
                )
            legal = legal & valid_mask

        encoding = self.boundary_encoder(token_states, text_mask)
        marginals = self.boundary_query_head(
            encoding.states,
            encoding.mask,
            token_states,
            text_mask,
            query_states,
            query_mask,
        )
        compatibility = self.boundary_proposer.score_explicit_pairs(encoding.states, query_states, indices, legal)
        proposals = BoundaryProposals(
            indices=indices,
            logits=None,
            valid_mask=legal,
            compat_logits=compatibility,
        )
        inside_prefix = marginals.inside_prefix if self.use_inside_evidence else None
        return self.pair_scorer(
            encoding.states,
            query_states,
            proposals,
            marginals.start_logits,
            marginals.end_logits,
            inside_prefix,
            text_lengths,
            token_states,
            text_mask,
            inside_prefix_mean=marginals.inside_prefix_mean,
        )

    def forward(
        self,
        token_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        targets=None,
        *,
        return_candidates: bool = True,
        gold_injection_prob: float | None = None,
        collect_diagnostics: bool | None = None,
    ) -> BoundaryHeadOutput:
        """Propose and rerank boundary spans.

        Args:
            token_states: Token states `[B, L, H]`.
            text_mask: Valid tokens `[B, L]`.
            query_states: Query states `[B, Q, H]`.
            query_mask: Valid queries `[B, Q]`.
            targets: Optional targets with `mention_pairs` and `mention_mask`.
                Gold is injected only while this module is training.
            return_candidates: Attach the sparse candidate batch.
            gold_injection_prob: Override for the stored injection probability.
            collect_diagnostics: Request proposal statistics.

        Returns:
            Marginal logits and, when requested, scored candidates.
        """
        batch = token_states.shape[0]
        text_lengths = text_mask.sum(dim=1).long()
        encoding = self.boundary_encoder(token_states, text_mask)
        marginals = self.boundary_query_head(
            encoding.states,
            encoding.mask,
            token_states,
            text_mask,
            query_states,
            query_mask,
        )
        gold_pairs = gold_mask = None
        if targets is not None:
            gold_pairs = targets.mention_pairs
            gold_mask = targets.mention_mask
        diagnostics = self.collect_diagnostics if collect_diagnostics is None else collect_diagnostics
        injection = self._gold_injection_prob if gold_injection_prob is None else gold_injection_prob
        inside_prefix = marginals.inside_prefix if self.use_inside_evidence else None
        pooled = None
        pooled_candidate_states = None
        pooled_logits = None
        if self.settings.candidate_pool == "shared":
            pooled = self.shared_pool_builder(
                encoding.states,
                encoding.mask,
                query_mask,
                marginals.start_logits,
                marginals.end_logits,
                gold_pairs=gold_pairs,
                gold_mask=gold_mask,
                gold_injection_prob=injection if self.training else 0.0,
                return_stats=diagnostics,
            )
            pooled_logits, _ = self.shared_pool_scorer(
                encoding.states,
                query_states,
                query_mask,
                pooled,
                marginals.start_logits,
                marginals.end_logits,
                inside_prefix,
                text_lengths,
                token_states,
                text_mask,
                inside_prefix_mean=marginals.inside_prefix_mean,
            )
            queries = query_states.shape[1]
            count = pooled.indices.shape[1]
            proposals = BoundaryProposals(
                indices=pooled.indices.unsqueeze(1).expand(batch, queries, count, 2),
                logits=(
                    pooled.proposal_logits.unsqueeze(1).expand(batch, queries, count)
                    if pooled.proposal_logits is not None
                    else None
                ),
                valid_mask=pooled.mask.unsqueeze(1).expand(batch, queries, count),
                gold_mask=(pooled.gold_mask.transpose(1, 2) if pooled.gold_mask is not None else None),
                stats=pooled.stats,
                compat_logits=(
                    pooled.compat_logits.unsqueeze(1).expand(batch, queries, count)
                    if pooled.compat_logits is not None
                    else None
                ),
            )
            pair_logits = pooled_logits.transpose(1, 2)
            if self.candidate_encoder is not None:
                start_states = gather_rows(encoding.states, pooled.indices[..., 0])
                end_states = gather_rows(encoding.states, pooled.indices[..., 1])
                pooled_candidate_states = self.candidate_encoder(
                    torch.cat((start_states, end_states), -1)
                ).masked_fill(~pooled.mask.unsqueeze(-1), 0.0)
        else:
            scorer_start_states, scorer_end_states = self.pair_scorer.project_endpoints(encoding.states)
            proposals = self.boundary_proposer(
                encoding.states,
                encoding.mask,
                query_states,
                query_mask,
                marginals.start_logits,
                marginals.end_logits,
                gold_pairs=gold_pairs,
                gold_mask=gold_mask,
                return_stats=diagnostics,
                return_proposal_logits=self.training,
                gold_injection_prob=injection,
                scorer_start_states=scorer_start_states,
                scorer_end_states=scorer_end_states,
            )
            pair_logits = self.pair_scorer(
                encoding.states,
                query_states,
                proposals,
                marginals.start_logits,
                marginals.end_logits,
                inside_prefix,
                text_lengths,
                token_states,
                text_mask,
                inside_prefix_mean=marginals.inside_prefix_mean,
            )
        self._last_proposal_stats = proposals.stats
        null_logits = self.null_projection(query_states).squeeze(-1) if self.null_projection is not None else None
        count_log_rates = self.count_head(query_states).squeeze(-1) if self.count_head is not None else None

        candidates = None
        if return_candidates:
            candidate_states = pooled_candidate_states
            if self.candidate_encoder is not None and pooled is None:
                gathered_start = gather_boundary_states(encoding.states, proposals.indices[..., 0])
                gathered_end = gather_boundary_states(encoding.states, proposals.indices[..., 1])
                candidate_states = self.candidate_encoder(torch.cat([gathered_start, gathered_end], dim=-1))
                candidate_states = candidate_states.masked_fill(~proposals.valid_mask.unsqueeze(-1), 0.0)
            candidates = (
                pooled.to_candidate_batch(pooled_logits, query_mask, candidate_states)
                if pooled is not None
                else CandidateTensorBatch(
                    indices=proposals.indices,
                    proposal_logits=proposals.logits,
                    pair_logits=pair_logits,
                    valid_mask=proposals.valid_mask,
                    query_mask=query_mask,
                    candidate_states=candidate_states,
                )
            )
        return BoundaryHeadOutput(
            start_logits=marginals.start_logits,
            end_logits=marginals.end_logits,
            inside_logits=marginals.inside_logits,
            candidates=candidates,
            null_logits=null_logits,
            count_log_rates=count_log_rates,
            batch_size=batch,
        )


__all__ = [
    "BoundaryHead",
    "RecordHead",
    "SparseRelationScorer",
    "TypedRelationPairGenerator",
]
