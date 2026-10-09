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

import torch
import torch.nn.functional as F

from ..models.gliner2.decoding_gliner2 import linear_sum_assignment


ENTITY_TASK_ID = 1
CLASSIFICATION_TASK_ID = 4
MAX_SPAN_COUNT = 19
MASK_LOGIT = -1.0e4


class TargetCapacityError(ValueError):
    """Raised when gold records exceed the instance hypotheses."""


def _require_finite(value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).all().item():
        raise ValueError("loss is not finite")
    return value


def supervises_count(task_id: int) -> bool:
    """Return whether this task contributes 20-way count cross-entropy."""
    return int(task_id) not in (ENTITY_TASK_ID, CLASSIFICATION_TASK_ID)


def clamp_gold_count(count) -> int:
    """Clamp a gold instance count onto the 20-way count head."""
    count = int(count.item() if isinstance(count, torch.Tensor) else count)
    if count < 0:
        raise ValueError(f"span count must be >= 0, got {count}")
    return min(count, MAX_SPAN_COUNT)


def invalid_span_mask(length: int, max_width: int, device) -> torch.Tensor:
    """Return True where `start + width` falls outside the word axis."""
    starts = torch.arange(length, device=device).unsqueeze(1)
    widths = torch.arange(max_width, device=device).unsqueeze(0)
    return (starts + widths >= length).reshape(-1)


def span_count_loss(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Summed 20-way count cross-entropy."""
    if logits.dim() != 2 or logits.shape[-1] != 20 or counts.shape != logits.shape[:1]:
        raise ValueError(f"count logits must be [N, 20] with N targets, got {tuple(logits.shape)}")
    return F.cross_entropy(logits, counts.to(device=logits.device, dtype=torch.long), reduction="sum")


def span_structure_loss(
    scores: torch.Tensor, structure, span_mask: torch.Tensor, *, training: bool, masking_rate: float = 0.5
) -> torch.Tensor:
    """Summed span BCE. Training drops each negative with `rand_like < rate`."""
    labels = _span_label_tensor(scores, structure)
    flat_mask = span_mask.reshape(-1).bool()
    if flat_mask.numel() != scores.shape[-2] * scores.shape[-1]:
        raise ValueError("span_mask does not match the span grid")
    if masking_rate > 0.0 and training:
        keep = (~((labels == 0) & (torch.rand_like(scores) < masking_rate))).to(scores.dtype)
    else:
        keep = torch.ones_like(scores)
    loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none") * keep
    return (loss.reshape(loss.shape[0], loss.shape[1], -1) * (~flat_mask).to(loss.dtype)).sum()


def _span_label_tensor(scores: torch.Tensor, structure) -> torch.Tensor:
    if torch.is_tensor(structure):
        if structure.shape != scores.shape:
            raise ValueError(f"span targets {tuple(structure.shape)} != scores {tuple(scores.shape)}")
        return structure.to(device=scores.device, dtype=scores.dtype)
    gold_count = clamp_gold_count(structure[0])
    if scores.shape[0] != gold_count or len(structure[1]) < gold_count:
        raise ValueError(f"span score count {scores.shape[0]} != gold count {gold_count}")
    labels = torch.zeros_like(scores)
    for index in range(gold_count):
        for field_index, span in enumerate(structure[1][index]):
            for sub in [span] if isinstance(span, tuple) else span or ():
                if sub is None:
                    continue
                start, width = int(sub[0]), int(sub[1]) - int(sub[0])
                if 0 <= start < labels.shape[2] and 0 <= width < labels.shape[3]:
                    labels[index, field_index, start, width] = 1
    return labels


def _aligned(predictions, targets):
    if torch.is_tensor(predictions) and torch.is_tensor(targets):
        yield predictions, targets
        return
    if predictions is None or torch.is_tensor(predictions) or len(predictions) != len(targets):
        raise ValueError("classification_targets must align with the model outputs")
    for prediction, target in zip(predictions, targets):
        yield from _aligned(prediction, target)


def _anchor_zero(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.sum() * 0.0
    for item in value:
        return _anchor_zero(item)
    raise ValueError("cannot build a zero loss without tensors")


def classification_bce(logits, targets, weight: float | None = None) -> torch.Tensor:
    """BCE over aligned classification groups.

    Args:
        logits: Classification logits, or nested lists aligned with `targets`.
        targets: Targets with the same nesting as `logits`.
        weight: When given, the BCE is averaged over labels and scaled by `weight`; otherwise it is summed.

    Returns:
        Scalar loss. A mean over no labels is a zero attached to `logits`.
    """
    total, count = None, 0
    for prediction, target in _aligned(logits, targets):
        if prediction.numel() == 0 and target.numel() == 0:
            continue
        if prediction.shape != target.shape:
            raise ValueError(f"classification_targets shape {tuple(target.shape)} != logits {tuple(prediction.shape)}")
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        term = F.binary_cross_entropy_with_logits(prediction, target, reduction="sum")
        total = term if total is None else total + term
        count += int(target.numel())
    if total is None:
        if weight is None:
            raise ValueError("classification_targets did not match any logits")
        return _require_finite(_anchor_zero(logits) * weight)
    return _require_finite(total if weight is None else weight * total / count)


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    negative_weight: float = 1.0,
    query_mask=None,
    reduction: str = "global",
) -> torch.Tensor:
    """Masked BCE reduced over kept positions.

    Args:
        logits: Unnormalized scores.
        targets: Labels aligned with `logits`.
        mask: Positions that contribute. Other positions are zeroed before the BCE.
        negative_weight: Weight of targets that are not positive.
        query_mask: Optional query filter. Applied by `"sum"` and `"per_query"`.
        reduction: `"sum"`, `"global"`, or `"per_query"`.
    """
    keep = mask.bool()
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    safe_targets = torch.where(keep, targets, torch.zeros_like(targets))
    elementwise = F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")
    if float(negative_weight) != 1.0:
        elementwise = elementwise * torch.where(
            targets > 0.5, torch.ones_like(targets), torch.full_like(targets, negative_weight)
        )
    return _reduce(elementwise, keep, query_mask, reduction)


def _reduce(elementwise: torch.Tensor, keep: torch.Tensor, query_mask, mode: str) -> torch.Tensor:
    keep_f = keep.to(elementwise.dtype)
    if mode == "sum":
        if query_mask is not None and keep_f.dim() >= 2:
            keep_f = keep_f * query_mask.unsqueeze(-1).to(keep_f.dtype)
        reduced = (elementwise * keep_f).sum()
    elif mode == "global":
        reduced = (elementwise * keep_f).sum() / keep_f.sum().clamp_min(1)
    elif mode == "per_query":
        per_query = (elementwise * keep_f).sum(-1) / keep_f.sum(-1).clamp_min(1)
        active = keep.any(-1) if query_mask is None else keep.any(-1) & query_mask
        active_f = active.to(per_query.dtype)
        reduced = (per_query * active_f).sum() / active_f.sum().clamp_min(1)
    else:
        raise ValueError(f"unknown reduction mode {mode!r}")
    return _require_finite(reduced)


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma_positive: float = 0.0,
    gamma_negative: float = 2.0,
    clip: float = 0.05,
    negative_weight: float = 1.0,
    query_mask=None,
    reduction: str = "global",
) -> torch.Tensor:
    """Asymmetric focal loss for sparse multi-label boundaries."""
    probability = torch.sigmoid(torch.where(valid_mask, logits, torch.zeros_like(logits)))
    negative_prob = (1 - probability + clip).clamp(max=1.0) if clip > 0 else 1 - probability
    positive = targets * torch.log(probability.clamp_min(1e-8)) * ((1 - probability) ** gamma_positive)
    negative = (
        (1 - targets) * torch.log(negative_prob.clamp_min(1e-8)) * (probability**gamma_negative) * negative_weight
    )
    return _reduce(-(positive + negative), valid_mask, query_mask, reduction)


def build_candidate_labels(indices, mask, gold_pairs, gold_mask, *, return_iou: bool = False):
    """Label candidates that exactly match a gold span, optionally with soft IoU."""
    pooled = indices.dim() == 3
    cand = indices[:, :, None, None] if pooled else indices.unsqueeze(3)
    gold = gold_pairs.unsqueeze(1 if pooled else 2)
    gold_keep = gold_mask.unsqueeze(1 if pooled else 2)
    weight = (mask.unsqueeze(-1) if pooled else mask).to(torch.float)
    labels = ((cand == gold).all(dim=-1) & gold_keep).any(-1).to(torch.float) * weight
    if not return_iou:
        return labels
    intersection = (torch.minimum(cand[..., 1], gold[..., 1]) - torch.maximum(cand[..., 0], gold[..., 0])).clamp_min(0)
    union = cand[..., 1] - cand[..., 0] + gold[..., 1] - gold[..., 0] - intersection
    iou = intersection.to(torch.float) / union.clamp_min(1).to(torch.float) * gold_keep.to(torch.float)
    return labels, iou.amax(-1) * weight


def select_hard_negative_candidates(
    pair_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    negatives_per_positive: int,
    minimum_negatives: int,
    keep_all_when_no_positive: bool = False,
) -> torch.Tensor:
    """Keep every positive and the highest-scoring negatives per query."""
    positive = (labels > 0.5) & valid_mask
    negative = (labels <= 0.5) & valid_mask
    n_positive = positive.sum(-1, keepdim=True)
    floor = torch.finfo(pair_logits.dtype).min
    order = torch.argsort(pair_logits.masked_fill(~negative, floor), dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    n_keep = (negatives_per_positive * n_positive).clamp_min(minimum_negatives)
    if keep_all_when_no_positive:
        n_keep = torch.where(n_positive == 0, torch.full_like(n_keep, pair_logits.shape[-1]), n_keep)
    return positive | (negative & (rank < n_keep))


def proposal_listwise_loss(proposal_logits, gold_mask, valid_mask, query_mask) -> torch.Tensor:
    """Rank gold candidates above the other valid proposals."""
    logits = proposal_logits.masked_fill(~valid_mask, MASK_LOGIT)
    all_lse = torch.logsumexp(logits, dim=-1)
    gold_lse = torch.logsumexp(logits.masked_fill(~gold_mask, MASK_LOGIT), dim=-1)
    has_gold = gold_mask.any(-1) & query_mask
    loss = torch.where(has_gold, all_lse - gold_lse, torch.zeros_like(all_lse))
    return _require_finite(loss.sum() / has_gold.to(loss.dtype).sum().clamp_min(1))


def marginal_pair_consistency_loss(pair_logits, indices, valid_mask, start_logits, end_logits, boundary_keep):
    """Match boundary marginals to candidate noisy-OR probabilities."""
    probabilities = torch.sigmoid(pair_logits) * valid_mask.to(pair_logits.dtype)
    eps = max(1e-6, torch.finfo(probabilities.dtype).eps)
    log_survival = torch.log1p(-probabilities.clamp(max=1.0 - eps))
    n_boundary = start_logits.shape[-1]
    result = pair_logits.new_zeros(())
    for side, marginal in ((0, start_logits), (1, end_logits)):
        index = indices[..., side].clamp(0, n_boundary - 1)
        total = torch.zeros(*start_logits.shape, dtype=log_survival.dtype, device=log_survival.device)
        total.scatter_add_(2, index, log_survival)
        count = torch.zeros_like(total).scatter_add_(2, index, valid_mask.to(total.dtype))
        keep = (count > 0) & boundary_keep
        target = torch.sigmoid(torch.where(keep, marginal, torch.zeros_like(marginal)))
        result = result + (((1.0 - torch.exp(total)) - target) ** 2 * keep).sum() / keep.sum().clamp_min(1)
    return _require_finite(result * 0.5)


def abstention_loss(null_logits: torch.Tensor, mention_mask: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """BCE for a gate that is positive when the query has no mention."""
    return masked_bce(null_logits, (~mention_mask.any(-1)).to(null_logits.dtype), query_mask)


def count_log_rate_loss(count_log_rate, mention_mask, query_mask) -> torch.Tensor:
    """Poisson NLL of a count head that predicts log-rate."""
    target = mention_mask.sum(-1).to(count_log_rate.dtype)
    elementwise = F.poisson_nll_loss(count_log_rate, target, log_input=True, full=False, reduction="none")
    keep = query_mask.to(elementwise.dtype)
    return _require_finite((elementwise * keep).sum() / keep.sum().clamp_min(1))


def _fit_marginal(targets: torch.Tensor, width: int) -> torch.Tensor:
    if targets.shape[-1] >= width:
        return targets[..., :width]
    return F.pad(targets, (0, width - targets.shape[-1]))


def _negative_query_mask(labels: torch.Tensor, query_mask: torch.Tensor, settings) -> torch.Tensor:
    positive_queries = labels.any(2) & query_mask
    absent_queries = query_mask & ~positive_queries
    priorities = torch.rand(absent_queries.shape, device=absent_queries.device).masked_fill(~absent_queries, -1.0)
    order = priorities.reshape(-1).argsort(descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=order.device)
    n_keep = (
        (positive_queries.sum() * settings.negative_query_ratio)
        .ceil()
        .clamp(min=1, max=settings.max_negative_queries_per_batch)
    )
    return positive_queries | ((rank.view_as(absent_queries) < n_keep) & absent_queries)


def matched_gold_mask(indices, valid, pairs, pair_mask) -> torch.Tensor:
    """Mark candidates whose half-open span equals a gold mention."""
    same = (indices.unsqueeze(-2) == pairs.unsqueeze(-3)).all(-1)
    return (same & pair_mask.unsqueeze(-2) & valid.unsqueeze(-1)).any(-1) & valid


def dense_targets_from_pairs(pairs: torch.Tensor, mask: torch.Tensor, text_length: int):
    """Build start, end, and inside targets from half-open mention pairs `[B, Q, G, 2]`."""
    batch, queries = pairs.shape[:2]
    valid = mask & (pairs[..., 0] >= 0) & (pairs[..., 1] > pairs[..., 0]) & (pairs[..., 1] <= text_length)
    weights = valid.to(torch.float32)
    starts = pairs[..., 0].masked_fill(~mask, 0).clamp(0, text_length)
    ends = pairs[..., 1].masked_fill(~mask, 0).clamp(0, text_length)
    start_targets = torch.zeros(batch, queries, text_length + 1, dtype=torch.float32, device=pairs.device)
    end_targets = torch.zeros_like(start_targets)
    start_targets.scatter_add_(2, starts, weights).clamp_(max=1.0)
    end_targets.scatter_add_(2, ends, weights).clamp_(max=1.0)
    difference = torch.zeros(batch, queries, text_length + 2, dtype=torch.float32, device=pairs.device)
    difference.scatter_add_(2, starts, weights)
    difference.scatter_add_(2, ends, -weights)
    inside_targets = (difference[..., : text_length + 1].cumsum(-1)[..., :text_length] > 0.5).to(torch.float32)
    return start_targets, end_targets, inside_targets


def boundary_training_loss(
    *,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    inside_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_valid: torch.Tensor,
    proposal_logits: torch.Tensor | None,
    proposal_gold: torch.Tensor | None,
    null_logits: torch.Tensor | None,
    count_log_rates: torch.Tensor | None,
    mention_pairs: torch.Tensor,
    mention_mask: torch.Tensor,
    query_mask: torch.Tensor,
    text_mask: torch.Tensor,
    settings,
    training: bool,
    soft_iou_scale: float = 1.0,
    consistency_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Boundary objectives with the weights stored on `settings`."""
    if mention_pairs.dim() != 4 or mention_pairs.shape[-1] != 2 or mention_mask.shape != mention_pairs.shape[:-1]:
        raise ValueError(f"mention_pairs must be [batch, queries, gold, 2], got {tuple(mention_pairs.shape)}")
    device = start_logits.device
    mention_pairs = mention_pairs.to(device)
    mention_mask = mention_mask.to(device).bool()
    query_mask = query_mask.to(device).bool()
    text_mask = text_mask.to(device).bool()
    candidate_indices = candidate_indices.to(device)
    n_boundary = start_logits.shape[-1]
    safe_pairs = mention_pairs.clamp(0, max(n_boundary - 1, 0))
    start_targets, end_targets, inside_targets = dense_targets_from_pairs(
        safe_pairs, mention_mask, text_mask.shape[-1]
    )
    boundary_mask = torch.arange(n_boundary, device=device).unsqueeze(0) <= text_mask.long().sum(-1).unsqueeze(1)
    boundary_keep = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    negative_weight = float(getattr(settings, "boundary_negative_weight", 1.0))
    marginal = getattr(settings, "boundary_marginal_loss", "bce")
    reduction = getattr(settings, "loss_reduction", "global")
    if marginal not in ("bce", "asymmetric_focal"):
        raise ValueError(f"unknown boundary_marginal_loss {marginal!r}")

    def marginal_loss(logits, target, keep):
        options = {"negative_weight": negative_weight, "query_mask": query_mask, "reduction": reduction}
        target = _fit_marginal(target, logits.shape[-1]).to(logits.dtype)
        if marginal == "bce":
            return masked_bce(logits, target, keep, **options)
        return asymmetric_focal_loss(
            logits,
            target,
            keep,
            gamma_positive=float(settings.boundary_focal_gamma_positive),
            gamma_negative=float(settings.boundary_focal_gamma_negative),
            clip=float(settings.boundary_focal_clip),
            **options,
        )

    zero = pair_logits.new_zeros(())
    loss_valid = candidate_valid.to(device).bool() & query_mask.unsqueeze(-1)
    use_soft_iou = float(settings.soft_iou_aux_weight) > 0
    labels = build_candidate_labels(
        candidate_indices, loss_valid, mention_pairs, mention_mask, return_iou=use_soft_iou
    )
    labels, soft_iou_targets = labels if use_soft_iou else (labels, None)
    hard = select_hard_negative_candidates(
        pair_logits.detach(),
        labels,
        loss_valid,
        negatives_per_positive=int(settings.hard_negatives_per_positive),
        minimum_negatives=int(settings.minimum_hard_negatives),
        keep_all_when_no_positive=bool(settings.hard_negative_keep_all_when_absent),
    )
    pair_query_mask = query_mask
    if training and float(settings.negative_query_ratio) > 0:
        pair_query_mask = _negative_query_mask(labels, query_mask, settings)
    effective = loss_valid & ((labels > 0.5) | hard)
    terms = {
        "start_loss": marginal_loss(start_logits, start_targets, boundary_keep),
        "end_loss": marginal_loss(end_logits, end_targets, boundary_keep),
        "pair_loss": masked_bce(pair_logits, labels, effective, query_mask=pair_query_mask, reduction=reduction),
        "soft_iou_loss": zero,
        "rerank_listwise_loss": zero,
        "inside_loss": masked_bce(
            inside_logits,
            _fit_marginal(inside_targets, inside_logits.shape[-1]).to(inside_logits.dtype),
            text_mask.unsqueeze(1) & query_mask.unsqueeze(-1),
            negative_weight=negative_weight,
            query_mask=query_mask,
            reduction=reduction,
        ),
        "proposal_loss": zero,
        "consistency_loss": zero,
        "abstention_loss": zero,
        "count_loss": zero,
    }
    if soft_iou_targets is not None and soft_iou_scale > 0:
        terms["soft_iou_loss"] = masked_bce(
            pair_logits, soft_iou_targets, effective, query_mask=pair_query_mask, reduction=reduction
        )
    if float(settings.rerank_listwise_weight) > 0:
        terms["rerank_listwise_loss"] = proposal_listwise_loss(
            pair_logits, (labels > 0.5) & loss_valid, loss_valid, pair_query_mask
        )
    if proposal_logits is not None and float(settings.proposal_loss_weight) > 0:
        if proposal_gold is None:
            proposal_gold = matched_gold_mask(candidate_indices, loss_valid, mention_pairs, mention_mask)
        terms["proposal_loss"] = proposal_listwise_loss(
            proposal_logits.to(device), proposal_gold.to(device), loss_valid, query_mask
        )
    if float(settings.consistency_loss_weight) > 0:
        terms["consistency_loss"] = marginal_pair_consistency_loss(
            pair_logits, candidate_indices, loss_valid, start_logits, end_logits, boundary_keep
        )
    if null_logits is not None and float(settings.abstention_loss_weight) > 0:
        terms["abstention_loss"] = abstention_loss(null_logits, mention_mask, query_mask)
    if count_log_rates is not None and float(settings.count_loss_weight) > 0:
        terms["count_loss"] = count_log_rate_loss(count_log_rates, mention_mask, query_mask)
    weights = getattr(settings, "loss_weights", None) or {}
    total = (
        float(weights.get("start", 1.0)) * terms["start_loss"]
        + float(weights.get("end", 1.0)) * terms["end_loss"]
        + float(weights.get("pair", 1.0)) * terms["pair_loss"]
        + float(weights.get("inside", 0.5)) * terms["inside_loss"]
        + float(settings.soft_iou_aux_weight) * float(soft_iou_scale) * terms["soft_iou_loss"]
        + float(settings.rerank_listwise_weight) * terms["rerank_listwise_loss"]
        + float(settings.proposal_loss_weight) * terms["proposal_loss"]
        + float(settings.consistency_loss_weight) * float(consistency_scale) * terms["consistency_loss"]
        + float(settings.abstention_loss_weight) * terms["abstention_loss"]
        + float(settings.count_loss_weight) * terms["count_loss"]
    )
    terms["loss"] = _require_finite(total)
    return terms


def sparse_relation_loss(logits: torch.Tensor, pairs, gold_pairs: torch.Tensor, gold_mask, weight: float):
    """Weighted BCE over sparse relation pairs."""
    if gold_pairs.dim() != 4 or gold_pairs.shape[-1] != 4 or gold_mask.shape != gold_pairs.shape[:-1]:
        raise ValueError(f"relation_gold_pairs must be [batch, relations, gold, 4], got {tuple(gold_pairs.shape)}")
    if logits.numel() == 0 or gold_pairs.shape[1] == 0 or gold_pairs.shape[0] == 0:
        return _require_finite(logits.sum() * 0.0)
    gold_pairs = gold_pairs.to(logits.device)
    gold_mask = gold_mask.to(logits.device).bool()
    batch, max_rel = gold_pairs.shape[:2]
    valid = (pairs.relation_index >= 0) & (pairs.relation_index < max_rel)
    valid &= (pairs.batch_index >= 0) & (pairs.batch_index < batch)
    safe_batch = pairs.batch_index.clamp(min=0, max=batch - 1)
    safe_rel = pairs.relation_index.clamp(min=0, max=max_rel - 1)
    coords = torch.stack((pairs.head_start, pairs.head_end, pairs.tail_start, pairs.tail_end), dim=-1)
    selected_mask = gold_mask[safe_batch, safe_rel] & valid.unsqueeze(-1)
    labels = ((coords.unsqueeze(1) == gold_pairs[safe_batch, safe_rel]).all(-1) & selected_mask).any(-1)
    pair_mask = pairs.pair_mask if pairs.pair_mask is not None else torch.ones_like(labels)
    pair_mask = pair_mask.to(logits.device) & valid
    if labels.shape != logits.shape or pair_mask.shape != logits.shape:
        raise ValueError(f"relation labels {tuple(labels.shape)} != logits {tuple(logits.shape)}")
    return _require_finite(weight * masked_bce(logits, labels.to(logits.dtype), pair_mask))


def _value_cols(value_alternatives, span_to_idx) -> list[int]:
    found = (span_to_idx.get((int(span[0]), int(span[1]))) for span in value_alternatives)
    return [index + 1 for index in found if index is not None]


def _scalar_field_nll(logits_row: torch.Tensor, target_cols) -> torch.Tensor:
    max_idx = logits_row.shape[-1] - 1
    cols = [col for col in target_cols if 0 <= col <= max_idx] or [0]
    idx = torch.tensor(cols, dtype=torch.long, device=logits_row.device)
    return -torch.logsumexp(F.log_softmax(logits_row, dim=-1)[idx], dim=-1)


def _list_field_bce(logits_row: torch.Tensor, positive_cols, reduction: str) -> torch.Tensor | None:
    cand_logits = logits_row[1:]
    if cand_logits.numel() == 0:
        return None
    target = torch.zeros_like(cand_logits)
    for col in positive_cols:
        if 0 <= col - 1 < cand_logits.shape[0]:
            target[col - 1] = 1.0
    return masked_bce(cand_logits, target, torch.ones_like(cand_logits, dtype=torch.bool), reduction=reduction)


def _field_terms(group, inst: int, record, span_indices):
    for field_index, field_spec in enumerate(group.field_specs):
        values = record.get(field_spec.query_id) or []
        span_to_idx = span_indices[field_index]
        if field_spec.is_scalar:
            cols = _value_cols(values[0], span_to_idx) if values else []
        else:
            cols = [col for value in values for col in _value_cols(value, span_to_idx)]
        yield group.assign_logits[field_index][inst], cols, field_spec.is_scalar


def _instance_field_loss(group, inst: int, record, span_indices) -> torch.Tensor:
    total = group.object_logits.new_zeros(())
    for row, cols, is_scalar in _field_terms(group, inst, record, span_indices):
        term = _scalar_field_nll(row, cols) if is_scalar else _list_field_bce(row, cols, "global")
        total = total + (term if term is not None else row.new_zeros(()))
    return total / max(len(group.field_specs), 1)


def _instance_field_logprob(group, inst: int, record, span_indices) -> torch.Tensor:
    total = group.object_logits.new_zeros(())
    for row, cols, is_scalar in _field_terms(group, inst, record, span_indices):
        term = _scalar_field_nll(row, cols) if is_scalar else _list_field_bce(row, cols, "sum")
        if term is not None:
            total = total - term
    return total


def compute_record_group_loss(group, records) -> dict[str, torch.Tensor]:
    """Object and field losses for one record group, with Hungarian assignment."""
    device = group.object_logits.device
    zero = group.object_logits.new_zeros(())
    span_indices = [
        {(int(spans[i, 0]), int(spans[i, 1])): i for i in range(spans.shape[0])} for spans in group.field_spans
    ]
    n_instances = group.num_instances
    if group.spec.mode == "natural":
        anchor_query = group.spec.anchor_query_id
        anchor_field = group.field_query_ids.index(anchor_query)
        seed_to_inst = {
            seed[1]: index
            for index, seed in enumerate(group.instance_seed)
            if seed is not None and seed[0] == anchor_field
        }
        field_loss, matched = zero, 0
        for record in records:
            anchor_values = record.get(anchor_query)
            cols = _value_cols(anchor_values[0], span_indices[anchor_field]) if anchor_values else []
            if not cols or (inst := seed_to_inst.get(cols[0] - 1)) is None:
                continue
            field_loss = field_loss + _instance_field_loss(group, inst, record, span_indices)
            matched += 1
        field_count = matched * len(group.field_specs)
        return {
            "object_loss": zero,
            "field_loss": field_loss / max(matched, 1),
            "object_count": 0,
            "field_count": field_count,
        }
    if not records:
        target = torch.zeros(n_instances, device=device)
        object_loss = F.binary_cross_entropy_with_logits(group.object_logits, target) if n_instances else zero
        return {"object_loss": object_loss, "field_loss": zero, "object_count": n_instances, "field_count": 0}
    if n_instances < len(records):
        raise TargetCapacityError(
            f"record group has {len(records)} gold instances but only {n_instances} hypotheses (mode={group.spec.mode})"
        )
    with torch.no_grad():
        obj_logp = F.logsigmoid(group.object_logits)
        cost = torch.zeros(n_instances, len(records), device=device)
        for row in range(n_instances):
            for col, record in enumerate(records):
                cost[row, col] = -(obj_logp[row] + _instance_field_logprob(group, row, record, span_indices))
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = [(int(row), int(col)) for row, col in zip(row_ind.tolist(), col_ind.tolist())]
    obj_target = torch.zeros(n_instances, device=device)
    obj_target[[row for row, _ in pairs]] = 1.0
    field_loss = zero
    for row, col in pairs:
        field_loss = field_loss + _instance_field_loss(group, row, records[col], span_indices)
    return {
        "object_loss": F.binary_cross_entropy_with_logits(group.object_logits, obj_target),
        "field_loss": field_loss / max(len(pairs), 1),
        "object_count": n_instances,
        "field_count": len(pairs) * len(group.field_specs),
    }


def aggregate_record_losses(parts, weight: float) -> dict[str, torch.Tensor]:
    """Average record groups by their object and field counts."""
    obj_total = parts[0]["object_loss"] * 0.0
    field_total = parts[0]["field_loss"] * 0.0
    for part in parts:
        obj_total = obj_total + part["object_loss"] * int(part["object_count"])
        field_total = field_total + part["field_loss"] * int(part["field_count"])
    obj = obj_total / max(sum(int(part["object_count"]) for part in parts), 1)
    field = field_total / max(sum(int(part["field_count"]) for part in parts), 1)
    return {"object": obj, "field": field, "total": float(weight) * (obj + field)}


def ForSchemaExtractionLoss(
    config,
    *,
    anchor,
    classification_logits=None,
    classification_targets=None,
    structure_loss=None,
    count_loss=None,
    boundary_terms=None,
    record_part_losses=None,
    relation_loss=None,
    **kwargs,
):
    """Sum the schema-extraction terms the model has already scored.

    Span checkpoints add summed classification BCE and the structure and count losses. Boundary checkpoints add
    `boundary_terms["loss"]`, label-mean classification BCE times `classification_loss_weight`, the aggregated
    record losses times `record_loss_weight`, and `relation_loss`.

    Args:
        config: `Gliner2Config`. Boundary weights are read from `config.boundary_config`.
        anchor: Zero scalar that fixes the device and dtype of an empty sum.
        classification_logits: Per-label logits, when classification targets are present.
        classification_targets: Dense classification targets.
        structure_loss: Span-structure loss already reduced by the span head.
        count_loss: Span-count loss already reduced by the span head.
        boundary_terms: Dict returned by `boundary_training_loss`, including its own `"loss"` key.
        record_part_losses: Per-group record losses for `aggregate_record_losses`.
        relation_loss: Already weighted relation loss.

    Returns:
        The total loss and the per-term dict, including `"loss"`.
    """
    has_classification = classification_targets is not None and classification_logits is not None
    if getattr(config, "architecture", "span") != "boundary":
        losses = {
            "classification_loss": classification_bce(classification_logits, classification_targets)
            if has_classification
            else anchor,
            "structure_loss": anchor if structure_loss is None else structure_loss,
            "count_loss": anchor if count_loss is None else count_loss,
        }
        losses["loss"] = losses["classification_loss"] + losses["structure_loss"] + losses["count_loss"]
        return losses["loss"], losses
    settings = config.boundary_config
    total = anchor
    losses = dict(boundary_terms or {})
    if boundary_terms is not None:
        total = total + boundary_terms["loss"]
    if has_classification:
        losses["classification_loss"] = classification_bce(
            classification_logits, classification_targets, settings.classification_loss_weight
        )
        total = total + losses["classification_loss"]
    if record_part_losses:
        packed = aggregate_record_losses(record_part_losses, settings.record_loss_weight)
        losses["record_object_loss"], losses["record_field_loss"] = packed["object"], packed["field"]
        total = total + packed["total"]
    if relation_loss is not None:
        losses["relation_loss"] = relation_loss
        total = total + relation_loss
    losses["loss"] = total
    return total, losses
