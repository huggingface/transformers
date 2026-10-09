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

from .decoding_gliner2 import linear_sum_assignment


ENTITY_TASK_ID = 1
CLASSIFICATION_TASK_ID = 4
MAX_SPAN_COUNT = 19
MASK_LOGIT = -1.0e4


class TargetCapacityError(ValueError):
    """Raised when gold records exceed the instance hypotheses."""


def finite_loss(value: torch.Tensor) -> torch.Tensor:
    """Replace non-finite loss values with zero."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def head_touch(modules, device=None) -> torch.Tensor:
    """Zero term so enabled optional heads stay in the autograd graph."""
    total = None
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            term = parameter.sum() * 0.0
            total = term if total is None else total + term
            device = parameter.device
    if total is None:
        if device is None:
            raise ValueError("head_touch requires a module or a device")
        return torch.zeros((), device=device)
    return total


def supervises_count(task_id: int) -> bool:
    """Return whether this task contributes 20-way count cross-entropy."""
    return int(task_id) not in (ENTITY_TASK_ID, CLASSIFICATION_TASK_ID)


def clamp_gold_count(count) -> int:
    """Clamp a gold instance count onto the 20-way count head."""
    if isinstance(count, torch.Tensor):
        if count.numel() != 1:
            raise ValueError("span count must be a scalar")
        count = int(count.item())
    count = int(count)
    if count < 0:
        raise ValueError(f"span count must be >= 0, got {count}")
    return min(count, MAX_SPAN_COUNT)


def invalid_span_mask(length: int, max_width: int, device) -> torch.Tensor:
    """Return True where `start + width` falls outside the word axis."""
    if length < 0 or max_width < 0:
        raise ValueError("span mask length and width must be >= 0")
    starts = torch.arange(length, device=device).unsqueeze(1)
    widths = torch.arange(max_width, device=device).unsqueeze(0)
    return (starts + widths >= length).reshape(-1)


def span_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Summed classification BCE."""
    return _bce_sum(logits, targets, "classification_targets")


def span_count_loss(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Summed 20-way count cross-entropy."""
    if logits.dim() != 2 or logits.shape[-1] != 20:
        raise ValueError(f"count logits must be [N, 20], got {tuple(logits.shape)}")
    if counts.shape != logits.shape[:1]:
        raise ValueError(f"count targets must be {tuple(logits.shape[:1])}, got {tuple(counts.shape)}")
    if int(counts.min()) < 0 or int(counts.max()) > MAX_SPAN_COUNT:
        raise ValueError("count class must be in 0..19")
    return F.cross_entropy(logits, counts.to(device=logits.device, dtype=torch.long), reduction="sum")


def span_structure_loss(
    scores: torch.Tensor,
    structure,
    span_mask: torch.Tensor,
    *,
    training: bool,
    masking_rate: float = 0.5,
) -> torch.Tensor:
    """Summed span BCE. Training drops each negative with `rand_like < rate`."""
    if scores.dim() != 4:
        raise ValueError(f"span scores must be [count, fields, words, width], got {tuple(scores.shape)}")
    labels = _span_label_tensor(scores, structure)
    if not torch.is_tensor(span_mask):
        raise ValueError("span_mask must be a tensor")
    flat_mask = span_mask.reshape(-1).bool()
    if flat_mask.numel() != scores.shape[-2] * scores.shape[-1]:
        raise ValueError("span_mask does not match the span grid")
    if masking_rate > 0.0 and training:
        negative = labels == 0
        random_mask = torch.rand_like(scores) < masking_rate
        keep = (~(negative & random_mask)).to(scores.dtype)
    else:
        keep = torch.ones_like(scores)
    loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none") * keep
    valid = (~flat_mask).to(loss.dtype)
    loss = loss.reshape(loss.shape[0], loss.shape[1], -1) * valid
    return loss.sum()


def _span_label_tensor(scores: torch.Tensor, structure) -> torch.Tensor:
    if torch.is_tensor(structure):
        if tuple(structure.shape) != tuple(scores.shape):
            raise ValueError(f"span targets {tuple(structure.shape)} != scores {tuple(scores.shape)}")
        return structure.to(device=scores.device, dtype=scores.dtype)
    if not isinstance(structure, (list, tuple)) or len(structure) < 2:
        raise ValueError("span structure must be [count, spans] or a label tensor")
    gold_count = clamp_gold_count(structure[0])
    if scores.shape[0] != gold_count:
        raise ValueError(f"span score count {scores.shape[0]} != gold count {gold_count}")
    labels = torch.zeros_like(scores)
    instances = structure[1]
    if len(instances) < gold_count:
        raise ValueError("span structure has fewer instances than its count")
    for index in range(gold_count):
        gold_spans = instances[index]
        if len(gold_spans) > scores.shape[1]:
            raise ValueError("span structure has more fields than the score tensor")
        for field_index, span in enumerate(gold_spans):
            _write_span_label(labels, index, field_index, span)
    return labels


def _write_span_label(labels: torch.Tensor, index: int, field_index: int, span) -> None:
    if span is None or span == (-1, -1):
        return
    if isinstance(span, tuple):
        _write_one_span(labels, index, field_index, span)
    elif isinstance(span, list):
        for sub in span:
            if sub is None or sub == (-1, -1):
                continue
            _write_one_span(labels, index, field_index, sub)
    else:
        raise ValueError("span labels must be (start, end), a list of spans, or None")


def _write_one_span(labels: torch.Tensor, index: int, field_index: int, span) -> None:
    if len(span) != 2:
        raise ValueError("a span label must be (start, end)")
    start, end = int(span[0]), int(span[1])
    width = end - start
    if 0 <= start < labels.shape[2] and 0 <= width < labels.shape[3]:
        labels[index, field_index, start, width] = 1


def _bce_sum(logits: torch.Tensor, targets: torch.Tensor, name: str) -> torch.Tensor:
    if tuple(logits.shape) != tuple(targets.shape):
        raise ValueError(f"{name} shape {tuple(targets.shape)} != logits {tuple(logits.shape)}")
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")


def iter_aligned_logits(predictions, targets, path: str):
    """Yield logit/target pairs from nested batch and group lists."""
    if predictions is None:
        raise ValueError(f"{path} requires model logits")
    if torch.is_tensor(predictions):
        if not torch.is_tensor(targets):
            raise ValueError(f"{path} must be a tensor aligned with logits")
        yield predictions, targets
        return
    try:
        pred_len = len(predictions)
        target_len = len(targets)
    except TypeError as exc:
        raise ValueError(f"{path} must align with the model outputs") from exc
    if pred_len != target_len:
        raise ValueError(f"{path} length {target_len} != outputs {pred_len}")
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        yield from iter_aligned_logits(prediction, target, f"{path}[{index}]")


def summed_classification_loss(logits, targets) -> torch.Tensor:
    """Summed BCE over every aligned classification group."""
    total = None
    for prediction, target in iter_aligned_logits(logits, targets, "classification_targets"):
        if prediction.numel() == 0 and target.numel() == 0:
            continue
        term = span_classification_loss(prediction, target)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("classification_targets did not match any logits")
    return total


def mean_classification_loss(logits, targets, weight: float) -> torch.Tensor:
    """Label-normalized BCE scaled by `classification_loss_weight`."""
    total = None
    count = 0
    for prediction, target in iter_aligned_logits(logits, targets, "classification_targets"):
        if prediction.numel() == 0 and target.numel() == 0:
            continue
        term = span_classification_loss(prediction, target)
        total = term if total is None else total + term
        count += int(target.numel())
    if total is None or count == 0:
        if logits is None:
            raise ValueError("classification_targets require classification logits")
        return _anchor_zero(logits) * weight
    return weight * total / count


def _anchor_zero(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.sum() * 0.0
    for item in value:
        return _anchor_zero(item)
    raise ValueError("cannot build a zero loss without tensors")


def _to_query_candidate(tensor: torch.Tensor, query_axis: int, candidate_axis: int) -> torch.Tensor:
    return torch.movedim(tensor, (query_axis, candidate_axis), (1, 2))


def _from_query_candidate(tensor: torch.Tensor, query_axis: int, candidate_axis: int) -> torch.Tensor:
    return torch.movedim(tensor, (1, 2), (query_axis, candidate_axis))


def _safe_bce(logits: torch.Tensor, targets: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    safe_targets = torch.where(keep, targets, torch.zeros_like(targets))
    return F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")


def _reduce(elementwise: torch.Tensor, keep: torch.Tensor, query_mask, mode: str) -> torch.Tensor:
    keep_f = keep.to(elementwise.dtype)
    if mode == "sum":
        if query_mask is not None and keep_f.dim() >= 2:
            keep_f = keep_f * query_mask.unsqueeze(-1).to(keep_f.dtype)
        return (elementwise * keep_f).sum()
    if mode == "global":
        return (elementwise * keep_f).sum() / keep_f.sum().clamp_min(1)
    if mode != "per_query":
        raise ValueError(f"unknown reduction mode {mode!r}")
    numerator = (elementwise * keep_f).sum(-1)
    denominator = keep_f.sum(-1).clamp_min(1)
    per_query = numerator / denominator
    active = keep.any(-1)
    if query_mask is not None:
        active = active & query_mask
    active_f = active.to(per_query.dtype)
    return (per_query * active_f).sum() / active_f.sum().clamp_min(1)


def balanced_multilabel_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    negative_weight: float = 1.0,
    query_mask=None,
    reduction: str = "global",
) -> torch.Tensor:
    """Mean multi-label BCE over valid positions."""
    bce = _safe_bce(logits, targets, valid_mask)
    weight = torch.where(targets > 0.5, torch.ones_like(targets), torch.full_like(targets, negative_weight))
    return _reduce(bce * weight, valid_mask, query_mask, reduction)


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
    safe_logits = torch.where(valid_mask, logits, torch.zeros_like(logits))
    probability = torch.sigmoid(safe_logits)
    if clip > 0:
        negative_prob = (1 - probability + clip).clamp(max=1.0)
    else:
        negative_prob = 1 - probability
    positive = targets * torch.log(probability.clamp_min(1e-8)) * ((1 - probability) ** gamma_positive)
    negative = (
        (1 - targets) * torch.log(negative_prob.clamp_min(1e-8)) * (probability**gamma_negative) * negative_weight
    )
    return _reduce(-(positive + negative), valid_mask, query_mask, reduction)


def build_candidate_labels(
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    gold_pairs: torch.Tensor,
    gold_mask: torch.Tensor,
    *,
    return_iou: bool = False,
    query_axis: int = 1,
    candidate_axis: int = 2,
):
    """Label candidates that exactly match a gold span, optionally with soft IoU."""
    _check_gold_pairs(gold_pairs, gold_mask)
    pooled = candidate_indices.dim() == 3
    if pooled:
        cand = candidate_indices.unsqueeze(2).unsqueeze(3)
        gold = gold_pairs.unsqueeze(1)
        same = (cand == gold).all(dim=-1) & gold_mask.unsqueeze(1)
        labels = same.any(-1).to(torch.float) * candidate_mask.unsqueeze(-1)
        candidate_weight = candidate_mask.unsqueeze(-1).to(labels.dtype)
    else:
        canonical_indices = torch.movedim(candidate_indices, (query_axis, candidate_axis), (1, 2))
        canonical_mask = _to_query_candidate(candidate_mask, query_axis, candidate_axis)
        cand = canonical_indices.unsqueeze(3)
        gold = gold_pairs.unsqueeze(2)
        same = (cand == gold).all(dim=-1) & gold_mask.unsqueeze(2)
        canonical_labels = same.any(dim=-1).to(torch.float) * canonical_mask.to(torch.float)
        labels = _from_query_candidate(canonical_labels, query_axis, candidate_axis)
        candidate_weight = canonical_mask.to(canonical_labels.dtype)
    if not return_iou:
        return labels
    intersection = (torch.minimum(cand[..., 1], gold[..., 1]) - torch.maximum(cand[..., 0], gold[..., 0])).clamp_min(0)
    union = cand[..., 1] - cand[..., 0] + gold[..., 1] - gold[..., 0] - intersection
    iou = intersection.to(torch.float) / union.clamp_min(1).to(torch.float)
    iou = iou * (gold_mask.unsqueeze(1) if pooled else gold_mask.unsqueeze(2)).to(iou.dtype)
    canonical_soft = iou.amax(-1) * candidate_weight
    soft_targets = canonical_soft if pooled else _from_query_candidate(canonical_soft, query_axis, candidate_axis)
    return labels, soft_targets


def select_hard_negative_candidates(
    pair_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    negatives_per_positive: int,
    minimum_negatives: int,
    keep_all_when_no_positive: bool = False,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """Keep every positive and the highest-scoring negatives per query."""
    original_query, original_candidate = query_axis, candidate_axis
    pair_logits = _to_query_candidate(pair_logits, query_axis, candidate_axis)
    labels = _to_query_candidate(labels, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    positive = (labels > 0.5) & valid_mask
    negative = (labels <= 0.5) & valid_mask
    n_positive = positive.sum(-1, keepdim=True)
    floor = torch.finfo(pair_logits.dtype).min
    order = torch.argsort(pair_logits.masked_fill(~negative, floor), dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    n_keep = (negatives_per_positive * n_positive).clamp_min(minimum_negatives)
    if keep_all_when_no_positive:
        n_keep = torch.where(n_positive == 0, torch.full_like(n_keep, pair_logits.shape[-1]), n_keep)
    selected = positive | (negative & (rank < n_keep))
    return _from_query_candidate(selected, original_query, original_candidate)


def candidate_pair_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    hard_negative_mask: torch.Tensor | None = None,
    *,
    query_mask=None,
    reduction: str = "global",
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """BCE over candidates, optionally restricted to hard negatives."""
    logits = _to_query_candidate(logits, query_axis, candidate_axis)
    labels = _to_query_candidate(labels, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    if hard_negative_mask is not None:
        hard_negative_mask = _to_query_candidate(hard_negative_mask, query_axis, candidate_axis)
        effective = valid_mask & ((labels > 0.5) | hard_negative_mask)
    else:
        effective = valid_mask
    return _reduce(_safe_bce(logits, labels, effective), effective, query_mask, reduction)


def inside_consistency_loss(
    inside_logits: torch.Tensor,
    inside_targets: torch.Tensor,
    text_mask: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    negative_weight: float = 1.0,
    reduction: str = "global",
) -> torch.Tensor:
    """BCE for inside-span marginals."""
    keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    return balanced_multilabel_bce(
        inside_logits,
        inside_targets,
        keep,
        negative_weight=negative_weight,
        query_mask=query_mask,
        reduction=reduction,
    )


def proposal_listwise_loss(
    proposal_logits: torch.Tensor,
    gold_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """Rank gold candidates above the other valid proposals."""
    proposal_logits = _to_query_candidate(proposal_logits, query_axis, candidate_axis)
    gold_mask = _to_query_candidate(gold_mask, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    logits = proposal_logits.masked_fill(~valid_mask, MASK_LOGIT)
    all_lse = torch.logsumexp(logits, dim=-1)
    gold_lse = torch.logsumexp(logits.masked_fill(~gold_mask, MASK_LOGIT), dim=-1)
    has_gold = gold_mask.any(-1) & query_mask
    loss = torch.where(has_gold, all_lse - gold_lse, torch.zeros_like(all_lse))
    return loss.sum() / has_gold.to(loss.dtype).sum().clamp_min(1)


def reranker_listwise_loss(
    pair_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """Listwise gold-mass loss over reranked candidates."""
    gold_mask = (labels > 0.5) & valid_mask
    return proposal_listwise_loss(
        pair_logits,
        gold_mask,
        valid_mask,
        query_mask,
        query_axis=query_axis,
        candidate_axis=candidate_axis,
    )


def marginal_pair_consistency_loss(
    pair_logits: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    boundary_keep: torch.Tensor,
) -> torch.Tensor:
    """Match boundary marginals to candidate noisy-OR probabilities."""
    probabilities = torch.sigmoid(pair_logits) * valid_mask.to(pair_logits.dtype)
    eps = max(1e-6, torch.finfo(probabilities.dtype).eps)
    log_survival = torch.log1p(-probabilities.clamp(max=1.0 - eps))
    batch, queries, n_boundary = start_logits.shape

    def accumulate(index: torch.Tensor):
        safe_index = index.clamp(0, n_boundary - 1)
        total = torch.zeros(batch, queries, n_boundary, dtype=log_survival.dtype, device=log_survival.device)
        total.scatter_add_(2, safe_index, log_survival)
        count = torch.zeros_like(total)
        count.scatter_add_(2, safe_index, valid_mask.to(total.dtype))
        return 1.0 - torch.exp(total), count > 0

    predicted_start, reached_start = accumulate(indices[..., 0])
    predicted_end, reached_end = accumulate(indices[..., 1])
    result = pair_logits.new_zeros(())
    for predicted, marginal, reached in (
        (predicted_start, start_logits, reached_start),
        (predicted_end, end_logits, reached_end),
    ):
        keep = reached & boundary_keep
        target = torch.sigmoid(torch.where(keep, marginal, torch.zeros_like(marginal)))
        squared = (predicted - target) ** 2
        result = result + (squared * keep).sum() / keep.sum().clamp_min(1)
    return result * 0.5


def abstention_loss(null_logits: torch.Tensor, mention_mask: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """BCE for a gate that is positive when the query has no mention."""
    target = (~mention_mask.any(-1)).to(null_logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(null_logits, target, reduction="none")
    keep = query_mask.to(elementwise.dtype)
    return (elementwise * keep).sum() / keep.sum().clamp_min(1)


def count_log_rate_loss(
    count_log_rate: torch.Tensor, mention_mask: torch.Tensor, query_mask: torch.Tensor
) -> torch.Tensor:
    """Poisson NLL of a count head that predicts log-rate."""
    target = mention_mask.sum(-1).to(count_log_rate.dtype)
    elementwise = F.poisson_nll_loss(count_log_rate, target, log_input=True, full=False, reduction="none")
    keep = query_mask.to(elementwise.dtype)
    return (elementwise * keep).sum() / keep.sum().clamp_min(1)


def _check_gold_pairs(gold_pairs: torch.Tensor, gold_mask: torch.Tensor) -> None:
    if gold_pairs.dim() != 4 or gold_pairs.shape[-1] != 2:
        raise ValueError(f"mention_pairs must be [batch, queries, gold, 2], got {tuple(gold_pairs.shape)}")
    if tuple(gold_mask.shape) != tuple(gold_pairs.shape[:-1]):
        raise ValueError("mention_mask must match mention_pairs without the span axis")


def _fit_marginal(targets: torch.Tensor, width: int) -> torch.Tensor:
    if targets.shape[-1] == width:
        return targets
    if targets.shape[-1] > width:
        return targets[..., :width]
    return F.pad(targets, (0, width - targets.shape[-1]))


def _negative_query_mask(
    labels: torch.Tensor, query_mask: torch.Tensor, settings, candidate_axis: int
) -> torch.Tensor:
    positive_queries = labels.any(candidate_axis) & query_mask
    absent_queries = query_mask & ~positive_queries
    priorities = torch.rand(absent_queries.shape, device=absent_queries.device).masked_fill(~absent_queries, -1.0)
    flat_priorities = priorities.reshape(-1)
    order = flat_priorities.argsort(descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=order.device)
    n_keep = (
        (positive_queries.sum() * settings.negative_query_ratio)
        .ceil()
        .clamp(min=1, max=settings.max_negative_queries_per_batch)
    )
    selected = (rank.view_as(absent_queries) < n_keep) & absent_queries
    return positive_queries | selected


def matched_gold_mask(
    indices: torch.Tensor, valid: torch.Tensor, pairs: torch.Tensor, pair_mask: torch.Tensor
) -> torch.Tensor:
    """Mark candidates whose half-open span equals a gold mention."""
    same = (indices.unsqueeze(-2) == pairs.unsqueeze(-3)).all(-1)
    same = same & pair_mask.unsqueeze(-2) & valid.unsqueeze(-1)
    return same.any(-1) & valid


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
    start_targets: torch.Tensor | None = None,
    end_targets: torch.Tensor | None = None,
    inside_targets: torch.Tensor | None = None,
    soft_iou_scale: float = 1.0,
    consistency_scale: float = 1.0,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> dict[str, torch.Tensor]:
    """Boundary objectives with the weights stored on `settings`."""
    _check_gold_pairs(mention_pairs, mention_mask)
    device = start_logits.device
    mention_pairs = mention_pairs.to(device)
    mention_mask = mention_mask.to(device).bool()
    query_mask = query_mask.to(device).bool()
    text_mask = text_mask.to(device).bool()
    n_boundary = start_logits.shape[-1]
    text_length = text_mask.shape[-1]
    if start_targets is None or end_targets is None or inside_targets is None:
        from .processing_gliner2 import dense_targets_from_pairs

        safe_pairs = mention_pairs.clamp(0, max(n_boundary - 1, 0))
        built_start, built_end, built_inside = dense_targets_from_pairs(safe_pairs, mention_mask, text_length)
        start_targets = built_start if start_targets is None else start_targets
        end_targets = built_end if end_targets is None else end_targets
        inside_targets = built_inside if inside_targets is None else inside_targets
    start_targets = _fit_marginal(start_targets, n_boundary).to(start_logits.dtype)
    end_targets = _fit_marginal(end_targets, n_boundary).to(end_logits.dtype)
    inside_targets = _fit_marginal(inside_targets, inside_logits.shape[-1]).to(inside_logits.dtype)
    lengths = text_mask.long().sum(-1)
    boundary_index = torch.arange(n_boundary, device=device)
    boundary_mask = boundary_index.unsqueeze(0) <= lengths.unsqueeze(1)
    boundary_keep = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    negative_weight = float(getattr(settings, "boundary_negative_weight", 1.0))
    marginal = getattr(settings, "boundary_marginal_loss", "bce")
    reduction = getattr(settings, "loss_reduction", "global")

    def marginal_loss(logits, target, keep):
        if marginal == "asymmetric_focal":
            return asymmetric_focal_loss(
                logits,
                target,
                keep,
                gamma_positive=float(settings.boundary_focal_gamma_positive),
                gamma_negative=float(settings.boundary_focal_gamma_negative),
                clip=float(settings.boundary_focal_clip),
                negative_weight=negative_weight,
                query_mask=query_mask,
                reduction=reduction,
            )
        if marginal != "bce":
            raise ValueError(f"unknown boundary_marginal_loss {marginal!r}")
        return balanced_multilabel_bce(
            logits, target, keep, negative_weight=negative_weight, query_mask=query_mask, reduction=reduction
        )

    start_loss = marginal_loss(start_logits, start_targets, boundary_keep)
    end_loss = marginal_loss(end_logits, end_targets, boundary_keep)
    loss_valid = candidate_valid.to(device).bool() & query_mask.unsqueeze(-1)
    use_soft_iou = float(settings.soft_iou_aux_weight) > 0
    label_output = build_candidate_labels(
        candidate_indices.to(device),
        loss_valid,
        mention_pairs,
        mention_mask,
        return_iou=use_soft_iou,
        query_axis=query_axis,
        candidate_axis=candidate_axis,
    )
    if use_soft_iou:
        labels, soft_iou_targets = label_output
    else:
        labels = label_output
        soft_iou_targets = None
    hard = select_hard_negative_candidates(
        pair_logits.detach(),
        labels,
        loss_valid,
        negatives_per_positive=int(settings.hard_negatives_per_positive),
        minimum_negatives=int(settings.minimum_hard_negatives),
        keep_all_when_no_positive=bool(settings.hard_negative_keep_all_when_absent),
        query_axis=query_axis,
        candidate_axis=candidate_axis,
    )
    pair_query_mask = query_mask
    if training and float(settings.negative_query_ratio) > 0:
        pair_query_mask = _negative_query_mask(labels, query_mask, settings, candidate_axis)
    pair_loss = candidate_pair_loss(
        pair_logits,
        labels,
        loss_valid,
        hard_negative_mask=hard,
        query_mask=pair_query_mask,
        reduction=reduction,
        query_axis=query_axis,
        candidate_axis=candidate_axis,
    )
    soft_iou_loss = pair_logits.new_zeros(())
    if soft_iou_targets is not None and soft_iou_scale > 0:
        effective = loss_valid & ((labels > 0.5) | hard)
        soft_iou_loss = candidate_pair_loss(
            pair_logits,
            soft_iou_targets,
            effective,
            query_mask=pair_query_mask,
            reduction=reduction,
            query_axis=query_axis,
            candidate_axis=candidate_axis,
        )
    rerank_loss = pair_logits.new_zeros(())
    if float(settings.rerank_listwise_weight) > 0:
        rerank_loss = reranker_listwise_loss(
            pair_logits, labels, loss_valid, pair_query_mask, query_axis=query_axis, candidate_axis=candidate_axis
        )
    inside_loss = inside_consistency_loss(
        inside_logits, inside_targets, text_mask, query_mask, negative_weight=negative_weight, reduction=reduction
    )
    proposal_loss = pair_logits.new_zeros(())
    if proposal_logits is not None and float(settings.proposal_loss_weight) > 0:
        if proposal_gold is None:
            proposal_gold = matched_gold_mask(candidate_indices.to(device), loss_valid, mention_pairs, mention_mask)
        proposal_loss = proposal_listwise_loss(
            proposal_logits.to(device),
            proposal_gold.to(device),
            loss_valid,
            query_mask,
            query_axis=query_axis,
            candidate_axis=candidate_axis,
        )
    consistency_loss = pair_logits.new_zeros(())
    if float(settings.consistency_loss_weight) > 0:
        consistency_loss = marginal_pair_consistency_loss(
            pair_logits, candidate_indices.to(device), loss_valid, start_logits, end_logits, boundary_keep
        )
    null_loss = pair_logits.new_zeros(())
    if null_logits is not None and float(settings.abstention_loss_weight) > 0:
        null_loss = abstention_loss(null_logits, mention_mask, query_mask)
    count_loss = pair_logits.new_zeros(())
    if count_log_rates is not None and float(settings.count_loss_weight) > 0:
        count_loss = count_log_rate_loss(count_log_rates, mention_mask, query_mask)
    weights = getattr(settings, "loss_weights", None) or {}
    terms = {
        "start_loss": finite_loss(start_loss),
        "end_loss": finite_loss(end_loss),
        "pair_loss": finite_loss(pair_loss),
        "soft_iou_loss": finite_loss(soft_iou_loss),
        "rerank_listwise_loss": finite_loss(rerank_loss),
        "inside_loss": finite_loss(inside_loss),
        "proposal_loss": finite_loss(proposal_loss),
        "consistency_loss": finite_loss(consistency_loss),
        "abstention_loss": finite_loss(null_loss),
        "count_loss": finite_loss(count_loss),
    }
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
    terms["loss"] = total
    return terms


def sparse_relation_loss(
    logits: torch.Tensor,
    pairs,
    gold_pairs: torch.Tensor,
    gold_mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Weighted BCE over sparse relation pairs."""
    if gold_pairs.dim() != 4 or gold_pairs.shape[-1] != 4:
        raise ValueError(f"relation_gold_pairs must be [batch, relations, gold, 4], got {tuple(gold_pairs.shape)}")
    if tuple(gold_mask.shape) != tuple(gold_pairs.shape[:-1]):
        raise ValueError("relation_gold_mask must match relation_gold_pairs without the coordinate axis")
    if logits.numel() == 0 or gold_pairs.shape[1] == 0 or gold_pairs.shape[0] == 0:
        return logits.sum() * 0.0
    gold_pairs = gold_pairs.to(logits.device)
    gold_mask = gold_mask.to(logits.device).bool()
    max_rel = gold_pairs.shape[1]
    relation_index = pairs.relation_index
    valid_rel = (relation_index >= 0) & (relation_index < max_rel)
    safe_rel = relation_index.clamp(min=0, max=max_rel - 1)
    safe_batch = pairs.batch_index.clamp(min=0, max=gold_pairs.shape[0] - 1)
    valid_batch = (pairs.batch_index >= 0) & (pairs.batch_index < gold_pairs.shape[0])
    coords = torch.stack((pairs.head_start, pairs.head_end, pairs.tail_start, pairs.tail_end), dim=-1)
    selected_gold = gold_pairs[safe_batch, safe_rel]
    selected_mask = gold_mask[safe_batch, safe_rel] & (valid_rel & valid_batch).unsqueeze(-1)
    labels = ((coords.unsqueeze(1) == selected_gold).all(-1) & selected_mask).any(-1).to(logits.dtype)
    if labels.shape != logits.shape:
        raise ValueError(f"relation labels {tuple(labels.shape)} != logits {tuple(logits.shape)}")
    pair_mask = pairs.pair_mask if pairs.pair_mask is not None else torch.ones_like(labels, dtype=torch.bool)
    pair_mask = pair_mask.to(logits.device) & valid_rel & valid_batch
    if pair_mask.shape != logits.shape:
        raise ValueError(f"relation pair_mask {tuple(pair_mask.shape)} != logits {tuple(logits.shape)}")
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    reduced = (loss * pair_mask.to(loss.dtype)).sum() / pair_mask.sum().clamp_min(1)
    return weight * reduced


def _span_index(field_spans: torch.Tensor) -> dict:
    return {(int(field_spans[i, 0]), int(field_spans[i, 1])): i for i in range(field_spans.shape[0])}


def _resolve_value_cols(value_alternatives, span_to_idx):
    columns = []
    for span in value_alternatives:
        found = span_to_idx.get((int(span[0]), int(span[1])))
        if found is not None:
            columns.append(found + 1)
    return columns


def _scalar_field_nll(logits_row: torch.Tensor, target_cols) -> torch.Tensor:
    logp = F.log_softmax(logits_row, dim=-1)
    cols = target_cols or [0]
    max_idx = logits_row.shape[-1] - 1
    cols = [min(col, max_idx) for col in cols if 0 <= col <= max_idx] or [0]
    idx = torch.tensor(cols, dtype=torch.long, device=logits_row.device)
    return -torch.logsumexp(logp[idx], dim=-1)


def _list_field_bce(logits_row: torch.Tensor, positive_cols) -> torch.Tensor:
    cand_logits = logits_row[1:]
    if cand_logits.numel() == 0:
        return logits_row.new_zeros(())
    target = torch.zeros_like(cand_logits)
    for col in positive_cols:
        idx = col - 1
        if 0 <= idx < cand_logits.shape[0]:
            target[idx] = 1.0
    return F.binary_cross_entropy_with_logits(cand_logits, target, reduction="mean")


def _field_target_cols(field_spec, record, span_to_idx):
    target = record.field_for_query(field_spec.query_id)
    if field_spec.cardinality.is_scalar:
        columns = _resolve_value_cols(target.values[0], span_to_idx) if target is not None and target.values else []
        return columns, True
    columns = []
    if target is not None:
        for value in target.values:
            columns.extend(_resolve_value_cols(value, span_to_idx))
    return columns, False


def _instance_field_loss(group, inst: int, record, span_indices) -> torch.Tensor:
    total = group.object_logits.new_zeros(())
    n_fields = 0
    for field_index, field_spec in enumerate(group.field_specs):
        cols, is_scalar = _field_target_cols(field_spec, record, span_indices[field_index])
        row = group.assign_logits[field_index][inst]
        total = total + (_scalar_field_nll(row, cols) if is_scalar else _list_field_bce(row, cols))
        n_fields += 1
    return total / max(n_fields, 1)


def _instance_field_logprob(group, inst: int, record, span_indices) -> torch.Tensor:
    total = group.object_logits.new_zeros(())
    for field_index, field_spec in enumerate(group.field_specs):
        cols, is_scalar = _field_target_cols(field_spec, record, span_indices[field_index])
        row = group.assign_logits[field_index][inst]
        if is_scalar:
            total = total - _scalar_field_nll(row, cols)
        else:
            cand_logits = row[1:]
            if cand_logits.numel() == 0:
                continue
            target = torch.zeros_like(cand_logits)
            for col in cols:
                idx = col - 1
                if 0 <= idx < cand_logits.shape[0]:
                    target[idx] = 1.0
            total = total - F.binary_cross_entropy_with_logits(cand_logits, target, reduction="sum")
    return total


def compute_record_group_loss(group, records) -> dict[str, torch.Tensor]:
    """Object and field losses for one record group, with Hungarian assignment."""
    device = group.object_logits.device
    zero = group.object_logits.new_zeros(())
    span_indices = [_span_index(spans) for spans in group.field_spans]
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
            anchor_target = record.field_for_query(anchor_query)
            if anchor_target is None or not anchor_target.values:
                continue
            cols = _resolve_value_cols(anchor_target.values[0], span_indices[anchor_field])
            if not cols or (inst := seed_to_inst.get(cols[0] - 1)) is None:
                continue
            field_loss = field_loss + _instance_field_loss(group, inst, record, span_indices)
            matched += 1
        return {
            "object_loss": zero,
            "field_loss": field_loss / max(matched, 1),
            "object_count": 0,
            "field_count": matched * len(group.field_specs),
        }
    count = len(records)
    if count == 0:
        target = torch.zeros(n_instances, device=device)
        object_loss = F.binary_cross_entropy_with_logits(group.object_logits, target) if n_instances else zero
        return {"object_loss": object_loss, "field_loss": zero, "object_count": n_instances, "field_count": 0}
    if n_instances < count:
        task = getattr(group.spec, "task_index", 0)
        raise TargetCapacityError(
            f"record group task={task} has {count} gold instances but only {n_instances} hypotheses "
            f"(mode={group.spec.mode})"
        )
    with torch.no_grad():
        obj_logp = F.logsigmoid(group.object_logits)
        cost = torch.zeros(n_instances, count, device=device)
        for row in range(n_instances):
            for col, record in enumerate(records):
                cost[row, col] = -(obj_logp[row] + _instance_field_logprob(group, row, record, span_indices))
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_rows = {int(row) for row in row_ind.tolist() if 0 <= int(row) < n_instances}
    obj_target = torch.zeros(n_instances, device=device)
    for row in matched_rows:
        obj_target[row] = 1.0
    object_loss = F.binary_cross_entropy_with_logits(group.object_logits, obj_target)
    valid_pairs = [
        (int(row), int(col))
        for row, col in zip(row_ind.tolist(), col_ind.tolist())
        if 0 <= int(row) < n_instances and 0 <= int(col) < len(records)
    ]
    field_loss = zero
    for row, col in valid_pairs:
        field_loss = field_loss + _instance_field_loss(group, row, records[col], span_indices)
    return {
        "object_loss": object_loss,
        "field_loss": field_loss / max(len(valid_pairs), 1),
        "object_count": n_instances,
        "field_count": len(valid_pairs) * len(group.field_specs),
    }


def aggregate_record_losses(parts, weight: float) -> dict[str, torch.Tensor]:
    """Average record groups by their object and field counts."""
    if not parts:
        raise ValueError("record loss requires at least one group")
    obj_total = parts[0]["object_loss"] * 0.0
    field_total = parts[0]["field_loss"] * 0.0
    object_count = 0
    field_count = 0
    for part in parts:
        n_obj = int(part["object_count"])
        n_field = int(part["field_count"])
        obj_total = obj_total + part["object_loss"] * n_obj
        field_total = field_total + part["field_loss"] * n_field
        object_count += n_obj
        field_count += n_field
    obj = obj_total / max(object_count, 1)
    field = field_total / max(field_count, 1)
    return {"object": obj, "field": field, "total": float(weight) * (obj + field)}


def build_dense_record_matching_cost(
    object_logits: torch.Tensor,
    assign_logits: torch.Tensor,
    gold_indicator: torch.Tensor,
    scalar_fields: torch.Tensor,
    instance_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized record cost `[..., instances, gold]` for dense pools."""
    present = gold_indicator.any(-1)
    target = torch.cat(((~present).unsqueeze(-1), gold_indicator), -1)
    logp = F.log_softmax(assign_logits, -1)
    scalar_logprob = torch.logsumexp(logp.unsqueeze(-3).masked_fill(~target.unsqueeze(-4), MASK_LOGIT), -1)
    candidates = assign_logits[..., 1:]
    list_logprob = -F.binary_cross_entropy_with_logits(
        candidates.unsqueeze(-3).expand(
            *candidates.shape[:-3],
            candidates.shape[-3],
            gold_indicator.shape[-3],
            candidates.shape[-2],
            candidates.shape[-1],
        ),
        gold_indicator.unsqueeze(-4)
        .expand(
            *gold_indicator.shape[:-3],
            assign_logits.shape[-3],
            gold_indicator.shape[-3],
            gold_indicator.shape[-2],
            gold_indicator.shape[-1],
        )
        .to(candidates.dtype),
        reduction="none",
    ).sum(-1)
    field_logprob = torch.where(scalar_fields.unsqueeze(-2).unsqueeze(-2), scalar_logprob, list_logprob).sum(-1)
    cost = -(F.logsigmoid(object_logits).unsqueeze(-1) + field_logprob)
    if instance_mask is not None:
        cost = cost.masked_fill(~instance_mask.unsqueeze(-1), -MASK_LOGIT)
    return cost


def _filter_match_indices(indices, shape, instance_mask=None):
    if len(indices) != len(shape):
        raise ValueError(f"index rank {len(indices)} does not match tensor rank {len(shape)}")
    if not indices:
        return ()
    if len({int(index.numel()) for index in indices}) != 1:
        raise ValueError("advanced-index tensors must have equal lengths")
    valid = torch.ones_like(indices[0], dtype=torch.bool)
    for index, size in zip(indices, shape):
        valid = valid & (index >= 0) & (index < int(size))
    if instance_mask is not None and valid.any():
        if len(indices) < 3:
            raise ValueError("instance_mask requires at least three index axes")
        if any(int(size) <= 0 for size in instance_mask.shape):
            return tuple(index[:0] for index in indices)
        for index, size in zip(indices[:3], instance_mask.shape):
            valid = valid & (index >= 0) & (index < int(size))
        safe = tuple(
            index.clamp(min=0, max=max(int(size) - 1, 0)) for index, size in zip(indices[:3], instance_mask.shape)
        )
        valid = valid & instance_mask[safe]
    return tuple(index[valid] for index in indices)


def dense_record_batch_loss(output) -> dict[str, torch.Tensor]:
    """Vectorized record loss. Hungarian assignment stays per group."""
    device = output.object_logits.device
    record_mask = output.record_mask.to(device)
    gold_indicator = output.gold_indicator.to(device)
    present = gold_indicator.any(-1)
    target = torch.cat(((~present).unsqueeze(-1), gold_indicator), -1)
    logp = F.log_softmax(output.assign_logits, -1)
    scalar_nll = -torch.logsumexp(logp.unsqueeze(3).masked_fill(~target.unsqueeze(2), MASK_LOGIT), -1)
    candidates = output.assign_logits[..., 1:]
    list_nll = F.binary_cross_entropy_with_logits(
        candidates.unsqueeze(3).expand(
            *candidates.shape[:3], gold_indicator.shape[2], candidates.shape[3], candidates.shape[4]
        ),
        gold_indicator.unsqueeze(2)
        .expand(
            *gold_indicator.shape[:2],
            output.object_logits.shape[2],
            gold_indicator.shape[2],
            gold_indicator.shape[3],
            gold_indicator.shape[4],
        )
        .to(candidates.dtype),
        reduction="none",
    ).mean(-1)
    field_nll = torch.where(output.scalar_fields[:, :, None, None, :], scalar_nll, list_nll)
    field_nll = (field_nll * output.field_mask[:, :, None, None, :].to(field_nll.dtype)).sum(
        -1
    ) / output.field_mask.sum(-1)[:, :, None, None].clamp_min(1)
    with torch.no_grad():
        cost = build_dense_record_matching_cost(
            output.object_logits,
            output.assign_logits,
            gold_indicator,
            output.scalar_fields,
            output.instance_mask,
        )
        cost_cpu = cost.detach().cpu()
        metadata = torch.stack(
            (
                output.instance_mask.sum(-1),
                record_mask.sum(-1),
                output.modes,
                output.group_mask.to(torch.long),
            ),
            -1,
        ).cpu()
        anchor = output.anchor_fields.clamp(min=0, max=gold_indicator.shape[3] - 1)
        natural_gold = (
            gold_indicator.gather(
                3, anchor[..., None, None, None].expand(*gold_indicator.shape[:3], 1, gold_indicator.shape[-1])
            )
            .squeeze(3)
            .cpu()
        )
        matched_batch, matched_group, matched_rows, matched_cols = [], [], [], []
        batch_size, groups = output.group_mask.shape
        for batch_index in range(batch_size):
            for group_index in range(groups):
                available, count, mode, valid = metadata[batch_index, group_index].tolist()
                if not valid:
                    continue
                if available < count:
                    raise TargetCapacityError(
                        f"record group batch={batch_index} group={group_index} has {count} gold instances "
                        f"but only {available} hypotheses"
                    )
                if count == 0:
                    continue
                if mode == 0:
                    anchors = natural_gold[batch_index, group_index, : int(count)]
                    columns = torch.nonzero(anchors.any(-1), as_tuple=False).flatten()
                    rows = anchors[columns].to(torch.long).argmax(-1)
                else:
                    rows, columns = linear_sum_assignment(cost_cpu[batch_index, group_index, :, : int(count)])
                matched_batch.extend([batch_index] * len(rows))
                matched_group.extend([group_index] * len(rows))
                matched_rows.extend(rows.tolist())
                matched_cols.extend(columns.tolist())
    index = tuple(
        torch.as_tensor(values, dtype=torch.long, device=device)
        for values in (matched_batch, matched_group, matched_rows, matched_cols)
    )
    index = _filter_match_indices(index, field_nll.shape, instance_mask=output.instance_mask)
    object_target = torch.zeros(
        output.object_logits.shape, device=output.object_logits.device, dtype=output.object_logits.dtype
    )
    non_natural = (
        output.modes[index[0], index[1]] != 0 if index[0].numel() else torch.zeros(0, dtype=torch.bool, device=device)
    )
    row_idx = index[2][non_natural]
    if row_idx.numel():
        object_target[index[0][non_natural], index[1][non_natural], row_idx] = 1.0
    object_keep = output.instance_mask & output.group_mask[..., None] & (output.modes != 0)[..., None]
    object_terms = F.binary_cross_entropy_with_logits(output.object_logits, object_target, reduction="none")
    object_loss = (object_terms * object_keep.to(object_terms.dtype)).sum() / object_keep.sum().clamp_min(1)
    field_loss = field_nll[index].mean() if index[0].numel() else output.object_logits.new_zeros(())
    return {"object_loss": object_loss, "field_loss": field_loss}
