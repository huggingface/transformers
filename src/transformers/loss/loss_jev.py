# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import torch
import torch.nn.functional as F


def JevForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    is_score: torch.Tensor,
    rps_weight: float = 0.5,
    brier_weight: float = 0.0,
    loss_weights: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
    **kwargs,
) -> torch.Tensor:
    """The KL + ordinal RPS objective of `shibing624/train-jev`, over the active options of each question."""
    logits, labels = logits.float(), labels.to(logits.device).reshape(-1)
    sizes = num_options.reshape(-1).tolist()
    soft_labels = labels.is_floating_point()
    expected_labels = logits.numel() if soft_labels else len(sizes)
    if labels.numel() != expected_labels:
        raise ValueError(f"Expected {expected_labels} labels, got {labels.numel()}.")
    targets = labels.split(sizes) if soft_labels else labels.unbind()
    is_score = is_score.reshape(-1)
    if is_score.numel() != len(sizes):
        raise ValueError("Expected one question type per question.")
    weights = logits.new_ones(len(sizes)) if loss_weights is None else loss_weights.to(logits).reshape(-1)
    if weights.numel() != len(sizes):
        raise ValueError("Expected one loss weight per question.")
    losses = []
    for index, (scores, target) in enumerate(zip(logits.split(sizes), targets)):
        if soft_labels:
            target = target.float().clamp_min(0)
        elif target == -100:
            continue
        else:
            target = F.one_hot(target.long(), num_classes=scores.numel()).to(scores.dtype)
        log_probabilities = F.log_softmax(scores, dim=-1)
        # Keep the target entropy term: soft-target CE has the same gradients, but a different loss value.
        loss = (target * (target.clamp_min(1e-12).log() - log_probabilities)).sum()
        probabilities = log_probabilities.exp()
        if rps_weight and is_score[index] and scores.numel() > 1:
            difference = probabilities.cumsum(-1) - target.cumsum(-1)
            loss = loss + rps_weight * difference[:-1].square().sum() / (scores.numel() - 1)
        if brier_weight:
            loss = loss + brier_weight * (probabilities - target).square().sum()
        losses.append(loss * weights[index])
    if not losses:
        return logits.sum() * 0.0
    normalizer = len(losses) if num_items_in_batch is None else num_items_in_batch
    if isinstance(normalizer, torch.Tensor):
        normalizer = normalizer.to(logits.device)
    return torch.stack(losses).sum() / normalizer
