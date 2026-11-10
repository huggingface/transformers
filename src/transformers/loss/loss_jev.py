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
from torch.nn import functional as F


def JevForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    is_score: torch.Tensor,
    rps_weight: float = 0.5,
    loss_weights: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """AutoTrust JEV's documented KL(target || model) + ordinal RPS objective over each question's active slots.

    Source: autotrust/JEV-9B@7ac669b8fb476ba926491e3d1c39f9330d68dbed, README.md, "Training details".
    """
    logits, labels = logits.float(), labels.to(logits.device).reshape(-1)
    num_options = num_options.to(logits.device).reshape(-1)
    positions = torch.arange(num_options.max(), device=logits.device)
    option_mask = positions[None] < num_options[:, None]
    starts = num_options.cumsum(0) - num_options
    indices = (starts[:, None] + positions).clamp_max(logits.numel() - 1)
    scores = logits[indices].masked_fill(~option_mask, -torch.inf)
    # Exclude the maximum before log1p to retain tiny losses and target gradients at confident predictions.
    # https://jonbarron.info/softmaxfix/ (softmax collapse; the separate soft-target fix needs teacher logits).
    maximum = scores.argmax(-1, keepdim=True)
    is_maximum = positions[None] == maximum
    shifted = torch.where(is_maximum, 0.0, scores - scores.gather(-1, maximum))
    others = shifted.exp().masked_fill(is_maximum, 0.0).sum(-1, keepdim=True)
    log_probabilities = (shifted - torch.log1p(others)).masked_fill(~option_mask, 0.0)
    if labels.is_floating_point():
        if labels.numel() != logits.numel():
            raise ValueError("Soft labels must contain one probability per active option.")
        targets = labels[indices].float().clamp_min(0).masked_fill(~option_mask, 0.0)
        valid = torch.ones_like(num_options, dtype=torch.bool)
    else:
        if labels.numel() != num_options.numel():
            raise ValueError("Integer labels must contain one option index per question.")
        valid = labels != ignore_index
        if torch.any(valid & ((labels < 0) | (labels >= num_options))):
            raise ValueError("Each integer label must select an active option of its question.")
        targets = F.one_hot(labels.masked_fill(~valid, 0).long(), num_classes=positions.shape[0]).float()
        targets = targets * valid[:, None]
    # Keep target entropy: soft-target CE has the same gradients, but a different loss value.
    losses = F.kl_div(log_probabilities, targets, reduction="none").sum(-1)
    if rps_weight:
        probabilities = log_probabilities.exp() * option_mask
        difference = probabilities.cumsum(-1) - targets.cumsum(-1)
        gaps = positions[None] < (num_options - 1)[:, None]
        rps = (difference.square() * gaps).sum(-1) / (num_options - 1).clamp_min(1)
        losses = losses + rps_weight * rps * is_score.to(logits.device).reshape(-1)
    if loss_weights is not None:
        losses = losses * loss_weights.to(logits.device).reshape(-1)
    # Normalize over valid questions, including when question weights are supplied.
    normalizer = valid.sum().clamp_min(1) if num_items_in_batch is None else num_items_in_batch
    normalizer = torch.as_tensor(normalizer, device=logits.device)
    return (losses * valid).sum() / normalizer
