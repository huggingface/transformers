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


def masked_log_softmax(scores: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Compute stable log probabilities over the active options of each question."""
    scores = scores.float().masked_fill(~option_mask, -torch.inf)
    # Exclude the maximum before log1p to retain tiny losses and target gradients at confident predictions.
    maximum = scores.argmax(-1, keepdim=True)
    is_maximum = torch.arange(scores.shape[-1], device=scores.device) == maximum
    shifted = torch.where(is_maximum, 0.0, scores - scores.gather(-1, maximum))
    others = shifted.exp().masked_fill(is_maximum, 0.0).sum(-1, keepdim=True)
    return (shifted - torch.log1p(others)).masked_fill(~option_mask, 0.0)


def ranked_probability_score(
    probabilities: torch.Tensor, targets: torch.Tensor, num_options: torch.Tensor
) -> torch.Tensor:
    """Average the squared CDF error over the gaps between adjacent ranks."""
    difference = probabilities.cumsum(-1) - targets.cumsum(-1)
    gaps = torch.arange(probabilities.shape[-1], device=probabilities.device) < (num_options - 1)[:, None]
    return (difference.square() * gaps).sum(-1) / (num_options - 1).clamp_min(1)


def JevForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    is_score: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """AutoTrust JEV's documented KL(target || model) + ordinal RPS objective over each question's active slots.

    Source: https://huggingface.co/autotrust/JEV-9B/blob/7ac669b8fb476ba926491e3d1c39f9330d68dbed/README.md#training-details
    """
    labels = labels.to(logits.device).reshape(-1)
    num_options = num_options.to(logits.device).reshape(-1)
    soft_labels = labels.is_floating_point()
    expected_size = logits.numel() if soft_labels else num_options.numel()
    if labels.numel() != expected_size:
        raise ValueError("Expected one soft label per active option or one integer label per question.")

    valid = torch.ones_like(num_options, dtype=torch.bool) if soft_labels else labels != ignore_index
    if not soft_labels and torch.any(valid & ((labels < 0) | (labels >= num_options))):
        raise ValueError("Each integer label must select an active option of its question.")

    positions = torch.arange(num_options.max(), device=logits.device)
    option_mask = positions < num_options[:, None]
    starts = num_options.cumsum(0) - num_options
    indices = (starts[:, None] + positions).clamp_max(logits.numel() - 1)
    log_probabilities = masked_log_softmax(logits[indices], option_mask)

    if soft_labels:
        targets = labels[indices].float().clamp_min(0).masked_fill(~option_mask, 0.0)
    else:
        targets = F.one_hot(labels.masked_fill(~valid, 0).long(), num_classes=positions.numel()).float()
        targets = targets * valid[:, None]

    # Keep target entropy: soft-target CE has the same gradients, but a different loss value.
    losses = F.kl_div(log_probabilities, targets, reduction="none").sum(-1)
    rps = ranked_probability_score(log_probabilities.exp() * option_mask, targets, num_options)
    losses = losses + 0.5 * rps * is_score.to(logits.device).reshape(-1)

    if loss_weights is not None:
        losses = losses * loss_weights.to(logits.device).reshape(-1)

    # Normalize over valid questions, including when question weights are supplied.
    normalizer = valid.sum().clamp_min(1) if num_items_in_batch is None else num_items_in_batch
    normalizer = torch.as_tensor(normalizer, device=logits.device)
    return (losses * valid).sum() / normalizer
