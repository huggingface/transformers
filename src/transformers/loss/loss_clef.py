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


def ClefForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_mask: torch.Tensor,
    is_score: torch.Tensor,
    label_smoothing: float = 0.0,
    brier_weight: float = 0.0,
    ordinal_weight: float = 0.0,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    The loss of the questions of a decision request, with `Q` questions of at most `K` options.

    Args:
        logits (`torch.Tensor` of shape `(Q, K)`): scores of the options, anything on the padded cells.
        labels (`torch.LongTensor` of shape `(Q,)` or `torch.FloatTensor` of shape `(Q, K)`): the index of the right
            option of each question (`ignore_index` to skip the question), or a distribution over its options.
        option_mask (`torch.BoolTensor` of shape `(Q, K)`): the options that exist.
        is_score (`torch.BoolTensor` of shape `(Q,)`): the score questions, whose options are ordered.
        label_smoothing: the weight of the uniform distribution over the options in the target of the cross entropy.
        brier_weight: the weight of the Brier score, the squared distance of the probabilities to the target.
        ordinal_weight: the weight of the expected distance `|i - j|` of the options to the target, on score questions.
        num_items_in_batch: the number of questions the summed cross entropy and Brier terms are divided by, instead
            of the number of questions that are not ignored.
    """
    logits = logits.float()
    questions, width = logits.shape
    if labels.is_floating_point():
        target, valid = labels.float(), torch.ones(questions, dtype=torch.bool, device=logits.device)
    else:
        valid = labels != ignore_index
        target = torch.nn.functional.one_hot(labels.clamp(min=0), width).float()
    # Padded cells and ignored questions contribute nothing: they are zeroed in every term
    cells = option_mask & valid[:, None]
    target = target * cells
    options = option_mask.sum(-1, keepdim=True).float()
    log_probs = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min).log_softmax(-1)
    probs = log_probs.exp() * cells
    log_probs = log_probs * cells

    smoothed = (
        (1 - label_smoothing) * target + label_smoothing * cells / options.clamp(min=1) if label_smoothing else target
    )
    loss = -(smoothed * log_probs).sum()
    if brier_weight:
        loss = loss + brier_weight * ((probs - target) ** 2 * cells).sum()
    normaliser = valid.sum().clamp(min=1) if num_items_in_batch is None else num_items_in_batch
    loss = loss / normaliser

    if ordinal_weight:
        positions = torch.arange(width, device=logits.device)
        distance = (positions[:, None] - positions[None]).abs().float()
        expected = ((probs @ distance) * target).sum(-1) / (options[:, 0] - 1).clamp(min=1)
        scored = is_score & valid
        loss = loss + ordinal_weight * (expected * scored).sum() / scored.sum().clamp(min=1)
    # A batch with nothing to learn from is zero, and still has a gradient
    return loss + 0.0 * logits.masked_fill(~option_mask, 0).sum()
