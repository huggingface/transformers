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
from torch import nn


def VonForDecisionLoss(
    logits: torch.Tensor, labels: torch.Tensor, brier_weight: float = 0.5, **kwargs
) -> torch.Tensor:
    """The CE + Brier objective of `wfzyx/von/training/train_option_marker.py::compute_marker_rlcd_loss`."""
    scores, labels = logits.float(), labels.to(logits.device).reshape(-1)
    if labels.is_floating_point():
        if labels.numel() != scores.numel():
            raise ValueError(f"Expected {scores.numel()} probabilities, got {labels.numel()}.")
        target = labels.float()
    else:
        if labels.numel() != 1:
            raise ValueError("Von expects one label for the question of the prompt.")
        if labels[0] == -100:
            return scores.sum() * 0.0
        target = nn.functional.one_hot(labels[0].long(), num_classes=scores.numel()).to(scores.dtype)
    probabilities = scores.softmax(dim=-1)
    # Retain the upstream epsilon: replacing this with log_softmax changes saturated predictions and their gradients.
    cross_entropy = -(target * (probabilities + 1e-8).log()).sum()
    brier = (probabilities - target).square().sum()
    return cross_entropy + brier_weight * brier
