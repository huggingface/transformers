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


def KevForDecisionLoss(
    logits: torch.Tensor, labels: torch.Tensor, num_options: torch.Tensor, **kwargs
) -> torch.Tensor:
    """The default objective of `jaredpalmer/kev/kev/train.py::question_loss`, averaged over questions."""
    logits, labels = logits.float(), labels.to(logits.device).reshape(-1)
    sizes = num_options.reshape(-1).tolist()
    soft_labels = labels.is_floating_point()
    expected_labels = logits.numel() if soft_labels else len(sizes)
    if labels.numel() != expected_labels:
        raise ValueError(f"Expected {expected_labels} labels, got {labels.numel()}.")
    targets = labels.split(sizes) if soft_labels else labels.unbind()
    losses = []
    for scores, target in zip(logits.split(sizes), targets):
        if soft_labels:
            losses.append(-(target.float() * nn.functional.log_softmax(scores, dim=-1)).sum())
        elif target != -100:
            losses.append(nn.functional.cross_entropy(scores[None], target.long()[None]))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0
