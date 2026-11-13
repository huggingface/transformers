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


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    maximum = logits.argmax(-1, keepdim=True)
    shifted = (logits - logits.gather(-1, maximum)).scatter(-1, maximum, 0.0)
    others = shifted.exp().scatter(-1, maximum, 0.0).sum(-1, keepdim=True)
    return shifted - torch.log1p(others)


def ClmForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
    **kwargs,
) -> torch.Tensor:
    """CLM's bidirectional InfoNCE over the in-batch candidate pool.

    Source: Contrastive-LM/CLM@d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6, train/finetune.py, _clm_loss and infonce_loss.
    """
    logits = logits.float()
    labels = labels.to(logits.device)
    targets = F.one_hot(labels.long(), logits.shape[-1]).float() if labels.ndim == 1 else labels.float()
    if group_ids is not None:
        group_ids = group_ids.to(logits.device)
        same_group = group_ids[:, None] == group_ids[None, :]
        same_group.fill_diagonal_(False)
        logits = logits.masked_fill(same_group, -torch.inf)
    state_log_probabilities = _log_softmax(logits).masked_fill(targets == 0, 0.0)
    state_loss = -(targets * state_log_probabilities).sum(-1).mean()
    reverse_targets = targets.T
    positive_candidates = reverse_targets.sum(-1) > 0
    reverse_targets = reverse_targets[positive_candidates]
    reverse_targets = reverse_targets / reverse_targets.sum(-1, keepdim=True)
    action_log_probabilities = _log_softmax(logits.T[positive_candidates]).masked_fill(reverse_targets == 0, 0.0)
    action_loss = -(reverse_targets * action_log_probabilities).sum(-1).mean()
    loss = (state_loss + action_loss) / 2
    if num_items_in_batch is not None:
        loss = loss * logits.shape[0] / num_items_in_batch
    return loss
