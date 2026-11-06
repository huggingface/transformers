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


def ClmForDecisionLoss(
    logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor | None = None, **kwargs
) -> torch.Tensor:
    """CLM's symmetric state-to-candidate and candidate-to-state contrastive objective."""
    labels = labels.to(logits.device)
    targets = F.one_hot(labels, logits.shape[1]).float() if labels.ndim == 1 else labels.float()
    loss_logits = logits.float()
    if group_ids is not None:
        group_ids = group_ids.to(logits.device)
        same_group = group_ids[:, None] == group_ids[None, :]
        same_group.fill_diagonal_(False)
        loss_logits = loss_logits.masked_fill(same_group, -torch.inf)
    state_log_probs = loss_logits.log_softmax(-1).masked_fill(targets == 0, 0.0)
    state_loss = -(targets * state_log_probs).sum(-1).mean()
    reverse_targets = targets.T
    positive_candidates = reverse_targets.sum(-1) > 0
    reverse_targets = reverse_targets[positive_candidates]
    reverse_targets = reverse_targets / reverse_targets.sum(-1, keepdim=True)
    action_log_probs = loss_logits.T[positive_candidates].log_softmax(-1)
    action_log_probs = action_log_probs.masked_fill(reverse_targets == 0, 0.0)
    action_loss = -(reverse_targets * action_log_probs).sum(-1).mean()
    return (state_loss + action_loss) / 2
