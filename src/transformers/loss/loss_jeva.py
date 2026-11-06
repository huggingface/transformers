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
# Jeva: System One Models from First Principles
# Copyright 2026 Qi Li and Xinchao Wang.
import torch
import torch.nn.functional as F


def JevaForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coverage_logits: torch.Tensor,
    option_mask: torch.Tensor,
    allow_null: torch.Tensor | None = None,
    aux_loss: torch.Tensor | None = None,
    router_aux_loss_coef: float = 0.0,
    num_items_in_batch: torch.Tensor | int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Jeva's candidate cross-entropy, optional binary coverage loss, and weighted router auxiliary loss."""
    logits, coverage_logits = logits.float(), coverage_logits.float()
    candidates = labels[:, 1:].to(device=logits.device, dtype=torch.float32)
    candidate_loss = -(candidates * logits.masked_fill(~option_mask, 0.0)).sum(-1)
    coverage = candidates.sum(-1)
    coverage_loss = coverage * F.softplus(-coverage_logits) + (1 - coverage) * F.softplus(coverage_logits)
    if allow_null is None:
        allow_null = torch.zeros_like(coverage, dtype=torch.bool)
    per_example_loss = candidate_loss + coverage_loss * allow_null.to(logits.device)
    normalizer = len(per_example_loss) if num_items_in_batch is None else num_items_in_batch
    loss = per_example_loss.sum() / normalizer
    if aux_loss is not None:
        loss = loss + router_aux_loss_coef * aux_loss
    return loss
