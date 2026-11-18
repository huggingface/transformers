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


def JevaForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coverage_logits: torch.Tensor,
    option_mask: torch.Tensor,
    allow_null: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Jeva's candidate cross-entropy plus coverage loss for nullable decisions.

    Source: LiQiiiii/Jeva@e2104e0720100f979f2cdd456a1cbbee96da0abd, src/jeva/finetune.py, supervised_loss.
    """
    candidates = labels[:, 1:].to(device=logits.device, dtype=torch.float32)
    log_probabilities = logits.float().masked_fill(~option_mask, 0.0)
    losses = -(candidates * log_probabilities).sum(-1)
    if allow_null is not None:
        coverage = candidates.sum(-1)
        coverage_logits = coverage_logits.float()
        coverage_loss = coverage * F.softplus(-coverage_logits) + (1 - coverage) * F.softplus(coverage_logits)
        losses = losses + coverage_loss * allow_null.to(logits.device)
    normalizer = len(losses) if num_items_in_batch is None else num_items_in_batch
    return losses.sum() / normalizer
