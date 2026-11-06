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


def D1OmniForDecisionLoss(logits, labels, option_mask, num_items_in_batch=None, ignore_index=-100, **kwargs):
    """Plain cross-entropy adaptation from AutoModel; the released D1 code is inference-only."""
    labels = labels.to(logits.device).reshape(-1)
    scores = logits.float().masked_fill(~option_mask.bool(), float("-inf"))
    valid = labels != ignore_index
    if torch.any(valid & ~option_mask.bool().gather(1, labels.clamp_min(0)[:, None]).squeeze(1)):
        raise ValueError("A label must select a declared option.")
    # A graph-connected zero also supports batches whose labels are all ignored.
    per_question = F.cross_entropy(scores, labels, reduction="none", ignore_index=ignore_index)
    denominator = valid.sum().clamp_min(1) if num_items_in_batch is None else num_items_in_batch
    return per_question.sum() / torch.as_tensor(denominator, device=scores.device)
