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
import torch.nn as nn


def ForCTCLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_lengths: torch.Tensor,
    blank_token_id: int,
    reduction: str = "mean",
    zero_infinity: bool = False,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    Compute the CTC (Connectionist Temporal Classification) loss.

    Args:
        logits: Logits of shape `(batch, T, vocab_size)`.
        labels: Target labels of shape `(batch, U)`. Padded positions are expected to be filled with either
            `ignore_index` or `blank_token_id` (a CTC target never contains the blank token), and are masked out.
        logit_lengths: Number of valid frames in `logits` of shape `(batch,)`.
        blank_token_id: Blank token id.
        reduction: Loss reduction method. One of `"mean"`, `"sum"`, or `"none"`. Defaults to `"mean"`.
        zero_infinity: Whether to zero infinite losses and the associated gradients. Defaults to `False`.
        ignore_index: Label value to ignore. Defaults to `-100`.

    Returns:
        Scalar loss tensor (or per-example losses if `reduction="none"`).
    """
    labels = labels.to(logits.device)
    labels_mask = (labels != ignore_index) & (labels != blank_token_id)
    target_lengths = labels_mask.sum(-1)
    flattened_targets = labels.masked_select(labels_mask)

    # ctc_loss doesn't support fp16
    log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

    with torch.backends.cudnn.flags(enabled=False):
        return nn.functional.ctc_loss(
            log_probs,
            flattened_targets,
            logit_lengths,
            target_lengths,
            blank=blank_token_id,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
