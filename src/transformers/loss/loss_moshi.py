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
"""Moshi depth decoder loss."""

import torch
from torch import nn


def MoshiDepthDecoderLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    Moshi's depth decoder gets the text token first, so its position `k` predicts audio codebook `k`: the labels are not
    shifted. `config.audio_vocab_size` is the id of the padding audio code.
    """
    labels = labels.masked_fill(labels == config.audio_vocab_size, ignore_index)
    # Upcast to float to avoid potential precision issues
    logits = logits.float().reshape(-1, config.audio_vocab_size)
    labels = labels.reshape(-1).to(logits.device)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss
