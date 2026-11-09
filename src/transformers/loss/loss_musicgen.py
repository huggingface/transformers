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
"""Musicgen per-codebook loss."""

import torch
from torch import nn


def count_musicgen_labels(labels: torch.Tensor, config=None, ignore_index: int = -100) -> torch.Tensor:
    """Number of labels [`MusicgenForCausalLMLoss`] sums over: those of every codebook, apart from padding."""
    return (labels.ne(ignore_index) & labels.ne(config.pad_token_id)).sum()


def MusicgenForCausalLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    Musicgen predicts one token per codebook at each step: `logits` of shape `(batch_size, num_codebooks, sequence_length,
    vocab_size)` for `labels` of shape `(batch_size, sequence_length, num_codebooks)`, where `config.pad_token_id` marks
    padding. The labels are not shifted, the decoder gets them already shifted by its encoder-decoder model. The loss is
    the cross-entropy averaged over the labels of every codebook, or summed and divided by `num_items_in_batch` when
    given.
    """
    # The decoder may prepend the conditioning hidden states: keep the last positions, those of the labels
    logits = logits[:, :, -labels.shape[1] :].reshape(-1, logits.shape[-1])
    labels = labels.masked_fill(labels == config.pad_token_id, ignore_index)
    labels = labels.transpose(1, 2).reshape(-1).to(logits.device)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss
