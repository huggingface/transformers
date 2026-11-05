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
    padding. The labels are not shifted, the decoder gets them already shifted by its encoder-decoder model. As in
    AudioCraft, the loss is the average over the codebooks of their cross-entropy. With `num_items_in_batch`, which counts
    the labels of every codebook, the codebook losses are summed and divided by it instead.
    """
    # The decoder may prepend the conditioning hidden states: keep the last positions, those of the labels
    logits = logits[:, :, -labels.shape[1] :]
    labels = labels.masked_fill(labels == config.pad_token_id, ignore_index)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = 0
    for codebook in range(config.num_codebooks):
        codebook_logits = logits[:, codebook].reshape(-1, logits.shape[-1])
        codebook_labels = labels[..., codebook].reshape(-1).to(codebook_logits.device)
        loss = loss + nn.functional.cross_entropy(
            codebook_logits, codebook_labels, ignore_index=ignore_index, reduction=reduction
        )

    if num_items_in_batch is None:
        return loss / config.num_codebooks
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    return loss / num_items_in_batch
