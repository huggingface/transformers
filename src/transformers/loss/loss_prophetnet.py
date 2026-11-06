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
"""ProphetNet n-gram loss."""

import torch
from torch import nn


def ProphetNetForCausalLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    ProphetNet predicts the next `config.ngram` tokens from every position: each predicting stream of `logits`, of shape
    `(batch_size, ngram, sequence_length, vocab_size)`, is compared with the same `labels`, with `config.eps` label
    smoothing. The labels are not shifted. With `num_items_in_batch`, the losses are summed and divided by it times the
    number of predicted n-grams, so they add up to one average over the batches of an optimizer step.
    """
    expend_targets = labels.new_zeros(config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)
    for i in range(config.ngram):
        if i > 0 and config.disable_ngram_loss:
            break
        expend_targets[i, :, :] = labels

    logits = logits.transpose(0, 1).contiguous()
    lprobs = nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction=reduction)

    if config.eps > 0.0:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
        smooth_loss = smooth_loss[non_masked_tokens]
        smooth_loss = smooth_loss.sum() if num_items_in_batch is not None else smooth_loss.mean()

        eps_i = config.eps / lprobs.size(-1)
        loss = (1.0 - config.eps) * loss + eps_i * smooth_loss

    if num_items_in_batch is not None:
        num_ngrams = 1 if config.disable_ngram_loss else config.ngram
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / (num_items_in_batch * num_ngrams)
    return loss
