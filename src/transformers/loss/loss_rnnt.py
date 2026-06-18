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

from ..utils import is_torchaudio_available, logging


logger = logging.get_logger(__name__)

if is_torchaudio_available():
    import torchaudio


def rnnt_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token_id: int,
    reduction: str = "mean_volume",
) -> torch.Tensor:
    """
    Compute standard RNN-T (RNN Transducer) loss (https://huggingface.co/papers/1211.3711).

    Thin wrapper around [`torchaudio.functional.rnnt_loss`]. torchaudio is queried with `reduction="none"` to get
    the per-sample negative log-likelihoods, and the requested reduction is applied here. The reduction names and
    formulas mirror NeMo's `RNNTLoss` (the reference implementation used to train/finetune Parakeet), so that loss
    magnitudes and gradient scaling match when finetuning other RNNT models like Parakeet:

    - `"mean_volume"`: sum of per-sample losses divided by the sum of target lengths (per-token average over the
      whole batch). This is what `nvidia/parakeet-rnnt-0.6b` is trained with (`rnnt_reduction: mean_volume`).
    - `"mean_batch"`: plain average of per-sample losses over the batch (NeMo's default).
    - `"mean"`: per-sample loss divided by its own target length, then averaged over the batch.
    - `"sum"`: sum of per-sample losses.
    - `"none"`: per-sample losses, unreduced.

    Args:
        logits: Joint token logits of shape `(batch, T, U+1, vocab_size)`.
        targets: Target labels of shape `(batch, U)`.
        logit_lengths: Encoder output lengths of shape `(batch,)`.
        target_lengths: Target lengths of shape `(batch,)`.
        blank_token_id: Blank token id.
        reduction: Loss reduction method. One of `"mean_volume"`, `"mean_batch"`, `"mean"`, `"sum"`, or `"none"`.

    Returns:
        Scalar loss tensor (or per-example losses if `reduction="none"`).

    """

    if not is_torchaudio_available():
        raise ImportError("Computing the RNN-T loss requires torchaudio. Install it with `pip install torchaudio`.")

    valid_reductions = ("mean_volume", "mean_batch", "mean", "sum", "none")
    if reduction not in valid_reductions:
        raise ValueError(
            f'Invalid reduction mode "{reduction}". Expected one of {", ".join(repr(r) for r in valid_reductions)}.'
        )

    target_lengths = target_lengths.to(logits.device)
    losses = torchaudio.functional.rnnt_loss(
        logits=logits.float().contiguous(),
        targets=targets.to(logits.device).int(),
        logit_lengths=logit_lengths.to(logits.device).int(),
        target_lengths=target_lengths.int(),
        blank=blank_token_id,
        reduction="none",
    )

    if reduction == "mean_volume":
        return losses.sum() / target_lengths.float().sum()
    elif reduction == "mean_batch":
        return losses.mean()
    elif reduction == "mean":
        return (losses / target_lengths.float()).mean()
    elif reduction == "sum":
        return losses.sum()
    return losses


def ParakeetForRNNTLoss(
    logits,
    labels,
    logit_lengths,
    label_lengths,
    blank_token_id,
    reduction="mean_volume",
    **kwargs,
):
    return rnnt_loss(
        logits=logits,
        targets=labels,
        logit_lengths=logit_lengths,
        target_lengths=label_lengths,
        blank_token_id=blank_token_id,
        reduction=reduction,
    )
