# Copyright 2026 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
import torch.distributed as dist
import torch.nn.functional as F


def _normalize_mask_count(num_masks: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(num_masks)
        num_masks = num_masks / dist.get_world_size()
    return num_masks.clamp_min(1.0)


def xiaomi_lens0_mask_loss(
    logits: list[torch.Tensor],
    targets: list[torch.Tensor],
    valid_targets_mask: list[torch.Tensor] | None,
    dice_weight: float,
    bce_weight: float,
    chunk_size: int,
) -> torch.Tensor:
    r"""
    Compute the weighted Dice and sigmoid binary cross-entropy loss over valid masks.

    Args:
        logits (`list[torch.Tensor]`):
            Non-empty list of raw mask logits in input-image order. Each tensor has shape
            `(num_masks, height, width)`; mask counts and spatial dimensions may differ between images.
            Images without mask tokens can have tensors with `num_masks=0`.
        targets (`list[torch.Tensor]`):
            Binary segmentation targets with the same list length and tensor shapes as `logits`.
            Masks within each image must follow the same order as the logits.
        valid_targets_mask (`list[torch.Tensor]` or `None`):
            Boolean or binary tensors matching `logits`, with 1 indicating a supervised pixel and 0 an
            ignored pixel. Masks with no valid pixels are excluded from the mask count. If `None`, all pixels
            and masks are included.
        dice_weight (`float`):
            Weight of the Dice loss computed from sigmoid probabilities, with a smoothing constant of 1.
        bce_weight (`float`):
            Weight of the binary cross-entropy loss, averaged over valid pixels within each mask.
        chunk_size (`int`):
            Number of masks processed per chunk for each image. Logits and targets are converted to
            float per chunk to limit memory usage.

    Returns:
        `torch.Tensor` of shape `()`: Scalar float32 loss. Weighted per-mask losses are summed locally and divided
        by the valid-mask count, clamped to at least 1. When distributed training is initialized, the divisor is
        the global valid-mask count divided by the world size, also clamped to at least 1.
    """
    total_loss = 0
    num_masks = logits[0].new_zeros((), dtype=torch.float32)
    for index, (image_logits, image_targets) in enumerate(zip(logits, targets)):
        image_valid_mask = None if valid_targets_mask is None else valid_targets_mask[index]
        num_masks += image_logits.shape[0] if image_valid_mask is None else image_valid_mask.flatten(1).any(-1).sum()

        for start in range(0, image_logits.shape[0], chunk_size):
            chunk_logits = image_logits[start : start + chunk_size].float().flatten(1)
            chunk_targets = image_targets[start : start + chunk_size].float().flatten(1)
            probabilities = chunk_logits.sigmoid()

            if image_valid_mask is None:
                bce_loss = (
                    F.binary_cross_entropy_with_logits(chunk_logits, chunk_targets, reduction="mean")
                    * chunk_logits.shape[0]
                )
            else:
                chunk_valid_mask = image_valid_mask[start : start + chunk_size].float().flatten(1)
                weights = chunk_valid_mask / chunk_valid_mask.sum(-1, keepdim=True).clamp_min(1.0)
                bce_loss = F.binary_cross_entropy_with_logits(
                    chunk_logits, chunk_targets, weight=weights, reduction="sum"
                )
                probabilities = probabilities * chunk_valid_mask
                chunk_targets = chunk_targets * chunk_valid_mask

            intersection = 2 * (probabilities * chunk_targets).sum(-1)
            denominator = probabilities.sum(-1) + chunk_targets.sum(-1)
            dice_loss = (1 - (intersection + 1) / (denominator + 1)).sum()
            total_loss = total_loss + dice_weight * dice_loss + bce_weight * bce_loss

    return total_loss / _normalize_mask_count(num_masks)


def XiaomiLens0ForConditionalGenerationLoss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    mask_logits: list[torch.Tensor] | None,
    mask_labels: list[torch.Tensor] | None,
    mask_labels_valid_mask: list[torch.Tensor] | None,
    config,
    **kwargs,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    r"""
    Compute the language modeling loss and weighted mask loss for Xiaomi-Lens-0.

    Every supervised call participates in mask-count normalization, including language-only batches, which
    contribute zero masks without running the mask decoder. In distributed training, all ranks must call this
    loss with language labels or mask labels. Distributed wrappers must support unused parameters when a rank
    has no mask supervision.

    Args:
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Raw language-model logits. Also determines the device for mask-count normalization when mask labels
            are absent.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)` or `None`):
            Language-model targets. Labels are shifted internally for next-token prediction, and values of -100
            are ignored by default. Pass `None` to omit the language modeling loss.
        mask_logits (`list[torch.Tensor]` or `None`):
            Raw mask logits grouped in input-image order, each of shape `(num_masks, height, width)`.
            Required when `mask_labels` is provided; otherwise ignored.
        mask_labels (`list[torch.Tensor]` or `None`):
            Binary segmentation targets matching the list length, shapes, and mask order of `mask_logits`.
            Pass `None` to omit the mask loss. Individual image tensors may contain zero masks.
        mask_labels_valid_mask (`list[torch.Tensor]` or `None`):
            Boolean or binary pixel-validity tensors matching `mask_labels`. Zero-valued pixels are ignored;
            masks with no valid pixels are excluded from normalization. Pass `None` to supervise all pixels.
            Only used when `mask_labels` is provided.
        config ([`XiaomiLens0Config`]):
            Model configuration supplying `text_config.vocab_size`, `mask_loss_weight`,
            `mask_dice_loss_weight`, `mask_bce_loss_weight`, and `mask_loss_chunk_size`.
        **kwargs:
            Additional arguments forwarded to `ForCausalLMLoss`, such as `num_items_in_batch`, `ignore_index`,
            and pre-shifted `shift_labels`.

    Returns:
        `tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]`: A tuple containing:

        - **loss** (`torch.Tensor` of shape `()` or `None`) -- Sum of the language modeling loss and weighted
          mask loss. If only one component is supervised, returns that component. Returns `None` if both
          `labels` and `mask_labels` are `None`.
        - **lm_loss** (`torch.Tensor` of shape `()` or `None`) -- Language modeling loss returned by
          `ForCausalLMLoss`, or `None` when `labels` is absent.
        - **mask_loss** (`torch.Tensor` of shape `()` or `None`) -- Dice and binary cross-entropy loss
          multiplied by `config.mask_loss_weight`, or `None` when `mask_labels` is absent.
    """
    # Import here because loss_utils registers this function in LOSS_MAPPING.
    from .loss_utils import ForCausalLMLoss

    loss = lm_loss = mask_loss = None
    if labels is not None:
        lm_loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=config.text_config.vocab_size, **kwargs)
        loss = lm_loss

    if mask_labels is not None:
        mask_loss = config.mask_loss_weight * xiaomi_lens0_mask_loss(
            mask_logits,
            mask_labels,
            valid_targets_mask=mask_labels_valid_mask,
            dice_weight=config.mask_dice_loss_weight,
            bce_weight=config.mask_bce_loss_weight,
            chunk_size=config.mask_loss_chunk_size,
        )
        loss = mask_loss if loss is None else loss + mask_loss
    elif labels is not None:
        # Match the collective on ranks with mask supervision without connecting the decoder to the loss.
        _normalize_mask_count(logits.new_zeros((), dtype=torch.float32))
    return loss, lm_loss, mask_loss
