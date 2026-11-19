# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch


def GS1ForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
    brier_weight: float = 1.0,
    num_items_in_batch: torch.Tensor | int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Mean distribution NLL plus expected multiclass Brier, one unit per question."""
    logits = logits.float()
    labels = labels.to(device=logits.device).reshape(-1)
    num_options = num_options.to(device=logits.device, dtype=torch.long).reshape(-1)
    if num_options.numel() == 0 or (num_options < 1).any() or num_options.sum() != logits.numel():
        raise ValueError("num_options must partition the flat candidate logits")
    if not labels.is_floating_point() or labels.numel() != logits.numel():
        raise ValueError("GS1 targets must provide a floating-point distribution per active candidate")
    labels = labels.float()
    if not torch.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Decision targets must be finite and nonnegative")

    losses, offset = [], 0
    for index, count in enumerate(num_options.tolist()):
        scores = logits[offset : offset + count]
        target = labels[offset : offset + count]
        if not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-6, rtol=0):
            raise ValueError("Each target distribution must sum to one")
        log_probability = scores.log_softmax(-1)
        probability = log_probability.exp()
        nll = -(target * log_probability).sum()
        brier = (probability - target).square().sum() + 1 - target.square().sum()
        weight = 1.0 if loss_weights is None else loss_weights.reshape(-1)[index].to(logits.device)
        losses.append((nll + brier_weight * brier) * weight)
        offset += count

    loss = torch.stack(losses).sum()
    if num_items_in_batch is not None:
        denominator = torch.as_tensor(num_items_in_batch, device=logits.device).clamp_min(1)
        return loss / denominator
    return loss / len(losses)


__all__ = ["GS1ForDecisionLoss"]
