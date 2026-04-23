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

from ..utils import logging


logger = logging.get_logger(__name__)


def _load_tdt_kernel():
    """Try to load the TDT loss CUDA kernel from the Hub. Returns None on failure."""
    try:
        from ..integrations.hub_kernels import lazy_load_kernel

        kernel = lazy_load_kernel("tdt-loss")
        if kernel is None or not hasattr(kernel, "tdt_loss"):
            logger.warning_once("Falling back to pure PyTorch implementation.")
            return None
        return kernel
    except (ImportError, ModuleNotFoundError):
        return None
    except Exception as e:
        logger.warning_once(f"Failed to load TDT CUDA kernel: {e}. Falling back to pure PyTorch implementation.")
        return None


def tdt_loss(
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token_id: int,
    durations: list[int],
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute TDT (Token-and-Duration Transducer) loss (https://arxiv.org/abs/2304.06795).

    Ported from NeMo's `TDTLossPytorch` with anti-diagonal processing. Unlike standard RNNT loss, this loss trains both
    the token prediction head and the duration prediction head. It uses vectorized anti-diagonal processing for
    efficiency: all (t, u) pairs on each anti-diagonal t+u=n are computed in parallel as batched tensor operations.

    When the ``kernels-community/tdt-loss`` CUDA kernel is installed, it is used automatically for GPU tensors,
    Falls back to the pure PyTorch implementation otherwise.

    Args:
        token_logits: Token logits of shape `(batch, T, U+1, vocab_size+1)`.
        duration_logits: Duration logits of shape `(batch, T, U+1, num_durations)`.
        targets: Target labels of shape `(batch, U)`.
        logit_lengths: Encoder output lengths of shape `(batch,)`.
        target_lengths: Target lengths of shape `(batch,)`.
        blank_token_id: Blank token id.
        durations: List of duration values (e.g., `[0, 1, 2, 3, 4]`).
        sigma: Logit undernormalization constant (see TDT paper). Defaults to `0.0`.
        reduction: Loss reduction method. One of `"mean"`, `"sum"`, or `"none"`. Defaults to `"mean"`.

    Returns:
        Scalar loss tensor (or per-example losses if `reduction="none"`).

    """
    kernel = _load_tdt_kernel() if token_logits.is_cuda else None
    if kernel is not None and hasattr(kernel, "tdt_loss"):
        durations_t = torch.tensor(durations, dtype=torch.int32, device=token_logits.device)
        return kernel.tdt_loss(
            token_logits,
            duration_logits,
            targets,
            logit_lengths,
            target_lengths,
            durations_t,
            blank_token_id,
            sigma,
            reduction,
        )

    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f'Invalid reduction mode "{reduction}". Expected one of "mean", "sum", or "none".')

    device = token_logits.device
    batch_size, max_t, max_u, _ = token_logits.shape

    token_logits = token_logits.float()
    duration_logits = duration_logits.float()

    # Apply log-softmax to get log probabilities
    # sigma only applies to token logits (undernormalization constant from the TDT paper)
    token_log_probs = torch.log_softmax(token_logits, dim=-1) - sigma
    duration_log_probs = torch.log_softmax(duration_logits, dim=-1)

    log_alpha = torch.full((batch_size, max_t, max_u), float("-inf"), device=device)
    log_alpha[:, 0, 0] = 0.0

    # Precompute blank and label log-probs for vectorized access
    blank_log_probs = token_log_probs[:, :, :, blank_token_id]

    if max_u > 1:
        targets_expanded = targets.unsqueeze(1).expand(-1, max_t, -1)  # (batch, T, U_labels)
        label_log_probs = torch.gather(
            token_log_probs[:, :, : max_u - 1, :],  # (batch, T, U-1, vocab)
            dim=3,
            index=targets_expanded.unsqueeze(-1),
        ).squeeze(-1)  # (batch, T, U-1)

    neg_inf = torch.tensor(float("-inf"), device=device)

    # Process anti-diagonals: all (t, u) with t + u = n have no mutual dependencies
    for n in range(1, max_t + max_u - 1):
        u_start = max(0, n - max_t + 1)
        u_end = min(n + 1, max_u)
        u_indices = torch.arange(u_start, u_end, device=device)

        t_indices = n - u_indices
        all_candidates = []
        for i, dur in enumerate(durations):
            t_prev = t_indices - dur
            valid_t = t_prev >= 0
            if not valid_t.any():
                continue
            t_src = t_prev.clamp(min=0)

            # Blank arcs (dur > 0): from (t-dur, u) to (t, u)
            if dur > 0:
                contrib = (
                    log_alpha[:, t_src, u_indices]
                    + blank_log_probs[:, t_src, u_indices]
                    + duration_log_probs[:, t_src, u_indices, i]
                )
                contrib = torch.where(valid_t.unsqueeze(0), contrib, neg_inf)
                all_candidates.append(contrib)

            # Label arcs: from (t-dur, u-1) to (t, u), only if u > 0
            valid_u = u_indices > 0
            valid_both = valid_t & valid_u
            if valid_both.any():
                u_src = (u_indices - 1).clamp(min=0)
                u_src_label = u_src.clamp(max=max_u - 2) if max_u > 1 else u_src

                contrib = (
                    log_alpha[:, t_src, u_src]
                    + label_log_probs[:, t_src, u_src_label]
                    + duration_log_probs[:, t_src, u_src, i]
                )
                contrib = torch.where(valid_both.unsqueeze(0), contrib, neg_inf)
                all_candidates.append(contrib)

        if all_candidates:
            stacked = torch.stack(all_candidates, dim=0)
            log_alpha[:, t_indices, u_indices] = torch.logsumexp(stacked, dim=0)

    # Terminal probability: sum over blank arcs that reach (T, U) from (T-dur, U)
    batch_idx = torch.arange(batch_size, device=device)
    log_probs = torch.full((batch_size,), float("-inf"), device=device)
    for i, dur in enumerate(durations):
        if dur == 0:
            continue
        t_final = logit_lengths - dur
        valid = t_final >= 0
        if not valid.any():
            continue

        t_clamped = t_final.clamp(min=0)
        terminal = (
            log_alpha[batch_idx, t_clamped, target_lengths]
            + token_log_probs[batch_idx, t_clamped, target_lengths, blank_token_id]
            + duration_log_probs[batch_idx, t_clamped, target_lengths, i]
        )
        combined = torch.stack([log_probs, terminal], dim=0)
        log_probs = torch.where(valid, torch.logsumexp(combined, dim=0), log_probs)

    losses = -log_probs

    if reduction == "mean":
        return (losses / target_lengths.float()).mean()
    elif reduction == "sum":
        return losses.sum()
    return losses


def ParakeetForTDTLoss(
    token_logits,
    duration_logits,
    labels,
    logit_lengths,
    label_lengths,
    blank_token_id,
    durations,
    sigma=0.0,
    reduction="mean",
    **kwargs,
):
    device = token_logits.device
    return tdt_loss(
        token_logits=token_logits,
        duration_logits=duration_logits,
        targets=labels.to(device).int(),
        logit_lengths=logit_lengths.to(device).int(),
        target_lengths=label_lengths.to(device).int(),
        blank_token_id=blank_token_id,
        durations=durations,
        sigma=sigma,
        reduction=reduction,
    )
