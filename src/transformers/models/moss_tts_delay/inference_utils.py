# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Utilities for MOSS-TTS delay-pattern inference."""

import torch
import torch.nn.functional as F


def apply_top_k(logits, top_k):
    batch_size, vocab_size = logits.shape
    top_k = min(top_k, vocab_size)
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    filtered_logits = torch.full_like(logits, float("-inf"))
    batch_indices = torch.arange(batch_size).unsqueeze(-1)
    filtered_logits[batch_indices, top_k_indices] = top_k_values
    return filtered_logits


def apply_top_p(logits, top_p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    batch_size = logits.shape[0]
    filtered_logits = logits.clone()
    for i in range(batch_size):
        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
        filtered_logits[i, indices_to_remove] = float("-inf")
    return filtered_logits


def apply_top_p_optimized(logits, top_p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits[indices_to_remove] = float("-inf")
    return logits


def apply_repetition_penalty_delay_pattern(
    logits: torch.Tensor,
    prev_tokens: torch.LongTensor,
    penalty: float,
):
    """
    logits: [B, H, V]  or [N, V]
    prev_tokens: [B, T, H] or [N, T] or [B, H]

    Apply the repetition penalty independently for each H (VQ head).
    """
    if penalty == 1.0 or prev_tokens is None:
        return logits

    # Case 1: regular [N, V] (text layer)
    if logits.dim() == 2:
        prev_tokens_flat = prev_tokens.reshape(-1)
        unique_tokens = torch.unique(prev_tokens_flat)

        token_logits = logits[:, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, unique_tokens] = token_logits
        return logits

    # Case 2: Delay Pattern audio [B, H, V]
    assert logits.dim() == 3, "Delay Pattern audio logits must be [B, H, V]"
    B, H, V = logits.shape

    for h in range(H):
        # prev_tokens_h: [B, T] or [B]
        prev_tokens_h = prev_tokens[..., h].reshape(-1)
        unique_tokens = torch.unique(prev_tokens_h)

        if unique_tokens.numel() == 0:
            continue

        token_logits = logits[:, h, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, h, unique_tokens] = token_logits

    return logits


def sample_token(
    logits,
    prev_tokens: torch.LongTensor | None = None,
    repetition_penalty: float = 1.0,
    top_p=None,
    top_k=None,
    do_sample=True,
):
    vocab_size = logits.size(-1)

    # ===== Repetition Penalty (before reshaping!) =====
    if prev_tokens is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty_delay_pattern(
            logits,
            prev_tokens,
            repetition_penalty,
        )

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    # ===== Only flatten after this, for top-k / top-p / multinomial =====
    original_shape = logits.shape
    reshaped_logits = logits.view(-1, vocab_size)

    if top_k is not None and top_k > 0:
        reshaped_logits = apply_top_k(reshaped_logits, top_k)

    if top_p is not None and top_p < 1.0:
        reshaped_logits = apply_top_p_optimized(reshaped_logits, top_p)

    probs = F.softmax(reshaped_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)

    return next_tokens.view(original_shape[:-1])


def find_last_equal_C(tensor, C):
    """
    tensor: torch.Tensor of shape [batch_size, seq_len]
    C: scalar value to match
    Returns: torch.Tensor of shape [batch_size] with last indices
    """
    mask = (tensor == C).int()  # Shape: [batch_size, seq_len], bool tensor
    flipped_mask = mask.flip(dims=[1])  # Flip along sequence dimension
    flipped_indices = flipped_mask.argmax(dim=1)  # First True in flipped
    seq_len = tensor.shape[1]
    last_indices = (seq_len - 1) - flipped_indices  # Convert to original indices

    # Optional: Handle cases with no C (set to -1), though problem assumes existence
    actual_values = tensor[torch.arange(tensor.shape[0]), last_indices]
    no_match = actual_values != C
    last_indices[no_match] = -1

    return last_indices
