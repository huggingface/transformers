# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.distributions as dists
import torch.nn.functional as F


def get_token_ids_from_config(config) -> dict[str, int]:
    """Extract all token IDs from the configuration object.

    Args:
        config: Configuration object (LocateAnythingConfig or similar)

    Returns:
        Dictionary containing all token IDs
    """
    token_ids = {}

    # Get from main config
    token_ids["box_start_token_id"] = getattr(config, "box_start_token_id", 151668)
    token_ids["box_end_token_id"] = getattr(config, "box_end_token_id", 151669)
    token_ids["coord_start_token_id"] = getattr(config, "coord_start_token_id", 151677)
    token_ids["coord_end_token_id"] = getattr(config, "coord_end_token_id", 152677)
    token_ids["ref_start_token_id"] = getattr(config, "ref_start_token_id", 151672)
    token_ids["ref_end_token_id"] = getattr(config, "ref_end_token_id", 151673)
    token_ids["none_token_id"] = getattr(config, "none_token_id", 4064)

    # Get from text_config
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        token_ids["null_token_id"] = getattr(text_config, "null_token_id", 152678)
        token_ids["im_end_token_id"] = getattr(text_config, "eos_token_id", 151645)
        token_ids["switch_token_id"] = getattr(text_config, "switch_token_id", 152679)
        token_ids["default_mask_token_id"] = getattr(text_config, "text_mask_token_id", 151676)
    else:
        token_ids["null_token_id"] = 152678
        token_ids["im_end_token_id"] = 151645
        token_ids["switch_token_id"] = 152679
        token_ids["default_mask_token_id"] = 151676

    return token_ids


def top_p_logits(logits: torch.Tensor, top_p: float | None = None) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor, input_ids: torch.Tensor, repetition_penalty: float = 1.0
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Args:
        logits: Shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
        input_ids: Previously generated token ids, shape [batch_size, seq_len]
        repetition_penalty: Penalty factor. > 1.0 penalizes repetition, < 1.0 encourages it.

    Returns:
        Modified logits with repetition penalty applied.
    """
    if repetition_penalty == 1.0:
        return logits

    # Convert to 3D for vectorized computation
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)  # [B, 1, V]
        squeeze_back = True
    else:
        squeeze_back = False

    batch_size, seq_len, vocab_size = logits.shape

    # Construct [B, V] bool mask marking tokens that have appeared in each batch
    device = logits.device
    token_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    for b in range(batch_size):
        # Apply penalty only based on tokens already generated in this batch
        unique_tokens = input_ids[b].unique()
        # Prevent out-of-bounds: only keep IDs within vocab range
        valid_tokens = unique_tokens[(unique_tokens >= 0) & (unique_tokens < vocab_size)]
        if valid_tokens.numel() > 0:
            token_mask[b, valid_tokens] = True

    # Expand to [B, L, V] to align with logits
    token_mask = token_mask.unsqueeze(1).expand(-1, seq_len, -1)

    # Divide positive values by penalty, multiply negative values by penalty
    positive = logits > 0
    negative = ~positive

    # Apply penalty only at mask positions
    logits = torch.where(token_mask & positive, logits / repetition_penalty, logits)
    logits = torch.where(token_mask & negative, logits * repetition_penalty, logits)

    if squeeze_back:
        logits = logits.squeeze(1)

    return logits


def sample_tokens(
    logits: torch.Tensor,
    generated: torch.Tensor,
    token_ids: dict[str, int],
    **generate_kwargs,
):
    batch_size, seq_len, vocab_size = logits.shape

    repetition_penalty = generate_kwargs.get("repetition_penalty", 1.0)
    temperature = generate_kwargs.get("temperature", 0)
    top_p = generate_kwargs.get("top_p")
    top_k = generate_kwargs.get("top_k")

    # Apply repetition penalty based on all previously generated tokens
    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if seq_len == 1:
        return probs, confidence, x0, None

    box_avg = []
    fallback_box = torch.zeros(1, dtype=x0.dtype, device=x0.device)

    for b in range(batch_size):
        decoded_box = decode_bbox_avg(
            logits[b],
            probs[b],
            token_ids,
            keep_k=generate_kwargs.get("keep_k_avg", 4),
            generation_mode=generate_kwargs.get("generation_mode", "hybrid"),
        )
        if decoded_box is not None:
            box_avg.append(decoded_box)
        else:
            out_ref = decode_ref(logits[b], probs[b], token_ids)
            if out_ref is not None:
                box_avg.append(torch.tensor(out_ref, dtype=x0.dtype, device=x0.device))
            else:
                box_avg.append(fallback_box)

    box_avg = torch.stack(box_avg)

    return probs, confidence, x0, box_avg


def sample_tokens_ar(
    logits: torch.Tensor,
    generated: torch.Tensor,
    token_ids: dict[str, int],
    **generate_kwargs,
):
    """
    Lightweight sampling function for AR single-step sampling only.

    Args:
        logits: [batch_size, vocab_size] or [batch_size, 1, vocab_size]
        generated: [batch_size, seq_len]
    """
    # Convert to 3D for reusing repetition penalty and clipping logic
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)  # [B, 1, V]
    batch_size, seq_len, vocab_size = logits.shape
    if seq_len != 1:
        raise ValueError("sample_tokens_ar only supports single-step AR sampling (seq_len == 1).")

    repetition_penalty = generate_kwargs.get("repetition_penalty", 1.0)
    temperature = generate_kwargs.get("temperature", 0)
    top_p = generate_kwargs.get("top_p")
    top_k = generate_kwargs.get("top_k")

    # Apply repetition penalty only based on historically generated tokens
    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        # For greedy: directly take the token with maximum probability
        confidence, x0 = probs.max(dim=-1)

    # Keep interface consistent with sample_tokens: return [B, 1, V] / [B, 1] shape
    return probs, confidence, x0, None, None


def is_valid_box_frame(
    probs,
    token_ids: dict[str, int],
    start_thresh=0.6,
    end_thresh=0.2,
    topk=5,
):
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    null_token_id = token_ids["null_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]
    none_token_id = token_ids["none_token_id"]  # none

    p_start = probs[0, box_start_token_id]
    if p_start >= start_thresh:
        if (
            probs[1, none_token_id] > 0.2
            and probs[2, box_end_token_id] > 0.2
            and probs[3, null_token_id] > 0.1
            and probs[4, null_token_id] > 0.1
        ):
            return "empty_box"

    end_target_ids = torch.tensor([box_end_token_id, null_token_id, im_end_token_id], device=probs.device)
    end_score = probs[5, end_target_ids].sum()

    if end_score >= end_thresh:
        return "legal_box"

    return "illegal_box"


def decode_bbox_avg(
    logits,
    probs,
    token_ids: dict[str, int],
    keep_k=5,
    start_thresh=0.7,
    end_thresh=0.2,
    generation_mode: str = "hybrid",
):
    """
    Decode bounding box coordinates using top-k weighted average.

    Args:
        logits: Logits of shape (6, vocab_size)
        probs: Probability distribution of shape (6, vocab_size)
        token_ids: Dictionary containing all token IDs
        keep_k: Number of top-k candidate tokens to keep at each position
        start_thresh: Confidence threshold for box start token
        end_thresh: Confidence threshold for box end token

    Returns:
        Decoded bounding box coordinate list in format [box_start, x1, x2, y1, y2, box_end],
        or None if decoding fails
    """
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    none_token_id = token_ids["none_token_id"]

    device = logits.device

    box_type = is_valid_box_frame(probs, token_ids, start_thresh=start_thresh, end_thresh=end_thresh, topk=keep_k)
    if box_type == "empty_box":
        # Handle the <box>none</box> case first
        return torch.tensor(
            [
                box_start_token_id,
                none_token_id,
                box_end_token_id,
                token_ids["null_token_id"],
                token_ids["null_token_id"],
                token_ids["null_token_id"],
            ],
            dtype=torch.long,
            device=probs.device,
        )
    elif box_type == "illegal_box":
        return None

    # Extract probabilities at positions 1-4 and compute Top-K for all 4 positions at once
    pos_probs, pos_ids = torch.topk(probs[1:5], k=keep_k, dim=-1)
    mask = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    has_valid = mask.any(dim=-1)  # shape: [4]
    if not has_valid.all():
        return None  # not a box, exit...

    first_valid_idx = mask.long().argmax(dim=-1, keepdim=True)  # [4, 1]
    # Extract highest-probability valid_probs[0] and corresponding valid_ids[0]
    first_valid_probs = pos_probs.gather(-1, first_valid_idx).squeeze(-1)  # [4]
    first_valid_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)  # [4]
    if generation_mode == "hybrid":
        valid_counts = mask.sum(dim=-1)  # [4]
        # Compute max/min of valid ids: fill invalid positions with extreme values to avoid interfering with max/min
        LARGE_NUM, SMALL_NUM = 999999, -999999
        valid_ids_for_max = torch.where(mask, pos_ids, torch.tensor(SMALL_NUM, device=device))
        valid_ids_for_min = torch.where(mask, pos_ids, torch.tensor(LARGE_NUM, device=device))

        valid_max = valid_ids_for_max.max(dim=-1)[0]
        valid_min = valid_ids_for_min.min(dim=-1)[0]

        is_abnormal = (first_valid_probs < 0.9) & (valid_counts > 1) & ((valid_max - valid_min) > 60)
        # is_abnormal = (first_valid_probs < 0.7) & (valid_counts > 1) & ((valid_max - valid_min) > 80)

        # Normal positions take top-1 (first_valid_ids); abnormal positions are replaced with 0
        final_coords = torch.where(is_abnormal, torch.tensor(0, device=pos_ids.device), first_valid_ids)
    elif generation_mode == "fast":
        final_coords = first_valid_ids

    start_t = torch.tensor([box_start_token_id], dtype=final_coords.dtype, device=device)
    end_t = torch.tensor([box_end_token_id], dtype=final_coords.dtype, device=device)

    return torch.cat([start_t, final_coords, end_t])


def decode_ref(
    logits,
    probs,
    token_ids: dict[str, int],
    keep_k=5,
    start_thresh=0.6,
):
    ref_start_token_id = token_ids.get("ref_start_token_id")
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    device = probs.device
    probs.size(0)

    # 1. Check if the first position is <ref> and its probability meets start_thresh
    # Note: we directly use the probability of the ref token at position 0 for the check
    if probs[0, ref_start_token_id] < start_thresh:
        return None

    # 2. Extract Top-K probabilities and token IDs for all subsequent positions
    pos_probs, pos_ids = torch.topk(probs[1:], k=keep_k, dim=-1)  # shape: [L-1, keep_k]

    # 3. Build mask: identify coordinate tokens (<0> ~ <1000>)
    is_coord = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    # Invert: valid tokens are non-coordinate tokens
    is_valid = ~is_coord  # shape: [L-1, keep_k]

    # Ensure each position has at least one non-coordinate valid token in its Top-K
    has_valid = is_valid.any(dim=-1)  # shape: [L-1]
    if not has_valid.all():
        return None

    # 4. Get the highest-probability valid token
    # Since topk results are sorted in descending order of probability,
    # argmax returns the first index where is_valid is True, i.e., the index of the most probable valid token
    first_valid_idx = is_valid.long().argmax(dim=-1, keepdim=True)  # shape: [L-1, 1]

    # Extract the final token IDs
    final_text_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)  # shape: [L-1]

    start_t = torch.tensor([ref_start_token_id], dtype=final_text_ids.dtype, device=device)

    return torch.cat([start_t, final_text_ids])


def handle_pattern(x0, token_ids: dict[str, int], generation_mode: str = "hybrid"):
    """
    Args:
        x0: Token ID list of length 6
        token_ids: Dictionary containing all token IDs
    """
    null_token_id = token_ids["null_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    none_token_id = token_ids["none_token_id"]
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    ref_end_token_id = token_ids["ref_end_token_id"]

    x0 = x0.tolist()

    if x0[0] == null_token_id:
        return {
            "type": "im_end",
            "tokens": [im_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": True,
        }
    elif x0[0] == im_end_token_id:
        return {
            "type": "im_end",
            "tokens": [im_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": True,
        }
    elif x0[:2] == [box_start_token_id, none_token_id]:
        return {
            "type": "empty_box",
            "tokens": [box_start_token_id, none_token_id, box_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": False,
        }
    elif x0[0] == box_start_token_id:
        coord_ix = 1
        for coord in x0[1:5]:
            if coord_start_token_id <= coord <= coord_end_token_id:
                coord_ix += 1
            else:
                break

        # Standard 4-coordinate bbox: <box><x1><x2><y1><y2></box>
        if coord_ix == 5 and x0[5] == box_end_token_id:
            return {
                "type": "coord_box",
                "tokens": x0,
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        # Two-coordinate pointing: <box><x><y></box>
        # Convention: the first two coordinates are valid coord tokens, the third token is box_end.
        # Remaining positions (if any) are not part of the pattern; truncate at box_end.
        elif coord_ix == 3 and x0[3] == box_end_token_id:
            return {
                "type": "point_box",
                "tokens": x0[:4],
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        else:
            if generation_mode == "fast":
                # fast mode: treat as coord_box, stay in MTP
                return {
                    "type": "coord_box",
                    "tokens": x0,
                    "need_switch_to_ar": False,
                    "is_terminal": False,
                }
            else:
                # hybrid mode: error_box, switch to AR
                return {
                    "type": "error_box",
                    "tokens": x0[:coord_ix],
                    "need_switch_to_ar": True,
                    "is_terminal": False,
                }

    else:
        for i, token in enumerate(x0):
            if token == null_token_id:
                x0 = x0[:i]
                break

        if len(x0) >= 2 and x0[-1] == x0[-2] == ref_end_token_id:
            x0 = x0[:-1]

        return {
            "type": "ref_object",
            "tokens": x0,
            "need_switch_to_ar": False,
            "is_terminal": False,
        }
