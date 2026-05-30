# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch


# MagiAttention attn_type_map convention
FULL, CAUSAL = 0, 1


def build_magi_ranges(kv_len: int, q_len: int, block_size: int, ar_decode: bool = False, device: str = "cpu"):
    """
    Fixed strategy:
      - use_cache=True: Mask blocked_k = (kv_len - block_size - 1) column
      - causal_attn=False: Window interior is FULL (bidirectional)
      - If q_len==kv_len: Use coarse prefix version (fewer ranges)
      - Otherwise: General decode version (recompute rows expanding visible region row by row)

    Conventions:
      - K/V global length kv_len: [0, kv_len)
      - Current Q is "last q_len tokens"
      - First r=q_len-block_size rows are recomputed; last block_size rows are window
    """
    if not 0 < q_len <= kv_len:
        raise ValueError(f"Expected 0 < q_len <= kv_len, got q_len={q_len} and kv_len={kv_len}.")

    if ar_decode:
        return {
            "q_ranges": torch.tensor([[0, q_len]], dtype=torch.int32, device=device).contiguous(),
            "k_ranges": torch.tensor([[0, kv_len]], dtype=torch.int32, device=device).contiguous(),
            "attn_type_map": torch.tensor([CAUSAL], dtype=torch.int32, device=device).contiguous(),
        }

    if not 0 < block_size <= q_len <= kv_len:
        raise ValueError(
            f"Expected 0 < block_size <= q_len <= kv_len, got block_size={block_size}, q_len={q_len}, kv_len={kv_len}."
        )
    B = block_size
    r = q_len - B
    q_global_start = kv_len - q_len

    window_start_k = kv_len - B
    blocked_k = window_start_k - 1  # The column that is blocked

    q_ranges, k_ranges, types = [], [], []

    # -------- prefix (q_len == kv_len) coarse-grained --------
    if q_len == kv_len:
        prefix_len = window_start_k  # kv_len - B

        # prefix->prefix: causal
        if prefix_len > 0:
            q_ranges += [[0, prefix_len]]
            k_ranges += [[0, prefix_len]]
            types += [CAUSAL]

        # window->prefix: full, but exclude blocked_k => keys [0, blocked_k)
        if prefix_len > 0 and blocked_k > 0:
            q_ranges += [[prefix_len, kv_len]]
            k_ranges += [[0, blocked_k]]
            types += [FULL]

        # window->window: full
        q_ranges += [[prefix_len, kv_len]]
        k_ranges += [[prefix_len, kv_len]]
        types += [FULL]

        return {
            "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
            "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
            "attn_type_map": torch.tensor(types, dtype=torch.int32, device=device).contiguous(),
        }

    # -------- decode / general (q_len < kv_len) --------

    # A) Recomputed rows: expand visible key cutoff row by row (use FULL + single-row q_range for precise shape)
    for i in range(r):
        g = q_global_start + i
        q_ranges.append([i, i + 1])
        k_ranges.append([0, g + 1])  # Allow keys [0, g]
        types.append(FULL)

    # B) Window rows: allow prefix but block blocked_k; window interior is full
    q_win = [r, q_len]

    # prefix keys [0, blocked_k)
    if blocked_k > 0:
        q_ranges.append(q_win)
        k_ranges.append([0, blocked_k])
        types.append(FULL)

    # window keys [window_start_k, kv_len)
    q_ranges.append(q_win)
    k_ranges.append([window_start_k, kv_len])
    types.append(FULL)

    return {
        "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
        "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
        "attn_type_map": torch.tensor(types, dtype=torch.int32, device=device).contiguous(),
    }
