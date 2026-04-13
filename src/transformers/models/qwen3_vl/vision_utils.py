# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Pure vision utility functions for Qwen3-VL and Qwen3.5.

All functions are standalone (no model weights) and compute data-dependent
tensors from ``grid_thw`` + config scalars. Used by both the processor
(to precompute before inference) and the vision model (as fallback).
"""

import torch

from ..qwen2_vl.vision_utils import get_cu_seqlens  # noqa: F401 — re-export


def get_rotary_pos_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Compute (row, col) position IDs for rotary embeddings in Qwen3-VL.

    Uses a different grid layout than Qwen2-VL: block-interleaved positions
    with intra-block offsets.

    Args:
        grid_thw: ``(num_images_or_videos, 3)``
        spatial_merge_size: merge block size from vision config.

    Returns:
        ``pos_ids``: ``(total_tokens, 2)`` long — (row, col) position per token.
    """
    m = spatial_merge_size
    device = grid_thw.device
    total_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw.tolist():
        num_frames, height, width = int(num_frames), int(height), int(width)
        merged_h, merged_w = height // m, width // m

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(m, device=device)
        intra_col = torch.arange(m, device=device)

        row_idx = (
            (block_rows[:, None, None, None] * m + intra_row[None, None, :, None])
            .expand(merged_h, merged_w, m, m)
            .reshape(-1)
        )
        col_idx = (
            (block_cols[None, :, None, None] * m + intra_col[None, None, None, :])
            .expand(merged_h, merged_w, m, m)
            .reshape(-1)
        )

        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    return pos_ids


def get_pos_embed_indices(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bilinear interpolation indices and weights for position embeddings.

    Args:
        grid_thw: ``(num_images_or_videos, 3)``
        num_grid_per_side: ``int(num_position_embeddings ** 0.5)`` from vision config.
        spatial_merge_size: merge block size from vision config.

    Returns:
        ``embed_indices``: ``(4, total_thw)`` long — bilinear corner indices into pos_embed table.
        ``bilinear_weights``: ``(4, total_thw)`` float — interpolation weights.
    """
    N = num_grid_per_side
    m = spatial_merge_size
    device = grid_thw.device

    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_idxs = torch.linspace(0, N - 1, h, device=device)
        w_idxs = torch.linspace(0, N - 1, w, device=device)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clamp(max=N - 1)
        w_ceil = (w_floor + 1).clamp(max=N - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        bh_f = h_floor * N
        bh_c = h_ceil * N

        raw_idx = [
            (bh_f[:, None] + w_floor[None, :]).flatten(),
            (bh_f[:, None] + w_ceil[None, :]).flatten(),
            (bh_c[:, None] + w_floor[None, :]).flatten(),
            (bh_c[:, None] + w_ceil[None, :]).flatten(),
        ]
        raw_w = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        ]

        # Compose spatial merge reorder into the indices
        h_idx = torch.arange(h, device=device).view(h // m, m)
        w_idx = torch.arange(w, device=device).view(w // m, m)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).permute(0, 2, 1, 3).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(raw_idx[i][reorder])
            weight_parts[i].append(raw_w[i][reorder])

    embed_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return embed_indices, bilinear_weights
