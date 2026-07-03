# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Vision utility functions for pre-computing very dynamic and
data-dependent tensors that can break model graph capturing.

All functions are standalone (no model weights) and compute tensors from
`grid_thw` + config scalars. They are used by vision encoders and can be
precomputed before `torch.compile` / `torch.export` tracing since they
use untraceable ops (`repeat_interleave`, `.tolist()`, `nonzero()`, loops).

Each `get_*` accepts an optional `kwargs` dict; if it contains the
precomputed tensor under the natural key (`"cu_seqlens"`, `"position_ids"`,
…), the function pops and returns it instead of computing. Vision encoders
write `x = get_vision_x(..., kwargs=kwargs)` and the matching key is
removed from the caller's kwargs as a side-effect of the pop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_vision_cu_seqlens(grid_thw: torch.Tensor, kwargs: dict | None = None) -> torch.Tensor:
    """Get cumulative sequence lengths from vision grid info, or pop from `kwargs` if precomputed.

    Args:
        grid_thw: `(num_images_or_videos, 3)` — temporal, height, width per entry.
        kwargs: optional caller kwargs — if it contains `"cu_seqlens"` it is popped and returned.

    Returns:
        `cu_seqlens`: `(total_patches + 1,)` int32 cumulative sequence boundaries.
    """
    if kwargs is not None and (cu_seqlens := kwargs.pop("cu_seqlens", None)) is not None:
        return cu_seqlens
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    )
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int | torch.Tensor,
    include_temporal: bool = False,
    kwargs: dict | None = None,
) -> torch.Tensor:
    """Get position IDs for vision rotary embeddings, or pop from `kwargs` if precomputed.

    Args:
        grid_thw: `(num_images_or_videos, 3)`
        spatial_merge_size: merge block size — either a single `int` (same for all images)
            or a `(num_images_or_videos,)` tensor (per-image).
        kwargs: optional caller kwargs — if it contains `"position_ids"` it is popped and returned.
        include_temporal: when ``True``, prepend a temporal-index column and return
            `(total_tokens, 3)` — for encoders whose rotary embedding rotates T/H/W axes
            (minimax_m3_vl). When ``False`` (default), return `(total_tokens, 2)` for the
            2-axis (h, w) case (qwen2_5_vl / qwen3_vl / glm4v / paddleocr_vl); the h/w
            indices are still repeated ``t`` times for video inputs.

    Returns:
        `position_ids`: `(total_tokens, 3)` long if ``include_temporal`` else `(total_tokens, 2)`,
        with the spatial indices laid out block-major over ``m×m`` spatial-merge blocks.
    """
    if kwargs is not None and (position_ids := kwargs.pop("position_ids", None)) is not None:
        return position_ids

    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        spatial_merge_size = torch.tensor([spatial_merge_size], device=device).expand(len(grid_thw))

    position_ids = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        if include_temporal:
            tpos_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            position_ids.append(torch.stack([tpos_ids, hpos_ids.repeat(t), wpos_ids.repeat(t)], dim=-1))
        else:
            position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    return torch.cat(position_ids, dim=0)


def get_vision_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get window attention indices, or pop `"window_index"`/`"cu_window_seqlens"` from `kwargs` if both precomputed.

    Args:
        grid_thw: `(num_images_or_videos, 3)`
        spatial_merge_size: merge block size from vision config.
        window_size: window size from vision config.
        patch_size: patch size from vision config.
        kwargs: optional caller kwargs — if it contains both `"window_index"` and `"cu_window_seqlens"` they are popped and returned.

    Returns:
        `window_index`: `(total_tokens,)` long — reorder indices for windowed attention.
        `cu_window_seqlens`: `(num_windows + 1,)` int32 — cumulative window boundaries.
    """
    if kwargs is not None:
        window_index = kwargs.pop("window_index", None)
        cu_window_seqlens = kwargs.pop("cu_window_seqlens", None)
        if window_index is not None and cu_window_seqlens is not None:
            return window_index, cu_window_seqlens
    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size**2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    window_index = torch.cat(window_index, dim=0)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=grid_thw.device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    return window_index, cu_window_seqlens


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get bilinear interpolation indices/weights, or pop `"bilinear_indices"`/`"bilinear_weights"` from `kwargs` if both precomputed.

    Args:
        grid_thw: `(num_images_or_videos, 3)`
        num_grid_per_side: `int(num_position_embeddings ** 0.5)` from vision config.
        spatial_merge_size: merge block size from vision config.
        kwargs: optional caller kwargs — if it contains both `"bilinear_indices"` and `"bilinear_weights"` they are popped and returned.

    Returns:
        `bilinear_indices`: `(4, total_thw)` long — bilinear corner indices into pos_embed table.
        `bilinear_weights`: `(4, total_thw)` float — interpolation weights.
    """
    if kwargs is not None:
        bilinear_indices = kwargs.pop("bilinear_indices", None)
        bilinear_weights = kwargs.pop("bilinear_weights", None)
        if bilinear_indices is not None and bilinear_weights is not None:
            return bilinear_indices, bilinear_weights
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device

    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)

        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)

        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights


def get_vision_nearest_position_ids(
    target_sizes: torch.Tensor, num_patches_per_side: int, kwargs: dict | None = None
) -> torch.Tensor:
    """Get nearest-neighbor position IDs into a `num_patches_per_side**2` 2-D table, or pop
    from `kwargs` if precomputed.

    For each image of size `(h, w)`, maps fractional grid coordinates `i/h` to the nearest
    bucket on a `num_patches_per_side` grid (via `bucketize`) and flattens to 1-D embedding
    indices, concatenated across all images. Used by NaViT-style packers (e.g. MiniCPM-V).

    Args:
        target_sizes: `(num_images, 2)` int — `(h, w)` per image.
        num_patches_per_side: side length of the learned 2-D position-embedding grid.
        kwargs: optional caller kwargs — if it contains `"position_ids"` it is popped and returned.

    Returns:
        `position_ids`: `(sum(h_i * w_i),)` long — flat indices into a `num_patches_per_side**2` table.
    """
    if kwargs is not None and (pos_ids := kwargs.pop("position_ids", None)) is not None:
        return pos_ids
    device = target_sizes.device
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device=device)
    pos_ids_list = []
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        h_coords = torch.arange(height, device=device) / height
        w_coords = torch.arange(width, device=device) / width
        bucket_h = torch.bucketize(h_coords, boundaries, right=True)
        bucket_w = torch.bucketize(w_coords, boundaries, right=True)
        pos_ids_list.append((bucket_h[:, None] * num_patches_per_side + bucket_w).flatten())
    return torch.cat(pos_ids_list)


def get_vision_merged_shape(
    target_sizes: torch.Tensor, window_kernel_size: tuple[int, int], kwargs: dict | None = None
) -> tuple[int, int]:
    """Get post-window-merge `(merged_h, merged_w)` Python ints, or pop from `kwargs` if precomputed.

    `.view()` needs Python ints, but `target_sizes[0].item()` is non-traceable. Callers must pop
    the precomputed value from `kwargs` when running under `torch.export`. Assumes uniform
    `target_sizes` across the batch (standard NaViT preprocessing output).

    Args:
        target_sizes: `(num_images, 2)` int — `(h, w)` per image.
        window_kernel_size: `(window_h, window_w)` window-attention kernel.
        kwargs: optional caller kwargs — if it contains `"merged_shape"` it is popped and returned.

    Returns:
        `(merged_h, merged_w)`: per-image grid size after window merging, as Python ints.
    """
    if kwargs is not None and (merged := kwargs.pop("merged_shape", None)) is not None:
        return merged
    window_h, window_w = window_kernel_size
    return int(target_sizes[0, 0].item()) // window_h, int(target_sizes[0, 1].item()) // window_w
