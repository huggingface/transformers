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

from .configuration_utils import PreTrainedConfig
from .utils import logging
from .utils.generic import get_max_seqlen


logger = logging.get_logger(__name__)


def get_vision_cu_seqlens(
    grid_thw: torch.Tensor, merge_temporal: bool = False, kwargs: dict | None = None
) -> torch.Tensor:
    """Get cumulative sequence lengths from vision grid info, or pop from `kwargs` if precomputed.

    Args:
        grid_thw: `(num_images_or_videos, 3)` — temporal, height, width per entry.
        merge_temporal: when `False` (default), each frame is its own attention segment (`h * w`
            per frame, `t` segments per entry — the qwen2_vl / glm4v convention). When `True`,
            the whole clip is a single segment (`t * h * w`), i.e. attention spans all frames
            jointly (the kimi_k25 convention).
        kwargs: optional caller kwargs — if it contains `"cu_seqlens"` it is popped and returned.

    Returns:
        `cu_seqlens`: `(num_segments + 1,)` int32 cumulative sequence boundaries.
    """
    if kwargs is not None and (cu_seqlens := kwargs.pop("cu_seqlens", None)) is not None:
        return cu_seqlens
    dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)


def get_vision_attention_seqlens(
    grid_thw: torch.Tensor,
    config: PreTrainedConfig,
    merge_temporal: bool = False,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, int | None]:
    """Get cumulative and maximum sequence lengths for packed vision attention. ``merge_temporal`` is
    forwarded to [`get_vision_cu_seqlens`] (``True`` for clip-level attention, e.g. kimi_k25)."""
    cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal=merge_temporal, kwargs=kwargs)
    max_seqlen = get_max_seqlen(cu_seqlens, config, kwargs=kwargs)
    return cu_seqlens, max_seqlen


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


def _interpolation_axis_taps_weights(
    index: torch.Tensor, size: torch.Tensor, side: int, mode: str, align_corners: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis interpolation taps into a `side`-length source table and their weights, for `index`
    target positions on an axis of length `size`. `mode` selects the kernel width — 2 taps
    (`"bilinear"`) or 4 taps (`"bicubic"`, Keys convolution kernel with `a=-0.75`). `size` may be a
    scalar or a per-element tensor (ragged batches)."""
    index = index.to(torch.float32)
    if align_corners:
        # Closed form of `torch.linspace(0, side-1, size)[index]` — endpoints map to 0 and side-1.
        # `clamp(min=1)` avoids a divide-by-zero when size == 1 (index is 0, so src is 0 too).
        src = index * (side - 1) / torch.clamp(size - 1, min=1)
    else:
        src = (index + 0.5) * side / size - 0.5  # half-pixel centres (align_corners=False)
    floor = torch.floor(src)
    if mode == "bilinear":
        offsets = torch.arange(0, 2, device=index.device)  # floor, floor+1
    elif mode == "bicubic":
        offsets = torch.arange(-1, 3, device=index.device)  # floor-1 .. floor+2
    else:
        raise ValueError(f"Unsupported interpolation mode {mode!r} (expected 'bilinear' or 'bicubic').")
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    if mode == "bilinear":
        weights = (1 - distance).clamp(min=0)  # linear hat kernel
    else:
        a = -0.75
        near = ((a + 2) * distance - (a + 3)) * distance * distance + 1
        far = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
        weights = torch.where(distance <= 1, near, far)
    return taps, weights


def get_vision_interpolation_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    mode: str = "bilinear",
    align_corners: bool = False,
    spatial_merge_size: int = 1,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-patch gather indices/weights that resample a square learned `(num_grid_per_side,
    num_grid_per_side)` position-embedding table to each image's `(h, w)` grid, or pop
    `"interp_indices"`/`"interp_weights"` from `kwargs` if both precomputed.

    Reproduces `F.interpolate(mode=mode, align_corners=align_corners)` as `(total_patches, n_taps)`
    indices + weights (`n_taps` = 4 for `"bilinear"`, 16 for `"bicubic"` — the Keys kernel with
    `a=-0.75`), consumed by a fused `F.embedding_bag` or a weighted `embedding` sum. Fully vectorised
    over packed patches (ragged `(h, w)` handled with `repeat_interleave`, no per-image loop), so it
    supports dynamic shapes like the other grid_thw precompute helpers. `spatial_merge_size > 1`
    emits patches in spatial-merge-block order (for encoders that consume merged tokens); `1` keeps
    raster order.

    Args:
        grid_thw: `(num_images_or_videos, 3)` — temporal, height, width per entry.
        num_grid_per_side: `int(num_position_embeddings ** 0.5)` from the vision config.
        mode: `"bilinear"` or `"bicubic"`.
        align_corners: matches the corresponding `F.interpolate` flag.
        spatial_merge_size: merge block size; `1` keeps raster patch order.
        kwargs: optional caller kwargs — if it contains both `"interp_indices"` and `"interp_weights"`
            they are popped and returned.

    Returns:
        `indices`: `(total_thw, n_taps)` long — gather indices into the flattened pos_embed table.
        `weights`: `(total_thw, n_taps)` float — interpolation weights.
    """
    if kwargs is not None:
        interp_indices = kwargs.pop("interp_indices", None)
        interp_weights = kwargs.pop("interp_weights", None)
        if interp_indices is not None and interp_weights is not None:
            return interp_indices, interp_weights

    side = num_grid_per_side
    merge = spatial_merge_size
    device = grid_thw.device

    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    # Position within a single frame's flat patch sequence (0 .. h*w-1), repeating across the t frames.
    within = (torch.arange(counts.sum(), device=device) - starts) % (heights * widths)
    # Decode `within` into (row, col): raster order when merge == 1, else spatial-merge-block order
    # (block_row, block_col, in_row, in_col) — the inverse of qwen/glm-style merge reordering.
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col

    h_taps, h_weights = _interpolation_axis_taps_weights(row, heights, side, mode, align_corners)
    w_taps, w_weights = _interpolation_axis_taps_weights(col, widths, side, mode, align_corners)
    n = h_taps.shape[1]  # taps per axis
    # 2D separable: outer product of the per-axis taps/weights → n*n taps per patch.
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, n * n)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, n * n)
    return indices, weights


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deprecated — use [`get_vision_interpolation_indices_and_weights`] with `mode="bilinear"`,
    `align_corners=True`. Returns the legacy `(4, total_thw)` layout (transpose of the unified helper's
    `(total_thw, 4)`)."""
    logger.warning_once(
        "`get_vision_bilinear_indices_and_weights` is deprecated and will be removed in v5.17. Use "
        "`get_vision_interpolation_indices_and_weights(..., mode='bilinear', align_corners=True)` "
        "instead (it returns `(total_thw, n_taps)`, the transpose of this function's output)."
    )
    if kwargs is not None:
        bilinear_indices = kwargs.pop("bilinear_indices", None)
        bilinear_weights = kwargs.pop("bilinear_weights", None)
        if bilinear_indices is not None and bilinear_weights is not None:
            return bilinear_indices, bilinear_weights
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid_thw, num_grid_per_side, mode="bilinear", align_corners=True, spatial_merge_size=spatial_merge_size
    )
    return indices.transpose(0, 1), weights.transpose(0, 1)


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
