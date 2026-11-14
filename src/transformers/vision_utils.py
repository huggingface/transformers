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

import itertools

import torch
import torch.nn.functional as F

from .configuration_utils import PreTrainedConfig
from .utils import logging
from .utils.generic import get_max_seqlen


logger = logging.get_logger(__name__)


def prepare_vision_inputs(grid_thw: torch.Tensor, config: PreTrainedConfig, kwargs: dict | None = None) -> dict:
    """Prepare packed Qwen-VL vision metadata on CPU, before device transfer and graph capture.

    Returns tensor indices plus Python tuples for non-Flash attention splits. Each distinct split
    layout can specialize a compiled graph; changing index values with the same layout does not.
    Caller-provided values are preserved. Window permutation and boundaries must be supplied together.
    """
    prepared = dict(kwargs) if kwargs is not None else {}
    if grid_thw.device.type != "cpu" or any(
        isinstance(value, torch.Tensor) and value.device.type != "cpu" for value in prepared.values()
    ):
        raise ValueError("Prepare vision metadata on CPU before moving the batch to the accelerator.")
    merge_size = config.spatial_merge_size
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.shape[0] == 0:
        raise ValueError("grid_thw must be a nonempty (num_images_or_videos, 3) tensor.")
    if grid_thw.dtype not in (torch.int32, torch.int64):
        raise ValueError("grid_thw must contain integer dimensions.")
    if (grid_thw <= 0).any() or (grid_thw[:, 1:] % merge_size != 0).any():
        raise ValueError("Grid dimensions must be positive and spatial dimensions divisible by spatial_merge_size.")

    def fill(key, compute):
        if prepared.get(key) is None:
            prepared[key] = compute()
        return prepared[key]

    def lengths(boundaries):
        if boundaries.ndim != 1 or boundaries.numel() < 2 or boundaries[0] != 0:
            raise ValueError("Cumulative sequence lengths must be one-dimensional and start at zero.")
        result = tuple((boundaries[1:] - boundaries[:-1]).tolist())
        if any(not isinstance(length, int) or length <= 0 for length in result):
            raise ValueError("Attention segments must have positive integer lengths.")
        if boundaries[-1] != grid_thw.prod(-1).sum():
            raise ValueError("Cumulative sequence lengths must cover all vision patches.")
        return result

    cu_seqlens = fill("cu_seqlens", lambda: get_vision_cu_seqlens(grid_thw))
    sequence_lengths = lengths(cu_seqlens)
    position_ids = fill("position_ids", lambda: get_vision_position_ids(grid_thw, merge_size))
    if position_ids.shape != (grid_thw.prod(-1).sum().item(), 2):
        raise ValueError("Vision position_ids must contain a height/width pair for each patch.")
    if fill("max_seqlen", lambda: max(sequence_lengths)) < max(sequence_lengths):
        raise ValueError("max_seqlen must cover the longest attention segment.")
    if fill("sequence_lengths", lambda: sequence_lengths) != sequence_lengths:
        raise ValueError("sequence_lengths must match cu_seqlens.")

    if getattr(config, "window_size", None) is not None:
        if (prepared.get("window_index") is None) != (prepared.get("cu_window_seqlens") is None):
            raise ValueError("Supply both window_index and cu_window_seqlens, or neither.")
        if prepared.get("window_index") is None:
            prepared["window_index"], prepared["cu_window_seqlens"] = get_vision_window_index(
                grid_thw, merge_size, config.window_size, config.patch_size
            )
        window_lengths = lengths(prepared["cu_window_seqlens"])
        if fill("max_window_seqlen", lambda: max(window_lengths)) < max(window_lengths):
            raise ValueError("max_window_seqlen must cover the longest window.")
        if fill("window_sequence_lengths", lambda: window_lengths) != window_lengths:
            raise ValueError("window_sequence_lengths must match cu_window_seqlens.")
    return prepared


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
    index: torch.Tensor, size: torch.Tensor, side: int, mode: str, align_corners: bool, padding: str = "border"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis interpolation taps into a `side`-length source table and their weights, for `index`
    target positions on an axis of length `size`. `mode` selects the kernel width — 2 taps
    (`"bilinear"`) or 4 taps (`"bicubic"`, Keys convolution kernel with `a=-0.75`). `size` may be a
    scalar or a per-element tensor (ragged batches). `padding` controls out-of-range taps: `"border"`
    clamps to the edge value (like `F.interpolate`), `"zeros"` gives them zero weight (like
    `F.grid_sample(padding_mode="zeros")`)."""
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
    raw_taps = floor.long()[:, None] + offsets
    taps = raw_taps.clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    if mode == "bilinear":
        weights = (1 - distance).clamp(min=0)  # linear hat kernel
    else:
        a = -0.75
        near = ((a + 2) * distance - (a + 3)) * distance * distance + 1
        far = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
        weights = torch.where(distance <= 1, near, far)
    if padding == "zeros":
        # Out-of-range taps were clamped to the border above; zero their weight so they contribute
        # nothing (the clamped index is then irrelevant) — `F.grid_sample(padding_mode="zeros")`.
        weights = weights * ((raw_taps >= 0) & (raw_taps <= side - 1))
    return taps, weights


def get_vision_interpolation_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    mode: str = "bilinear",
    align_corners: bool = False,
    spatial_merge_size: int = 1,
    padding: str = "border",
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

    h_taps, h_weights = _interpolation_axis_taps_weights(row, heights, side, mode, align_corners, padding)
    w_taps, w_weights = _interpolation_axis_taps_weights(col, widths, side, mode, align_corners, padding)
    n = h_taps.shape[1]  # taps per axis
    # 2D separable: outer product of the per-axis taps/weights → n*n taps per patch.
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, n * n)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, n * n)
    return indices, weights


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    align_corners: bool = True,
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


def get_multimodal_vision_position_ids(
    start_position: int,
    grid_thw: list[int, int, int] | torch.Tensor,
    temp_merge_size: int = 1,
    spatial_merge_size: int = 1,
    time_interval: float | torch.Tensor = 1.0,
    device: str | torch.device | None = None,
):
    """
    Compute 3D positional indices for vision tokens derived from a single image or video input.

    The positions are generated from the input grid defined by temporal (T), height (H), and
    width (W) dimensions. Temporal and spatial dimensions can be downscaled according to the
    merge sizes used in the vision backbone. The resulting positions are offset by `start_position`.

    Args:
        start_position (`int`):
            Offset added to all computed positional indices.
        grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
            The (T, H, W) grid representing the feature layout of the current image or video after patch embedding.
        temp_merge_size (`int`, *optional*):
            Factor by which the temporal dimension is reduced in the backbone. The temporal grid size is divided
            by this value. Defaults to 1.
        spatial_merge_size (`int`, *optional*):
            Factor by which the spatial dimensions (H and W) are reduced in the backbone. Both H and W are divided
            by this value. Defaults to 1.
        time_interval (`float` or scalar `torch.Tensor`, *optional*, defaults to 1.0):
            Spacing factor applied before quantizing temporal position indices.
        device (`str` or `torch.device`, *optional*):
            Device on which the resulting tensor is allocated. If `None`, uses the current default device.

    Returns:
        torch.LongTensor of shape (3, sequence_length):
            Positional indices for temporal, height, and width dimensions,
            flattened into sequence form and offset by `start_position`.
    """
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item() // temp_merge_size,
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )

    position_temporal = (torch.arange(llm_grid_t, device=device) * time_interval).long()
    position_height = torch.arange(llm_grid_h, device=device) + start_position
    position_width = torch.arange(llm_grid_w, device=device) + start_position

    T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
    vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
    vision_position_ids[0] += start_position  # must be after time_interval multiply
    return vision_position_ids


def get_multimodal_rope_index(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    spatial_merge_size: int = 1,
    tokens_per_second: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Qwen-style multimodal RoPE positions from token layout and grid metadata.

    This data-dependent bookkeeping can run on CPU before compiling the model.
    `tokens_per_second=None` uses unscaled temporal positions (Qwen2-VL).
    """

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    second_per_grid_ts = iter(second_per_grid_ts) if second_per_grid_ts is not None else iter([1] * input_ids.shape[1])
    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            # text == 0
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            # image == 1, video == 2
            else:
                grid_thw = next(grid_iters[modality_type])
                # Only apply temporal scaling for videos; still images have no
                # temporal dimension to space out (fixes #45325).
                if modality_type == 2 and tokens_per_second is not None:
                    time_interval = tokens_per_second * next(second_per_grid_ts)
                else:
                    time_interval = 1
                vision_position_ids = get_multimodal_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size, time_interval, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def prepare_qwen_vl_inputs(config: PreTrainedConfig, inputs: dict) -> dict:
    """Prepare a CPU prefill batch before device transfer, without constructing a model.

    Returns a new mapping with modality-prefixed vision metadata and multimodal positions.
    Tensor values can then be moved to the accelerator (for example with `BatchFeature.to`).
    Attention and feature split lengths stay as Python tuples: each distinct split layout
    may compile separately. This method is outside the compiled region and is not a decode
    input preparer. Existing eager calls do not require it.
    """
    prepared = dict(inputs)
    if prepared.get("past_key_values") is not None:
        raise ValueError("Prepare the prefill batch before creating or populating a KV cache.")
    vision_config = config.vision_config
    keys = (
        "cu_seqlens",
        "max_seqlen",
        "position_ids",
        "sequence_lengths",
        "window_index",
        "cu_window_seqlens",
        "max_window_seqlen",
        "window_sequence_lengths",
    )
    for modality, pixel_key in (("image", "pixel_values"), ("video", "pixel_values_videos")):
        grid = prepared.get(f"{modality}_grid_thw")
        if grid is None or prepared.get(pixel_key) is None:
            continue
        overrides = {}
        for key in keys:
            value = prepared.get(f"{modality}_{key}")
            if value is None and key != "position_ids":
                value = prepared.get(key)
            if value is not None:
                overrides[key] = value
        metadata = prepare_vision_inputs(grid, vision_config, overrides)
        prepared.update({f"{modality}_{key}": value for key, value in metadata.items()})
        split_key = f"{modality}_split_sizes"
        split_sizes = tuple((grid.prod(-1) // vision_config.spatial_merge_size**2).tolist())
        if prepared.get(split_key) is not None and tuple(prepared[split_key]) != split_sizes:
            raise ValueError(f"{split_key} must match the grid's merged feature lengths.")
        prepared[split_key] = split_sizes

    has_vision = prepared.get("image_grid_thw") is not None or prepared.get("video_grid_thw") is not None
    if has_vision and prepared.get("position_ids") is None and prepared.get("input_ids") is not None:
        if prepared.get("mm_token_type_ids") is None:
            raise ValueError("Preparing multimodal positions requires mm_token_type_ids when input_ids is provided.")
        rope_keys = (
            "input_ids",
            "mm_token_type_ids",
            "image_grid_thw",
            "video_grid_thw",
            "attention_mask",
            "second_per_grid_ts",
        )
        rope_inputs = {key: prepared[key] for key in rope_keys if prepared.get(key) is not None}
        if any(isinstance(value, torch.Tensor) and value.device.type != "cpu" for value in rope_inputs.values()):
            raise ValueError("Prepare multimodal positions on CPU before moving the batch to the accelerator.")
        positions, deltas = get_multimodal_rope_index(
            spatial_merge_size=vision_config.spatial_merge_size,
            tokens_per_second=getattr(vision_config, "tokens_per_second", None),
            **rope_inputs,
        )
        prepared["position_ids"] = positions
        if prepared.get("rope_deltas") is None:
            prepared["rope_deltas"] = deltas
    return prepared
