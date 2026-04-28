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
``grid_thw`` + config scalars. They are used by vision encoders and can be
precomputed before `torch.compile` / ``torch.export`` tracing since they
use untraceable ops (``repeat_interleave``, ``.tolist()``, ``nonzero()``, loops).

Each ``get_*`` accepts an optional ``kwargs`` dict; if it contains the
precomputed tensor under the natural key (``"cu_seqlens"``, ``"position_ids"``,
…), the function pops and returns it instead of computing. Vision encoders
write ``x = get_vision_x(..., kwargs=kwargs)`` and the matching key is
removed from the caller's kwargs as a side-effect of the pop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_vision_cu_seqlens(grid_thw: torch.Tensor, *, kwargs: dict | None = None) -> torch.Tensor:
    """Get cumulative sequence lengths from vision grid info, or pop from ``kwargs`` if precomputed.

    Args:
        grid_thw: ``(num_images_or_videos, 3)`` — temporal, height, width per entry.
        kwargs: optional caller kwargs — if it contains ``"cu_seqlens"`` it is popped and returned.

    Returns:
        ``cu_seqlens``: ``(total_patches + 1,)`` int32 cumulative sequence boundaries.
    """
    if kwargs is not None and (cu_seqlens := kwargs.pop("cu_seqlens", None)) is not None:
        return cu_seqlens
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    )
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int | torch.Tensor, *, kwargs: dict | None = None
) -> torch.Tensor:
    """Get (row, col) position IDs for vision rotary embeddings, or pop from ``kwargs`` if precomputed.

    Args:
        grid_thw: ``(num_images_or_videos, 3)``
        spatial_merge_size: merge block size — either a single ``int`` (same for all images)
            or a ``(num_images_or_videos,)`` tensor (per-image).
        kwargs: optional caller kwargs — if it contains ``"position_ids"`` it is popped and returned.

    Returns:
        ``position_ids``: ``(total_tokens, 2)`` long — (row, col) position per token.
    """
    if kwargs is not None and (position_ids := kwargs.pop("position_ids", None)) is not None:
        return position_ids
    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        spatial_merge_size = torch.tensor([spatial_merge_size], device=device).expand(len(grid_thw))

    position_ids = []
    for (t, h, w), m in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        t, h, w, m = int(t), int(h), int(w), int(m)
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()

        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    return torch.cat(position_ids, dim=0)


def get_vision_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
    spatial_merge_unit: int,
    *,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get window attention indices, or pop ``"window_index"``/``"cu_window_seqlens"`` from ``kwargs`` if both precomputed.

    Args:
        grid_thw: ``(num_images_or_videos, 3)``
        spatial_merge_size: merge block size from vision config.
        window_size: window size from vision config.
        patch_size: patch size from vision config.
        spatial_merge_unit: ``spatial_merge_size ** 2``.
        kwargs: optional caller kwargs — if it contains both ``"window_index"`` and ``"cu_window_seqlens"`` they are popped and returned.

    Returns:
        ``window_index``: ``(total_tokens,)`` long — reorder indices for windowed attention.
        ``cu_window_seqlens``: ``(num_windows + 1,)`` int32 — cumulative window boundaries.
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
    *,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get bilinear interpolation indices/weights, or pop ``"bilinear_indices"``/``"bilinear_weights"`` from ``kwargs`` if both precomputed.

    Args:
        grid_thw: ``(num_images_or_videos, 3)``
        num_grid_per_side: ``int(num_position_embeddings ** 0.5)`` from vision config.
        spatial_merge_size: merge block size from vision config.
        kwargs: optional caller kwargs — if it contains both ``"bilinear_indices"`` and ``"bilinear_weights"`` they are popped and returned.

    Returns:
        ``bilinear_indices``: ``(4, total_thw)`` long — bilinear corner indices into pos_embed table.
        ``bilinear_weights``: ``(4, total_thw)`` float — interpolation weights.
    """
    if kwargs is not None:
        bilinear_indices = kwargs.pop("bilinear_indices", None)
        bilinear_weights = kwargs.pop("bilinear_weights", None)
        if bilinear_indices is not None and bilinear_weights is not None:
            return bilinear_indices, bilinear_weights
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

        h_idx = torch.arange(h, device=device).view(h // m, m)
        w_idx = torch.arange(w, device=device).view(w // m, m)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).permute(0, 2, 1, 3).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(raw_idx[i][reorder])
            weight_parts[i].append(raw_w[i][reorder])

    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights
