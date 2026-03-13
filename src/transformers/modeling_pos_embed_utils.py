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

"""Shared sinusoidal positional embedding utilities.

There are two canonical families of sin/cos positional embeddings used across
the library:

**Family 1 — Grid-based 2D embeddings** (`build_2d_sinusoidal_position_embedding`)
    Encode the (h, w) position of every patch in a 2D image grid.
    Used by vision encoders such as ViT-MAE, AIMv2, RT-DETR, D-FINE.
    Returns a (H*W [+1], embed_dim) tensor; callers add a batch dimension
    when needed.

**Family 2 — Coordinate-based embeddings** (`encode_sinusoidal_position_embedding`)
    Encode normalized (x, y[, w, h]) coordinates of DETR-style decoder query
    anchors.  Works for any number of input coordinates.
    Used by Conditional-DETR, DAB-DETR, LW-DETR, Grounding-DINO, and variants.

All implementations share the same underlying frequency formula:
    omega_i = 1 / temperature^(i / pos_dim),  i in [0, pos_dim)
"""

import math

import torch


def build_2d_sinusoidal_position_embedding(
    height: int,
    width: int,
    embed_dim: int = 256,
    temperature: float = 10000.0,
    cls_token: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 2D sinusoidal position embeddings for an image patch grid.

    Each (h, w) grid position gets an ``embed_dim``-dimensional vector whose
    four D/4-wide blocks are laid out as:

        ``[sin_h | cos_h | sin_w | cos_w]``

    Frequencies follow ``omega_i = 1 / temperature^(i / (embed_dim // 4))``
    for ``i`` in ``[0, embed_dim // 4)``.

    The patch sequence is in row-major (H-outer, W-inner) order, matching the
    output of ``tensor.flatten(2).transpose(1, 2)`` on a ``(B, C, H, W)``
    feature map.

    Args:
        height (`int`): Grid height in patches.
        width (`int`): Grid width in patches.
        embed_dim (`int`, *optional*, defaults to 256):
            Total embedding dimension.  Must be divisible by 4.
        temperature (`float`, *optional*, defaults to 10000.0):
            Base for the frequency decay.
        cls_token (`bool`, *optional*, defaults to `False`):
            If `True`, prepend a zero row for a CLS token, yielding shape
            ``(1 + H*W, embed_dim)``.
        device (`torch.device`, *optional*):
            Target device; defaults to CPU.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Output dtype.  Frequency arithmetic is done in float64 to avoid
            precision loss, then cast at the end.

    Returns:
        `torch.Tensor` of shape ``(height * width, embed_dim)`` or
        ``(1 + height * width, embed_dim)`` when `cls_token` is `True`.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}")

    pos_dim = embed_dim // 4
    # Float64 for the frequency computation avoids precision loss when
    # temperature^(i/pos_dim) is close to 1.
    omega = torch.arange(pos_dim, dtype=torch.float64, device=device) / pos_dim
    omega = 1.0 / temperature**omega  # (D/4,)

    grid_h = torch.arange(height, dtype=torch.float64, device=device)
    grid_w = torch.arange(width, dtype=torch.float64, device=device)
    # "ij" indexing: grid_h[i, j] = i, grid_w[i, j] = j → clear naming, H outer
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")  # (H, W) each

    emb_h = grid_h.flatten().outer(omega)  # (H*W, D/4)
    emb_w = grid_w.flatten().outer(omega)  # (H*W, D/4)

    pos_embed = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim, dtype=torch.float64, device=device), pos_embed], dim=0)

    return pos_embed.to(dtype)


def encode_sinusoidal_position_embedding(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> torch.Tensor:
    """Generate sinusoidal position embeddings from normalized anchor coordinates.

    Each coordinate in `pos_tensor` is independently encoded with an
    interleaved sin/cos pattern using ``num_pos_feats`` frequency components,
    and the per-coordinate embeddings are concatenated.

    This handles 2-D inputs ``(x, y)`` as used by Conditional-DETR and N-D
    inputs ``(x, y, w, h)`` as used by DAB-DETR / LW-DETR without any changes
    to the function signature.

    Args:
        pos_tensor (`torch.Tensor`):
            Normalized coordinates in ``[0, 1]``, shape ``(..., n_coords)``.
        num_pos_feats (`int`, *optional*, defaults to 128):
            Embedding dimension per coordinate.  The total output last
            dimension is ``n_coords * num_pos_feats``.
        temperature (`int`, *optional*, defaults to 10000):
            Base for the frequency decay.
        exchange_xy (`bool`, *optional*, defaults to `True`):
            Swap the x and y embeddings in the output so the result is ordered
            ``[pos_y, pos_x, ...]`` rather than ``[pos_x, pos_y, ...]``.
            Matches the convention used in most DETR variants.

    Returns:
        `torch.Tensor` of shape ``(..., n_coords * num_pos_feats)``, cast to
        the same dtype as `pos_tensor`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    # Split along the last axis and encode each coordinate independently
    coords = pos_tensor.unbind(-1)  # list of (...,) tensors
    embeddings = [coord[..., None] * scale / dim_t for coord in coords]  # each (..., num_pos_feats)
    embeddings = [
        torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=-1).flatten(-2) for e in embeddings
    ]  # each (..., num_pos_feats)

    if exchange_xy and len(embeddings) >= 2:
        embeddings[0], embeddings[1] = embeddings[1], embeddings[0]

    return torch.cat(embeddings, dim=-1).to(pos_tensor.dtype)
