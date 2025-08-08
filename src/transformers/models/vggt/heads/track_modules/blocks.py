# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Modified from https://github.com/facebookresearch/co-tracker/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bilinear_sampler
from .modules import Mlp, AttnBlock, CrossAttnBlock, ResidualBlock


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
    ):
        super().__init__()

        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn

        # Add input LayerNorm before linear projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)

        # Add output LayerNorm before final projection
        self.output_norm = nn.LayerNorm(hidden_size)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks

        if self.add_space_attn:
            self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        else:
            self.virual_tracks = None

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None):
        # Apply input LayerNorm
        input_tensor = self.input_norm(input_tensor)
        tokens = self.input_transform(input_tensor)

        init_tokens = tokens

        B, _, T, _ = tokens.shape

        if self.add_space_attn:
            virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
            tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C

            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if self.add_space_attn and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0):
                space_tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)  # B N T C -> (B T) N C
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](virtual_tokens, point_tokens, mask=mask)
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](point_tokens, virtual_tokens, mask=mask)

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C
                j += 1

        if self.add_space_attn:
            tokens = tokens[:, : N - self.num_virtual_tracks]

        tokens = tokens + init_tokens

        # Apply output LayerNorm before final projection
        tokens = self.output_norm(tokens)
        flow = self.flow_head(tokens)

        return flow, None


class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4, multiple_track_feats=False, padding_mode="zeros"):
        """
        Build a pyramid of feature maps from the input.

        fmaps: Tensor (B, S, C, H, W)
        num_levels: number of pyramid levels (each downsampled by factor 2)
        radius: search radius for sampling correlation
        multiple_track_feats: if True, split the target features per pyramid level
        padding_mode: passed to grid_sample / bilinear_sampler
        """
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.num_levels = num_levels
        self.radius = radius
        self.padding_mode = padding_mode
        self.multiple_track_feats = multiple_track_feats

        # Build pyramid: each level is half the spatial resolution of the previous
        self.fmaps_pyramid = [fmaps]  # level 0 is full resolution
        current_fmaps = fmaps
        for i in range(num_levels - 1):
            B, S, C, H, W = current_fmaps.shape
            # Merge batch & sequence dimensions
            current_fmaps = current_fmaps.reshape(B * S, C, H, W)
            # Avg pool down by factor 2
            current_fmaps = F.avg_pool2d(current_fmaps, kernel_size=2, stride=2)
            _, _, H_new, W_new = current_fmaps.shape
            current_fmaps = current_fmaps.reshape(B, S, C, H_new, W_new)
            self.fmaps_pyramid.append(current_fmaps)

        # Precompute a delta grid (of shape (2r+1, 2r+1, 2)) for sampling.
        # This grid is added to the (scaled) coordinate centroids.
        r = self.radius
        dx = torch.linspace(-r, r, 2 * r + 1, device=fmaps.device, dtype=fmaps.dtype)
        dy = torch.linspace(-r, r, 2 * r + 1, device=fmaps.device, dtype=fmaps.dtype)
        # delta: for every (dy,dx) displacement (i.e. Δx, Δy)
        self.delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1)  # shape: (2r+1, 2r+1, 2)

    def corr_sample(self, targets, coords):
        """
        Instead of storing the entire correlation pyramid, we compute each level's correlation
        volume, sample it immediately, then discard it. This saves GPU memory.

        Args:
          targets: Tensor (B, S, N, C) — features for the current targets.
          coords: Tensor (B, S, N, 2) — coordinates at full resolution.

        Returns:
          Tensor (B, S, N, L) where L = num_levels * (2*radius+1)**2 (concatenated sampled correlations)
        """
        B, S, N, C = targets.shape

        # If you have multiple track features, split them per level.
        if self.multiple_track_feats:
            targets_split = torch.split(targets, C // self.num_levels, dim=-1)

        out_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            # Get current spatial resolution H, W for this pyramid level.
            B, S, C, H, W = fmaps.shape
            # Reshape feature maps for correlation computation:
            # fmap2s: (B, S, C, H*W)
            fmap2s = fmaps.view(B, S, C, H * W)
            # Choose appropriate target features.
            fmap1 = targets_split[i] if self.multiple_track_feats else targets  # shape: (B, S, N, C)

            # Compute correlation directly
            corrs = compute_corr_level(fmap1, fmap2s, C)
            corrs = corrs.view(B, S, N, H, W)

            # Prepare sampling grid:
            # Scale down the coordinates for the current level.
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / (2**i)
            # Make sure our precomputed delta grid is on the same device/dtype.
            delta_lvl = self.delta.to(coords.device).to(coords.dtype)
            # Now the grid for grid_sample is:
            # coords_lvl = centroid_lvl + delta_lvl   (broadcasted over grid)
            coords_lvl = centroid_lvl + delta_lvl.view(1, 2 * self.radius + 1, 2 * self.radius + 1, 2)

            # Sample from the correlation volume using bilinear interpolation.
            # We reshape corrs to (B * S * N, 1, H, W) so grid_sample acts over each target.
            corrs_sampled = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W), coords_lvl, padding_mode=self.padding_mode
            )
            # The sampled output is (B * S * N, 1, 2r+1, 2r+1). Flatten the last two dims.
            corrs_sampled = corrs_sampled.view(B, S, N, -1)  # Now shape: (B, S, N, (2r+1)^2)
            out_pyramid.append(corrs_sampled)

        # Concatenate all levels along the last dimension.
        out = torch.cat(out_pyramid, dim=-1).contiguous()
        return out


def compute_corr_level(fmap1, fmap2s, C):
    # fmap1: (B, S, N, C)
    # fmap2s: (B, S, C, H*W)
    corrs = torch.matmul(fmap1, fmap2s)  # (B, S, N, H*W)
    corrs = corrs.view(fmap1.shape[0], fmap1.shape[1], fmap1.shape[2], -1)  # (B, S, N, H*W)
    return corrs / math.sqrt(C)
