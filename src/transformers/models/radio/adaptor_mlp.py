# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from einops import rearrange
from torch import nn

from .adaptor_base import AdaptorModuleBase


class MLP(AdaptorModuleBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_inner: int = 0,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(requires_summary_and_spatial=False)
        self.fc1 = nn.Linear(input_size, hidden_size, device=device)
        self.norm = nn.LayerNorm(hidden_size, device=device)
        self.relu = nn.ReLU()

        inner = []
        for _ in range(num_inner):
            inner.extend(
                [
                    nn.Linear(hidden_size, hidden_size, device=device),
                    nn.LayerNorm(hidden_size, device=device),
                    nn.ReLU(),
                ]
            )
        if inner:
            self.inner = nn.Sequential(*inner)
        else:
            self.inner = nn.Identity()

        self.fc2 = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.inner(x)
        x = self.fc2(x)
        return x


class MLP2(AdaptorModuleBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_inner: int = 0,
        pre_norm: bool = False,
        device: torch.device = None,
        upsample_factor: int = 1,
        upsample_rank: int = None,
        from_config: bool = False,
        **kwargs,
    ):
        super().__init__(requires_summary_and_spatial=False)

        self.pre_norm = (
            nn.Sequential(
                nn.LayerNorm(input_size),
                nn.GELU(),
            )
            if pre_norm
            else nn.Identity()
        )

        self.upsample_factor = upsample_factor
        sq_ups = upsample_factor**2

        self._real_output_dim = output_size // sq_ups

        # hidden_size *= upsample_factor
        # output_size *= (upsample_factor ** 2)

        self.fc1 = nn.Linear(input_size, hidden_size, device=device)

        blocks = []
        for _ in range(num_inner):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_size, device=device),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size, device=device),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size, device=device),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(
        self, x: torch.Tensor, images: torch.Tensor | None = None, patch_size: int | None = None
    ) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)

        if self.upsample_factor > 1:
            if images is None:
                raise ValueError("`images` cannot be `None` when the head's `upsample_factor > 1`!")
            if patch_size is None:
                raise ValueError("`patch_size` cannot be `None` when the head's `upsample_factor > 1`!")
            h, w = tuple(d // patch_size for d in images.shape[-2:])
            x = rearrange(
                x,
                "b (h w) (u1 u2 c) -> b (h u1 w u2) c",
                h=h,
                w=w,
                u1=self.upsample_factor,
                u2=self.upsample_factor,
                c=self._real_output_dim,
            )

        return x
