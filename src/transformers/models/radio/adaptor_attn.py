# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from timm.models.vision_transformer import Block
from torch import nn

from .adaptor_base import AdaptorModuleBase
from .adaptor_mlp import MLP2


class AttnFDHead(AdaptorModuleBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_inner: int = 0,
        pre_norm: bool = False,
        device: torch.device = None,
        upsample_factor: int = 1,
        upsample_rank: int = 0,
        **kwargs,  # Ignore kwargs that might be to other "mlp" verions, e.g. teacher_summary_idxs
    ) -> None:
        super().__init__(requires_summary_and_spatial=False)

        self.blocks = nn.Sequential(*[Block(input_size, num_heads=16, init_values=1e-5) for _ in range(2)])
        self.mlp = MLP2(
            input_size,
            hidden_size,
            output_size,
            num_inner=0,
            pre_norm=pre_norm,
            device=device,
            upsample_factor=upsample_factor,
            upsample_rank=upsample_rank,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.blocks(x)
        x = self.mlp(x)
        return x
