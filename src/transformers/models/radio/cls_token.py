# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torch import nn


class ClsToken(nn.Module):
    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: int | None = None,
        num_registers: int | None = None,
    ):
        super().__init__()

        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if num_registers:
                self.num_registers = num_registers
            elif register_multiple:
                self.num_registers = register_multiple - (num_tokens % register_multiple)

            scale = ndim**-0.5
            self.token = nn.Parameter(torch.randn(num_tokens + self.num_registers, ndim) * scale)
        else:
            self.token = None

        self.num_patches = self.num_tokens + self.num_registers

    def disable(self):
        self.token = None
        self.enabled = False

    def forward(self, x: torch.Tensor):
        if self.token is None:
            return x

        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat(
            [
                token,
                x,
            ],
            dim=1,
        )

        return x

    def no_weight_decay(self):
        return [
            "token",
        ]
