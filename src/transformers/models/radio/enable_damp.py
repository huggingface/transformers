# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from torch import nn
from torch.nn.utils import parametrize


# For now, don't do anything
class DAMP(nn.Identity):
    def __init__(self, std: float):
        super().__init__()
        self.std = std


def enable_damp(model: nn.Module, std: float):
    if isinstance(model, (list, tuple)):
        for m in model:
            enable_damp(m, std)
        return

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parametrize.register_parametrization(module, "weight", DAMP(std))


def configure_damp_from_args(model: nn.Module, args):
    damp = getattr(args, "damp", None)
    if damp:
        enable_damp(model, damp)
