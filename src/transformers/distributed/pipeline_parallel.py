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
from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import is_torch_available

if TYPE_CHECKING:
    import torch.nn as nn

if is_torch_available():
    import torch
    import torch.nn as nn


class PPMissingLayer(nn.Identity):
    """A placeholder layer for missing layers in a pipeline parallel model."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


def apply_pipeline_parallelism(model: nn.Module, device_mesh: torch.distributed.device_mesh.DeviceMesh) -> nn.Module:
    """Naive even split of ``base_model.layers`` across PP ranks."""
    pp_size = device_mesh.size()
    if pp_size <= 1:
        return model

    pp_rank = device_mesh.get_local_rank()
    is_first_rank = pp_rank == 0
    is_last_rank = pp_rank == pp_size - 1

    base_model = getattr(model, model.base_model_prefix)
    layers = base_model.layers
    num_layers = len(layers)

    layers_per_rank = num_layers // pp_size
    start_layer = pp_rank * layers_per_rank
    end_layer = num_layers if is_last_rank else start_layer + layers_per_rank

    if not is_first_rank:
        base_model.embed_tokens = PPMissingLayer()

    for i in range(num_layers):
        if i < start_layer or i >= end_layer:
            layers[i] = PPMissingLayer()

    if not is_last_rank:
        base_model.norm = PPMissingLayer()
        model.lm_head = PPMissingLayer()

    model._pp_rank = pp_rank
    model._pp_size = pp_size
    return model


#TODO(3outeille): probably have to introduce pipeline_communicate here ?