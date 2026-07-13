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

import os
from typing import TYPE_CHECKING

from ..utils import is_torch_available, is_torch_greater_or_equal
from .utils import _ensure_torch_distributed


if TYPE_CHECKING:
    from .configuration_utils import DistributedConfig

if is_torch_available():
    import torch


def initialize_pipeline_parallelism(
    distributed_config: DistributedConfig,
):
    if not is_torch_greater_or_equal("2.5"):
        raise OSError("Pipeline parallelism with DistributedConfig requires `torch>=2.5`.")

    device_type = torch._C._get_accelerator().type
    _ensure_torch_distributed(device_type)

    world_size = torch.distributed.get_world_size()
    pp_size = distributed_config.pp_size
    if world_size != pp_size:
        raise RuntimeError(f"world_size ({world_size}) must be equal to pp_size ({pp_size})")

    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        getattr(torch, device_type).set_device(local_rank)
        device_map = torch.device(device_type, local_rank)
    else:
        device_map = torch.device(device_type)
    

    assert world_size == pp_size, f"world_size ({world_size}) must be equal to pp_size ({pp_size})"
    mesh = torch.distributed.init_device_mesh(device_type, (pp_size,), mesh_dim_names=("pp",))

    return device_map, mesh

#TODO(3outeille): probably have to introduce pipeline_communicate here ?