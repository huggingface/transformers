# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from torch.utils.data import DataLoader

from ..utils import is_torch_xla_available


def tpu_spmd_dataloader(dataloader: DataLoader):
    if is_torch_xla_available():
        import torch_xla.distributed.parallel_loader as pl

        assert isinstance(dataloader, pl.MpDeviceLoader), (
            "The dataloader must be a `torch_xla.distributed.parallel_loader.MpDeviceLoader`."
        )

        # This is to support PyTorch/XLA FSDP via SPMD.
        # Here we shard the input data's 0th dim across the fsdp axis.
        import torch_xla.distributed.spmd as xs

        sharding_spec = xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None))
        dataloader._parallel_loader_kwargs["input_sharding"] = sharding_spec
        return dataloader
    else:
        return dataloader
