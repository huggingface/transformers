# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import importlib.util
from dataclasses import dataclass, field

import torch

from transformers.file_utils import cached_property, is_sagemaker_distributed_available
from transformers.training_args import TrainingArguments
from transformers.utils import logging


logger = logging.get_logger(__name__)


def is_smdistributed_available():
    return importlib.util.find_spec("smdistributed") is not None


if is_smdistributed_available():
    import smdistributed.modelparallel.torch as smp


@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    mp_parameters: str = field(
        default="", metadata={"help": "Used by the SageMaker launcher to send mp-specific args."}
    )

    def __post_init__(self):
        super().__post_init__()
        if is_smdistributed_available() and self.mp_parameters != "":
            smp.init()

    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_smdistributed_available() and self.mp_parameters != "":
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_distributed_available():
            import smdistributed.dataparallel.torch.distributed as dist

            dist.init_process_group()
            self.local_rank = dist.get_local_rank()
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

    @property
    def world_size(self):
        if is_smdistributed_available() and self.mp_parameters != "":
            return smp.dp_size()

        return super().world_size

    @property
    def place_model_on_device(self):
        return not (is_smdistributed_available() and self.mp_parameters != "")

    @property
    def _no_sync_in_gradient_accumulation(self):
        return False
