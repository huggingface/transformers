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
import json
import os
import warnings
from dataclasses import dataclass, field

import torch

from transformers.training_args import TrainingArguments
from transformers.utils import cached_property, is_sagemaker_dp_enabled, logging


logger = logging.get_logger(__name__)

# TODO: should be moved to `utils` after refactoring of SageMakerTrainer


def is_sagemaker_model_parallel_available():
    # Get the sagemaker specific mp parameters from smp_options variable.
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # Parse it and check the field "partitions" is included, it is required for model parallel.
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # Get the sagemaker specific framework parameters from mpi_options variable.
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return importlib.util.find_spec("smdistributed") is not None


if is_sagemaker_model_parallel_available():
    import smdistributed.modelparallel.torch as smp

    smp.init()


@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in SageMakerTrainer"},
    )

    def __post_init__(self):
        super().__post_init__()
        warnings.warn(
            "`SageMakerTrainingArguments` is deprecated and will be removed in v5 of Transformers. You can use "
            "`TrainingArguments` instead.",
            FutureWarning,
        )

    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_sagemaker_model_parallel_available():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401

            torch.distributed.init_process_group(backend="smddp")
            self.local_rank = int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))
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
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

    @property
    def world_size(self):
        if is_sagemaker_model_parallel_available():
            return smp.dp_size()

        return super().world_size

    @property
    def place_model_on_device(self):
        return not is_sagemaker_model_parallel_available()

    @property
    def _no_sync_in_gradient_accumulation(self):
        return False
