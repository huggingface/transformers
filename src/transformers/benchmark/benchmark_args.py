# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from dataclasses import dataclass, field
from typing import Tuple

from ..file_utils import cached_property, is_torch_available, is_torch_tpu_available, torch_required
from .benchmark_args_utils import BenchmarkArguments


if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


logger = logging.getLogger(__name__)


@dataclass
class PyTorchBenchmarkArguments(BenchmarkArguments):
    no_cuda: bool = field(default=False, metadata={"help": "Whether to run on available cuda devices"})
    torchscript: bool = field(default=False, metadata={"help": "Trace the models using torchscript"})
    no_tpu: bool = field(default=False, metadata={"help": "Whether to run on available tpu devices"})
    fp16: bool = field(default=False, metadata={"help": "Use FP16 to accelerate inference."})
    tpu_print_metrics: bool = field(default=False, metadata={"help": "Use FP16 to accelerate inference."})

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        return device, n_gpu

    @property
    @torch_required
    def device_idx(self) -> int:
        return torch.cuda.current_device()

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]
