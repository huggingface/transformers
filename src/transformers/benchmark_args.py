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

from time import time
import dataclasses
from dataclasses import dataclass, field
from typing import List
import json


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class BenchmarkArguments:
    """
    BenchMarkArguments are arguments we use in our benchmark scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    models: List[str] = list_field(default=["gpt-2"], metadata={"help": "Model checkpoints to be provided to the AutoModel classes. Leave blank to benchmark the base version of all available model "})

    verbose: bool = field(default=False, metadata={"help": "Verbose memory tracing"})
    no_speed: bool = field(default=False, metadata={"help": "Don't perform speed measurments"})
    no_memory: bool = field(default=False, metadata={"help": "Don't perform memory measurments"})
    torch: bool = field(default=False, metadata={"help": "Benchmark the Pytorch version of the models"})
    torch_cuda: bool = field(default=False, metadata={"help": "Pytorch only: run on available cuda devices"})
    torchscript: bool = field(default=False, metadata={"help": "Pytorch only: trace the models using torchscript"})
    tensorflow: bool = field(default=False, metadata={"help": "Benchmark the TensorFlow version of the models. Will run on GPU if the correct dependencies are installed"})
    xla: bool = field(default=False, metadata={"help": "TensorFlow only: use XLA acceleration."})
    amp: bool = field(default=False, metadata={"help": "TensorFlow only: use automatic mixed precision acceleration."})
    fp16: bool = field(default=False, metadata={"help": "PyTorch only: use FP16 to accelerate inference."})
    keras_predict: bool = field(default=False, metadata={"help": "Whether to use model.predict instead of model() to do a forward pass."})
    save_to_csv: bool = field(default=False, metadata={"help": "Save result to a CSV file"})
    log_print: bool = field(default=False, metadata={"help": "Save all print statements in a log file"})
    csv_time_filename: str = field(default=f"time_{round(time())}.csv", metadata={"help": "CSV filename used if saving time results to csv."})
    csv_memory_filename: str = field(default=f"memory_{round(time())}.csv", metadata={"help": "CSV filename used if saving memory results to csv."})
    log_filename: str = field(default=f"log_{round(time())}.csv", metadata={"help": "Log filename used if print statements are saved in log."})
    average_over: int = field(default=3, metadata={"help": "Times an experiment will be run."})
    batch_sizes: List[int] = list_field(default=[1, 2, 4, 8], metadata={"help": "List of batch sizes for which memory and time performance will be evaluated"})
    sequence_lengths: List[int] = list_field(default=[8, 64, 128, 256, 512, 1024], metadata={"help": "List of sequence lengths for which memory and time performance will be evaluated"})

#    @property
#    def train_batch_size(self) -> int:
#        return self.per_gpu_train_batch_size * max(1, self.n_gpu)
#
#    @property
#    def eval_batch_size(self) -> int:
#        return self.per_gpu_eval_batch_size * max(1, self.n_gpu)
#
#    @cached_property
#    @torch_required
#    def _setup_devices(self) -> Tuple["torch.device", int]:
#        logger.info("PyTorch: setting up devices")
#        if self.no_cuda:
#            device = torch.device("cpu")
#            n_gpu = 0
#        elif is_tpu_available():
#            device = xm.xla_device()
#            n_gpu = 0
#        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
#            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#            n_gpu = torch.cuda.device_count()
#        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#            torch.distributed.init_process_group(backend="nccl")
#            device = torch.device("cuda", self.local_rank)
#            n_gpu = 1
#        return device, n_gpu
#
#    @property
#    @torch_required
#    def device(self) -> "torch.device":
#        return self._setup_devices[0]
#
#    @property
#    @torch_required
#    def n_gpu(self):
#        return self._setup_devices[1]

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)
