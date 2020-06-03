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

import dataclasses
import json
from dataclasses import dataclass, field
from time import time
from typing import List


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

    models: List[str] = list_field(
        default=[],
        metadata={
            "help": "Model checkpoints to be provided to the AutoModel classes. Leave blank to benchmark the base version of all available models"
        },
    )

    batch_sizes: List[int] = list_field(
        default=[8], metadata={"help": "List of batch sizes for which memory and time performance will be evaluated"}
    )

    sequence_lengths: List[int] = list_field(
        default=[8, 32, 128, 512],
        metadata={"help": "List of sequence lengths for which memory and time performance will be evaluated"},
    )

    no_inference: bool = field(default=False, metadata={"help": "Don't benchmark inference of model"})
    training: bool = field(default=False, metadata={"help": "Benchmark training of model"})
    verbose: bool = field(default=False, metadata={"help": "Verbose memory tracing"})
    no_speed: bool = field(default=False, metadata={"help": "Don't perform speed measurments"})
    no_memory: bool = field(default=False, metadata={"help": "Don't perform memory measurments"})
    trace_memory_line_by_line: bool = field(default=False, metadata={"help": "Trace memory line by line"})
    save_to_csv: bool = field(default=False, metadata={"help": "Save result to a CSV file"})
    log_print: bool = field(default=False, metadata={"help": "Save all print statements in a log file"})
    no_env_print: bool = field(default=False, metadata={"help": "Don't print environment information"})
    inference_time_csv_file: str = field(
        default=f"inference_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv."},
    )
    inference_memory_csv_file: str = field(
        default=f"inference_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv."},
    )
    train_time_csv_file: str = field(
        default=f"train_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv for training."},
    )
    train_memory_csv_file: str = field(
        default=f"train_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv for training."},
    )
    env_info_csv_file: str = field(
        default=f"env_info_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving environment information."},
    )
    log_filename: str = field(
        default=f"log_{round(time())}.csv",
        metadata={"help": "Log filename used if print statements are saved in log."},
    )
    repeat: int = field(default=3, metadata={"help": "Times an experiment will be run."})

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    @property
    def model_names(self):
        return self.models
