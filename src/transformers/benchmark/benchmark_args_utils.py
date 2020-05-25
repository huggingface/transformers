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

    models: List[str] = list_field(
        default=[],
        metadata={
            "help": "Model checkpoints to be provided to the AutoModel classes. Leave blank to benchmark the base version of all available model "
        },
    )

    no_inference: bool = field(default=False, metadata={"help": "Don't benchmark inference of model"})
    training: bool = field(default=False, metadata={"help": "Benchmark training of model"})
    verbose: bool = field(default=False, metadata={"help": "Verbose memory tracing"})
    no_speed: bool = field(default=False, metadata={"help": "Don't perform speed measurments"})
    no_memory: bool = field(default=False, metadata={"help": "Don't perform memory measurments"})
    trace_memory_line_by_line: bool = field(default=False, metadata={"help": "Trace memory line by line"})
    save_to_csv: bool = field(default=False, metadata={"help": "Save result to a CSV file"})
    log_print: bool = field(default=False, metadata={"help": "Save all print statements in a log file"})
    csv_time_filename_inference: str = field(
        default=f"inference_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv."},
    )
    csv_memory_filename_inference: str = field(
        default=f"inference_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv."},
    )
    csv_time_filename_train: str = field(
        default=f"train_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv for training."},
    )
    csv_memory_filename_train: str = field(
        default=f"train_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv for training."},
    )
    log_filename: str = field(
        default=f"log_{round(time())}.csv",
        metadata={"help": "Log filename used if print statements are saved in log."},
    )
    average_over: int = field(default=3, metadata={"help": "Times an experiment will be run."})
    #    batch_sizes: List[int] = list_field(default=[1, 2, 4, 8], metadata={"help": "List of batch sizes for which memory and time performance will be evaluated"})
    batch_sizes: List[int] = list_field(
        default=[1], metadata={"help": "List of batch sizes for which memory and time performance will be evaluated"}
    )
    #    sequence_lengths: List[int] = list_field(default=[8, 64, 128, 256, 512, 1024], metadata={"help": "List of sequence lengths for which memory and time performance will be evaluated"})
    sequence_lengths: List[int] = list_field(
        default=[8],
        metadata={"help": "List of sequence lengths for which memory and time performance will be evaluated"},
    )

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    @property
    def model_names(self):
        if len(self.models) == 0:
            return [
                "gpt2",
                "bert-base-cased",
                "xlnet-base-cased",
                "xlm-mlm-en-2048",
                "transfo-xl-wt103",
                "openai-gpt",
                "distilbert-base-uncased",
                "distilgpt2",
                "roberta-base",
                "ctrl",
                "t5-base",
                "facebook/bart-large",
                "google/reformer-enwik8"
            ]
        else:
            return self.models


@dataclass
class TensorflowBenchmarkArguments(BenchmarkArguments):
    xla: bool = field(default=False, metadata={"help": "TensorFlow only: use XLA acceleration."})
    amp: bool = field(default=False, metadata={"help": "TensorFlow only: use automatic mixed precision acceleration."})
    keras_predict: bool = field(default=False, metadata={"help": "Whether to use model.predict instead of model() to do a forward pass."})
