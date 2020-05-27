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
""" Benchmarking the library on inference and training """

# If checking the tensors placement
# tf.debugging.set_log_device_placement(True)

import csv
import logging
import timeit
from typing import Callable, Dict, List
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    MemorySummary,
    is_tf_available,
    is_torch_available,
    start_memory_tracing,
    stop_memory_tracing,
    BenchmarkArguments,
)

from .modeling_utils import PreTrainedModel


if is_tf_available():
    import tensorflow as tf
    from .modeling_tf_auto import TFAutoModel
    from .benchmark_args import TensorflowBenchmarkArguments

if is_torch_available():
    import torch
    from .modeling_auto import AutoModel
    from .benchmark_args import  PyTorchBenchmarkArguments


class Benchmarks(ABC):
    """
    Benchmarks is a simple but feature-complete benchmarking script
    to compare memory and time performance of models in Transformers.
    """

    args: BenchmarkArguments
    print_fn: Callable[[str], None]

    @abstractmethod
    def __init__(
        self,
        args: BenchmarkArguments = None,
    ):
        self.args = args
        self.print_fn = self.get_print_function(args)

    @abstractmethod
    def train(self, model_name, batch_size, sequence_length):
        pass

    @abstractmethod
    def inference(self, model_name, batch_size, sequence_length):
        pass

    def run(self):
        result = {model_name: {} for model_name in self.args.model_names}

        for c, model_name in enumerate(self.args.model_names):
            self.print_fn(f"{c + 1} / {len(self.args.model_names)}")

            result[model_name] = {"bs": self.args.batch_sizes, "ss": self.args.sequence_lengths, "time": {}, "memory": {}}
            result[model_name]["time"] = {i: {} for i in self.args.batch_sizes}
            result[model_name]["memory"] = {i: {} for i in self.args.batch_sizes}

            for batch_size in self.args.batch_sizes:
                for sequence_length in self.args.sequence_lengths:

                        if not self.args.no_memory:
                            if self.args.inference:
                                memory = self.inferencen(model_name, batch_size, sequence_length)
                                result[model_name]["memory"][batch_size][sequence_length] = memory

                            if not self.args.no_training:
                                memory = self.train(model_name, batch_size, sequence_length)
                                result[model_name]["memory"][batch_size][sequence_length] = memory

                        if not self.args.no_speed:
                            if self.args.inference:
                                runtimes = timeit.repeat(lambda: self.inference(model_name, batch_size, sequence_length), repeat=self.args.average_over, number=3)
                                average_time = sum(runtimes) / float(len(runtimes)) / 3.0
                                result[model_name]["time"][batch_size][sequence_length] = average_time

                            if not self.args.no_training:
                                runtimes = timeit.repeat(lambda: self.train(model_name, batch_size, sequence_length), repeat=self.args.average_over, number=3)
                                average_time = sum(runtimes) / float(len(runtimes)) / 3.0
                                result[model_name]["time"][batch_size][sequence_length] = average_time

        if self.args.is_print:
            self.print_results()

        if self.args.save_to_csv:
            self.save_to_csv()

    def get_print_function(self, args):
        if args.save_print_log:
            logging.basicConfig(
                level=logging.DEBUG,
                filename=args.log_filename,
                filemode="a+",
                format="%(asctime)-15s %(levelname)-8s %(message)s",
            )

            def print_with_print_log(*args):
                logging.info(*args)
                print(*args)

            return print_with_print_log
        else:
            return print

    def print_results(self):
        self.print_fn("=========== RESULTS ===========")
        for model_name in self.args.model_names:
            self.print_fn("\t" + f"======= MODEL CHECKPOINT: {model_name} =======")
            for batch_size in self.results[model_name]["bs"]:
                self.print_fn("\t\t" + f"===== BATCH SIZE: {batch_size} =====")
                for sequence_length in self.results[model_name]["ss"]:
                    time = self.results[model_name]["time"][batch_size][sequence_length]
                    memory = self.results[model_name]["memory"][batch_size][sequence_length]
                    if isinstance(time, str):
                        self.print_fn(f"\t\t{model_name}/{batch_size}/{sequence_length}: " f"{time} " f"{memory}")
                    else:
                        self.print_fn(
                            f"\t\t{model_name}/{batch_size}/{sequence_length}: "
                            f"{(round(1000 * time) / 1000)}"
                            f"s "
                            f"{memory}"
                        )

    def print_memory_trace_statistics(self, summary: MemorySummary):
        self.print_fn(
            "\nLines by line memory consumption:\n"
            + "\n".join(
                f"{state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.sequential
            )
        )
        self.print_fn(
            "\nLines with top memory consumption:\n"
            + "\n".join(
                f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.cumulative[:6]
            )
        )
        self.print_fn(
            "\nLines with lowest memory consumption:\n"
            + "\n".join(
                f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.cumulative[-6:]
            )
        )
        self.print_fn(f"\nTotal memory increase: {summary.total}")

    def save_to_csv(self):
        with open(self.args.csv_time_filename, mode="w") as csv_time_file, open(
            self.args.csv_memory_filename, mode="w"
        ) as csv_memory_file:

            assert len(self.args.model_names) > 0, "At least 1 model should be defined, but got {}".format(self.model_names)

            fieldnames = ["model", "batch_size", "sequence_length"]
            time_writer = csv.DictWriter(csv_time_file, fieldnames=fieldnames + ["time_in_s"])
            time_writer.writeheader()
            memory_writer = csv.DictWriter(csv_memory_file, fieldnames=fieldnames + ["memory"])
            memory_writer.writeheader()

            for model_name in self.args.model_names:
                time_dict = self.results[model_name]["time"]
                memory_dict = self.results[model_name]["memory"]
                for bs in time_dict:
                    for ss in time_dict[bs]:
                        time_writer.writerow(
                            {
                                "model": model_name,
                                "batch_size": bs,
                                "sequence_length": ss,
                                "time_in_s": "{:.4f}".format(time_dict[bs][ss]),
                            }
                        )

                for bs in memory_dict:
                    for ss in time_dict[bs]:
                        memory_writer.writerow(
                            {
                                "model": model_name,
                                "batch_size": bs,
                                "sequence_length": ss,
                                "memory": memory_dict[bs][ss],
                            }
                        )


class PyTorchBenchmarks(Benchmarks):

    args: PyTorchBenchmarkArguments
    models: PreTrainedModel
    train_fn: Callable[[int, int], int]
    inference_fn: Callable[[int, int], int]

    def __init__(
        self,
        args: PyTorchBenchmarkArguments,
        models: List[PreTrainedModel] = None,
        train_fn: Callable = None,
        inference_fn: Callable = None
    ):
        super().__init__(args=args)

        if models is None:
            self.model_dict = {model_name: AutoModel.from_pretrained(model_name) for model_name in self.args.model_names}
        else:
            self.model_dict = {model_name: model for model_name, model in zip(self.args.model_names, models)}

        self.train_fn = train_fn
        self.inference_fn = inference_fn

        if inference_fn is None:
            self.inference_fn = pytorch_default_inference

    def train(self, model_name, batch_size, sequence_length):
        model = self.model_dict[model_name]
        input_ids = torch.randint(model.config.vocab_size, batch_size, sequence_length)
        return self.train_fn(model, input_ids)

    def inference(self, model_name, batch_size, sequence_length):
        model = self.model_dict[model_name]
        input_ids = torch.randint(model.config.vocab_size, batch_size, sequence_length)
        return self.inference_fn(model, input_ids)


def pytorch_default_inference(model, input_ids):
    model(input_ids)
