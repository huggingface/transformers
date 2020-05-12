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
from typing import Callable, Dict
from abc import ABC, abstractmethod

from transformers import (
    AutoConfig,
    AutoTokenizer,
    MemorySummary,
    is_tf_available,
    is_torch_available,
    start_memory_tracing,
    stop_memory_tracing,
)


if is_tf_available():
    import tensorflow as tf
    from transformers import TFAutoModel

if is_torch_available():
    import torch
    from transformers import AutoModel


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
        args: BenchmarkArguments = None
    ):
        self.args = args
        self.print_fn = self.get_print_function(args)

    @abstractmethod
    def compute(self, result_dict: Dict[str]) -> Dict[str]:
        pass

    def run(self):
        result_dict = {model_name: {} for model_name in self.args.model_names}
        self.results = self.compute(result_dict)

        if self.args.is_print:
            self.print_results()

#        if self.args.do_trace:
#            self.print_memory_trace_statistics()

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
                for slice_size in self.results[model_name]["ss"]:
                    time = self.results[model_name]["time"][batch_size][slice_size]
                    memory = self.results[model_name]["memory"][batch_size][slice_size]
                    if isinstance(time, str):
                        self.print_fn(f"\t\t{model_name}/{batch_size}/{slice_size}: " f"{time} " f"{memory}")
                    else:
                        self.print_fn(
                            f"\t\t{model_name}/{batch_size}/{slice_size}: "
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

    args: PyTorchBenchMarkArguments
    models: PreTrainedModel

    def __init__(self, args: PyTorchBenchMarkArguments, models: PretrainedModel = None):
        super().__init__(args=args)
#        config = AutoConfig.from_pretrained(model_name, torchscript=self.args.torchscript)
#        model = AutoModel.from_pretrained(model_name, config=config)
        if models is None:
            self.models = [AutoModel.from_pretrained(model_name) for model_name in self.args.model_names]
        else:
            self.models = models

    def compute(self, result):
        for c, (model, model_name) in enumerate(zip(self.models, self.args.model_names)):
            self.print_fn(f"{c + 1} / {len(self.args.model_names)}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                max_input_size = tokenizer.max_model_input_sizes[model_name]
            except:
                # some models don't have a tokenizer
                max_input_size = model.config.max_position_embeddings

            result[model_name] = {"bs": self.args.batch_sizes, "ss": self.args.slice_sizes, "time": {}, "memory": {}}
            result[model_name]["time"] = {i: {} for i in batch_sizes}
            result[model_name]["memory"] = {i: {} for i in batch_sizes}

            self.print_fn("Using model {}".format(model))
            self.print_fn("Number of all parameters {}".format(model.num_parameters()))

            for batch_size in self.args.batch_sizes:
                if self.args.fp16:
                    model.half()
                model.to(device)
                model.eval()

                for sequence_length in self.args.sequence_lengths:
                    if max_input_size is not None and slice_size > max_input_size:
                        result[model_name]["time"][batch_size][slice_size] = "N/A"
                    else:
                        sequence = torch.randint(model.config.vocab_size, (batch_size, self.args.sequence_length), device=self.device)
                        try:
                            if self.args.torchscript:
                                self.print_fn("Tracing model with sequence size {}".format(sequence.shape))
                                inference = torch.jit.trace(model, sequence)
                                inference(sequence)
                            else:
                                inference = model
                                inference(sequence)

                            if not self.args.no_memory:
                                # model.add_memory_hooks()  # Forward method tracing (only for PyTorch models)

                                # Line by line memory tracing (all code in the module `transformers`) works for all models/arbitrary code

#                                if self.args.verbose:
#                                    trace = start_memory_tracing("transformers")
#                                    inference(sequence)
#                                    summary = stop_memory_tracing(trace)
#                                    print_summary_statistics(summary)

                                # TODO: change
#                                result[model_name]["memory"][batch_size][slice_size] = str(summary.total)
                                result[model_name]["memory"][batch_size][slice_size] = memory
                            else:
                                result[model_name]["memory"][batch_size][slice_size] = "N/A"

                            if not self.args.no_speed:
                                self.print_fn("Going through model with sequence of shape".format(sequence.shape))
                                runtimes = timeit.repeat(lambda: inference(sequence), repeat=self.args.average_over, number=3)
                                average_time = sum(runtimes) / float(len(runtimes)) / 3.0
                                result[model_name]["time"][batch_size][slice_size] = average_time
                            else:
                                result[model_name]["time"][batch_size][slice_size] = "N/A"

                        except RuntimeError as e:
                            self.print_fn("Doesn't fit on GPU. {}".format(e))
                            torch.cuda.empty_cache()
                            result[model_name]["time"][batch_size][slice_size] = "N/A"
                            result[model_name]["memory"][batch_size][slice_size] = "N/A"
        return result


class TensorflowBenchmarks(Benchmarks):

    args: TensorflowBenchMarkArguments

    def __init__(self, args: TensorflowBenchMarkArguments):
        super().__init__(args=args)

    def compute(self):
        pass
