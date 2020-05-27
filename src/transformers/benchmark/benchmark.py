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
"""
    Benchmarking the library on inference and training in PyTorch.
"""


import inspect
import logging
import timeit

from transformers import MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING, PretrainedConfig, is_torch_available

from .benchmark_utils import Benchmark, Memory, start_memory_tracing, stop_memory_tracing


if is_torch_available():
    import torch
    from .benchmark_args import PyTorchBenchmarkArguments


logger = logging.getLogger(__name__)


class PyTorchBenchmark(Benchmark):

    args: PyTorchBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "PyTorch"

    @property
    def framework_version(self):
        return torch.__version__

    def train(self, model_name, batch_size, sequence_length, trace_memory=False):
        try:
            config = self.config_dict[model_name]
            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)
            model.to(self.args.device)
            model.train()

            input_ids = torch.randint(
                model.config.vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device
            )

            def compute_loss_and_backprob():
                # TODO: Not all models call labels argument labels => this hack using the function signature should be corrected once all models have a common name for labels
                function_argument_names = inspect.getfullargspec(model.forward).args
                if "labels" in function_argument_names:
                    loss = model(input_ids, labels=input_ids)[0]
                elif "lm_labels" in function_argument_names:
                    loss = model(input_ids, lm_labels=input_ids)[0]
                elif "masked_lm_labels" in function_argument_names:
                    loss = model(input_ids, masked_lm_labels=input_ids)[0]
                else:
                    NotImplementedError(f"{model_name} does not seem to allow training with labels")

                loss.backward()
                model.zero_grad()

            if trace_memory is True:
                if self.args.trace_memory_line_by_line or self.args.n_gpu == 0:
                    trace = start_memory_tracing("transformers")
                else:
                    # clear cuda cache
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                # calculate loss and do backpropagation
                compute_loss_and_backprob()

                if self.args.trace_memory_line_by_line or self.args.n_gpu == 0:
                    summary = stop_memory_tracing(trace)
                    memory = summary.total
                else:
                    memory = Memory(torch.cuda.max_memory_reserved())

                return memory
            else:
                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(lambda: compute_loss_and_backprob(), repeat=self.args.repeat, number=10,)
                return min(runtimes) / 10.0
        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            return "N/A"

    def inference(self, model_name, batch_size, sequence_length, trace_memory=False):
        try:
            config = self.config_dict[model_name]
            model = MODEL_MAPPING[config.__class__](config)
            model.to(self.args.device)
            model.eval()

            input_ids = torch.randint(
                config.vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device
            )
            if trace_memory is True:
                if self.args.trace_memory_line_by_line or self.args.n_gpu == 0:
                    trace = start_memory_tracing("transformers")
                else:
                    # clear cuda cache
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "max_memory_reserved"):
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        logger.info(
                            "Please consider updating PyTorch to version 1.4 to get more accuracy on GPU memory usage"
                        )
                        torch.cuda.reset_max_memory_cached()

                model(input_ids)

                if self.args.trace_memory_line_by_line or self.args.n_gpu == 0:
                    summary = stop_memory_tracing(trace)
                    memory = summary.total
                else:
                    if hasattr(torch.cuda, "max_memory_reserved"):
                        memory = Memory(torch.cuda.max_memory_reserved())
                    else:
                        logger.info(
                            "Please consider updating PyTorch to version 1.4 to get more accuracy on GPU memory usage"
                        )
                        memory = Memory(torch.cuda.max_memory_cached())

                return memory
            else:
                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(lambda: model(input_ids), repeat=self.args.repeat, number=10,)
                return min(runtimes) / 10.0

        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            return "N/A"
