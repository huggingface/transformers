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

from typing import Callable


from .benchmark_utils import Benchmarks, start_memory_tracing, stop_memory_tracing

from transformers import (
    is_torch_available,
    PretrainedConfig,
    MODEL_MAPPING
)

if is_torch_available():
    import torch
    from .benchmark_args import PyTorchBenchmarkArguments


class PyTorchBenchmarks(Benchmarks):

    args: PyTorchBenchmarkArguments
    configs: PretrainedConfig
    train_fn: Callable[[int, int], int]
    inference_fn: Callable[[int, int], int]

    def train(self, model_name, batch_size, sequence_length, trace_memory=False):
        config = self.config_dict[model_name]
        model = MODEL_MAPPING[config.__class__](config)
        model.train()
        input_ids = torch.randint(model.config.vocab_size, batch_size, sequence_length)
        return self.train_fn(model, input_ids)

    def inference(self, model_name, batch_size, sequence_length, trace_memory=False):
        config = self.config_dict[model_name]
        model = MODEL_MAPPING[config.__class__](config)
        model.eval()
        input_ids = torch.randint(config.vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)
        if trace_memory is True:
            if self.args.trace_memory_line_by_line or self.args.n_gpu == 0:
                trace = start_memory_tracing("transformers")
            else:
                torch.cuda.emtpy_cache()

            model(input_ids)

            if self.args.trace_memory_line_by_line or self.args.n_gpu == 0:
                summary = stop_memory_tracing(trace)
                memory = summary.total
            else:
                memory = torch.cuda.max_memory_reserved()

            return memory
        else:
            model(input_ids)

        return None
