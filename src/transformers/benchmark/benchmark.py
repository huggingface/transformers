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
