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


import logging
import timeit

from transformers import (
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    PretrainedConfig,
    is_torch_available,
    is_torch_tpu_available,
)

from .benchmark_utils import Benchmark, Memory, measure_peak_memory_cpu, start_memory_tracing, stop_memory_tracing


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

            if self.args.torchscript:
                config.torchscript = True

            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)
            model.to(self.args.device)
            model.train()

            # encoder-decoder has vocab size saved differently
            vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
            input_ids = torch.randint(
                vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device
            )

            if self.args.torchscript:
                raise NotImplementedError("Training for torchscript is currently not implemented")
            else:
                train_model = model

            def compute_loss_and_backprob_encoder():
                loss = train_model(input_ids, labels=input_ids)[0]
                loss.backward()
                train_model.zero_grad()

            def compute_loss_and_backprob_encoder_decoder():
                loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
                loss.backward()
                train_model.zero_grad()

            _train = (
                compute_loss_and_backprob_encoder_decoder
                if config.is_encoder_decoder
                else compute_loss_and_backprob_encoder
            )

            if trace_memory is True:
                if self.args.trace_memory_line_by_line:
                    trace = start_memory_tracing("transformers")

                if self.args.n_gpu > 0:
                    # gpu
                    # clear gpu cache
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "max_memory_reserved"):
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        logger.info(
                            "Please consider updating PyTorch to version 1.4 to get more accuracy on GPU memory usage"
                        )
                        torch.cuda.reset_max_memory_cached()

                    # calculate loss and do backpropagation
                    _train()
                elif not self.args.no_tpu and is_torch_tpu_available():
                    # tpu
                    raise NotImplementedError(
                        "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.no_memory=True`"
                    )
                else:
                    # cpu
                    memory_bytes = measure_peak_memory_cpu(_train)
                    memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

                if self.args.trace_memory_line_by_line:
                    summary = stop_memory_tracing(trace)
                else:
                    summary = None

                if self.args.n_gpu > 0:
                    # gpu
                    if hasattr(torch.cuda, "max_memory_reserved"):
                        memory = Memory(torch.cuda.max_memory_reserved())
                    else:
                        logger.info(
                            "Please consider updating PyTorch to version 1.4 to get more accuracy on GPU memory usage"
                        )
                        memory = Memory(torch.cuda.max_memory_reserved())

                return memory, summary
            else:
                if (not self.args.no_tpu and is_torch_tpu_available()) or self.args.torchscript:
                    # run additional 10 times to stabilize compilation for tpu and torchscript
                    logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                    timeit.repeat(
                        _train, repeat=1, number=5,
                    )

                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(_train, repeat=self.args.repeat, number=10,)

                if not self.args.no_tpu and is_torch_tpu_available() and self.args.tpu_print_metrics:
                    import torch_xla.debug.metrics as met

                    self.print_fn(met.metrics_report())

                return min(runtimes) / 10.0
        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            if trace_memory:
                return "N/A", None
            else:
                return "N/A"

    def inference(self, model_name, batch_size, sequence_length, trace_memory=False):
        try:
            config = self.config_dict[model_name]
            model = None

            if self.args.torchscript:
                config.torchscript = True

            if self.args.with_lm_head:
                model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)
            else:
                model = MODEL_MAPPING[config.__class__](config)

            model.eval()
            model.to(self.args.device)

            # encoder-decoder has vocab size saved differently
            vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size

            input_ids = torch.randint(
                vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device
            )

            if self.args.torchscript:
                with torch.no_grad():
                    if config.is_encoder_decoder:
                        raise NotImplementedError("Torchscript is currently not supported for EncoderDecoder models")
                    else:
                        inference_model = torch.jit.trace(model, input_ids)
            else:
                inference_model = model

            def encoder_decoder_forward():
                with torch.no_grad():
                    inference_model(input_ids, decoder_input_ids=input_ids)

            def encoder_forward():
                with torch.no_grad():
                    inference_model(input_ids)

            _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward

            if trace_memory is True:
                if self.args.trace_memory_line_by_line:
                    trace = start_memory_tracing("transformers")

                if self.args.n_gpu > 0:
                    # gpu
                    # clear gpu cache
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "max_memory_reserved"):
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        logger.info(
                            "Please consider updating PyTorch to version 1.4 to get more accuracy on GPU memory usage"
                        )
                        torch.cuda.reset_max_memory_cached()

                    # run forward
                    _forward()
                elif not self.args.no_tpu and is_torch_tpu_available():
                    # tpu
                    raise NotImplementedError(
                        "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.no_memory=True`"
                    )
                else:
                    # cpu
                    memory_bytes = measure_peak_memory_cpu(_forward)
                    memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

                if self.args.trace_memory_line_by_line:
                    summary = stop_memory_tracing(trace)
                else:
                    summary = None

                if self.args.n_gpu > 0:
                    # gpu
                    if hasattr(torch.cuda, "max_memory_reserved"):
                        memory = Memory(torch.cuda.max_memory_reserved())
                    else:
                        logger.info(
                            "Please consider updating PyTorch to version 1.4 to get more accuracy on GPU memory usage"
                        )
                        memory = Memory(torch.cuda.max_memory_cached())

                return memory, summary
            else:

                if (not self.args.no_tpu and is_torch_tpu_available()) or self.args.torchscript:
                    # run additional 10 times to stabilize compilation for tpu and torchscript
                    logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                    timeit.repeat(
                        _forward, repeat=1, number=5,
                    )

                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(_forward, repeat=self.args.repeat, number=10,)

                if not self.args.no_tpu and is_torch_tpu_available() and self.args.tpu_print_metrics:
                    import torch_xla.debug.metrics as met

                    self.print_fn(met.metrics_report())

                return min(runtimes) / 10.0

        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            if trace_memory:
                return "N/A", None
            else:
                return "N/A"
