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


import random
import timeit
from functools import wraps
from typing import Callable, Optional

from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_tf_available, logging
from .benchmark_utils import (
    Benchmark,
    Memory,
    MemorySummary,
    measure_peak_memory_cpu,
    start_memory_tracing,
    stop_memory_tracing,
)


if is_tf_available():
    import tensorflow as tf
    from tensorflow.python.framework.errors_impl import ResourceExhaustedError

    from .benchmark_args_tf import TensorFlowBenchmarkArguments

if is_py3nvml_available():
    import py3nvml.py3nvml as nvml

logger = logging.get_logger(__name__)


def run_with_tf_optimizations(do_eager_mode: bool, use_xla: bool):
    def run_func(func):
        @wraps(func)
        def run_in_eager_mode(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        @tf.function(experimental_compile=use_xla)
        def run_in_graph_mode(*args, **kwargs):
            return func(*args, **kwargs)

        if do_eager_mode is True:
            assert (
                use_xla is False
            ), "Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`."
            return run_in_eager_mode
        else:
            return run_in_graph_mode

    return run_func


def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int) -> ["tf.Tensor"]:
    rng = random.Random()
    values = [rng.randint(0, vocab_size - 1) for i in range(batch_size * sequence_length)]
    return tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)


class TensorFlowBenchmark(Benchmark):

    args: TensorFlowBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "TensorFlow"

    @property
    def framework_version(self):
        return tf.__version__

    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # initialize GPU on separate process
        strategy = self.args.strategy
        assert strategy is not None, "A device strategy has to be initialized before using TensorFlow."
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_inference)

    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        strategy = self.args.strategy
        assert strategy is not None, "A device strategy has to be initialized before using TensorFlow."
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_train)

    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # initialize GPU on separate process
        if self.args.is_gpu:
            tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
        strategy = self.args.strategy
        assert strategy is not None, "A device strategy has to be initialized before using TensorFlow."
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_inference)

    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        if self.args.is_gpu:
            tf.config.experimental.set_memory_growth(self.args.gpu_list[self.args.device_idx], True)
        strategy = self.args.strategy
        assert strategy is not None, "A device strategy has to be initialized before using TensorFlow."

        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_train)

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        config = self.config_dict[model_name]

        if self.args.fp16:
            raise NotImplementedError("Mixed precision is currently not supported.")

        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                model_class = "TF" + config.architectures[0]  # prepend 'TF' for tensorflow model
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            model = TF_MODEL_MAPPING[config.__class__](config)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_decoder_forward():
            return model(input_ids, decoder_input_ids=input_ids, training=False)

        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_forward():
            return model(input_ids, training=False)

        _inference = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward

        return _inference

    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        config = self.config_dict[model_name]

        assert (
            self.args.eager_mode is False
        ), "Training cannot be done in eager mode. Please make sure that `args.eager_mode = False`."

        if self.args.fp16:
            raise NotImplementedError("Mixed precision is currently not supported.")

        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                model_class = "TF" + config.architectures[0]  # prepend 'TF' for tensorflow model
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            model = TF_MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = random_input_ids(batch_size, sequence_length, vocab_size)

        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_decoder_train():
            loss = model(input_ids, decoder_input_ids=input_ids, labels=input_ids, training=True)[0]
            gradients = tf.gradients(loss, model.trainable_variables)
            return gradients

        @run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
        def encoder_train():
            loss = model(input_ids, labels=input_ids, training=True)[0]
            gradients = tf.gradients(loss, model.trainable_variables)
            return gradients

        _train = encoder_decoder_train if config.is_encoder_decoder else encoder_train

        return _train

    def _measure_speed(self, func) -> float:
        with self.args.strategy.scope():
            try:
                if self.args.is_tpu or self.args.use_xla:
                    # run additional 10 times to stabilize compilation for tpu
                    logger.info("Do inference on TPU. Running model 5 times to stabilize compilation")
                    timeit.repeat(func, repeat=1, number=5)

                # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
                runtimes = timeit.repeat(
                    func,
                    repeat=self.args.repeat,
                    number=10,
                )

                return min(runtimes) / 10.0
            except ResourceExhaustedError as e:
                self.print_fn(f"Doesn't fit on GPU. {e}")

    def _measure_memory(self, func: Callable[[], None]) -> [Memory, MemorySummary]:
        logger.info(
            "Note that TensorFlow allocates more memory than "
            "it might need to speed up computation. "
            "The memory reported here corresponds to the memory "
            "reported by `nvidia-smi`, which can vary depending "
            "on total available memory on the GPU that is used."
        )
        with self.args.strategy.scope():
            try:
                if self.args.trace_memory_line_by_line:
                    assert (
                        self.args.eager_mode
                    ), "`args.eager_mode` is set to `False`. Make sure to run model in eager mode to measure memory consumption line by line."
                    trace = start_memory_tracing("transformers")

                if self.args.is_tpu:
                    # tpu
                    raise NotImplementedError(
                        "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.memory=False`"
                    )
                elif self.args.is_gpu:
                    # gpu
                    if not is_py3nvml_available():
                        logger.warning(
                            "py3nvml not installed, we won't log GPU memory usage. "
                            "Install py3nvml (pip install py3nvml) to log information about GPU."
                        )
                        memory = "N/A"
                    else:
                        logger.info(
                            "Measuring total GPU usage on GPU device. Make sure to not have additional processes running on the same GPU."
                        )
                        # init nvml
                        nvml.nvmlInit()
                        func()
                        handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                        meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                        max_bytes_in_use = meminfo.used
                        memory = Memory(max_bytes_in_use)
                        # shutdown nvml
                        nvml.nvmlShutdown()
                else:
                    # cpu
                    if self.args.trace_memory_line_by_line:
                        logger.info(
                            "When enabling line by line tracing, the max peak memory for CPU is inaccurate in TensorFlow."
                        )
                        memory = None
                    else:
                        memory_bytes = measure_peak_memory_cpu(func)
                        memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes
                if self.args.trace_memory_line_by_line:
                    summary = stop_memory_tracing(trace)
                    if memory is None:
                        memory = summary.total
                else:
                    summary = None

                return memory, summary
            except ResourceExhaustedError as e:
                self.print_fn(f"Doesn't fit on GPU. {e}")
                return "N/A", None
