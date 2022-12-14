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

from dataclasses import dataclass, field
from typing import Tuple

from ..utils import cached_property, is_tf_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments


if is_tf_available():
    import tensorflow as tf


logger = logging.get_logger(__name__)


@dataclass
class TensorFlowBenchmarkArguments(BenchmarkArguments):

    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]

    def __init__(self, **kwargs):
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        for deprecated_arg in self.deprecated_args:
            if deprecated_arg in kwargs:
                positive_arg = deprecated_arg[3:]
                kwargs[positive_arg] = not kwargs.pop(deprecated_arg)
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no-{positive_arg} or"
                    f" {positive_arg}={kwargs[positive_arg]}"
                )
        self.tpu_name = kwargs.pop("tpu_name", self.tpu_name)
        self.device_idx = kwargs.pop("device_idx", self.device_idx)
        self.eager_mode = kwargs.pop("eager_mode", self.eager_mode)
        self.use_xla = kwargs.pop("use_xla", self.use_xla)
        super().__init__(**kwargs)

    tpu_name: str = field(
        default=None,
        metadata={"help": "Name of TPU"},
    )
    device_idx: int = field(
        default=0,
        metadata={"help": "CPU / GPU device index. Defaults to 0."},
    )
    eager_mode: bool = field(default=False, metadata={"help": "Benchmark models in eager model."})
    use_xla: bool = field(
        default=False,
        metadata={
            "help": "Benchmark models using XLA JIT compilation. Note that `eager_model` has to be set to `False`."
        },
    )

    @cached_property
    def _setup_tpu(self) -> Tuple["tf.distribute.cluster_resolver.TPUClusterResolver"]:
        requires_backends(self, ["tf"])
        tpu = None
        if self.tpu:
            try:
                if self.tpu_name:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(self.tpu_name)
                else:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            except ValueError:
                tpu = None
        return tpu

    @cached_property
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", "tf.distribute.cluster_resolver.TPUClusterResolver"]:
        requires_backends(self, ["tf"])
        if self.is_tpu:
            tf.config.experimental_connect_to_cluster(self._setup_tpu)
            tf.tpu.experimental.initialize_tpu_system(self._setup_tpu)

            strategy = tf.distribute.TPUStrategy(self._setup_tpu)
        else:
            # currently no multi gpu is allowed
            if self.is_gpu:
                # TODO: Currently only single GPU is supported
                tf.config.set_visible_devices(self.gpu_list[self.device_idx], "GPU")
                strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{self.device_idx}")
            else:
                tf.config.set_visible_devices([], "GPU")  # disable GPU
                strategy = tf.distribute.OneDeviceStrategy(device=f"/cpu:{self.device_idx}")

        return strategy

    @property
    def is_tpu(self) -> bool:
        requires_backends(self, ["tf"])
        return self._setup_tpu is not None

    @property
    def strategy(self) -> "tf.distribute.Strategy":
        requires_backends(self, ["tf"])
        return self._setup_strategy

    @property
    def gpu_list(self):
        requires_backends(self, ["tf"])
        return tf.config.list_physical_devices("GPU")

    @property
    def n_gpu(self) -> int:
        requires_backends(self, ["tf"])
        if self.cuda:
            return len(self.gpu_list)
        return 0

    @property
    def is_gpu(self) -> bool:
        return self.n_gpu > 0
