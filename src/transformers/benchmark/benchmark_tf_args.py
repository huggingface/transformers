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

import logging
from dataclasses import dataclass, field

from ..file_utils import is_tf_available, tf_required
from .benchmark_args_utils import BenchmarkArguments


if is_tf_available():
    import tensorflow as tf


logger = logging.getLogger(__name__)


@dataclass
class TensorflowBenchmarkArguments(BenchmarkArguments):
    xla: bool = field(default=False, metadata={"help": "TensorFlow only: use XLA acceleration."})
    amp: bool = field(default=False, metadata={"help": "TensorFlow only: use automatic mixed precision acceleration."})
    keras_predict: bool = field(
        default=False, metadata={"help": "Whether to use model.predict instead of model() to do a forward pass."}
    )
