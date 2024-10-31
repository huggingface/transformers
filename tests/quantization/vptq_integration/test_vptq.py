# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import gc
import importlib
import tempfile
import unittest

from packaging import version

from transformers import VptqConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, StaticCache
from transformers.testing_utils import (
    require_accelerate,
    require_aqlm,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_vptq_available, is_torch_available

if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights

