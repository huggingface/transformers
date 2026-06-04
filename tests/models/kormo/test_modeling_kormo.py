# Copyright 2026 KORMo Team and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on HuggingFace's LLaMA implementation in this library.
# It has been modified to accommodate KORMo, which is architecturally identical to
# LLaMA except for the names of the two decoder-layer RMSNorms.
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
"""Testing suite for the PyTorch KORMo model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import (
        KORMoForCausalLM,
        KORMoModel,
    )


class KORMoModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = KORMoModel


@require_torch
class KORMoModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = KORMoModelTester
    _torch_compile_train_cls = KORMoForCausalLM if is_torch_available() else None


@require_torch_accelerator
@slow
class KORMoIntegrationTest(unittest.TestCase):
    pass
