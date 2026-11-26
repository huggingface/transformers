# Copyright 2026 the Hugging Face team. All rights reserved.
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
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


if is_torch_available():
    from transformers import (
        MiniCPM5MoeForCausalLM,
        MiniCPM5MoeModel,
    )


class MiniCPM5MoeModelTester(CausalLMModelTester):
    base_model_class = MiniCPM5MoeModel
    causal_lm_class = MiniCPM5MoeForCausalLM

    small_config_overrides = {"n_routed_experts": 4, "num_key_value_heads": 1, "attention_bias": True}


@require_torch
class MiniCPM5MoeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPM5MoeModelTester


@unittest.skip(reason="needs weights to be released")
@require_torch_accelerator
@slow
class MiniCPM5MoeIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    pass
