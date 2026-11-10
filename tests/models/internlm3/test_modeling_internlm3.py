# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for InternLM3, which uses the Llama implementation."""

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import InternLM3Config, is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import LlamaModel


class InternLM3ModelTester(CausalLMModelTester):
    config_class = InternLM3Config
    if is_torch_available():
        base_model_class = LlamaModel


@require_torch
class InternLM3ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = InternLM3ModelTester

    def test_original_bias_names(self):
        # The original config names, as found in the released checkpoints
        config = InternLM3Config(qkv_bias=True, bias=True)
        self.assertTrue(config.attention_bias)
        self.assertTrue(config.mlp_bias)
        with self.assertRaises(StrictDataclassClassValidationError):
            InternLM3Config(qkv_bias=True, bias=False)
