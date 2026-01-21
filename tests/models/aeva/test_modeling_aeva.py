# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch GLM-4.5, GLM-4.6, GLM-4.7 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import AevaForCausalLM, AevaModel


class AevaModelTester(CausalLMModelTester):
    def __init__(self, parent):
        super().__init__(
            parent,
            vocab_size=151552,
            hidden_size=1024,
        )


@require_torch
class AevaModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = AevaModelTester
    all_model_classes = (AevaModel, AevaForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": AevaModel, "text-generation": AevaForCausalLM} if is_torch_available() else {}
    )
    model_split_percents = [0.5, 0.85, 0.9]

    @unittest.skip("Aeva has MoE, dynamic routing leads to non-deterministic outputs")
    def test_model_outputs_equivalence(self):
        pass
