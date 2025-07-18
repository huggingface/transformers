# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GLM-4-MoE model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import (
        Glm4MoeConfig,
        Glm4MoeForCausalLM,
        Glm4MoeModel,
    )


class Glm4MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Glm4MoeConfig
        base_model_class = Glm4MoeModel
        causal_lm_class = Glm4MoeForCausalLM


@require_torch
class Glm4MoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Glm4MoeModel,
            Glm4MoeForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Glm4MoeModel,
            "text-generation": Glm4MoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = Glm4MoeModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Glm4MoeForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
@slow
class Glm4MoeIntegrationTest(unittest.TestCase):
    pass
