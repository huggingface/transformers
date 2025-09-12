# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team and the Swiss AI Initiative. All rights reserved.
#
# This code is based on HuggingFace's LLaMA implementation in this library.
# It has been modified from its original forms to accommodate minor architectural
# differences compared to LLaMA used by the Swiss AI Initiative that trained the model.
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
"""Testing suite for the PyTorch Apertus model."""

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
        ApertusConfig,
        ApertusForCausalLM,
        ApertusForTokenClassification,
        ApertusModel,
    )


class ApertusModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = ApertusConfig
        base_model_class = ApertusModel
        causal_lm_class = ApertusForCausalLM
        token_class = ApertusForTokenClassification


@require_torch
class ApertusModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            ApertusModel,
            ApertusForCausalLM,
            ApertusForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ApertusModel,
            "text-generation": ApertusForCausalLM,
            "token-classification": ApertusForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = ApertusModelTester

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = ApertusForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
@slow
class ApertusIntegrationTest(unittest.TestCase):
    pass
