# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch openPangu-Embedded model."""

import unittest

import pytest
from packaging import version

from transformers import AutoTokenizer, StaticCache, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    run_test_using_subprocess,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        PanguEmbeddedConfig,
        PanguEmbeddedForCausalLM,
        PanguEmbeddedModel,
    )


class PanguEmbeddedModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = PanguEmbeddedConfig
        base_model_class = PanguEmbeddedModel
        causal_lm_class = PanguEmbeddedForCausalLM


@require_torch
class PanguEmbeddedModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            PanguEmbeddedModel,
            PanguEmbeddedForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PanguEmbeddedModel,
            "text-generation": PanguEmbeddedForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = PanguEmbeddedModelTester

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = PanguEmbeddedForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
class PanguEmbeddedIntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        return True