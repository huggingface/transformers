# coding=utf-8
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
"""Testing suite for the PyTorch VaultGemma model."""

import unittest

from transformers import VaultGemmaConfig, is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    from transformers import VaultGemmaForCausalLM, VaultGemmaModel


class VaultGemmaModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = VaultGemmaConfig
        base_model_class = VaultGemmaModel
        causal_lm_class = VaultGemmaForCausalLM
    pipeline_model_mapping = (
        {
            "feature-extraction": VaultGemmaModel,
            "text-generation": VaultGemmaForCausalLM,
        }
        if is_torch_available()
        else {}
    )


@require_torch
class VaultGemmaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (VaultGemmaModel, VaultGemmaForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": VaultGemmaModel,
            "text-generation": VaultGemmaForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = VaultGemmaModelTester

    def setUp(self):
        self.model_tester = VaultGemmaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VaultGemmaConfig, hidden_size=37)
