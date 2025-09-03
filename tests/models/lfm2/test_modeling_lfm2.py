# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LLaMA model."""

import unittest

import pytest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import Lfm2Config, Lfm2ForCausalLM, Lfm2Model


class Lfm2ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Lfm2Config
        base_model_class = Lfm2Model
        causal_lm_class = Lfm2ForCausalLM

    def __init__(
        self,
        parent,
        layer_types=["full_attention", "conv"],
    ):
        super().__init__(parent)
        self.layer_types = layer_types


@require_torch
class Lfm2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (Lfm2Model, Lfm2ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Lfm2Model,
            "text-generation": Lfm2ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = Lfm2ModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Lfm2ForCausalLM if is_torch_available() else None

    @unittest.skip(
        "Lfm2 alternates between attention and conv layers, so attention are only returned for attention layers"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip("Lfm2 has a special cache format as it alternates between attention and conv layers")
    def test_past_key_values_format(self):
        pass

    @unittest.skip(
        "Lfm2 has a special cache format which is not compatible with compile as it has static address for conv cache"
    )
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass


@require_torch_accelerator
@require_read_token
@slow
class Lfm2IntegrationTest(unittest.TestCase):
    pass
