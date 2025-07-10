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

from transformers import is_torch_available
from transformers.testing_utils import (
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import LFM2Config, LFM2ForCausalLM, LFM2Model


class LFM2ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = LFM2Config
        base_model_class = LFM2Model
        causal_lm_class = LFM2ForCausalLM

    def __init__(
        self,
        parent,
        layer_types=["full_attention", "conv"],
    ):
        super().__init__(parent)
        self.layer_types = layer_types


@require_torch
class LFM2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (LFM2Model, LFM2ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": LFM2Model,
            "text-generation": LFM2ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = LFM2ModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = LFM2ForCausalLM if is_torch_available() else None

    @unittest.skip(
        "LFM2 alternates between attention and conv layers, so attention are only returned for attention layers"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip("LFM2 has a special cache format as it alternates between attention and conv layers")
    def test_past_key_values_format(self):
        pass

    @unittest.skip("LFM2 has a special cache format which is not compatible with contrastive search")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("LFM2 has a special cache format which is not compatible with contrastive search")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("LFM2 has a special cache format which is not compatible with contrastive search")
    def test_contrastive_generate_low_memory(self):
        pass


@require_torch_accelerator
@require_read_token
@slow
class LFM2IntegrationTest(unittest.TestCase):
    pass
