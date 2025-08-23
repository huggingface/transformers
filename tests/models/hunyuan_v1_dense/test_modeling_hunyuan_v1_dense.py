# Copyright (C) 2024 THL A29 Limited, a Tencent company and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HunYuanDenseV1 model."""

import unittest

from transformers import HunYuanDenseV1Config, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    from transformers import (
        HunYuanDenseV1ForCausalLM,
        HunYuanDenseV1ForSequenceClassification,
        HunYuanDenseV1Model,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class HunYuanDenseV1ModelTester(CausalLMModelTester):
    config_class = HunYuanDenseV1Config
    if is_torch_available():
        base_model_class = HunYuanDenseV1Model
        causal_lm_class = HunYuanDenseV1ForCausalLM
        sequence_class = HunYuanDenseV1ForSequenceClassification


@require_torch
class HunYuanDenseV1ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            HunYuanDenseV1Model,
            HunYuanDenseV1ForCausalLM,
            HunYuanDenseV1ForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = HunYuanDenseV1ModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": HunYuanDenseV1Model,
            "text-generation": HunYuanDenseV1ForCausalLM,
            "text-classification": HunYuanDenseV1ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True


@require_torch
class HunYuanDenseV1IntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        # TODO Need new Dense Model
        return True
