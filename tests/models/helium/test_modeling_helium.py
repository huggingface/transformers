# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Helium model."""

import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, HeliumConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_read_token,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ..gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester


if is_torch_available():
    import torch

    from transformers import (
        HeliumForCausalLM,
        HeliumForSequenceClassification,
        HeliumForTokenClassification,
        HeliumModel,
    )


class HeliumModelTester(GemmaModelTester):
    if is_torch_available():
        config_class = HeliumConfig
        model_class = HeliumModel
        for_causal_lm_class = HeliumForCausalLM
        for_sequence_class = HeliumForSequenceClassification
        for_token_class = HeliumForTokenClassification


@require_torch
class HeliumModelTest(GemmaModelTest, unittest.TestCase):
    all_model_classes = (
        (HeliumModel, HeliumForCausalLM, HeliumForSequenceClassification, HeliumForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": HeliumModel,
            "text-classification": HeliumForSequenceClassification,
            "token-classification": HeliumForTokenClassification,
            "text-generation": HeliumForCausalLM,
            "zero-shot": HeliumForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = HeliumModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HeliumConfig, hidden_size=37)


@slow
# @require_torch_gpu
class HeliumIntegrationTest(unittest.TestCase):
    input_text = ["Hello, today is a great day to"]

    @require_read_token
    def test_model_2b(self):
        model_id = "kyutai/helium-1-preview"
        expected_texts = Expectations(
            {
                ("rocm", (9, 5)): ["Hello, today is a great day to start a new project. I have been working on a new project for a while now, and I"],
                (None, None): ["Hello, today is a great day to start a new project. I have been working on a new project for a while now and I have"],
                ("cuda", 8): ['Hello, today is a great day to start a new project. I have been working on a new project for a while now, and I'],
            }
        )  # fmt: skip
        EXPECTED_TEXTS = expected_texts.get_expectation()

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, revision="refs/pr/1").to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision="refs/pr/1")
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
