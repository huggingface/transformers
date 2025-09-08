# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Testing suite for the PyTorch Nemotron model."""

import unittest

from transformers import NemotronConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        NemotronForCausalLM,
        NemotronForQuestionAnswering,
        NemotronForSequenceClassification,
        NemotronForTokenClassification,
        NemotronModel,
    )


class NemotronModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = NemotronConfig
        base_model_class = NemotronModel
        causal_lm_class = NemotronForCausalLM
        sequence_class = NemotronForSequenceClassification
        token_class = NemotronForTokenClassification


@require_torch
class NemotronModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = NemotronModelTester
    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]
    all_model_classes = (
        (
            NemotronModel,
            NemotronForCausalLM,
            NemotronForSequenceClassification,
            NemotronForQuestionAnswering,
            NemotronForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": NemotronModel,
            "text-classification": NemotronForSequenceClassification,
            "text-generation": NemotronForCausalLM,
            "zero-shot": NemotronForSequenceClassification,
            "question-answering": NemotronForQuestionAnswering,
            "token-classification": NemotronForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = NemotronForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = NemotronModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronConfig, hidden_size=37)

    @unittest.skip("Eager and SDPA do not produce the same outputs, thus this test fails")
    def test_model_outputs_equivalence(self, **kwargs):
        pass


@require_torch_accelerator
class NemotronIntegrationTest(unittest.TestCase):
    @slow
    @require_read_token
    def test_nemotron_8b_generation_sdpa(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_nemotron_8b_generation_eager(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer: What is the name of the 19",
                ],
                ("cuda", 7): [
                    "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_nemotron_8b_generation_fa2(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)
