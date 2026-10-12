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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        HunYuanDenseV1Model,
    )
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class HunYuanDenseV1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HunYuanDenseV1Model


@require_torch
class HunYuanDenseV1ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HunYuanDenseV1ModelTester

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
    model_id = "tencent/HY-MT1.5-1.8B"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        # TODO Need new Dense Model
        return True

    @slow
    @require_torch_accelerator
    def test_rope_frequencies_past_max_position_embeddings(self):
        """The NTK-alpha frequencies are static, so decoding past the context window must not touch them."""
        config = AutoConfig.from_pretrained(self.model_id)
        # Shrink the context window so that a short prompt already decodes past it
        config.max_position_embeddings = 16

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config, dtype=torch.bfloat16).to(
            torch_device
        )
        rotary = model.model.rotary_emb

        # The checkpoint asks for `dynamic`, the module resolves it to the variant it actually implements
        self.assertEqual(config.rope_parameters["rope_type"], "dynamic")
        self.assertEqual(rotary.rope_type, "ntk_alpha")

        head_dim = config.head_dim
        base = config.rope_parameters["rope_theta"] * config.rope_parameters["alpha"] ** (head_dim / (head_dim - 2))
        expected_inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        torch.testing.assert_close(rotary.inv_freq.cpu(), expected_inv_freq, rtol=0, atol=0)

        inputs = tokenizer("Machine translation is", return_tensors="pt").to(torch_device)
        model.generate(**inputs, max_new_tokens=32, min_new_tokens=32, do_sample=False)

        # Before this was fixed, `@dynamic_rope_update` overwrote these with standard NTK frequencies
        torch.testing.assert_close(rotary.inv_freq.cpu(), expected_inv_freq, rtol=0, atol=0)
