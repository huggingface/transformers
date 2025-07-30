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

import pytest
import torch
from packaging import version

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import AutoTokenizer, Glm4MoeConfig, Glm4MoeForCausalLM, Glm4MoeModel


class Glm4MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Glm4MoeConfig
        base_model_class = Glm4MoeModel
        causal_lm_class = Glm4MoeForCausalLM

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=8,
    ):
        super().__init__(parent=parent, num_experts_per_tok=num_experts_per_tok)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group


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
    # used in `test_torch_compile_for_training`. Skip as "Dynamic control flow in MoE"
    _torch_compile_train_cls = None


@require_torch_accelerator
@require_read_token
@slow
class Glm4MoeIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # See LlamaIntegrationTest.tearDown(). Can be removed once LlamaIntegrationTest.tearDown() is removed.
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            'hello, world!\'\'\')\nprint(\'hello, world!\')\nprint("hello, world!")\nprint("hello, world!")\nprint("hello, world!")\nprint("hello, world!")\nprint("hello, world!")\n',
            "tell me the story of the first Thanksgiving. commonly known as the Pilgrims, arrived in the autumn of 1620. They were seeking religious freedom and a new life in the Plymouth Colony. Their first",
        ]

        prompts = ["[gMASK]<sop>hello", "[gMASK]<sop>tell me"]
        tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5")
        model = Glm4MoeForCausalLM.from_pretrained("zai-org/GLM-4.5", device_map=torch_device, dtype=torch.bfloat16)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)
