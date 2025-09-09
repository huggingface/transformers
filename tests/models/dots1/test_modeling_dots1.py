# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch dots1 model."""

import gc
import unittest

from transformers import AutoTokenizer, Dots1Config, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Dots1ForCausalLM,
        Dots1Model,
    )


class Dots1ModelTester(CausalLMModelTester):
    config_class = Dots1Config
    if is_torch_available():
        base_model_class = Dots1Model
        causal_lm_class = Dots1ForCausalLM

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
class Dots1ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Dots1Model,
            Dots1ForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Dots1Model,
            "text-generation": Dots1ForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    model_tester_class = Dots1ModelTester


@require_torch_accelerator
class Dots1IntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    def tearDown(self):
        # See LlamaIntegrationTest.tearDown(). Can be removed once LlamaIntegrationTest.tearDown() is removed.
        cleanup(torch_device, gc_collect=False)

    @slow
    def test_model_15b_a2b_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            """To be or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe"""
        )
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("redmoe-ai-v1/dots.llm1.test", use_fast=False)
        model = Dots1ForCausalLM.from_pretrained("redmoe-ai-v1/dots.llm1.test", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()
