# Copyright 2025 Upstage and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SolarOpen model."""

import unittest

import torch

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import AutoTokenizer, SolarOpenForCausalLM, SolarOpenModel


class SolarOpenModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = SolarOpenModel

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        use_qk_norm=False,
    ):
        super().__init__(parent=parent, num_experts_per_tok=num_experts_per_tok)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm


@require_torch
class SolarOpenModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = SolarOpenModelTester
    # used in `test_torch_compile_for_training`. Skip as "Dynamic control flow in MoE"
    _torch_compile_train_cls = None
    model_split_percents = [0.5, 0.85, 0.9]  # it tries to offload everything with the default value


@require_torch_accelerator
@slow
class SolarOpenIntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=False)

    def test_batch_generation_bf16(self):
        prompts = [
            "<|begin|>assistant<|content|>How are you?<|end|><|begin|>assistant",
            "<|begin|>assistant<|content|>What is LLM?<|end|><|begin|>assistant",
        ]
        model_id = "upstage/Solar-Open-100B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = SolarOpenForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for text in generated_texts:
            # Check that the generated continuation starts with <|think|>
            generated_part = text.split("<|begin|>assistant")[-1]
            self.assertTrue(generated_part.startswith("<|think|>"))
