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
"""Testing suite for the PyTorch GLM-4.5, GLM-4.6, GLM-4.7 model."""

import tempfile
import unittest

import pytest

from transformers import is_torch_available
from transformers.testing_utils import (
    backend_device_count,
    get_cpu_ram_total_gib,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, Glm4MoeForCausalLM, Glm4MoeModel


class Glm4MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Glm4MoeModel

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
    model_tester_class = Glm4MoeModelTester
    # used in `test_torch_compile_for_training`. Skip as "Dynamic control flow in MoE"
    _torch_compile_train_cls = None
    model_split_percents = [0.5, 0.85, 0.9]  # it tries to offload everything with the default value


@require_torch_accelerator
@slow
class Glm4MoeIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    MODEL_ID = "zai-org/GLM-4.5-Air"
    NUM_TOKENS_TO_GENERATE = 5
    EXPECTED_TEXT_COMPLETION = ['hello world" -> "world', "tell me about the history of the"]

    @classmethod
    def setUpClass(cls):
        cls.model = None
        cls.tokenizer = None
        cls.offload_dir = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.offload_dir = tempfile.TemporaryDirectory()
            # A 70% per-GPU max_memory cap reserves the headroom to avoid CUDA OOM on
            # multi-GPU runners related to MergeModulelist.
            n = backend_device_count(torch_device)
            if n > 0 and torch_device != "cpu":
                torch_accel = getattr(torch, torch_device)
                per_device = int(
                    min(torch_accel.get_device_properties(i).total_memory for i in range(n)) * 0.70 / 1024**3
                )
                max_memory = dict.fromkeys(range(n), f"{per_device}GiB")
                max_memory["cpu"] = (
                    f"{int(get_cpu_ram_total_gib() * 0.9)}GiB"  # To avoid runner failing with exit code 137.
                )
            else:
                max_memory = None
            cls.model = Glm4MoeForCausalLM.from_pretrained(
                cls.MODEL_ID,
                dtype="auto",
                device_map="auto",
                max_memory=max_memory,
                offload_folder=cls.offload_dir.name,
            )
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_ID)
        return cls.model, cls.tokenizer

    @classmethod
    def tearDownClass(cls):
        if cls.offload_dir is not None:
            cls.offload_dir.cleanup()
        super().tearDownClass()

    def test_1_dynamic_cache(self):
        model, tokenizer = self.get_model()
        prompts = ["[gMASK]<sop>hello", "[gMASK]<sop>tell me"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=self.NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(self.EXPECTED_TEXT_COMPLETION, dynamic_text)

    def test_2_static_cache(self):
        model, tokenizer = self.get_model()
        prompts = ["[gMASK]<sop>hello", "[gMASK]<sop>tell me"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(
            **inputs, max_new_tokens=self.NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(self.EXPECTED_TEXT_COMPLETION, static_text)

        # clear cache object, initialized when we pass `cache_implementation="static"`
        model._cache = None

    @unittest.skip("Offloaded models cannot be compiled with torch.compile")
    @pytest.mark.torch_compile_test
    def test_3_compile_static_cache(self):
        model, tokenizer = self.get_model()
        prompts = ["[gMASK]<sop>hello", "[gMASK]<sop>tell me"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=self.NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(self.EXPECTED_TEXT_COMPLETION, static_compiled_text)
