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
"""Testing suite for the PyTorch Ernie4.5 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
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
        AutoTokenizer,
        Ernie4_5Config,
        Ernie4_5ForCausalLM,
        Ernie4_5Model,
    )
    from transformers.models.ernie4_5.modeling_ernie4_5 import Ernie4_5RotaryEmbedding


class Ernie4_5ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Ernie4_5Config
        base_model_class = Ernie4_5Model
        causal_lm_class = Ernie4_5ForCausalLM


@require_torch
class Ernie4_5ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Ernie4_5Model,
            Ernie4_5ForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Ernie4_5Model,
            "text-generation": Ernie4_5ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = Ernie4_5ModelTester
    rotary_embedding_layer = Ernie4_5RotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Ernie4_5ForCausalLM if is_torch_available() else None


@require_torch_accelerator
class Ernie4_5IntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_ernie4_5_0p3B(self):
        """
        An integration test for Ernie 4.5 0.3B.
        """
        expected_texts = Expectations(
            {
                ("cuda", None): "User: Hey, are you conscious? Can you talk to me?\nAssistant: Hey! I'm here to help you with whatever you need. Are you feeling a bit overwhelmed or stressed? I'm here to listen and provide support.",
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-0.3B-PT", revision="refs/pr/3")
        model = Ernie4_5ForCausalLM.from_pretrained(
            "baidu/ERNIE-4.5-0.3B-PT",
            device_map="auto",
            dtype=torch.bfloat16,
        )

        prompt = "Hey, are you conscious? Can you talk to me?"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=128,
            do_sample=False,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip("\n")
        self.assertEqual(generated_text, EXPECTED_TEXT)
