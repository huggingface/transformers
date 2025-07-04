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

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
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
@require_read_token
class Ernie4_5IntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    # TODO: overwrite to something with ernie4_5 model
    # e.g. (30 tokens A100) "Hey, are you conscious? Can you talk to me?" with template --> "Hey! I'm here to help you with whatever you need. Are you feeling a bit overwhelmed or stressed? I'm here to listen"
    @slow
    def test_ernie4_5_3_1_hard(self):
        """
        An integration test for ernie4_5 3.1. It tests against a long output to ensure the subtle numerical differences
        from ernie4_5 3.1.'s RoPE can be detected
        """
        expected_texts = Expectations(
            {
                ("rocm", (9, 5)): 'Tell me about the french revolution. The french revolution was a period of radical social and political upheaval in France that lasted from 1789 until 1799. It was a time of great change and upheaval, marked by the overthrow of the monarchy, the rise of the middle class, and the eventual establishment of the First French Republic.\nThe revolution began in 1789 with the Estates-General, a representative assembly that had not met since 1614. The Third Estate, which represented the common people, demanded greater representation and eventually broke away to form the National Assembly. This marked the beginning of the end of the absolute monarchy and the rise of the middle class.\n',
                ("cuda", None): 'Tell me about the french revolution. The french revolution was a period of radical political and social upheaval in France that lasted from 1789 until 1799. It was a time of great change and upheaval, marked by the overthrow of the monarchy, the rise of the middle class, and the eventual establishment of the First French Republic.\nThe revolution began in 1789 with the Estates-General, a representative assembly that had not met since 1614. The Third Estate, which represented the common people, demanded greater representation and eventually broke away to form the National Assembly. The National Assembly adopted the Declaration of the Rights of Man and of the Citizen, which enshr',
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained("meta-ernie4_5/Meta-Ernie4_5-3.1-8B-Instruct")
        model = Ernie4_5ForCausalLM.from_pretrained(
            "meta-ernie4_5/Meta-Ernie4_5-3.1-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)
