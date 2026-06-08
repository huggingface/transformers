# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Cohere2Moe model"""

import unittest

from parameterized import parameterized
from pytest import mark

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Cohere2MoeConfig,
    Cohere2VisionForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flash_attn_2_available,
    is_kernels_available,
    is_torch_xpu_available,
    require_flash_attn,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import Cohere2MoeForCausalLM, Cohere2MoeModel


class Cohere2MoeModelTester(CausalLMModelTester):
    config_class = Cohere2MoeConfig
    if is_torch_available():
        base_model_class = Cohere2MoeModel
        causal_lm_class = Cohere2MoeForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_types = ["full_attention", "sliding_attention"]
        self.mlp_layer_types = ["dense", "sparse"]  # first layer will be MLP, 2nd will be MoE
        self.logit_scale = 1.0  # needed for `test_training_overfit` - otherwise the loss does not go down fast enough


@require_torch
class Cohere2MoeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Cohere2MoeModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Cohere2MoeForCausalLM if is_torch_available() else None
    # Raise the split thresholds so accelerate can place the model weight into multiple devices.
    model_split_percents = [0.5, 0.8, 0.9]


@slow
@require_torch_large_accelerator
class Cohere2MoeIntegrationTest(unittest.TestCase):
    """Integration tests for the cohere2moe text backbone via the Command A+ Model.

    Cohere2VisionForConditionalGeneration wraps the cohere2moe language model; running it with
    text-only inputs exercises the text backbone without requiring a separate text-only checkpoint.
    """

    model_id = "CohereLabs/command-a-plus-05-2026"
    input_text = ["Hello I am doing", "Hi today"]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_model(self, dtype, attn_implementation="eager", text_config_overrides=None):
        """Load the vision model (cohere2moe backbone) distributed across all available GPUs.

        text_config_overrides: optional dict of attributes to set on config.text_config before loading
        (e.g. {"sliding_window": 1024}).
        """
        if text_config_overrides:
            config = AutoConfig.from_pretrained(self.model_id)
            for k, v in text_config_overrides.items():
                setattr(config.text_config, k, v)
        else:
            config = None
        kwargs = {"torch_dtype": dtype, "attn_implementation": attn_implementation, "device_map": "auto"}
        if config is not None:
            kwargs["config"] = config
        return Cohere2VisionForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()

    def test_model_bf16(self):
        EXPECTED_TEXTS = [
            "<BOS_TOKEN>Hello I am doing a project on the history of the internet. I am trying to ARexx script a program that",
            '<PAD><PAD><BOS_TOKEN>Hi today we are going to discuss about the concept of "Self-Confidence". Self-confidence is a term that',
        ]

        model = self._load_model(torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_fp16(self):
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                (None, None): [
                    '<BOS_TOKEN>Hello I am doing a project on the history of the internet. I am trying to ARexx script a program that',
                    '<PAD><PAD><BOS_TOKEN>Hi today we are going to discuss about the concept of "Self-Confidence". Self-confidence is a term that',
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        # fmt: on

        model = self._load_model(torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_flash_attn
    @mark.flash_attn_test
    def test_model_flash_attn(self):
        # fmt: off
        EXPECTED_TEXTS = [
            '<BOS_TOKEN>Hello I am doing a project on the history of the internet. I am trying to ARexx script a program that will display a comment and then a progress bar that moves across the2009-09-30\n\nHello, I am doing a project on the history of the internet. I am trying to ARexx script a program that will display a comment and then a progress bar that moves across the screen. I have a question about the "wait" command. I have been using "wait 1"',
            '<PAD><PAD><BOS_TOKEN>Hi today we are going to discuss about the concept of "Self-Confidence". Self-confidence is a term that many people use to describe a state of mind where one feels confident in their abilities, decisions, and actions. It\'s a feeling of trust in one\'s own judgment and abilities. Self-confidence is not about being arrogant or overconfident; it\'s about having a realistic and positive view of oneself and one\'s capabilities.\n\nSelf-confidence can be developed and improved over time through various practices such as setting and achieving goals, learning',
        ]
        # fmt: on

        model = self._load_model(torch.float16, attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Verify that generation beyond the sliding window produces coherent output
        with all supported attention backends.
        """
        if (
            attn_implementation == "flash_attention_2"
            and not is_flash_attn_2_available()
            and not (is_torch_xpu_available() and is_kernels_available())
        ):
            self.skipTest("FlashAttention2 is required for this test.")

        EXPECTED_COMPLETIONS = [
            " but I think it's a nice place. This is a nice place. This is a nice place.",
            ", green, yellow, orange, purple, pink, brown, black, white.\n\nWe need to",
        ]

        input_text = [
            "This is a nice place. " * 200 + "I really enjoy the scenery,",
            "A list of colors: red, blue",
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to("cuda:0")

        model = self._load_model(
            torch.float16, attn_implementation=attn_implementation, text_config_overrides={"sliding_window": 1024}
        )

        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.text_config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)
