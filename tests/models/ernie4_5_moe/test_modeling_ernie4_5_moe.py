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
"""Testing suite for the PyTorch Ernie4.5 MoE model."""

import tempfile
import unittest

import pytest

from transformers import Ernie4_5_MoeConfig, is_torch_available
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_large_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        Ernie4_5_MoeForCausalLM,
        Ernie4_5_MoeModel,
    )
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class Ernie4_5_MoeModelTester(CausalLMModelTester):
    config_class = Ernie4_5_MoeConfig
    if is_torch_available():
        base_model_class = Ernie4_5_MoeModel
        causal_lm_class = Ernie4_5_MoeForCausalLM


@require_torch
class Ernie4_5_MoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Ernie4_5_MoeModel,
            Ernie4_5_MoeForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Ernie4_5_MoeModel,
            "text-generation": Ernie4_5_MoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    test_all_params_have_gradient = False
    model_tester_class = Ernie4_5_MoeModelTester

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @is_flaky()
    @slow
    def test_flash_attn_2_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(reason="Model does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="eager"
                )
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_fa = outputs_fa.hidden_states[-1]

                # higher tolerance, not sure where it stems from
                assert torch.allclose(logits_fa, logits, atol=1e-2, rtol=1e-2)

    # Ignore copy
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 8
        config.expert_interval = 2
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = Ernie4_5_MoeForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = 1000
        # Add padding tokens (assume that pad_token_id=1) to input_ids
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())


# Run on runners with larger accelerators (for example A10 instead of T4) with a lot of CPU RAM (e.g. g5-12xlarge)
@require_torch_multi_accelerator
@require_torch_large_accelerator
@require_torch
class Ernie4_5_MoeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Ernie4_5_MoeForCausalLM.from_pretrained(
                "baidu/ERNIE-4.5-21B-A3B-PT",
                device_map="auto",
                load_in_4bit=True,
            )

        return cls.model

    @require_bitsandbytes
    @slow
    def test_model_21b_a3b_generation(self):
        EXPECTED_TEXT_COMPLETION = "User: Hey, are you conscious? Can you talk to me?\nAssistant:  I don't have consciousness in the way humans do. I'm a text-based AI created to process and generate responses based on patterns in data."  # fmt: skip

        model = self.get_model()
        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT", revision="refs/pr/11")
        prompt = "Hey, are you conscious? Can you talk to me?"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=32,
            do_sample=False,
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip("\n")
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
