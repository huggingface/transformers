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

from transformers import BitsAndBytesConfig, is_torch_available
from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import load_balancing_loss_func
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_large_accelerator,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        Ernie4_5_MoeForCausalLM,
        Ernie4_5_MoeModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class Ernie4_5_MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Ernie4_5_MoeModel


@require_torch
class Ernie4_5_MoeModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = Ernie4_5_MoeModelTester

    @require_flash_attn
    @require_torch_accelerator
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

                model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16, attn_implementation="eager")
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_fa = outputs_fa.hidden_states[-1]

                # higher tolerance, not sure where it stems from
                assert torch.allclose(logits_fa, logits, atol=1e-2, rtol=1e-2)

    @is_flaky(max_attempts=2)
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        set_seed(42)
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 3
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(config.pad_token_id).to(torch_device)
        model = Ernie4_5_MoeForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        bs, seqlen = input_ids.shape
        self.assertEqual(result.router_logits[0].shape, (bs * seqlen, config.num_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        # (This length is selected from experiments)
        pad_length = input_ids.shape[1] * 4
        # Add extra tokens to input_ids and mask them out to simulate left padding
        padding_block = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(input_ids.shape[0], pad_length),
            dtype=torch.int32,
            device=torch_device,
        )
        padding_block[padding_block == config.pad_token_id] = (config.pad_token_id + 1) % config.vocab_size
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)
        padded_attention_mask = torch.zeros_like(padded_input_ids, dtype=torch.long)
        padded_attention_mask[:, pad_length:] = 1

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that masking can change the loss using a deterministic synthetic example.
        # This avoids flakiness when the model routes tokens uniformly.
        num_experts = 3
        top_k = 1
        synthetic_logits = torch.tensor(
            [
                [10.0, 0.0, 0.0],  # unmasked token -> expert 0
                [10.0, 0.0, 0.0],  # unmasked token -> expert 0
                [0.0, 10.0, 0.0],  # masked token -> expert 1
                [0.0, 10.0, 0.0],  # masked token -> expert 1
            ],
            device=torch_device,
        )
        synthetic_mask = torch.tensor([[1, 1, 0, 0]], device=torch_device)
        masked_loss = load_balancing_loss_func((synthetic_logits,), num_experts, top_k, synthetic_mask)
        unmasked_loss = load_balancing_loss_func((synthetic_logits,), num_experts, top_k, attention_mask=None)
        self.assertNotAlmostEqual(masked_loss.item(), unmasked_loss.item(), places=6)


@slow
@require_torch
class Ernie4_5_MoeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_large_model(cls):
        cls.model = Ernie4_5_MoeForCausalLM.from_pretrained(
            "baidu/ERNIE-4.5-21B-A3B-PT",
            device_map="auto",
            experts_implementation="eager",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        return cls.model

    @classmethod
    def get_small_model(cls):
        cls.model = Ernie4_5_MoeForCausalLM.from_pretrained(
            "hf-internal-testing/ERNIE-4.5-Small-Moe", device_map="auto", dtype="auto", experts_implementation="eager"
        )

        return cls.model

    @require_torch_large_accelerator(memory=48)  # Tested on A100 but requires around 48GiB
    @require_bitsandbytes
    def test_model_21b_a3b_generation(self):
        EXPECTED_TEXT_COMPLETION = "User: Hey, are you conscious? Can you talk to me?\nAssistant: \nI don't have consciousness in the way humans do. I don't feel emotions, have thoughts, or experience awareness. However, I'm"  # fmt: skip

        model = self.get_large_model()
        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")
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

    def test_shortened_model_generation(self):
        # This is gibberish which is expected as the model are the first x layers of the original 28B model
        EXPECTED_TEXT_COMPLETION = 'User: Hey, are you conscious? Can you talk to me?\nAssistant: 不了的 tongues说话 dagat绵席裹着头phones<mask:11>odikèkèk<mask:11><mask:11>bun褶席席地说起来这么说的话的话retti upside upsideolate疡疡疡'  # fmt: skip

        model = self.get_small_model()
        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")
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
