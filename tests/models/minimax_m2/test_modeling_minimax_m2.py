# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch MiniMaxM2 model."""

import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flaky,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        MiniMaxM2ForCausalLM,
        MiniMaxM2Model,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MiniMaxM2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniMaxM2Model


@require_torch
class MiniMaxM2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniMaxM2ModelTester

    @is_flaky(max_attempts=2)
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 3
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(config.pad_token_id).to(torch_device)
        model = MiniMaxM2ForCausalLM(config)
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
        # Add padding tokens to input_ids
        padding_block = config.pad_token_id * torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(
            torch_device
        )
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(config.pad_token_id).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())


@slow
@require_torch
class MiniMaxM2IntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    @require_torch_accelerator
    def test_small_model_logits_batched(self):
        model_id = "hf-internal-testing/MiniMax-M2-Small"
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = MiniMaxM2ForCausalLM.from_pretrained(
            model_id, dtype="auto", device_map="auto", experts_implementation="eager"
        )

        EXPECTED_LOGITS_LEFT_UNPADDED = Expectations(
            {
                ("cuda", 8): [[1.1094, -1.5352, -1.5811], [1.9395, 0.1461, -1.5537], [1.7803, 0.2466, -0.4316]],
            }
        )
        expected_left_unpadded = torch.tensor(EXPECTED_LOGITS_LEFT_UNPADDED.get_expectation(), device=torch_device)

        EXPECTED_LOGITS_RIGHT_UNPADDED = Expectations(
            {
                ("cuda", 8): [[0.8135, -1.8164, -1.5898], [0.0663, -1.3408, -0.5435], [0.5396, 0.3293, -1.7529]],
            }
        )
        expected_right_unpadded = torch.tensor(EXPECTED_LOGITS_RIGHT_UNPADDED.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()

        torch.testing.assert_close(
            logits[0, -3:, -3:],
            expected_left_unpadded,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            logits[1, -3:, -3:],
            expected_right_unpadded,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_small_model_generation(self):
        expected_texts = Expectations(
            {
                ("cuda", 8): 'Tell me about the french revolution. Pemkab Pemkab المتاحة/journal blinded blindedébé抓算不上 blinded blinded healthiest.Clébé Bronx开启了 Bronx Bronx抽样ikat糜 BronxSources TODOSources parfum Bronx parfum donde donde donde او',
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2")
        model = MiniMaxM2ForCausalLM.from_pretrained(
            "hf-internal-testing/MiniMax-M2-Small", device_map="auto", dtype="auto", experts_implementation="eager"
        )
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)
