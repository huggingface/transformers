# Copyright 2026 JetBrains and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Mellum model."""

import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        MellumForCausalLM,
        MellumModel,
    )
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MellumModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MellumModel


@require_torch
class MellumModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = MellumModelTester

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("RoPE-scaling-from-config test doesn't match Mellum's nested per-layer-type rope_parameters.")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def test_model_rope_scaling_frequencies(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["full_attention", "sliding_attention"]

        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = ["rotary_emb"]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        x = torch.randn(1, dtype=torch.float32, device=torch_device)
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)

        # Sanity check original RoPE
        rope_params = {"rope_type": "default", "rope_theta": 10_000.0}
        config.rope_parameters = {
            "full_attention": rope_params,
            "sliding_attention": rope_params,
        }
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="sliding_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        rope_params = {
            "rope_type": "linear",
            "factor": scaling_factor,
            "rope_theta": 10_000.0,
        }
        config.rope_parameters = {
            "full_attention": rope_params,
            "sliding_attention": rope_params,
        }
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(
                linear_cos_long[:, new_position, :],
                original_cos_long[:, original_position, :],
            )
            torch.testing.assert_close(
                linear_sin_long[:, new_position, :],
                original_sin_long[:, original_position, :],
            )

        # Sanity check Dynamic NTK RoPE scaling
        rope_params = {
            "rope_type": "dynamic",
            "factor": scaling_factor,
            "rope_theta": 10_000.0,
        }
        config.rope_parameters = {
            "full_attention": rope_params,
            "sliding_attention": rope_params,
        }
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue(
            (ntk_scaling_rope.sliding_attention_inv_freq <= original_rope.sliding_attention_inv_freq).all()
        )

        # Sanity check Yarn RoPE scaling
        rope_params = {
            "rope_type": "yarn",
            "factor": scaling_factor,
            "rope_theta": 10_000.0,
        }
        config.rope_parameters = {
            "full_attention": rope_params,
            "sliding_attention": rope_params,
        }
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 3
        config.expert_interval = 2
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MellumForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_experts))
        torch.testing.assert_close(
            result.aux_loss.cpu(),
            torch.tensor(2, dtype=torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )

        # First, we make sure that adding padding tokens doesn't change the loss
        pad_length = input_ids.shape[1] * 4
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())


@require_torch
class MellumIntegrationTest(unittest.TestCase):
    checkpoint = "JetBrains/Mellum2-12B-A2.5B-Base"

    def setUp(self):
        cleanup(torch_device, gc_collect=False)

    def tearDown(self):
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_torch_accelerator
    def test_model_generation(self):
        model = MellumForCausalLM.from_pretrained(self.checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        prompt = "def fibonacci(n):"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertTrue(len(output) > len(prompt))
        self.assertTrue(output.startswith(prompt))
