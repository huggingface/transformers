# coding=utf-8
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
"""Testing suite for the PyTorch Olmo3 model."""

import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import is_torch_available, set_seed
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        Olmo3ForCausalLM,
        Olmo3Model,
    )
    from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding


class Olmo3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Olmo3Model


@require_torch
class Olmo3ModelTest(CausalLMModelTest, unittest.TestCase):
    fx_compatible = False
    test_all_params_have_gradient = False
    model_tester_class = Olmo3ModelTester

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Olmo3ForCausalLM if is_torch_available() else None

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Rope only gets applied to full attention layers in Olmo3, so make all layers full attention.
        config.layer_types = ["full_attention"] * len(config.layer_types)

        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = self.model_tester_class.base_model_class(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_parameters = {"rope_type": scaling_type, "factor": 10.0, "rope_theta": 10_000.0}
        scaled_model = self.model_tester_class.base_model_class(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Parent test class's attempt to find Olmo3 rope fails, so we pass here explicitly.
        rope_class = Olmo3RotaryEmbedding

        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        config.rope_parameters = {"rope_type": "default", "rope_theta": 10_000.0}
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        config.rope_parameters = {"rope_type": "linear", "factor": scaling_factor, "rope_theta": 10_000.0}
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        config.rope_parameters = {"rope_type": "dynamic", "factor": scaling_factor, "rope_theta": 10_000.0}
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        config.rope_parameters = {"rope_type": "yarn", "factor": scaling_factor, "rope_theta": 10_000.0}
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short)
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long)
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


@require_torch
class Olmo3IntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = Olmo3ForCausalLM.from_pretrained("shanearora/2025-sep-a-base-model").to(
            torch_device, dtype=torch.bfloat16
        )
        out = model(torch.tensor(input_ids, device=torch_device)).logits.float()
        # Expected mean on dim = -1
        expectations = Expectations(
            {
                ("cuda", 8): [[1.9575, -2.4659, 0.5985, 1.3795, -0.5207, -0.9844, -2.7795, -1.0069]],
            }
        )
        EXPECTED_MEAN = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        expectations = Expectations(
            {
                ("cuda", 8): [8.5625, 5.7812, 4.4688, 2.7031, 3.1094, 4.8125, 5.7188, 3.4219, 2.3906, 2.0938, 3.9844, 5.4688, 3.5312, 5.0938, 2.7656, 8.8125, 9.4375, 9.0625, 8.5000, 8.1875, 7.8750, 7.5312, 7.3125, 7.2812, 7.0000, 2.5625, 4.0312, 3.1719, 7.6562, 4.5625],
            }
        )  # fmt: skip
        EXPECTED_SLICE = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @slow
    def test_model_7b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same for all observers, and 2) the speed of light is the same for all observers. The first part of the theory is called the principle of relativity, and the second part is called the principle of the constancy of the speed of light. The theory of rel"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", device_map="auto")
        model = Olmo3ForCausalLM.from_pretrained("shanearora/2025-sep-a-base-model", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @pytest.mark.torch_export_test
    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        olmo3_model = "shanearora/2025-sep-a-base-model"

        tokenizer = AutoTokenizer.from_pretrained(olmo3_model, pad_token="</s>", padding_side="right")
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the laws of physics are the same for all observers, and 2",
        ]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_generation_length,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_generation_length,
            },
        )
        model = Olmo3ForCausalLM.from_pretrained(
            olmo3_model,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=generation_config,
        )

        prompts = ["Simply put, the theory of relativity states that "]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + eager
        eager_generated_ids = model.generate(
            **prompt_tokens, max_new_tokens=max_new_tokens, do_sample=False, cache_implementation=cache_implementation
        )
        eager_generated_text = tokenizer.batch_decode(eager_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, eager_generated_text)

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
