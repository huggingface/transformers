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

import tempfile
import unittest

import pytest

from transformers import is_torch_available, set_seed
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_tensor_parallel_test,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torchao_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_tensor_parallel_mixin import _init_distributed, _test_tp_generation_quantized_impl


if is_torch_available():
    import torch

    from transformers import (
        Olmo3ForCausalLM,
        Olmo3ForSequenceClassification,
        Olmo3Model,
    )


class Olmo3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Olmo3Model
        sequence_classification_class = Olmo3ForSequenceClassification

    def __init__(
        self,
        parent,
        layer_types=[
            "full_attention",
            "sliding_attention",
        ],  # we want to test both types
        **kwargs,
    ):
        super().__init__(parent=parent, layer_types=layer_types, **kwargs)


@require_torch
class Olmo3ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = Olmo3ModelTester

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Olmo3ForCausalLM if is_torch_available() else None

    @is_tensor_parallel_test
    def test_tp_generation_quantized(self):
        # If model uses rope-theta 50k (default value), the test fails
        # Override and set `theta=10K`
        self._skip_if_not_supported()

        if not is_torchao_available():
            self.skipTest("Test requires torchao")

        config = self.model_tester.get_config()
        config.rope_parameters["full_attention"]["rope_theta"] = 10_000.0
        config.rope_parameters["sliding_attention"]["rope_theta"] = 10_000.0

        model_class = self._get_tp_model_class()
        max_new_tokens = 25

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_generation_quantized_impl)(
                tmp_dir, model_class, max_new_tokens
            )

    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        # Olmo3 has different RoPE configs per layer type
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Retrieves the RoPE layer class from the base model class. Uses `.named_modules()` to avoid hardcoding the
        # named location of the RoPE layer class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",  # most common case
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

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
        rope_params = {"rope_type": "default", "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="sliding_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        rope_params = {"rope_type": "linear", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        rope_params = {"rope_type": "dynamic", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
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
        # Scaling should be over the entire input
        rope_params = {"rope_type": "yarn", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
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


@slow
@require_torch
class Olmo3InternalIntegrationTest(unittest.TestCase):
    # Uses someone's personal repo, keeping it to have extensive testing
    model = None
    processor = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.model = Olmo3ForCausalLM.from_pretrained("shanearora/2025-sep-a-base-model", device_map="auto")
        cls.tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        out = self.model(torch.tensor(input_ids, device=torch_device)).logits.float()
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

    def test_model_7b_greedy_generation(self):
        expectations = Expectations(
            {
                ("cuda", None): """Simply put, the theory of relativity states that 1) the laws of physics are the same for all observers, and 2) the speed of light is the same for all observers. The first part of the theory is called the principle of relativity, and the second part is called the principle of the constancy of the speed of light. The theory of rel""",
            }
        )  # fmt: skip
        prompt = "Simply put, the theory of relativity states that "
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        # greedy generation outputs
        generated_ids = self.model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(expectations.get_expectation(), text)

    @pytest.mark.torch_export_test
    def test_export_static_cache(self):
        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the laws of physics are the same for all observers, and 2",
        ]
        max_generation_length = self.tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model on CPU, dont use `self.model` on `torch_device`
        # TODO (Ilyas / export experts): should be on `torch_device`, but causes GPU OOM
        device = "cpu"
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
            "shanearora/2025-sep-a-base-model",
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=generation_config,
        )

        prompts = ["Simply put, the theory of relativity states that "]
        prompt_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + eager
        eager_generated_ids = model.generate(
            **prompt_tokens, max_new_tokens=max_new_tokens, do_sample=False, cache_implementation=cache_implementation
        )
        eager_generated_text = self.tokenizer.batch_decode(eager_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, eager_generated_text)

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = self.tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)


@slow
@require_torch
class Olmo3IntegrationTest(unittest.TestCase):
    model_id = "allenai/Olmo-3-7B-Instruct"
    model = None
    processor = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.model = Olmo3ForCausalLM.from_pretrained(cls.model_id, device_map="auto")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_real_model_7b_greedy_generation(self):
        expectations = Expectations(
            {
                ("cuda", None): 'system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>\nuser\nWho would win in a fight - a dinosaur or a cow named Moo Moo?\nassistant\nThis is a fun and imaginative question! Let’s break it down:\n\n### 1. **A Dinosaur (General Case)**\nDinosaurs were a huge and diverse group, spanning from tiny feathered raptors to massive sauropods like *Brachiosaurus* or *Tyrannosaurus rex',
            }
        )  # fmt: skip

        message = [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}]
        inputs = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(expectations.get_expectation(), text)

    def test_real_model_7b_greedy_generation_batched(self):
        expectations = Expectations(
            {
                ("cuda", None): [
                    'system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>\nuser\nWho would win in a fight - a dinosaur or a cow named Moo Moo?\nassistant\nThis is a fun and imaginative question! Let’s break it down:\n\n### 1. **A Dinosaur (General Case)**\nDinosaurs were a huge and diverse group, spanning from tiny feathered raptors to massive sauropods like *Brachiosaurus* or *Tyrannosaurus rex',
                    'system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>\nuser\nSimply put, the theory of relativity\nassistant\nSure! In simple terms, **the theory of relativity** is Einstein’s explanation of how space, time, and gravity work. It has two main parts:\n\n1. **Special Relativity (1905):**  \n   This says that the laws of physics are the same for everyone moving at a constant speed (',
                ],
            }
        )  # fmt: skip

        message = [
            [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}],
            [{"role": "user", "content": "Simply put, the theory of relativity"}],
        ]
        inputs = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(expectations.get_expectation(), texts)

    def test_generate_beyond_sliding_window(self):
        expectations = Expectations(
            {
                ("cuda", None): """It looks like you've pasted a very lengthy and repetitive list of "This is a nice place""",
            }
        )  # fmt: skip

        # This is larger than 4096 tokens
        message = [
            {
                "role": "user",
                "content": "This is a nice place. " * 800 + "I really enjoy the scenery,",
            }
        ]
        inputs = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > self.model.config.sliding_window)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20, top_p=None, temperature=1, do_sample=False)
        text = self.tokenizer.decode(generated_ids[0, input_size:], skip_special_tokens=True)
        self.assertEqual(expectations.get_expectation(), text)
