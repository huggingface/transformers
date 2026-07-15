# Copyright 2026 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GraniteSWA model."""

import tempfile
import unittest

import pytest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, GraniteSWAForCausalLM, GraniteSWAModel


class GraniteSWAModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GraniteSWAModel
        # With the default `num_hidden_layers=2`, `layer_types` resolves to
        # ["full_attention", "sliding_attention"], so both attention paths are exercised. The default
        # `sliding_window` stays larger than the short test sequences (matching gemma2/gpt_oss); the
        # functional beyond-the-window behavior is covered by the slow integration tests.


@require_torch
class GraniteSWAModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = GraniteSWAModelTester

    @require_kernels
    @pytest.mark.flash_attn_test
    @require_torch_gpu
    # Copied from gpt_oss (swa backend handling)
    def test_default_flash_implementation_auto_correction(self):
        """An unsupported flash implementation is auto-corrected to `_compatible_flash_implementations`."""
        from kernels import get_kernel

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        expected_kernel = "kernels-community/vllm-flash-attn3"
        if get_kernel(expected_kernel) is None:
            self.skipTest(f"{expected_kernel} is not available, skipping auto-correction test.")

        # Auto correction on setting config at init time.
        config._attn_implementation = "flash_attention_2"
        tmp_model = GraniteSWAModel(config).to(device=torch_device, dtype=torch.bfloat16)
        self.assertEqual(tmp_model.config._attn_implementation, expected_kernel)

        # Auto correction at load time.
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_model.save_pretrained(tmp_dir_name)
            model = GraniteSWAModel.from_pretrained(tmp_dir_name, attn_implementation="flash_attention_2").to(
                device=torch_device
            )
            self.assertEqual(model.config._attn_implementation, expected_kernel)

        # Auto correction via `set_attn_implementation`.
        model.set_attn_implementation("eager")
        self.assertEqual(model.config._attn_implementation, "eager")
        model.set_attn_implementation("flash_attention_2")
        self.assertEqual(model.config._attn_implementation, expected_kernel)

        with torch.no_grad():
            output = model(**inputs_dict)
        self.assertIsNotNone(output)

    @unittest.skip("GraniteSWA does not support FlashAttention-2 (only FA3/FA4).")
    def test_flash_attn_2_equivalence(self):
        pass


@slow
@require_torch_accelerator
class GraniteSWAIntegrationTest(unittest.TestCase):
    model_id = "ibm-granite/granite-swash-2b"
    input_text = "The capital of France is"

    def test_model_logits_bf16(self):
        model = GraniteSWAForCausalLM.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        input_ids = tokenizer(self.input_text, return_tensors="pt").input_ids.to(torch_device)

        with torch.no_grad():
            out = model(input_ids)

        # fmt: off
        EXPECTED_MEANS = Expectations(
            {
                ("cuda", 8): torch.tensor([[-0.2178, -0.6719, -0.1885, 0.6484, -2.4375]]),
            }
        )
        EXPECTED_SLICES = Expectations(
            {
                ("cuda", 8): torch.tensor([[2.3125, 5.6562, 1.3047, 2.2969, 3.1562, 0.3711, 4.2812, 1.4688, 3.4531, 3.4531, 2.7188, 5.8125, 3.7812, 4.9062, 2.3906]]),
            }
        )
        # fmt: on
        torch.testing.assert_close(
            EXPECTED_MEANS.get_expectation().to(torch_device), out.logits.mean(-1).float(), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            EXPECTED_SLICES.get_expectation().to(torch_device), out.logits[0, 0, :15].float(), rtol=1e-3, atol=1e-3
        )

    def test_model_generation(self):
        model = GraniteSWAForCausalLM.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        EXPECTED_TEXT = (
            "The capital of France is Paris.\nThe capital of France is located in the north of the "
            "country.\nThe capital of France is"
        )
        self.assertEqual(generated_text, EXPECTED_TEXT)
