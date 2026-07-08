# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GraniteMoeSWA model."""

import tempfile
import unittest

import pytest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_kernels,
    require_torch,
    require_torch_gpu,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import GraniteMoeSWAModel


class GraniteMoeSWAModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GraniteMoeSWAModel
        # With the default `num_hidden_layers=2`, `layer_types` resolves to
        # ["full_attention", "sliding_attention"], so both attention paths are exercised. The default
        # `sliding_window` stays larger than the short test sequences (matching gemma2/gpt_oss). Shared
        # experts stay disabled (`shared_intermediate_size=0`), matching the model's default.


@require_torch
class GraniteMoeSWAModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = GraniteMoeSWAModelTester

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
        tmp_model = GraniteMoeSWAModel(config).to(device=torch_device, dtype=torch.bfloat16)
        self.assertEqual(tmp_model.config._attn_implementation, expected_kernel)

        # Auto correction at load time.
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_model.save_pretrained(tmp_dir_name)
            model = GraniteMoeSWAModel.from_pretrained(tmp_dir_name, attn_implementation="flash_attention_2").to(
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

    @unittest.skip("GraniteMoeSWA forcefully disables SDPA due to the attention sink.")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("GraniteMoeSWA eager and SDPA attention outputs are expected to differ (sink).")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip("GraniteMoeSWA does not support FlashAttention-2 (only FA3/FA4).")
    def test_flash_attn_2_equivalence(self):
        pass

    # The tensor-parallel test mixin hardcodes `attn_implementation="sdpa"`, which GraniteMoeSWA disables
    # because of the attention sink. The model still ships a valid `base_model_tp_plan` for real TP usage.
    @unittest.skip("TP test mixin forces attn_implementation='sdpa', unsupported by GraniteMoeSWA (sink).")
    def test_tp_forward(self):
        pass

    @unittest.skip("TP test mixin forces attn_implementation='sdpa', unsupported by GraniteMoeSWA (sink).")
    def test_tp_backward(self):
        pass

    @unittest.skip("TP test mixin forces attn_implementation='sdpa', unsupported by GraniteMoeSWA (sink).")
    def test_tp_generation(self):
        pass
