# Copyright 2025 HuggingFace Inc.
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

import unittest

import torch

from transformers import AutoModelForCausalLM, set_seed
from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.executorch import (
    TorchExportableModuleForDecoderOnlyLM,
    TorchExportableModuleWithHybridCache,
    TorchExportableModuleWithStaticCache,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_3
from transformers.testing_utils import require_torch


@require_torch
class ExecutorchTest(unittest.TestCase):
    def setUp(self):
        if not is_torch_greater_or_equal_than_2_3:
            self.skipTest("torch >= 2.3 is required")

        set_seed(0)
        self.model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.model.eval()

        # Create generation config with static cache for the model
        self.model.generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": 32, "device": "cpu"},
        )

        self.input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        self.inputs_embeds = torch.randn(1, 3, self.model.config.hidden_size)
        self.cache_position = torch.arange(3, dtype=torch.long)

    def test_static_cache_module_forward(self):
        """Test TorchExportableModuleWithStaticCache forward with both input types"""
        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": 32, "device": "cpu"},
        )

        # Set generation config on model
        self.model.generation_config = generation_config
        module = TorchExportableModuleWithStaticCache(self.model)

        # Test with input_ids
        eager_output_ids = self.model(input_ids=self.input_ids, use_cache=False).logits
        wrapped_output_ids = module.forward(input_ids=self.input_ids, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_ids, wrapped_output_ids, atol=1e-4, rtol=1e-4)

        # Test with inputs_embeds
        eager_output_embeds = self.model(inputs_embeds=self.inputs_embeds, use_cache=False).logits
        wrapped_output_embeds = module.forward(inputs_embeds=self.inputs_embeds, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_embeds, wrapped_output_embeds, atol=1e-4, rtol=1e-4)

    def test_hybrid_cache_module_forward(self):
        """Test TorchExportableModuleWithHybridCache forward with both input types"""
        config = self.model.config
        config.sliding_window = 16
        config.layer_types = ["full_attention"] * config.num_hidden_layers

        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="hybrid",
            cache_config={"batch_size": 1, "max_cache_len": 32, "device": "cpu"},
        )

        # Set generation config on model
        self.model.generation_config = generation_config
        module = TorchExportableModuleWithHybridCache(self.model)

        # Test with input_ids
        eager_output_ids = self.model(input_ids=self.input_ids, use_cache=False).logits
        wrapped_output_ids = module.forward(input_ids=self.input_ids, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_ids, wrapped_output_ids, atol=1e-4, rtol=1e-4)

        # Test with inputs_embeds
        eager_output_embeds = self.model(inputs_embeds=self.inputs_embeds, use_cache=False).logits
        wrapped_output_embeds = module.forward(inputs_embeds=self.inputs_embeds, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_embeds, wrapped_output_embeds, atol=1e-4, rtol=1e-4)

    def test_decoder_only_lm_export_validation(self):
        """Test TorchExportableModuleForDecoderOnlyLM export validation"""
        module = TorchExportableModuleForDecoderOnlyLM(self.model)

        # Should fail with both input_ids and inputs_embeds
        with self.assertRaises(ValueError):
            module.export(input_ids=self.input_ids, inputs_embeds=self.inputs_embeds)

        # Should fail with neither
        with self.assertRaises(ValueError):
            module.export()

    def test_decoder_only_lm_export(self):
        """Test TorchExportableModuleForDecoderOnlyLM export with both input types"""
        module = TorchExportableModuleForDecoderOnlyLM(self.model)

        # Test export with input_ids
        exported_program_ids = module.export(input_ids=self.input_ids, cache_position=self.cache_position)
        eager_output_ids = self.model(input_ids=self.input_ids, use_cache=False).logits
        exported_output_ids = exported_program_ids.module()(
            input_ids=self.input_ids, cache_position=self.cache_position
        )
        torch.testing.assert_close(eager_output_ids, exported_output_ids, atol=1e-4, rtol=1e-4)

        # Test export with inputs_embeds
        exported_program_embeds = module.export(inputs_embeds=self.inputs_embeds, cache_position=self.cache_position)
        eager_output_embeds = self.model(inputs_embeds=self.inputs_embeds, use_cache=False).logits
        exported_output_embeds = exported_program_embeds.module()(
            inputs_embeds=self.inputs_embeds, cache_position=self.cache_position
        )
        torch.testing.assert_close(eager_output_embeds, exported_output_embeds, atol=1e-4, rtol=1e-4)
