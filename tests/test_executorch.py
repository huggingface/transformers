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
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from transformers import AutoModelForCausalLM, set_seed
from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.executorch import (
    TorchExportableModuleForDecoderOnlyLM,
    TorchExportableModuleWithHybridCache,
    TorchExportableModuleWithStaticCache,
    _build_example_modality_inputs,
    _make_example_audio_inputs,
    _make_example_vision_inputs,
)
from transformers.testing_utils import require_torch, slow


@require_torch
class ExecutorchTest(unittest.TestCase):
    def setUp(self):
        set_seed(42)
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


@require_torch
class ExampleModalityInputsTest(unittest.TestCase):
    """Tests for modality input helpers used by convert_and_export_with_cache."""

    def _make_model_with_vision_config(self, image_size=32, num_channels=3):
        model = MagicMock()
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=image_size, num_channels=num_channels)
        )
        return model

    def _make_model_with_audio_config(self, num_mel_bins=80):
        model = MagicMock()
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(num_mel_bins=num_mel_bins)
        return model

    def _make_text_only_model(self):
        model = MagicMock()
        model.device = torch.device("cpu")
        model.config = SimpleNamespace()
        return model

    def test_vision_inputs_shape(self):
        model = self._make_model_with_vision_config(image_size=32, num_channels=3)
        result = _make_example_vision_inputs(model, model.device)
        self.assertIn("pixel_values", result)
        self.assertEqual(result["pixel_values"].shape, (1, 3, 32, 32))
        self.assertEqual(result["pixel_values"].dtype, torch.float32)

    def test_vision_inputs_custom_channels(self):
        model = self._make_model_with_vision_config(image_size=64, num_channels=1)
        result = _make_example_vision_inputs(model, model.device)
        self.assertEqual(result["pixel_values"].shape, (1, 1, 64, 64))

    def test_audio_inputs_shape(self):
        model = self._make_model_with_audio_config(num_mel_bins=80)
        result = _make_example_audio_inputs(model, model.device)
        self.assertIn("input_features", result)
        self.assertEqual(result["input_features"].shape, (1, 80, 3000))
        self.assertEqual(result["input_features"].dtype, torch.float32)

    def test_dispatch_text_only(self):
        model = self._make_text_only_model()
        result = _build_example_modality_inputs(model, model.device)
        self.assertEqual(result, {})

    def test_dispatch_vision(self):
        model = self._make_model_with_vision_config()
        result = _build_example_modality_inputs(model, model.device)
        self.assertIn("pixel_values", result)
        self.assertNotIn("input_features", result)

    def test_dispatch_audio(self):
        model = self._make_model_with_audio_config()
        result = _build_example_modality_inputs(model, model.device)
        self.assertIn("input_features", result)
        self.assertNotIn("pixel_values", result)

    def test_vision_takes_priority_over_audio(self):
        """vision_config check comes before num_mel_bins — multimodal models with both should get pixel_values."""
        model = MagicMock()
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=32, num_channels=3),
            num_mel_bins=80,
        )
        result = _build_example_modality_inputs(model, model.device)
        self.assertIn("pixel_values", result)
        self.assertNotIn("input_features", result)


@require_torch
class StaticCacheModuleModalityTest(unittest.TestCase):
    """Tests that TorchExportableModuleWithStaticCache.forward accepts modality inputs."""

    def setUp(self):
        set_seed(42)
        self.model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.model.eval()
        self.model.generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": 32, "device": "cpu"},
        )
        self.input_ids = torch.tensor([[1]], dtype=torch.long)
        self.cache_position = torch.tensor([0], dtype=torch.long)

    def test_forward_accepts_none_pixel_values(self):
        """pixel_values=None must not change output vs not passing it at all."""
        module = TorchExportableModuleWithStaticCache(self.model)
        out_default = module.forward(input_ids=self.input_ids, cache_position=self.cache_position)
        out_explicit_none = module.forward(
            input_ids=self.input_ids, cache_position=self.cache_position, pixel_values=None
        )
        torch.testing.assert_close(out_default, out_explicit_none)

    def test_forward_accepts_none_input_features(self):
        """input_features=None must not change output vs not passing it at all."""
        module = TorchExportableModuleWithStaticCache(self.model)
        out_default = module.forward(input_ids=self.input_ids, cache_position=self.cache_position)
        out_explicit_none = module.forward(
            input_ids=self.input_ids, cache_position=self.cache_position, input_features=None
        )
        torch.testing.assert_close(out_default, out_explicit_none)

    def test_forward_passes_pixel_values_to_model(self):
        """pixel_values and input_features are forwarded to the underlying model when non-None."""
        module = TorchExportableModuleWithStaticCache(self.model)

        pixel_values = torch.zeros(1, 3, 32, 32)
        input_features = torch.zeros(1, 80, 3000)

        captured = {}

        original_forward = self.model.forward

        def capturing_forward(*args, **kwargs):
            captured.update(kwargs)
            return original_forward(*args, **kwargs)

        self.model.forward = capturing_forward

        # pixel_values forwarded
        module.forward(input_ids=self.input_ids, cache_position=self.cache_position, pixel_values=pixel_values)
        self.assertIn("pixel_values", captured)
        self.assertTrue(torch.equal(captured["pixel_values"], pixel_values))

        captured.clear()

        # input_features forwarded
        module.forward(input_ids=self.input_ids, cache_position=self.cache_position, input_features=input_features)
        self.assertIn("input_features", captured)
        self.assertTrue(torch.equal(captured["input_features"], input_features))

        captured.clear()

        # Neither forwarded when both are None
        module.forward(input_ids=self.input_ids, cache_position=self.cache_position)
        self.assertNotIn("pixel_values", captured)
        self.assertNotIn("input_features", captured)

        self.model.forward = original_forward
