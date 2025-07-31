# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from unittest.mock import Mock, patch

import torch

from transformers import HfArgumentParser
from transformers.integrations.executorch import (
    ImageEncoderExportableModule,
    TorchExportableModuleForImageTextLM,
    TorchExportableModuleWithHybridCache,
)
from transformers.testing_utils import require_torch


@require_torch
class ExecuTorchMultimodalTest(unittest.TestCase):
    def setUp(self):
        # Mock multimodal model configuration
        self.mock_config = Mock()
        self.mock_config.text_config = Mock()
        self.mock_config.text_config.use_cache = True
        self.mock_config.text_config.hidden_size = 768
        self.mock_config.text_config.num_hidden_layers = 12
        self.mock_config.vision_config = Mock()
        self.mock_config.vision_config.image_size = 224

        # Mock model
        self.mock_model = Mock()
        self.mock_model.config = self.mock_config
        self.mock_model.device = torch.device("cpu")
        self.mock_model.dtype = torch.float32

    def test_hybrid_cache_inputs_embeds_support(self):
        """Test that TorchExportableModuleWithHybridCache supports inputs_embeds"""
        with patch("transformers.integrations.executorch.HybridCache") as MockCache:
            # Create exportable module
            exportable = TorchExportableModuleWithHybridCache(self.mock_model)
            
            # Test forward with inputs_embeds
            batch_size, seq_len, hidden_size = 1, 3, 768
            inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
            cache_position = torch.arange(seq_len)
            
            # Mock model output
            mock_output = Mock()
            mock_output.logits = torch.randn(batch_size, seq_len, 32000)  # vocab_size
            self.mock_model.return_value = mock_output
            
            # Call forward
            result = exportable.forward(inputs_embeds=inputs_embeds, cache_position=cache_position)
            
            # Verify model was called with inputs_embeds
            self.mock_model.assert_called_once()
            call_kwargs = self.mock_model.call_args[1]
            self.assertIn("inputs_embeds", call_kwargs)
            self.assertIsNone(call_kwargs["input_ids"])
            torch.testing.assert_close(call_kwargs["inputs_embeds"], inputs_embeds)

    def test_hybrid_cache_multimodal_config(self):
        """Test that TorchExportableModuleWithHybridCache uses text_config for multimodal models"""
        with patch("transformers.integrations.executorch.HybridCache") as MockCache:
            # Create exportable module
            exportable = TorchExportableModuleWithHybridCache(self.mock_model)
            
            # Verify HybridCache was initialized with text_config
            MockCache.assert_called_once()
            call_args = MockCache.call_args[1]
            self.assertEqual(call_args["config"], self.mock_config.text_config)

    def test_image_text_lm_module(self):
        """Test TorchExportableModuleForImageTextLM initialization"""
        with patch("transformers.integrations.executorch.TorchExportableModuleWithHybridCache") as MockWrapper:
            with patch("transformers.integrations.executorch.ALL_MASK_ATTENTION_FUNCTIONS"):
                with patch("transformers.integrations.executorch.ALL_ATTENTION_FUNCTIONS"):
                    # Create image-text LM module
                    exportable = TorchExportableModuleForImageTextLM(self.mock_model)
                    
                    # Verify it creates the appropriate wrapper
                    MockWrapper.assert_called_once_with(self.mock_model, 1, 4096)

    def test_image_encoder_module(self):
        """Test ImageEncoderExportableModule"""
        # Mock vision model
        mock_vision_tower = Mock()
        mock_vision_outputs = Mock()
        mock_vision_outputs.last_hidden_state = torch.randn(1, 196, 768)  # 14x14 patches
        mock_vision_tower.return_value = mock_vision_outputs
        
        mock_projector = Mock()
        mock_projector.return_value = torch.randn(1, 196, 768)  # projected features
        
        mock_model = Mock()
        mock_model.vision_tower = mock_vision_tower
        mock_model.multi_modal_projector = mock_projector
        
        # Create encoder module
        encoder = ImageEncoderExportableModule(mock_model)
        
        # Test forward pass
        pixel_values = torch.randn(1, 3, 224, 224)
        result = encoder.forward(pixel_values)
        
        # Verify calls
        mock_vision_tower.assert_called_once_with(pixel_values=pixel_values)
        mock_projector.assert_called_once_with(mock_vision_outputs.last_hidden_state)

    def test_error_handling(self):
        """Test error handling for invalid configurations"""
        # Test missing cache configuration
        bad_config = Mock()
        bad_config.text_config = Mock()
        bad_config.text_config.use_cache = False
        
        bad_model = Mock()
        bad_model.config = bad_config
        
        with self.assertRaises(ValueError):
            TorchExportableModuleForImageTextLM(bad_model)

    def test_forward_validation(self):
        """Test input validation in forward method"""
        with patch("transformers.integrations.executorch.HybridCache"):
            exportable = TorchExportableModuleWithHybridCache(self.mock_model)
            
            # Test missing both input_ids and inputs_embeds
            with self.assertRaises(ValueError):
                exportable.forward(cache_position=torch.tensor([0]))
            
            # Test missing cache_position
            with self.assertRaises(ValueError):
                exportable.forward(input_ids=torch.tensor([[1]]))


if __name__ == "__main__":
    unittest.main()

