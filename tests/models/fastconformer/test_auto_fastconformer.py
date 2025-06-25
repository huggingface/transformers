# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for FastConformer auto integration."""

import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModel,
    )
    from transformers.models.fastconformer import (
        FastConformerConfig,
        FastConformerFeatureExtractor,
        FastConformerModel,
    )


@require_torch
class FastConformerAutoIntegrationTest(unittest.TestCase):
    """Test that FastConformer integrates properly with the Auto classes."""

    def test_config_auto_integration(self):
        """Test that FastConformerConfig works with AutoConfig."""
        # Test from_pretrained with model_type
        config_dict = {
            "model_type": "fastconformer",
            "d_model": 256,
            "encoder_layers": 4,
            "encoder_attention_heads": 8,
            "num_mel_bins": 80,
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save config
            config = FastConformerConfig(**config_dict)
            config.save_pretrained(tmp_dir)
            
            # Load with AutoConfig
            auto_config = AutoConfig.from_pretrained(tmp_dir)
            
            self.assertIsInstance(auto_config, FastConformerConfig)
            self.assertEqual(auto_config.model_type, "fastconformer")
            self.assertEqual(auto_config.d_model, 256)
            self.assertEqual(auto_config.encoder_layers, 4)
            self.assertEqual(auto_config.encoder_attention_heads, 8)
            self.assertEqual(auto_config.num_mel_bins, 80)

    def test_feature_extractor_auto_integration(self):
        """Test that FastConformerFeatureExtractor works with AutoFeatureExtractor."""
        # Test direct instantiation
        feature_extractor = FastConformerFeatureExtractor(
            feature_size=80,
            sampling_rate=16000,
            normalize="per_feature",
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save feature extractor
            feature_extractor.save_pretrained(tmp_dir)
            
            # Load with AutoFeatureExtractor
            auto_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir)
            
            self.assertIsInstance(auto_feature_extractor, FastConformerFeatureExtractor)
            self.assertEqual(auto_feature_extractor.feature_size, 80)
            self.assertEqual(auto_feature_extractor.sampling_rate, 16000)
            self.assertEqual(auto_feature_extractor.normalize, "per_feature")

    def test_model_auto_integration(self):
        """Test that FastConformerModel works with AutoModel."""
        # Create a small config for testing
        config = FastConformerConfig(
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
            num_mel_bins=80,
        )
        
        # Create model
        model = FastConformerModel(config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            model.save_pretrained(tmp_dir)
            
            # Load with AutoModel
            auto_model = AutoModel.from_pretrained(tmp_dir)
            
            self.assertIsInstance(auto_model, FastConformerModel)
            self.assertEqual(auto_model.config.model_type, "fastconformer")
            self.assertEqual(auto_model.config.d_model, 64)
            self.assertEqual(auto_model.config.encoder_layers, 2)

    def test_end_to_end_auto_pipeline(self):
        """Test end-to-end processing using Auto classes."""
        # Create config and model
        config = FastConformerConfig(
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
            num_mel_bins=80,
        )
        
        model = FastConformerModel(config)
        feature_extractor = FastConformerFeatureExtractor(feature_size=80)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save everything
            model.save_pretrained(tmp_dir)
            feature_extractor.save_pretrained(tmp_dir)
            
            # Load with Auto classes
            auto_config = AutoConfig.from_pretrained(tmp_dir)
            auto_model = AutoModel.from_pretrained(tmp_dir)
            auto_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir)
            
            # Test processing
            auto_model.to(torch_device)
            auto_model.eval()
            
            # Create dummy audio
            raw_audio = torch.randn(2, 8000)  # 0.5 seconds of audio
            audio_lengths = torch.tensor([8000, 6000])
            
            # Pad to same length
            padded_audio = torch.zeros(2, 8000)
            padded_audio[0] = raw_audio[0]
            padded_audio[1, :6000] = raw_audio[1, :6000]
            
            # Process through feature extractor
            features = auto_feature_extractor(
                padded_audio, audio_lengths=audio_lengths, return_tensors="pt"
            )
            
                    # Process through model
        with torch.no_grad():
            outputs = auto_model(
                input_features=features.input_features.to(torch_device),
                attention_mask=features.attention_mask.to(torch_device),
                input_lengths=features.input_lengths.to(torch_device),
            )
            
            # Check outputs
            self.assertIsNotNone(outputs.last_hidden_state)
            self.assertEqual(outputs.last_hidden_state.shape[0], 2)  # batch size
            self.assertEqual(outputs.last_hidden_state.shape[2], 64)  # hidden size

    def test_auto_model_type_registration(self):
        """Test that 'fastconformer' model type is properly registered."""
        # Test that the model type maps to the correct classes
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
        from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
        
        # Check that fastconformer is in the mappings
        self.assertIn("fastconformer", CONFIG_MAPPING_NAMES)
        self.assertEqual(CONFIG_MAPPING_NAMES["fastconformer"], "FastConformerConfig")
        self.assertIn("fastconformer", FEATURE_EXTRACTOR_MAPPING_NAMES)
        self.assertEqual(FEATURE_EXTRACTOR_MAPPING_NAMES["fastconformer"], "FastConformerFeatureExtractor")
        self.assertIn("fastconformer", MODEL_MAPPING_NAMES)
        self.assertEqual(MODEL_MAPPING_NAMES["fastconformer"], "FastConformerModel")

    def test_config_auto_instantiation(self):
        """Test that AutoConfig can instantiate FastConformerConfig from model_type."""
        # Create config dict with just model_type
        config_dict = {"model_type": "fastconformer"}
        
        # Use the CONFIG_MAPPING to get the config class
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        config_class = CONFIG_MAPPING["fastconformer"]
        config = config_class(**config_dict)
        
        self.assertIsInstance(config, FastConformerConfig)
        self.assertEqual(config.model_type, "fastconformer")
        # Check some default values
        self.assertEqual(config.d_model, 1024)
        self.assertEqual(config.encoder_layers, 24)
        self.assertEqual(config.num_mel_bins, 128)

    def test_feature_extractor_from_model_type(self):
        """Test creating feature extractor from model type."""
        # This tests the internal mapping
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "hf-internal-testing/tiny-random-fastconformer", trust_remote_code=False
        ) if False else None  # Skip if no test model available
        
        # For now, just test that the class is properly mapped
        from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING
        
        # Check that fastconformer maps to FastConformerFeatureExtractor
        fastconformer_config = FastConformerConfig()
        mapped_class = FEATURE_EXTRACTOR_MAPPING[type(fastconformer_config)]
        self.assertEqual(mapped_class, FastConformerFeatureExtractor)

    def test_model_from_config_type(self):
        """Test creating model from config type using AutoModel."""
        config = FastConformerConfig(
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=2,
            num_mel_bins=40,
        )
        
        # Create model using AutoModel
        model = AutoModel.from_config(config)
        
        self.assertIsInstance(model, FastConformerModel)
        self.assertEqual(model.config.d_model, 32)
        self.assertEqual(model.config.encoder_layers, 1)

    def test_auto_classes_consistency(self):
        """Test that all Auto classes work together consistently."""
        # Create a complete setup
        config = FastConformerConfig(
            d_model=48,
            encoder_layers=2,
            encoder_attention_heads=3,
            num_mel_bins=64,
        )
        
        # Test AutoModel.from_config
        model1 = AutoModel.from_config(config)
        
        # Test FastConformerModel directly
        model2 = FastConformerModel(config)
        
        # Both should be the same type
        self.assertEqual(type(model1), type(model2))
        self.assertEqual(model1.config.d_model, model2.config.d_model)
        
        # Test that they produce the same architecture
        self.assertEqual(
            sum(p.numel() for p in model1.parameters()),
            sum(p.numel() for p in model2.parameters())
        )

    def test_save_and_load_consistency(self):
        """Test that save/load cycle preserves model functionality."""
        # Create original model
        config = FastConformerConfig(
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=2,
            num_mel_bins=40,
        )
        
        original_model = FastConformerModel(config)
        original_model.to(torch_device)
        original_model.eval()
        
                # Create test input
        test_input = torch.randn(1, 100, 40).to(torch_device)  # (batch, time, mel_bins)
        test_lengths = torch.tensor([100]).to(torch_device)

        # Get original output
        with torch.no_grad():
            original_output = original_model(test_input, input_lengths=test_lengths)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save using original model
            original_model.save_pretrained(tmp_dir)
            
            # Load using AutoModel
            loaded_model = AutoModel.from_pretrained(tmp_dir)
            loaded_model.to(torch_device)
            loaded_model.eval()
            
            # Get loaded output
            with torch.no_grad():
                loaded_output = loaded_model(test_input, input_lengths=test_lengths)
            
            # Outputs should be identical
            self.assertTrue(torch.allclose(
                original_output.last_hidden_state,
                loaded_output.last_hidden_state,
                atol=1e-6
            ))

    def test_model_type_inference(self):
        """Test that model type is correctly inferred from config."""
        config_dict = {
            "model_type": "fastconformer",
            "d_model": 128,
            "encoder_layers": 3,
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save both config and model
            config = FastConformerConfig(**config_dict)
            model = FastConformerModel(config)
            
            config.save_pretrained(tmp_dir)
            model.save_pretrained(tmp_dir)
            
            # AutoConfig should infer the type correctly
            inferred_config = AutoConfig.from_pretrained(tmp_dir)
            
            self.assertEqual(inferred_config.model_type, "fastconformer")
            self.assertIsInstance(inferred_config, FastConformerConfig)
            
            # AutoModel should create the right model type
            auto_model = AutoModel.from_pretrained(tmp_dir)
            self.assertIsInstance(auto_model, FastConformerModel) 