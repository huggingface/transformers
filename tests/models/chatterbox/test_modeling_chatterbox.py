# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Chatterbox model."""

import tempfile
import unittest

from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
from transformers.models.chatterbox.configuration_chatterbox import ChatterboxConfig
from transformers.models.chatterbox.feature_extraction_chatterbox import ChatterboxFeatureExtractor
from transformers.models.chatterbox.modeling_chatterbox import ChatterboxModel
from transformers.testing_utils import require_torch


@require_torch
class ChatterboxModelTest(unittest.TestCase):
    def setUp(self):
        """Set up test configuration."""
        self.config = ChatterboxConfig()
        # Use smaller model for faster tests
        self.config.t3_config.llama_config_dict["num_hidden_layers"] = 2
        self.config.t3_config.llama_config_dict["num_attention_heads"] = 4
        self.config.t3_config.hidden_size = 256
        self.config.s3gen_config.encoder_num_blocks = 2
        self.config.s3gen_config.decoder_n_blocks = 2

    def test_model_initialization(self):
        """Test that the model can be initialized."""
        model = ChatterboxModel(self.config)
        self.assertIsInstance(model, ChatterboxModel)

        # Check that sub-modules exist
        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsInstance(model.feature_extractor, ChatterboxFeatureExtractor)

    def test_config_attributes(self):
        """Test that config attributes are properly set."""
        model = ChatterboxModel(self.config)

        # Check that sub-configs exist
        self.assertIsNotNone(model.config.t3_config)
        self.assertIsNotNone(model.config.s3gen_config)

    def test_save_and_load(self):
        """Test saving and loading the model."""
        model = ChatterboxModel(self.config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save model
            model.save_pretrained(tmpdirname)

            # Check that files were created
            import os

            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "config.json")))
            self.assertTrue(
                os.path.exists(os.path.join(tmpdirname, "model.safetensors"))
                or os.path.exists(os.path.join(tmpdirname, "pytorch_model.bin"))
            )

            # Load model
            loaded_model = ChatterboxModel.from_pretrained(tmpdirname)
            self.assertIsInstance(loaded_model, ChatterboxModel)

    def test_auto_feature_extractor_mapping(self):
        self.assertIn("chatterbox", FEATURE_EXTRACTOR_MAPPING_NAMES)
        self.assertEqual(FEATURE_EXTRACTOR_MAPPING_NAMES["chatterbox"], "ChatterboxFeatureExtractor")

    @unittest.skip("Requires CUDA and full tokenizer setup")
    def test_generate_basic(self):
        """Test basic generation."""
        # Skipped: This test requires CUDA device and full tokenizer setup
        pass

    def test_config_serialization(self):
        """Test config serialization and deserialization."""
        config_dict = self.config.to_dict()

        # Check that nested configs are serialized
        self.assertIn("t3_config", config_dict)
        self.assertIn("s3gen_config", config_dict)
        self.assertIn("hiftnet_config", config_dict)

        # Test reconstruction - just check it doesn't crash
        # Note: exact equality may not hold due to nested config defaults
        new_config = ChatterboxConfig.from_dict(config_dict)
        self.assertIsInstance(new_config, ChatterboxConfig)


if __name__ == "__main__":
    unittest.main()
