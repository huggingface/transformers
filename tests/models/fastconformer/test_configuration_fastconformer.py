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
"""Testing suite for the FastConformer configuration."""

import unittest

from transformers.models.fastconformer import FastConformerConfig


class FastConformerConfigTester:
    def __init__(
        self,
        parent,
        vocab_size=1024,
        d_model=256,
        encoder_layers=4,
        encoder_attention_heads=4,
        encoder_ffn_dim=1024,
        encoder_layerdrop=0.1,
        activation_function="silu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        initializer_range=0.02,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        use_bias=False,
        num_mel_bins=128,
        xscaling=False,
        dropout_emb=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layerdrop = encoder_layerdrop
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.use_bias = use_bias
        self.num_mel_bins = num_mel_bins
        self.xscaling = xscaling
        self.dropout_emb = dropout_emb
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def create_and_test_config_common_properties(self):
        config = self.get_config()
        self.parent.assertTrue(hasattr(config, "vocab_size"))
        self.parent.assertTrue(hasattr(config, "d_model"))
        self.parent.assertTrue(hasattr(config, "encoder_layers"))
        self.parent.assertTrue(hasattr(config, "encoder_attention_heads"))

    def get_config(self):
        return FastConformerConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_layerdrop=self.encoder_layerdrop,
            activation_function=self.activation_function,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            initializer_range=self.initializer_range,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            use_bias=self.use_bias,
            num_mel_bins=self.num_mel_bins,
            xscaling=self.xscaling,
            dropout_emb=self.dropout_emb,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        inputs_dict = {}
        return config, inputs_dict


class FastConformerConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_tester = FastConformerConfigTester(self)

    def test_config(self):
        self.config_tester.create_and_test_config_common_properties()

    def test_config_common_properties(self):
        config = FastConformerConfig()
        self.assertTrue(hasattr(config, "vocab_size"))
        self.assertTrue(hasattr(config, "d_model"))
        self.assertTrue(hasattr(config, "encoder_layers"))
        self.assertTrue(hasattr(config, "encoder_attention_heads"))

    def test_config_defaults(self):
        """Test that the default configuration matches expected values."""
        config = FastConformerConfig()

        # Test default values
        self.assertEqual(config.vocab_size, 1024)
        self.assertEqual(config.d_model, 1024)
        self.assertEqual(config.encoder_layers, 24)
        self.assertEqual(config.encoder_attention_heads, 8)
        self.assertEqual(config.encoder_ffn_dim, 4096)
        self.assertEqual(config.encoder_layerdrop, 0.1)
        self.assertEqual(config.activation_function, "silu")
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.attention_dropout, 0.1)
        self.assertEqual(config.activation_dropout, 0.1)
        self.assertEqual(config.initializer_range, 0.02)
        self.assertEqual(config.conv_kernel_size, 9)
        self.assertEqual(config.subsampling_factor, 8)
        self.assertEqual(config.subsampling_conv_channels, 256)
        self.assertEqual(config.use_bias, False)
        self.assertEqual(config.num_mel_bins, 128)
        self.assertEqual(config.xscaling, False)
        self.assertEqual(config.dropout_emb, 0.0)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.bos_token_id, 1)
        self.assertEqual(config.eos_token_id, 2)

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = FastConformerConfig(
            vocab_size=2048,
            d_model=512,
            encoder_layers=12,
            encoder_attention_heads=16,
            encoder_ffn_dim=2048,
            activation_function="relu",
            dropout=0.2,
            conv_kernel_size=15,
            subsampling_factor=4,
            num_mel_bins=80,
            xscaling=True,
            use_bias=True,
        )

        self.assertEqual(config.vocab_size, 2048)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.encoder_layers, 12)
        self.assertEqual(config.encoder_attention_heads, 16)
        self.assertEqual(config.encoder_ffn_dim, 2048)
        self.assertEqual(config.activation_function, "relu")
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.conv_kernel_size, 15)
        self.assertEqual(config.subsampling_factor, 4)
        self.assertEqual(config.num_mel_bins, 80)
        self.assertEqual(config.xscaling, True)
        self.assertEqual(config.use_bias, True)

    def test_config_model_type(self):
        """Test that model_type is correctly set."""
        config = FastConformerConfig()
        self.assertEqual(config.model_type, "fastconformer")

    def test_config_attribute_map(self):
        """Test that attribute mapping works correctly."""
        config = FastConformerConfig()

        # Test attribute map
        self.assertEqual(config.attribute_map["num_attention_heads"], "encoder_attention_heads")
        self.assertEqual(config.attribute_map["hidden_size"], "d_model")

        # Test that mapped attributes work
        self.assertEqual(config.num_attention_heads, config.encoder_attention_heads)
        self.assertEqual(config.hidden_size, config.d_model)

    def test_config_keys_to_ignore_at_inference(self):
        """Test that keys_to_ignore_at_inference is properly set."""
        config = FastConformerConfig()
        self.assertEqual(config.keys_to_ignore_at_inference, ["past_key_values"])

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = FastConformerConfig(
            d_model=512,
            encoder_layers=6,
            encoder_attention_heads=8,
            num_mel_bins=80,
        )

        # Test to_dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict["d_model"], 512)
        self.assertEqual(config_dict["encoder_layers"], 6)
        self.assertEqual(config_dict["encoder_attention_heads"], 8)
        self.assertEqual(config_dict["num_mel_bins"], 80)
        self.assertEqual(config_dict["model_type"], "fastconformer")

        # Test from_dict
        new_config = FastConformerConfig.from_dict(config_dict)
        self.assertEqual(new_config.d_model, 512)
        self.assertEqual(new_config.encoder_layers, 6)
        self.assertEqual(new_config.encoder_attention_heads, 8)
        self.assertEqual(new_config.num_mel_bins, 80)

    def test_config_nemo_compatibility(self):
        """Test NeMo-specific configuration attributes."""
        config = FastConformerConfig()

        # Test NeMo-specific attributes
        self.assertTrue(hasattr(config, "subsampling_factor"))
        self.assertTrue(hasattr(config, "subsampling_conv_channels"))
        self.assertTrue(hasattr(config, "conv_kernel_size"))
        self.assertTrue(hasattr(config, "xscaling"))
        self.assertTrue(hasattr(config, "dropout_emb"))

        # Test that these can be set to NeMo-typical values
        config = FastConformerConfig(
            subsampling_factor=8,
            subsampling_conv_channels=256,
            conv_kernel_size=9,
            xscaling=True,
            dropout_emb=0.1,
        )

        self.assertEqual(config.subsampling_factor, 8)
        self.assertEqual(config.subsampling_conv_channels, 256)
        self.assertEqual(config.conv_kernel_size, 9)
        self.assertEqual(config.xscaling, True)
        self.assertEqual(config.dropout_emb, 0.1)

    def test_config_validation(self):
        """Test configuration validation edge cases."""
        # Test valid configuration
        config = FastConformerConfig(
            d_model=768,
            encoder_attention_heads=12,
        )
        # Should not raise error if d_model is divisible by num_attention_heads
        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.encoder_attention_heads, 12)

        # Test odd kernel size (should be valid)
        config = FastConformerConfig(conv_kernel_size=7)
        self.assertEqual(config.conv_kernel_size, 7)

        # Test power-of-2 subsampling factor
        config = FastConformerConfig(subsampling_factor=4)
        self.assertEqual(config.subsampling_factor, 4)

        config = FastConformerConfig(subsampling_factor=16)
        self.assertEqual(config.subsampling_factor, 16)
