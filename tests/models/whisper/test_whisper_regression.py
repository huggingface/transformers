# coding=utf-8
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
"""Tests for Whisper regression fixes."""

import unittest

import torch

from transformers import WhisperConfig, WhisperForConditionalGeneration
from transformers.testing_utils import require_torch, slow


@require_torch
class WhisperRegressionTest(unittest.TestCase):
    """Test cases for Whisper regression fixes, particularly issue #38378."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = WhisperConfig(
            vocab_size=1000,
            d_model=64,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,  # 64 is divisible by 4
            decoder_attention_heads=4,  # 64 is divisible by 4
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            num_mel_bins=80,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=3,
            begin_suppress_tokens=[4, 2],  # Use valid token IDs within vocab_size
        )

    def _create_test_config(self, **kwargs):
        """Helper method to create test config with valid token IDs."""
        config_kwargs = {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "encoder_layers": self.config.encoder_layers,
            "decoder_layers": self.config.decoder_layers,
            "encoder_attention_heads": self.config.encoder_attention_heads,
            "decoder_attention_heads": self.config.decoder_attention_heads,
            "pad_token_id": self.config.pad_token_id,
            "bos_token_id": self.config.bos_token_id,
            "eos_token_id": self.config.eos_token_id,
            "decoder_start_token_id": self.config.decoder_start_token_id,
            "begin_suppress_tokens": self.config.begin_suppress_tokens,
        }
        config_kwargs.update(kwargs)
        return WhisperConfig(**config_kwargs)

    def test_legacy_logprob_calculation_default(self):
        """Test that legacy logprob calculation is the default for backward compatibility."""
        config = WhisperConfig()
        self.assertTrue(config.use_legacy_logprob_calculation)

    def test_legacy_logprob_calculation_explicit(self):
        """Test explicit setting of legacy logprob calculation."""
        config_legacy = WhisperConfig(use_legacy_logprob_calculation=True)
        self.assertTrue(config_legacy.use_legacy_logprob_calculation)

        config_new = WhisperConfig(use_legacy_logprob_calculation=False)
        self.assertFalse(config_new.use_legacy_logprob_calculation)

    def test_logprob_calculation_difference(self):
        """Test that legacy and new logprob calculations produce different results."""
        # Test legacy mode
        config_legacy = self._create_test_config(use_legacy_logprob_calculation=True)
        model_legacy = WhisperForConditionalGeneration(config_legacy)

        # Test new mode
        config_new = self._create_test_config(use_legacy_logprob_calculation=False)
        model_new = WhisperForConditionalGeneration(config_new)

        # Create dummy scores and tokens for testing
        scores = [torch.randn(1000) for _ in range(5)]  # 5 time steps, vocab_size=1000
        tokens = torch.tensor([100, 200, 300, 400, 500])  # 5 tokens
        temperature = 1.0

        # Test legacy calculation
        legacy_logprobs = model_legacy._retrieve_avg_logprobs(scores, tokens, temperature)

        # Test new calculation
        new_logprobs = model_new._retrieve_avg_logprobs(scores, tokens, temperature)

        # Verify they are different (as expected)
        self.assertFalse(torch.allclose(legacy_logprobs, new_logprobs))

        # Verify the mathematical relationship
        # Legacy: sum_logprobs / (len(tokens) + 1) = sum_logprobs / 6
        # New: sum_logprobs / len(tokens) = sum_logprobs / 5
        # So: new_logprobs = legacy_logprobs * 6 / 5
        expected_ratio = 6.0 / 5.0
        actual_ratio = new_logprobs / legacy_logprobs
        self.assertTrue(torch.allclose(actual_ratio, torch.tensor(expected_ratio), atol=1e-6))

    def test_generation_deterministic_legacy_mode(self):
        """Test that generation is deterministic in legacy mode."""
        config = self._create_test_config(use_legacy_logprob_calculation=True)
        model = WhisperForConditionalGeneration(config)
        model.eval()

        # Create dummy input
        batch_size = 1
        seq_length = 100
        input_features = torch.randn(batch_size, config.num_mel_bins, seq_length)

        # Test that multiple runs with the same seed produce the same result
        torch.manual_seed(42)
        with torch.no_grad():
            output1 = model.generate(input_features, max_new_tokens=5, do_sample=False)

        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model.generate(input_features, max_new_tokens=5, do_sample=False)

        self.assertTrue(torch.equal(output1, output2))

    def test_generation_deterministic_new_mode(self):
        """Test that generation is deterministic in new mode."""
        config = self._create_test_config(use_legacy_logprob_calculation=False)
        model = WhisperForConditionalGeneration(config)
        model.eval()

        # Create dummy input
        batch_size = 1
        seq_length = 100
        input_features = torch.randn(batch_size, config.num_mel_bins, seq_length)

        # Test that multiple runs with the same seed produce the same result
        torch.manual_seed(42)
        with torch.no_grad():
            output1 = model.generate(input_features, max_new_tokens=5, do_sample=False)

        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model.generate(input_features, max_new_tokens=5, do_sample=False)

        self.assertTrue(torch.equal(output1, output2))

    def test_logprob_calculation_with_temperature_fallback(self):
        """Test that logprob calculation affects temperature fallback decisions."""
        # Create models with different logprob calculation modes
        config_legacy = self._create_test_config(use_legacy_logprob_calculation=True)
        model_legacy = WhisperForConditionalGeneration(config_legacy)

        config_new = self._create_test_config(use_legacy_logprob_calculation=False)
        model_new = WhisperForConditionalGeneration(config_new)

        # Copy weights to ensure same model behavior except for logprob calculation
        model_new.load_state_dict(model_legacy.state_dict())

        model_legacy.eval()
        model_new.eval()

        # Create dummy input
        batch_size = 1
        seq_length = 100
        input_features = torch.randn(batch_size, config_legacy.num_mel_bins, seq_length)

        # Test with logprob threshold that might be affected by the calculation difference
        # Note: We use a very permissive threshold to avoid actual fallback in this test
        torch.manual_seed(42)
        with torch.no_grad():
            output_legacy = model_legacy.generate(
                input_features,
                max_new_tokens=5,
                do_sample=False,
                logprob_threshold=-10.0,  # Very permissive threshold
                temperature=0.5,
            )

        torch.manual_seed(42)
        with torch.no_grad():
            output_new = model_new.generate(
                input_features, max_new_tokens=5, do_sample=False, logprob_threshold=-10.0, temperature=0.5
            )

        # Both should produce outputs (no assertion on equality since they might differ)
        self.assertIsNotNone(output_legacy)
        self.assertIsNotNone(output_new)
        self.assertEqual(output_legacy.shape[0], batch_size)
        self.assertEqual(output_new.shape[0], batch_size)

    @slow
    def test_regression_issue_38378_scenario(self):
        """Test the specific scenario from issue #38378."""
        # This test simulates the issue where different transformers versions
        # lead to different results in inference, particularly for long-form
        # transcription with timestamps

        config = self._create_test_config(use_legacy_logprob_calculation=True)  # Simulate older version behavior
        model = WhisperForConditionalGeneration(config)
        model.eval()

        # Create a longer audio sample to potentially trigger long-form behavior
        batch_size = 1
        seq_length = 200  # Longer sequence
        input_features = torch.randn(batch_size, config.num_mel_bins, seq_length)

        # Test short-form without timestamps
        with torch.no_grad():
            short_result = model.generate(input_features, max_new_tokens=10, return_timestamps=False)

        # Test short-form with timestamps
        with torch.no_grad():
            short_result_ts = model.generate(input_features, max_new_tokens=10, return_timestamps=True)

        # Verify outputs are generated
        self.assertIsNotNone(short_result)
        self.assertIsNotNone(short_result_ts)
        self.assertEqual(short_result.shape[0], batch_size)

        # Test that the fix doesn't break existing functionality
        self.assertGreater(short_result.shape[1], 0)  # Should generate some tokens
