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
"""Testing suite for the PyTorch S3Gen model."""

import unittest

import torch

from transformers.models.s3gen.configuration_s3gen import S3GenConfig
from transformers.models.s3gen.modeling_s3gen import S3GenModel
from transformers.testing_utils import require_torch, torch_device


@require_torch
class S3GenModelTest(unittest.TestCase):
    def setUp(self):
        self.config = S3GenConfig(
            vocab_size=6561,
            token_embed_dim=512,
            speaker_embed_dim=192,
            encoder_output_size=512,
            encoder_attention_heads=8,
            encoder_linear_units=2048,
            encoder_num_blocks=6,
            decoder_in_channels=320,
            decoder_out_channels=80,
            decoder_channels=[256],
            decoder_n_blocks=4,
            decoder_num_mid_blocks=12,
            sampling_rate=24000,
            mel_bins=80,
        )

    def test_model_initialization(self):
        """Test that the model can be initialized."""
        model = S3GenModel(self.config)
        self.assertIsInstance(model, S3GenModel)

        # Check that sub-modules exist
        self.assertIsNotNone(model.tokenizer)
        self.assertIsNotNone(model.speaker_encoder)
        self.assertIsNotNone(model.flow)
        self.assertIsNotNone(model.mel2wav)

    def test_forward_pass_with_ref_wav(self):
        """Test forward pass with reference audio."""
        model = S3GenModel(self.config)
        model.eval()
        model.to(torch_device)

        batch_size = 1
        seq_len = 50
        audio_len = 8000

        # Create dummy inputs
        speech_tokens = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=torch_device)
        ref_wav = torch.randn(batch_size, audio_len, device=torch_device)
        ref_sr = 16000

        # Run forward pass
        with torch.no_grad():
            output = model(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, finalize=True)

        # Check output shape
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.config.mel_bins)
        self.assertGreater(output.shape[2], 0)  # Time dimension

    def test_forward_pass_with_ref_dict(self):
        """Test forward pass with pre-computed reference embeddings."""
        model = S3GenModel(self.config)
        model.eval()
        model.to(torch_device)

        batch_size = 1
        seq_len = 50
        audio_len = 8000

        # Create dummy inputs
        speech_tokens = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=torch_device)
        ref_wav = torch.randn(batch_size, audio_len, device=torch_device)
        ref_sr = 16000

        # Extract reference embeddings
        ref_dict = model.embed_ref(ref_wav, ref_sr)

        # Run forward pass with ref_dict
        with torch.no_grad():
            output = model(speech_tokens, ref_dict=ref_dict, finalize=True)

        # Check output shape
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], batch_size)

    def test_embed_ref(self):
        """Test reference embedding extraction."""
        model = S3GenModel(self.config)
        model.eval()
        model.to(torch_device)

        batch_size = 1
        audio_len = 8000

        ref_wav = torch.randn(batch_size, audio_len, device=torch_device)
        ref_sr = 16000

        ref_dict = model.embed_ref(ref_wav, ref_sr)

        # Check that all required keys are present
        self.assertIn("prompt_token", ref_dict)
        self.assertIn("prompt_token_len", ref_dict)
        self.assertIn("prompt_feat", ref_dict)
        self.assertIn("embedding", ref_dict)

        # Check shapes
        self.assertEqual(len(ref_dict["embedding"].shape), 2)
        self.assertEqual(ref_dict["embedding"].shape[0], batch_size)
        self.assertEqual(ref_dict["embedding"].shape[1], self.config.speaker_embed_dim)

    def test_end_to_end_inference(self):
        """Test end-to-end inference (tokens → waveform)."""
        model = S3GenModel(self.config)
        model.eval()
        model.to(torch_device)

        batch_size = 1
        seq_len = 50
        audio_len = 8000

        speech_tokens = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=torch_device)
        ref_wav = torch.randn(batch_size, audio_len, device=torch_device)
        ref_sr = 16000

        # Run end-to-end inference
        with torch.no_grad():
            wavs, sources = model.inference(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, finalize=True)

        # Check output shapes
        self.assertEqual(len(wavs.shape), 2)
        self.assertEqual(wavs.shape[0], batch_size)
        self.assertGreater(wavs.shape[1], 0)  # Audio samples

        self.assertEqual(len(sources.shape), 3)
        self.assertEqual(sources.shape[0], batch_size)

    def test_save_and_load(self):
        """Test saving and loading the model."""
        import os
        import tempfile

        model = S3GenModel(self.config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save model
            model.save_pretrained(tmpdirname)

            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "config.json")))
            self.assertTrue(
                os.path.exists(os.path.join(tmpdirname, "model.safetensors"))
                or os.path.exists(os.path.join(tmpdirname, "pytorch_model.bin"))
            )

            # Load model
            loaded_model = S3GenModel.from_pretrained(tmpdirname)
            self.assertIsInstance(loaded_model, S3GenModel)

    def test_different_token_lengths(self):
        """Test with different token sequence lengths."""
        model = S3GenModel(self.config)
        model.eval()
        model.to(torch_device)

        ref_wav = torch.randn(1, 8000, device=torch_device)
        ref_sr = 16000

        for seq_len in [10, 50, 100]:
            speech_tokens = torch.randint(0, self.config.vocab_size, (1, seq_len), device=torch_device)

            with torch.no_grad():
                output = model(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, finalize=True)

            self.assertEqual(output.shape[0], 1)
            self.assertEqual(output.shape[1], self.config.mel_bins)

    def test_config_attributes(self):
        """Test that config attributes are properly set."""
        model = S3GenModel(self.config)

        self.assertEqual(model.config.vocab_size, 6561)
        self.assertEqual(model.config.speaker_embed_dim, 192)
        self.assertEqual(model.config.mel_bins, 80)
        self.assertEqual(model.config.sampling_rate, 24000)


if __name__ == "__main__":
    unittest.main()
