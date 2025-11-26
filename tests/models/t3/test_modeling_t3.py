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
"""Testing suite for the PyTorch T3 model."""

import tempfile
import unittest

import numpy as np
import torch

from transformers.models.t3.configuration_t3 import T3Config
from transformers.models.t3.modeling_t3 import T3Cond, T3Model
from transformers.testing_utils import require_torch, slow, torch_device


@require_torch
class T3ModelTest(unittest.TestCase):
    def setUp(self):
        """Set up test configuration."""
        self.config = T3Config.english_only()
        # Use smaller model for faster tests
        self.config.llama_config_dict["num_hidden_layers"] = 2
        self.config.llama_config_dict["num_attention_heads"] = 4
        self.config.hidden_size = 256
        self.config.speaker_embed_size = 128
        self.config.perceiver_num_latents = 8

    def test_model_initialization(self):
        """Test that the model can be initialized."""
        model = T3Model(self.config)
        self.assertIsInstance(model, T3Model)

        # Check that sub-modules exist
        self.assertIsNotNone(model.tfmr)
        self.assertIsNotNone(model.text_emb)
        self.assertIsNotNone(model.speech_emb)
        self.assertIsNotNone(model.text_head)
        self.assertIsNotNone(model.speech_head)
        self.assertIsNotNone(model.voice_encoder)
        self.assertIsNotNone(model.cond_enc)

    def test_english_only_config(self):
        """Test English-only configuration."""
        config = T3Config.english_only()
        self.assertEqual(config.text_tokens_dict_size, 704)
        self.assertFalse(config.is_multilingual)
        self.assertFalse(config.use_alignment_analyzer)

    def test_multilingual_config(self):
        """Test multilingual configuration."""
        config = T3Config.multilingual()
        self.assertEqual(config.text_tokens_dict_size, 2454)
        self.assertTrue(config.is_multilingual)
        self.assertTrue(config.use_alignment_analyzer)

    def test_forward_pass(self):
        """Test forward pass with conditioning."""
        model = T3Model(self.config)
        model.eval()
        model.to(torch_device)

        batch_size = 2
        text_len = 10
        speech_len = 20

        # Create dummy inputs
        text_tokens = torch.randint(0, self.config.text_tokens_dict_size, (batch_size, text_len), device=torch_device)
        text_token_lens = torch.tensor([text_len, text_len - 2], device=torch_device)

        speech_tokens = torch.randint(
            0, self.config.speech_tokens_dict_size, (batch_size, speech_len), device=torch_device
        )
        speech_token_lens = torch.tensor([speech_len, speech_len - 3], device=torch_device)

        # Create conditioning
        speaker_emb = torch.randn(batch_size, self.config.speaker_embed_size, device=torch_device)
        emotion_adv = torch.ones(batch_size, 1, 1, device=torch_device) * 0.5

        t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv)

        # Run forward pass
        with torch.no_grad():
            output = model(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                text_token_lens=text_token_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_token_lens,
            )

        # Check output shapes
        self.assertEqual(output["text_logits"].shape, (batch_size, text_len, self.config.text_tokens_dict_size))
        self.assertEqual(output["speech_logits"].shape, (batch_size, speech_len, self.config.speech_tokens_dict_size))

    def test_forward_pass_with_speech_prompt(self):
        """Test forward pass with speech conditioning prompt."""
        model = T3Model(self.config)
        model.eval()
        model.to(torch_device)

        batch_size = 1
        text_len = 10
        speech_len = 20
        prompt_len = self.config.speech_cond_prompt_len

        # Create inputs
        text_tokens = torch.randint(0, self.config.text_tokens_dict_size, (batch_size, text_len), device=torch_device)
        text_token_lens = torch.tensor([text_len], device=torch_device)

        speech_tokens = torch.randint(
            0, self.config.speech_tokens_dict_size, (batch_size, speech_len), device=torch_device
        )
        speech_token_lens = torch.tensor([speech_len], device=torch_device)

        # Create conditioning with speech prompt
        speaker_emb = torch.randn(batch_size, self.config.speaker_embed_size, device=torch_device)
        emotion_adv = torch.ones(batch_size, 1, 1, device=torch_device) * 0.5
        cond_prompt_speech_tokens = torch.randint(
            0, self.config.speech_tokens_dict_size, (batch_size, prompt_len), device=torch_device
        )

        t3_cond = T3Cond(
            speaker_emb=speaker_emb, emotion_adv=emotion_adv, cond_prompt_speech_tokens=cond_prompt_speech_tokens
        )

        # Run forward pass
        with torch.no_grad():
            output = model(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                text_token_lens=text_token_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_token_lens,
            )

        # Check output shapes
        self.assertIn("text_logits", output)
        self.assertIn("speech_logits", output)

    def test_loss_computation(self):
        """Test loss computation."""
        model = T3Model(self.config)
        model.train()
        model.to(torch_device)

        batch_size = 2
        text_len = 10
        speech_len = 20

        # Create inputs with start/stop tokens
        text_tokens = torch.randint(0, self.config.text_tokens_dict_size, (batch_size, text_len), device=torch_device)
        text_tokens[:, 0] = self.config.start_text_token
        text_tokens[:, -1] = self.config.stop_text_token
        text_token_lens = torch.tensor([text_len, text_len], device=torch_device)

        speech_tokens = torch.randint(
            0, self.config.speech_tokens_dict_size, (batch_size, speech_len), device=torch_device
        )
        speech_token_lens = torch.tensor([speech_len, speech_len], device=torch_device)

        # Create conditioning
        speaker_emb = torch.randn(batch_size, self.config.speaker_embed_size, device=torch_device)
        emotion_adv = torch.ones(batch_size, 1, 1, device=torch_device) * 0.5
        t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv)

        # Compute loss
        loss_text, loss_speech = model.loss(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
        )

        # Check losses are scalars and finite
        self.assertEqual(loss_text.ndim, 0)
        self.assertEqual(loss_speech.ndim, 0)
        self.assertTrue(torch.isfinite(loss_text))
        self.assertTrue(torch.isfinite(loss_speech))

    def test_inference_basic(self):
        """Test basic inference."""
        model = T3Model(self.config)
        model.eval()
        model.to(torch_device)

        batch_size = 1
        text_len = 10

        # Create inputs
        text_tokens = torch.randint(0, self.config.text_tokens_dict_size, (batch_size, text_len), device=torch_device)
        text_tokens[:, 0] = self.config.start_text_token
        text_tokens[:, -1] = self.config.stop_text_token

        # Create conditioning
        speaker_emb = torch.randn(batch_size, self.config.speaker_embed_size, device=torch_device)
        emotion_adv = torch.ones(batch_size, 1, 1, device=torch_device) * 0.5
        t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv)

        # Run inference (with small max_new_tokens for speed)
        with torch.no_grad():
            speech_tokens = model.inference(
                t3_cond=t3_cond, text_tokens=text_tokens[0], max_new_tokens=10, cfg_weight=0.5
            )

        # Check output
        self.assertEqual(len(speech_tokens.shape), 1)  # Should be 1D
        self.assertGreater(len(speech_tokens), 0)  # Should have generated tokens

    def test_voice_encoder(self):
        """Test voice encoder functionality."""
        model = T3Model(self.config)
        model.to(torch_device)
        voice_encoder = model.voice_encoder

        # Create dummy waveforms
        sample_rate = 16000
        duration = 2.0  # seconds
        num_samples = int(sample_rate * duration)
        wavs = [np.random.randn(num_samples).astype(np.float32) for _ in range(2)]

        # Extract embeddings
        embeds = voice_encoder.embeds_from_wavs(wavs, sample_rate=sample_rate)

        # Check output shape
        self.assertEqual(embeds.shape[0], 2)
        self.assertEqual(embeds.shape[1], voice_encoder.config.speaker_embed_size)

        # Check embeddings are L2-normalized
        norms = np.linalg.norm(embeds, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)

    def test_save_and_load(self):
        """Test saving and loading the model."""
        model = T3Model(self.config)

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
            loaded_model = T3Model.from_pretrained(tmpdirname)
            self.assertIsInstance(loaded_model, T3Model)

            # Check config is preserved
            self.assertEqual(loaded_model.config.text_tokens_dict_size, self.config.text_tokens_dict_size)
            self.assertEqual(loaded_model.config.speech_tokens_dict_size, self.config.speech_tokens_dict_size)

    def test_config_attributes(self):
        """Test that config attributes are properly set."""
        model = T3Model(self.config)

        self.assertEqual(model.config.text_tokens_dict_size, 704)
        self.assertEqual(model.config.speech_tokens_dict_size, 8194)
        self.assertEqual(model.config.start_text_token, 255)
        self.assertEqual(model.config.stop_text_token, 0)
        self.assertEqual(model.config.start_speech_token, 6561)
        self.assertEqual(model.config.stop_speech_token, 6562)

    def test_t3_cond_to_device(self):
        """Test T3Cond device casting."""
        speaker_emb = torch.randn(1, 256)
        emotion_adv = torch.ones(1, 1, 1)
        cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv)

        # Cast to device
        cond_device = cond.to(device=torch_device)

        self.assertEqual(cond_device.speaker_emb.device.type, torch_device.split(":")[0])
        self.assertEqual(cond_device.emotion_adv.device.type, torch_device.split(":")[0])

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        model = T3Model(self.config)
        model.eval()
        model.to(torch_device)

        for batch_size in [1, 2, 4]:
            text_len = 10
            speech_len = 20

            text_tokens = torch.randint(
                0, self.config.text_tokens_dict_size, (batch_size, text_len), device=torch_device
            )
            text_token_lens = torch.full((batch_size,), text_len, device=torch_device)

            speech_tokens = torch.randint(
                0, self.config.speech_tokens_dict_size, (batch_size, speech_len), device=torch_device
            )
            speech_token_lens = torch.full((batch_size,), speech_len, device=torch_device)

            speaker_emb = torch.randn(batch_size, self.config.speaker_embed_size, device=torch_device)
            emotion_adv = torch.ones(batch_size, 1, 1, device=torch_device) * 0.5
            t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv)

            with torch.no_grad():
                output = model(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens,
                )

            self.assertEqual(output["text_logits"].shape[0], batch_size)
            self.assertEqual(output["speech_logits"].shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()
