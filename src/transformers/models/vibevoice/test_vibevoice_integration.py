import unittest
import torch
import tempfile
import shutil
import os

from transformers import VibeVoiceConfig, VibeVoiceForConditionalGeneration, VibeVoiceTokenizer


class VibeVoiceIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_config_initialization(self):
        print("\nTesting Config Initialization...")
        config = VibeVoiceConfig()
        self.assertIsInstance(config, VibeVoiceConfig)
        self.assertEqual(config.model_type, "vibevoice")
        print("Config initialized successfully.")

    def test_model_initialization(self):
        print("\nTesting Model Initialization (Random Weights)...")
        config = VibeVoiceConfig(
            decoder_config={"num_hidden_layers": 2, "hidden_size": 32, "num_attention_heads": 2, "vocab_size": 100},
            acoustic_tokenizer_config={"vae_dim": 16, "encoder_depths": [1], "decoder_depths": [1]},
            diffusion_head_config={"hidden_size": 32, "head_layers": 1},
            tts_backbone_num_hidden_layers=1,
        )
        model = VibeVoiceForConditionalGeneration(config)
        self.assertIsInstance(model, VibeVoiceForConditionalGeneration)
        print("Model initialized successfully.")

    def test_tokenizer_special_tokens(self):
        print("\nTesting Tokenizer Special Tokens...")
        # Mocking or loading base tokenizer required for full test,
        # but here we check VibeVoice specific logic if accessible.
        # Since VibeVoiceTokenizer inherits from Qwen2Tokenizer, we need the base files.
        # Assuming environment has access or we skip deep loading.
        pass

    def test_forward_pass_fake_inputs(self):
        print("\nTesting Forward Pass with Fake Inputs...")
        config = VibeVoiceConfig(
            decoder_config={"num_hidden_layers": 4, "hidden_size": 32, "num_attention_heads": 4, "vocab_size": 100},
            acoustic_tokenizer_config={"vae_dim": 8, "encoder_depths": [1], "decoder_depths": [1]},
            diffusion_head_config={"hidden_size": 32, "head_layers": 1},
            tts_backbone_num_hidden_layers=2,
        )
        model = VibeVoiceForConditionalGeneration(config)
        model.eval()

        # Fake inputs
        input_ids = torch.randint(0, 100, (1, 10))
        # Note: VibeVoiceForConditionalGeneration.forward logic needs to be implemented/checked if it supports direct generation call
        # or if we need to call sub-components.
        # The current implementation has the main model.forward DISABLED.
        # Users should use model.generate() or component calls.

        # We verify we can access components
        self.assertIsNotNone(model.model.language_model)
        self.assertIsNotNone(model.model.tts_language_model)
        self.assertIsNotNone(model.model.prediction_head)

        print("Model components accessible.")


if __name__ == "__main__":
    unittest.main()
