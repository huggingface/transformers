# coding=utf-8
"""Tests for BEATs model."""

import unittest
import torch
from transformers.models.beats import BEATsConfig, BEATsModel, BEATsForAudioClassification


class BEATsConfigTest(unittest.TestCase):

    def test_default_config(self):
        config = BEATsConfig()
        self.assertEqual(config.model_type, "beats")
        self.assertEqual(config.encoder_layers, 12)
        self.assertEqual(config.encoder_embed_dim, 768)
        self.assertEqual(config.num_classes, 527)

    def test_custom_config(self):
        config = BEATsConfig(encoder_layers=6, num_classes=264)
        self.assertEqual(config.encoder_layers, 6)
        self.assertEqual(config.num_classes, 264)


class BEATsModelTest(unittest.TestCase):

    def setUp(self):
        self.config = BEATsConfig(
            input_patch_size=16,
            embed_dim=64,
            encoder_layers=2,
            encoder_embed_dim=64,
            encoder_ffn_embed_own=128,
            encoder_attention_heads=2,
            num_classes=10,
        )

    def test_model_forward(self):
        model = BEATsModel(self.config)
        model.eval()
        fbank = torch.randn(2, 100, 128)
        with torch.no_grad():
            output = model(fbank)
        self.assertEqual(output.shape[-1], self.config.encoder_embed_dim)

    def test_classification_forward(self):
        model = BEATsForAudioClassification(self.config)
        model.eval()
        fbank = torch.randn(2, 100, 128)
        with torch.no_grad():
            output = model(fbank)
        self.assertIn("logits", output)
        self.assertEqual(output["logits"].shape, (2, self.config.num_classes))

    def test_classification_with_labels(self):
        model = BEATsForAudioClassification(self.config)
        model.eval()
        fbank = torch.randn(2, 100, 128)
        labels = torch.zeros(2, self.config.num_classes)
        labels[0, 3] = 1.0
        with torch.no_grad():
            output = model(fbank, labels=labels)
        self.assertIn("loss", output)
        self.assertIn("logits", output)

    def test_output_values_between_0_and_1(self):
        model = BEATsForAudioClassification(self.config)
        model.eval()
        fbank = torch.randn(2, 100, 128)
        with torch.no_grad():
            output = model(fbank)
        self.assertTrue((output["logits"] >= 0).all())
        self.assertTrue((output["logits"] <= 1).all())


if __name__ == "__main__":
    unittest.main()