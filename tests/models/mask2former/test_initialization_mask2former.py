import unittest
import torch
from transformers import Mask2FormerConfig, Mask2FormerModel
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerMaskedAttentionDecoderLayer,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
)

class TestMask2FormerInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Mask2FormerConfig()
        cls.model = Mask2FormerModel(cls.config)

    def test_embedding_initialization(self):
        """Test that embeddings are initialized with std=1.0 (PyTorch default)"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                # Calculate empirical standard deviation
                std = torch.std(module.weight.data).item()
                self.assertAlmostEqual(std, 1.0, places=1)

    def test_mlp_bias_initialization(self):
        """Test that MLP biases are properly initialized"""
        for name, module in self.model.named_modules():
            if isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
                for param in module.parameters():
                    if param.dim() == 1:  # Bias terms
                        self.assertFalse(torch.all(param.data == 0))

    def test_multiscale_deformable_attention(self):
        """Test that multiscale deformable attention is properly initialized"""
        for name, module in self.model.named_modules():
            if isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
                # Check sampling offsets initialization
                self.assertTrue(torch.all(module.sampling_offsets.weight.data == 0.0))
                # Check attention weights
                self.assertTrue(torch.all(module.attention_weights.weight.data == 0.0))
                # Check value projection
                self.assertFalse(torch.all(module.value_proj.weight.data == 0.0))
