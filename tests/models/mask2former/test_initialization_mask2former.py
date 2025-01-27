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
