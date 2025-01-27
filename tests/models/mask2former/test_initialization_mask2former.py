import unittest
import torch
from transformers import Mask2FormerConfig, Mask2FormerModel
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerMaskedAttentionDecoderLayer,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
)
