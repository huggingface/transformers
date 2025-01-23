import unittest
import torch
import requests
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

@require_torch
@require_vision
class TestPaliGemmaAttentionMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
        cls.model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-pt-224",
            torch_dtype=torch.float16,
            device_map="auto"
        )