import unittest
import torch
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

class TestAttentionMaskIssue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        cls.model = GPT2LMHeadModel.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)