import unittest
import torch
from transformers.trainer_utils import neftune_post_forward_hook
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestNEFTune(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        self.embed = self.model.get_input_embeddings()
        self.embed.neftune_noise_alpha = 0.1
        # Register the hook
        self.hook_handle = self.embed.register_forward_hook(neftune_post_forward_hook)
        self.embed.train()  # Set to training mode

    def tearDown(self):
        # Remove the hook after tests
        self.hook_handle.remove()