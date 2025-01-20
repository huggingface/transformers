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
    
    def test_single_sequence(self):
        # Test regular non-packed behavior
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        # Store attention mask as attribute since we can't pass it to embedding
        self.embed._neftune_attention_mask = None
        emb1 = self.embed(input_ids)
        emb2 = self.embed(input_ids)
        
        # Verify different noise is applied
        self.assertFalse(torch.allclose(emb1, emb2))
        
        # Verify noise magnitude is correct
        diff = (emb1 - emb2).abs().max()
        expected_max = self.embed.neftune_noise_alpha / torch.sqrt(torch.tensor(emb1.size(1) * emb1.size(2)))
        self.assertLess(diff, expected_max * 2)