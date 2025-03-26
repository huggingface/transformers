import unittest

import torch

from transformers import AutoConfig, AutoModelForCausalLM


class TestResizeEmbeddings(unittest.TestCase):
    def setUp(self):
        self.model_name = "EleutherAI/pythia-410m"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def test_resize_embeddings_no_reinit(self):
        """Test that resized embeddings aren't reinitialized after post_init"""
        # Get initial weights
        old_num_tokens = self.model.get_input_embeddings().num_embeddings
        initial_lm_head = self.model.get_output_embeddings().weight.clone()

        # Resize embeddings
        new_num_tokens = old_num_tokens + 10
        self.model.resize_token_embeddings(new_num_tokens)
        post_resize_weights = self.model.get_output_embeddings().weight[:old_num_tokens].clone()

        # Verify original token weights unchanged after resize
        self.assertTrue(torch.allclose(initial_lm_head, post_resize_weights))

        # Call post_init
        self.model.post_init()
        post_init_weights = self.model.get_output_embeddings().weight[:old_num_tokens].clone()

        # Verify weights still match after post_init
        self.assertTrue(torch.allclose(initial_lm_head, post_init_weights))

    def test_new_tokens_initialization(self):
        """Test that new token embeddings are properly initialized"""
        old_num_tokens = self.model.get_input_embeddings().num_embeddings
        new_num_tokens = old_num_tokens + 10

        self.model.resize_token_embeddings(new_num_tokens)
        new_token_weights = self.model.get_output_embeddings().weight[old_num_tokens:]

        # Verify new tokens have reasonable values (not all zeros or extremely large)
        self.assertTrue(torch.any(new_token_weights != 0))
        self.assertTrue(torch.all(torch.abs(new_token_weights) < 100))

    def test_resize_embeddings_with_bias(self):
        """Test that resizing works correctly when lm_head has bias"""
        config = AutoConfig.from_pretrained(self.model_name)
        config.tie_word_embeddings = False
        model = AutoModelForCausalLM.from_pretrained(self.model_name, config=config)

        # Add bias to lm_head
        old_lm_head = model.get_output_embeddings()
        bias = torch.nn.Parameter(torch.zeros(old_lm_head.weight.size(0)))
        old_lm_head.bias = bias

        initial_weights = old_lm_head.weight.clone()
        initial_bias = old_lm_head.bias.clone()

        # Resize and verify
        model.resize_token_embeddings(initial_weights.size(0) + 10)
        model.post_init()

        new_lm_head = model.get_output_embeddings()
        self.assertTrue(torch.allclose(initial_weights, new_lm_head.weight[: initial_weights.size(0)]))
        self.assertTrue(torch.allclose(initial_bias, new_lm_head.bias[: initial_bias.size(0)]))
