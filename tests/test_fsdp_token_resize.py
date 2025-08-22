#!/usr/bin/env python3

"""Tests resizing token embeddings with FSDP conditions."""

import unittest
from unittest.mock import patch

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import is_fsdp_enabled


class TestFSDPTokenResize(unittest.TestCase):
    def setUp(self):
        self.model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    def test_fsdp_token_resize_with_meta_tensors(self):
        """Token resize should work when covariance checks would hit meta tensors."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        special_tokens = ["<SPECIAL1>", "<SPECIAL2>", "<SPECIAL3>"]
        tokenizer.add_tokens(special_tokens)
        new_vocab_size = len(tokenizer)

        # Simulate FSDP conditions
        with patch("transformers.modeling_utils.is_fsdp_enabled", return_value=True):

            def mock_init_method(old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens):
                old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
                mean_embeddings = torch.mean(old_embeddings_weight, axis=0)
                # Covariance on meta device -> triggers fallback
                covariance = torch.empty(old_embedding_dim, old_embedding_dim, device="meta")
                epsilon = 1e-9
                if is_fsdp_enabled() and covariance.device.type == "meta":
                    is_covariance_psd = False
                else:
                    is_covariance_psd = torch.distributions.constraints.positive_definite.check(
                        epsilon * covariance
                    ).all()
                if is_covariance_psd:
                    distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embeddings, covariance_matrix=epsilon * covariance
                    )
                    new_embeddings.weight.data[-added_num_tokens:, :] = distribution.sample(
                        sample_shape=(added_num_tokens,)
                    ).to(old_embeddings.weight.dtype)
                else:
                    new_embeddings.weight.data[-added_num_tokens:, :] = (
                        mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
                    )

            model._init_added_embeddings_weights_with_mean = mock_init_method

            try:
                model.resize_token_embeddings(new_vocab_size)
                resize_success = True
            except RuntimeError as e:
                if "Tensor.item() cannot be called on meta tensors" in str(e):
                    resize_success = False
                else:
                    raise

            self.assertTrue(resize_success)
            self.assertEqual(model.get_input_embeddings().num_embeddings, new_vocab_size)

    def test_normal_token_resize(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.add_tokens(["<NORMAL1>", "<NORMAL2>"])
        new_vocab_size = len(tokenizer)

        with patch("transformers.modeling_utils.is_fsdp_enabled", return_value=False):
            model.resize_token_embeddings(new_vocab_size)
            self.assertEqual(model.get_input_embeddings().num_embeddings, new_vocab_size)

    def test_fsdp_resize_without_mean_resizing(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.add_tokens(["<NOMEAN1>", "<NOMEAN2>"])
        new_vocab_size = len(tokenizer)

        with patch("transformers.modeling_utils.is_fsdp_enabled", return_value=True):
            model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
            self.assertEqual(model.get_input_embeddings().num_embeddings, new_vocab_size)


if __name__ == "__main__":
    unittest.main()
