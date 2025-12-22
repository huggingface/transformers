import unittest

import torch

from transformers import HumanVConfig, HumanVForCausalLM
from transformers.testing_utils import require_torch


class HumanVModelTest(unittest.TestCase):
    @require_torch
    def test_forward_shape(self):
        config = HumanVConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=64,
            layer_types=["full_attention", "full_attention"],
            use_cache=False,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )
        model = HumanVForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 11))
        outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.logits.shape, (2, 11, config.vocab_size))

    @require_torch
    def test_generate(self):
        config = HumanVConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=64,
            layer_types=["full_attention", "full_attention"],
            use_cache=False,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )
        model = HumanVForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 11))
        out = model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False)
        self.assertEqual(out.shape, (2, 14))
