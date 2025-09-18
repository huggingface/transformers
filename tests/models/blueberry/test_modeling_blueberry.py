import unittest

import torch

from transformers import BlueberryConfig
from transformers.models.blueberry.modeling_blueberry import BlueberryForCausalLM, BlueberryModel


class BlueberryModelTest(unittest.TestCase):
    def test_forward_output_shape(self):
        config = BlueberryConfig(n_positions=64, n_embd=64, n_layer=2, n_head=4, vocab_size=1000)
        model = BlueberryModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.last_hidden_state.shape, (2, 8, config.hidden_size))

    def test_causallm_head(self):
        config = BlueberryConfig(n_positions=64, n_embd=64, n_layer=2, n_head=4, vocab_size=1000)
        model = BlueberryForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 7))
        outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.logits.shape[-1], config.vocab_size)

    def test_layer_types_switching(self):
        # sliding and full attention layers alternate by default; ensure no error
        config = BlueberryConfig(n_positions=64, n_embd=64, n_layer=4, n_head=4, vocab_size=1000)
        model = BlueberryModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 12))
        _ = model(input_ids=input_ids)


class BlueberryModelIntegrationTest(unittest.TestCase):
    def test_from_pretrained_config(self):
        config = BlueberryConfig()
        model = BlueberryForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        outputs = model(input_ids=input_ids)
        self.assertIsNotNone(outputs.logits)


if __name__ == "__main__":
    unittest.main()

