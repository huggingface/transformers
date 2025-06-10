import unittest
import torch
from transformers.models.fireflies import FirefliesModel, FirefliesConfig 
class FirefliesModelTest(unittest.TestCase):
    def test_forward_pass(self):
        config = FirefliesConfig(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            num_layers=2
        )
        model = FirefliesModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        outputs = model(input_ids)
        self.assertEqual(outputs["logits"].shape, (2, 16, config.vocab_size))
