import unittest
import torch
from transformers import FirefliesConfig, FirefliesModel

class FirefliesModelTest(unittest.TestCase):
    def test_model_forward(self):
        config = FirefliesConfig(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            num_layers=2,
        )
        model = FirefliesModel(config)
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids)
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape, (2, 10, config.vocab_size))

if __name__ == "__main__":
    unittest.main()
