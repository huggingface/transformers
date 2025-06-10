import unittest

import torch

from transformers import FirefliesConfig, FirefliesModel


all_model_classes = (FirefliesModel,)


class FirefliesModelTest(unittest.TestCase):
    def test_forward_pass(self):
        config = FirefliesConfig()
        model = FirefliesModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        outputs = model(input_ids)
        self.assertEqual(outputs["last_hidden_state"].shape, (2, 10, config.hidden_size))
