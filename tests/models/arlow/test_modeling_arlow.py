import unittest

import torch
import torch.nn as nn

from transformers import ArlowConfig, ArlowForCausalLM, ArlowModel
from transformers.testing_utils import require_torch, torch_device


all_model_classes = (ArlowModel, ArlowForCausalLM)


@require_torch
class ArlowModelingTest(unittest.TestCase):
    def setUp(self):
        self.config = ArlowConfig(
            vocab_size=131072,
            hidden_size=2304,
            intermediate_size=9216,
            num_attention_heads=12,
            num_hidden_layers=2,
            pad_token_id=0,
        )
        self.batch_size = 2
        self.seq_len = 16
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(torch_device)

    def test_model_forward(self):
        model = ArlowModel(self.config).to(torch_device).half().eval()  # .half() here
        with torch.no_grad():
            outputs = model(self.input_ids)
        self.assertTrue(hasattr(outputs, "last_hidden_state"))
        self.assertEqual(outputs.last_hidden_state.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_causal_lm_forward(self):
        model = ArlowForCausalLM(self.config).to(torch_device).half().eval()
        with torch.no_grad():
            outputs = model(input_ids=self.input_ids)
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_len, self.config.vocab_size))

    def test_causal_lm_loss(self):
        model = ArlowForCausalLM(self.config).to(torch_device).half().eval()
        labels = self.input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids=self.input_ids, labels=labels)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_len, self.config.vocab_size))

    def test_pretrained_model_weight_init(self):
        from transformers import ArlowPreTrainedModel

        class DummyLinearModel(ArlowPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.linear = nn.Linear(config.hidden_size, config.hidden_size)
                self.post_init()

            def forward(self, x):
                return self.linear(x)

        dummy_config = ArlowConfig(hidden_size=64)
        model = DummyLinearModel(dummy_config)
        weight_std = dummy_config.initializer_range
        linear_std = model.linear.weight.std().item()

        # Should be close to expected std
        self.assertTrue(abs(linear_std - weight_std) < 0.05)
