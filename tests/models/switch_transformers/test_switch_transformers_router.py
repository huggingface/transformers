import unittest
import torch
from transformers import SwitchTransformersConfig
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersTop1Router,
    SwitchTransformersSparseMLP
)

class SwitchTransformersRouterTest(unittest.TestCase):
    def setUp(self):
        self.config = SwitchTransformersConfig(
            num_experts=2,
            hidden_size=32,
            d_ff=16,
            jitter_noise=0.2,
            expert_capacity=4
        )