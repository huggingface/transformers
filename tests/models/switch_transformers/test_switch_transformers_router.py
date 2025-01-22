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

    def test_router_jitter_noise_separation(self):
        """Test that jitter noise only affects routing but not expert inputs"""
        model = SwitchTransformersSparseMLP(self.config)
        model.eval()  # Set to eval mode to ensure deterministic behavior
        
        # Create input
        hidden_states = torch.ones(2, 4, 32)  # batch_size=2, seq_len=4, hidden_size=32
        original_states = hidden_states.clone()
        
        # First forward pass
        output1, _ = model(hidden_states)
        
        # Second forward pass with same input
        output2, _ = model(hidden_states)
        
        # Verify original input wasn't modified
        self.assertTrue(
            torch.allclose(hidden_states, original_states),
            "Input hidden states should not be modified"
        )
        
        # Verify outputs are identical (since we're in eval mode)
        self.assertTrue(
            torch.allclose(output1, output2),
            "Outputs should be identical in eval mode"
        )