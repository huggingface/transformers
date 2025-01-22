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

    def test_router_training_mode(self):
        """Test that jitter noise is only applied during training"""
        model = SwitchTransformersSparseMLP(self.config)
        model.train()  # Set to training mode
        
        # Create input
        hidden_states = torch.ones(2, 4, 32)
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
        
        # Verify outputs are different (due to jitter noise in training)
        self.assertFalse(
            torch.allclose(output1, output2),
            "Outputs should differ in training mode due to jitter noise"
        )

    def test_expert_inputs_consistency(self):
        """Test that expert inputs are consistent and not affected by jitter"""
        model = SwitchTransformersSparseMLP(self.config)
        model.train()  # Set to training mode to enable jitter
        
        # Create input
        hidden_states = torch.randn(2, 4, 32)
        
        # Store expert inputs during forward pass
        expert_inputs = []
        
        def hook_fn(module, input, output):
            expert_inputs.append(input[0].clone())
        
        # Register forward hook on first expert
        handle = model.experts.expert_0.register_forward_hook(hook_fn)
        
        # Multiple forward passes
        for _ in range(3):
            model(hidden_states.clone())
        
        # Remove hook
        handle.remove()
        
        # Verify all expert inputs are identical
        for i in range(1, len(expert_inputs)):
            self.assertTrue(
                torch.allclose(expert_inputs[0], expert_inputs[i], atol=1e-5),
                f"Expert inputs differ between run 0 and run {i}"
            )

if __name__ == '__main__':
    unittest.main() 