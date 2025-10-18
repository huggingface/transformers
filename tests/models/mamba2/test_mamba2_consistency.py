import unittest
import torch
from transformers import Mamba2Config, Mamba2Model


class TestMamba2Consistency(unittest.TestCase):
    
    def setUp(self):
        self.config = Mamba2Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=1,
            expand=2,
            num_heads=16,
            head_dim=8,
            state_size=16,
        )
        self.input_ids = torch.randint(0, 1000, (1, 4))
    
    def test_training_inference_consistency(self):
        model = Mamba2Model(self.config)
        torch.manual_seed(42)
        
        model.eval()
        with torch.no_grad():
            output_inference = model(self.input_ids, use_cache=False)
        
        model.train()
        output_training = model(self.input_ids, use_cache=False)
        
        max_diff = torch.max(torch.abs(
            output_inference.last_hidden_state - output_training.last_hidden_state.detach()
        ))
        
        self.assertLess(max_diff.item(), 1e-5, 
                       f"Training/inference outputs differ by {max_diff.item()}")
    
    def test_deterministic_output(self):
        model = Mamba2Model(self.config)
        model.eval()
        
        torch.manual_seed(42)
        with torch.no_grad():
            output_1 = model(self.input_ids, use_cache=False)
            output_2 = model(self.input_ids, use_cache=False)
        
        max_diff = torch.max(torch.abs(
            output_1.last_hidden_state - output_2.last_hidden_state
        ))
        
        self.assertLess(max_diff.item(), 1e-7,
                       f"Outputs are not deterministic: {max_diff.item()}")


if __name__ == '__main__':
    unittest.main()