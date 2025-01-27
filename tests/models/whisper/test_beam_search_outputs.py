import unittest
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.testing_utils import require_torch, torch_device

@require_torch
class TestWhisperBeamSearchOutputs(unittest.TestCase):
    def test_beam_search_generation_outputs(self):
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(torch_device)
        
        # Create dummy input features with correct length (3000)
        batch_size = 2
        input_features = torch.randn(batch_size, 80, 3000).to(torch_device)
        attention_mask = torch.ones((batch_size, 3000), dtype=torch.long, device=torch_device)
        
        num_beams = 5
        num_return_sequences = 2
        
        # Generate with beam search
        outputs = model.generate(
            input_features,
            attention_mask=attention_mask,
            max_new_tokens=20,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            language="en",
            task="transcribe"
        )
        
        # Test presence and types of outputs
        self.assertIsNotNone(outputs.sequences)
        self.assertIsNotNone(outputs.sequences_scores)
        self.assertIsNotNone(outputs.beam_indices)
        
        # Test shapes
        self.assertEqual(outputs.sequences.shape[0], batch_size * num_return_sequences)
        self.assertEqual(outputs.sequences_scores.shape[0], batch_size * num_return_sequences)
        
        # Test values
        self.assertTrue(torch.all(outputs.sequences_scores <= 0))  # Log probabilities should be <= 0
        
        # Check beam indices more carefully
        if outputs.beam_indices is not None:
            print(f"Beam indices shape: {outputs.beam_indices.shape}")
            print(f"Beam indices min: {outputs.beam_indices.min()}")
            print(f"Beam indices max: {outputs.beam_indices.max()}")
            
            # Check if beam indices are valid
            batch_size_with_returns = batch_size * num_return_sequences
            for i in range(batch_size_with_returns):
                batch_beam_indices = outputs.beam_indices[i]
                # Mask out padding positions (which have -1)
                valid_positions = batch_beam_indices != -1
                if valid_positions.any():
                    valid_indices = batch_beam_indices[valid_positions]
                    # For multiple batches, each batch's beam indices should be in [0, num_beams)
                    print(f"Batch {i}: Valid indices range: {valid_indices.min().item()} to {valid_indices.max().item()}")
                    
                    # Check that indices are within the valid range [0, num_beams)
                    self.assertTrue(
                        torch.all((valid_indices >= 0) & (valid_indices < num_beams)),
                        f"Beam indices for batch {i} outside valid range [0, {num_beams})"
                    )
    
    def test_beam_search_short_form(self):
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(torch_device)
        
        # Create input features with correct length (3000)
        input_features = torch.randn(1, 80, 3000).to(torch_device)
        attention_mask = torch.ones((1, 3000), dtype=torch.long, device=torch_device)
        
        num_beams = 5
        gen_kwargs = {
            "max_new_tokens": 20,
            "num_beams": num_beams,
            "num_return_sequences": 1,
            "return_dict_in_generate": True,
            "output_scores": True,
            "language": "en",
            "task": "transcribe",
            "attention_mask": attention_mask
        }
        
        outputs = model.generate(input_features, **gen_kwargs)
        
        # Verify output structure matches the reported issue
        self.assertTrue(hasattr(outputs, 'sequences'))
        self.assertTrue(hasattr(outputs, 'sequences_scores'))
        self.assertTrue(hasattr(outputs, 'beam_indices'))
        
        # Verify outputs are not None
        self.assertIsNotNone(outputs.sequences_scores)
        self.assertIsNotNone(outputs.beam_indices)
        
        # Check beam indices
        if outputs.beam_indices is not None:
            valid_positions = outputs.beam_indices != -1
            if valid_positions.any():
                valid_indices = outputs.beam_indices[valid_positions]
                print(f"Single batch indices range: {valid_indices.min().item()} to {valid_indices.max().item()}")
                # For single batch, indices should be in [0, num_beams)
                self.assertTrue(
                    torch.all((valid_indices >= 0) & (valid_indices < num_beams)),
                    f"Beam indices outside valid range [0, {num_beams})"
                )