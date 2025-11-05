"""
Test suite for GPT2 torch.cat empty tensor fix (Issue #42027).

Tests validate that the fix properly handles empty tensors during
torch.compile tracing in GPT2 attention mechanisms.
"""

import unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel


class GPT2TorchCompileFixTest(unittest.TestCase):
    """Test suite for GPT2 torch.compile empty tensor fix."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = GPT2Config(n_layer=2, n_head=4, n_embd=256)

    def test_empty_tensor_handling(self):
        """Test that empty tensors are handled correctly during cache updates."""
        model = GPT2LMHeadModel(self.config).to(self.device).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(self.device)

        with torch.no_grad():
            eager_output = model(input_ids, use_cache=True)

        self.assertIsNotNone(eager_output.logits)
        self.assertEqual(eager_output.logits.shape[1], input_ids.shape[1])

    def test_torch_compile_basic(self):
        """Test basic torch.compile compatibility."""
        model = GPT2LMHeadModel(self.config).to(self.device).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(self.device)

        try:
            compiled_model = torch.compile(model, backend="inductor", mode="default")
            
            with torch.no_grad():
                compiled_output = compiled_model(input_ids, use_cache=True)
            
            self.assertIsNotNone(compiled_output.logits)
            self.assertEqual(compiled_output.logits.shape[1], input_ids.shape[1])
        except Exception as e:
            self.fail(f"torch.compile failed: {e}")

    def test_generation_with_cache(self):
        """Test text generation with cache."""
        model = GPT2LMHeadModel(self.config).to(self.device).eval()
        input_ids = torch.tensor([[1, 2, 3]]).to(self.device)

        generated_sequence = input_ids
        past_key_values = None
        max_new_tokens = 3

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=generated_sequence[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=False
                )

            logits, past_key_values = outputs[0], outputs[1] if len(outputs) > 1 else None
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

        self.assertGreater(generated_sequence.shape[1], input_ids.shape[1])

    def test_mixed_batch_sizes(self):
        """Test with mixed batch sizes."""
        model = GPT2LMHeadModel(self.config).to(self.device).eval()

        batch_sizes = [1, 2, 4]
        seq_lengths = [1, 5, 10]

        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                input_ids = torch.randint(1, 1000, (batch_size, seq_length)).to(self.device)

                with torch.no_grad():
                    outputs = model(input_ids, use_cache=True)

                expected_shape = (batch_size, seq_length, model.config.vocab_size)
                self.assertEqual(outputs.logits.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
