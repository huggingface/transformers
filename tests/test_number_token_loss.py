# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for Number Token Loss implementation.
"""

import unittest
import torch
from transformers import AutoTokenizer
from transformers.loss.number_token_loss import (
    extract_numerical_value,
    build_token_to_number_map,
    ntl_was_loss,
    ntl_mse_loss,
    ForCausalLMWithNTLWAS,
    ForCausalLMWithNTLMSE,
)


class TestNumberTokenLoss(unittest.TestCase):
    """Test cases for Number Token Loss."""

    def test_extract_numerical_value(self):
        """Test numerical value extraction from tokens."""
        # Test regular numbers
        self.assertEqual(extract_numerical_value("123"), 123.0)
        self.assertEqual(extract_numerical_value("3.14"), 3.14)
        self.assertEqual(extract_numerical_value("-42"), -42.0)
        self.assertEqual(extract_numerical_value("1,000"), 1000.0)
        
        # Test number words
        self.assertEqual(extract_numerical_value("five"), 5.0)
        self.assertEqual(extract_numerical_value("twenty"), 20.0)
        self.assertEqual(extract_numerical_value("hundred"), 100.0)
        
        # Test ordinal words
        self.assertEqual(extract_numerical_value("first"), 1.0)
        self.assertEqual(extract_numerical_value("tenth"), 10.0)
        
        # Test with suffixes
        self.assertEqual(extract_numerical_value("5th"), 5.0)
        self.assertEqual(extract_numerical_value("1st"), 1.0)
        
        # Test non-numerical tokens
        self.assertIsNone(extract_numerical_value("hello"))
        self.assertIsNone(extract_numerical_value("the"))
        self.assertIsNone(extract_numerical_value(""))

    def test_build_token_to_number_map(self):
        """Test building token-to-number mapping."""
        # Use a simple tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        token_to_number = build_token_to_number_map(tokenizer)
        
        # Should be a dictionary
        self.assertIsInstance(token_to_number, dict)
        
        # Should contain some numerical tokens
        self.assertGreater(len(token_to_number), 0)
        
        # Check that some common numbers are present
        # Note: This depends on the specific tokenizer vocabulary
        for token_id, num_value in token_to_number.items():
            self.assertIsInstance(token_id, int)
            self.assertIsInstance(num_value, float)
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, tokenizer.vocab_size)

    def test_ntl_was_loss(self):
        """Test NTL-WAS loss computation."""
        vocab_size = 1000
        batch_size = 2
        seq_len = 10
        
        # Create dummy logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create a simple token-to-number mapping
        token_to_number = {0: 1.0, 1: 2.0, 2: 3.0}  # Only first few tokens are numerical
        
        # Compute loss
        loss = ntl_was_loss(logits, labels, token_to_number, vocab_size, alpha=0.1)
        
        # Should be a tensor
        self.assertIsInstance(loss, torch.Tensor)
        
        # Should be a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Should be positive
        self.assertGreater(loss.item(), 0)

    def test_ntl_mse_loss(self):
        """Test NTL-MSE loss computation."""
        vocab_size = 1000
        batch_size = 2
        seq_len = 10
        
        # Create dummy logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create a simple token-to-number mapping
        token_to_number = {0: 1.0, 1: 2.0, 2: 3.0}  # Only first few tokens are numerical
        
        # Compute loss
        loss = ntl_mse_loss(logits, labels, token_to_number, vocab_size, alpha=0.1)
        
        # Should be a tensor
        self.assertIsInstance(loss, torch.Tensor)
        
        # Should be a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Should be positive
        self.assertGreater(loss.item(), 0)

    def test_for_causal_lm_with_ntl_was(self):
        """Test ForCausalLMWithNTLWAS function."""
        # Get actual vocab size from tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        batch_size = 2
        seq_len = 10
        
        # Create dummy logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test with tokenizer
        loss = ForCausalLMWithNTLWAS(
            logits, labels, vocab_size, tokenizer=tokenizer, alpha=0.1
        )
        
        # Should be a tensor
        self.assertIsInstance(loss, torch.Tensor)
        
        # Should be a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Should be positive
        self.assertGreater(loss.item(), 0)
        
        # Test without tokenizer (should fall back to CE)
        loss_no_tokenizer = ForCausalLMWithNTLWAS(
            logits, labels, vocab_size, tokenizer=None
        )
        
        # Should still be a valid loss
        self.assertIsInstance(loss_no_tokenizer, torch.Tensor)
        self.assertGreater(loss_no_tokenizer.item(), 0)

    def test_for_causal_lm_with_ntl_mse(self):
        """Test ForCausalLMWithNTLMSE function."""
        # Get actual vocab size from tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        batch_size = 2
        seq_len = 10
        
        # Create dummy logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test with tokenizer
        loss = ForCausalLMWithNTLMSE(
            logits, labels, vocab_size, tokenizer=tokenizer, alpha=0.1
        )
        
        # Should be a tensor
        self.assertIsInstance(loss, torch.Tensor)
        
        # Should be a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Should be positive
        self.assertGreater(loss.item(), 0)
        
        # Test without tokenizer (should fall back to CE)
        loss_no_tokenizer = ForCausalLMWithNTLMSE(
            logits, labels, vocab_size, tokenizer=None
        )
        
        # Should still be a valid loss
        self.assertIsInstance(loss_no_tokenizer, torch.Tensor)
        self.assertGreater(loss_no_tokenizer.item(), 0)

    def test_loss_with_ignore_index(self):
        """Test that ignore_index is handled correctly."""
        vocab_size = 1000
        batch_size = 2
        seq_len = 10
        
        # Create dummy logits and labels with some ignored positions
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, 5:] = -100  # Ignore some positions
        
        # Create a simple token-to-number mapping
        token_to_number = {0: 1.0, 1: 2.0, 2: 3.0}
        
        # Test both loss variants
        loss_was = ntl_was_loss(logits, labels, token_to_number, vocab_size, ignore_index=-100)
        loss_mse = ntl_mse_loss(logits, labels, token_to_number, vocab_size, ignore_index=-100)
        
        # Both should be valid losses
        self.assertIsInstance(loss_was, torch.Tensor)
        self.assertIsInstance(loss_mse, torch.Tensor)
        self.assertGreater(loss_was.item(), 0)
        self.assertGreater(loss_mse.item(), 0)


if __name__ == "__main__":
    unittest.main() 