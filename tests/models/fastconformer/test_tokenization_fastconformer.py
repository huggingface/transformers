# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the FastConformer tokenizer."""

import json
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer
from transformers.models.fastconformer import FastConformerTokenizer
from transformers.testing_utils import require_torch


class FastConformerTokenizationTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with a simple vocabulary."""
        self.test_vocab = {
            "<unk>": 0,
            "▁the": 1,
            "▁to": 2,
            "▁and": 3,
            "▁a": 4,
            "▁of": 5,
            "t": 6,
            "h": 7,
            "e": 8,
            "o": 9,
            "▁hello": 10,
            "▁world": 11,
            " ": 12,
        }
        
        # Create temporary vocab file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vocab_file = Path(self.temp_dir.name) / "vocab.json"
        
        with open(self.vocab_file, 'w') as f:
            json.dump(self.test_vocab, f, indent=2)
    
    def tearDown(self):
        """Clean up temp files."""
        self.temp_dir.cleanup()
    
    def test_tokenizer_initialization(self):
        """Test that tokenizer initializes correctly."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            unk_token="<unk>",
            blank_token_id=len(self.test_vocab),  # 13
        )
        
        self.assertEqual(tokenizer.vocab_size, len(self.test_vocab))
        self.assertEqual(tokenizer.blank_token_id, 13)
        self.assertEqual(tokenizer.unk_token_id, 0)
        self.assertEqual(tokenizer.unk_token, "<unk>")
    
    def test_vocab_access(self):
        """Test vocabulary access methods."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            unk_token="<unk>",
        )
        
        vocab = tokenizer.get_vocab()
        self.assertEqual(len(vocab), len(self.test_vocab))
        self.assertIn("▁the", vocab)
        self.assertEqual(vocab["▁the"], 1)
        
        # Test token-to-id conversion
        self.assertEqual(tokenizer._convert_token_to_id("▁the"), 1)
        self.assertEqual(tokenizer._convert_token_to_id("unknown"), 0)  # UNK
        
        # Test id-to-token conversion
        self.assertEqual(tokenizer._convert_id_to_token(1), "▁the")
        self.assertEqual(tokenizer._convert_id_to_token(999), "<unk>")
    
    def test_ctc_decoding(self):
        """Test CTC decoding functionality."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        # Test case: [blank, "▁the", "▁the", blank, "▁to", blank, blank]
        ctc_output = [13, 1, 1, 13, 2, 13, 13]
        decoded_ids = tokenizer.ctc_decode_ids(ctc_output)
        
        # Should be [1, 2] after removing blanks and collapsing
        expected_ids = [1, 2]
        self.assertEqual(decoded_ids, expected_ids)
        
        # Test full CTC decoding to text
        decoded_text = tokenizer.decode_ctc_tokens(ctc_output)
        expected_text = "the to"
        self.assertEqual(decoded_text, expected_text)
    
    def test_consecutive_collapse(self):
        """Test consecutive token collapse in CTC decoding."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        # Multiple consecutive identical tokens
        ctc_output = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        decoded_ids = tokenizer.ctc_decode_ids(ctc_output)
        
        # Should collapse to [1, 2, 3]
        expected_ids = [1, 2, 3]
        self.assertEqual(decoded_ids, expected_ids)
    
    def test_sentencepiece_processing(self):
        """Test SentencePiece-style token processing."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        # Test ▁ to space conversion
        tokens = ["▁hello", "▁world"]
        text = tokenizer.convert_tokens_to_string(tokens)
        expected = "hello world"
        self.assertEqual(text, expected)
        
        # Test with CTC decoding
        ctc_output = [10, 11]  # "▁hello", "▁world"
        decoded_text = tokenizer.decode_ctc_tokens(ctc_output)
        self.assertEqual(decoded_text, "hello world")
    
    def test_decode_methods(self):
        """Test various decode methods."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        token_ids = [13, 1, 1, 13, 2, 13]
        
        # Test CTC decode (default)
        ctc_text = tokenizer.decode(token_ids, ctc_decode=True)
        self.assertEqual(ctc_text, "the to")
        
        # Test non-CTC decode
        non_ctc_text = tokenizer.decode(token_ids, ctc_decode=False)
        # Should include all tokens (except those >= vocab_size)
        self.assertIn("the", non_ctc_text)
        self.assertIn("to", non_ctc_text)
        
        # Test with single token
        single_text = tokenizer.decode(1, ctc_decode=True)
        self.assertEqual(single_text, "the")
    
    def test_batch_decode(self):
        """Test batch decoding functionality."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        batch_outputs = [
            [13, 1, 13, 2, 13],  # "the to"
            [13, 3, 13, 4, 13],  # "and a"
            [10, 11],             # "hello world"
        ]
        
        batch_decoded = tokenizer.batch_decode(batch_outputs, ctc_decode=True)
        expected = ["the to", "and a", "hello world"]
        self.assertEqual(batch_decoded, expected)
    
    def test_save_and_load(self):
        """Test saving and loading tokenizer."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        # Save tokenizer
        save_dir = Path(self.temp_dir.name) / "saved_tokenizer"
        tokenizer.save_pretrained(save_dir)
        
        # Check that files are saved
        self.assertTrue((save_dir / "tokenizer_config.json").exists())
        self.assertTrue((save_dir / "vocab.json").exists())
        
        # Load tokenizer
        loaded_tokenizer = FastConformerTokenizer.from_pretrained(save_dir)
        
        # Test that loaded tokenizer works the same
        ctc_output = [13, 1, 1, 13, 2, 13]
        original_text = tokenizer.decode_ctc_tokens(ctc_output)
        loaded_text = loaded_tokenizer.decode_ctc_tokens(ctc_output)
        
        self.assertEqual(original_text, loaded_text)
        self.assertEqual(loaded_tokenizer.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded_tokenizer.blank_token_id, tokenizer.blank_token_id)
    
    @require_torch
    def test_auto_tokenizer_integration(self):
        """Test AutoTokenizer integration."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        # Save tokenizer
        save_dir = Path(self.temp_dir.name) / "auto_tokenizer_test"
        tokenizer.save_pretrained(save_dir)
        
        # Load via AutoTokenizer
        auto_tokenizer = AutoTokenizer.from_pretrained(save_dir)
        
        # Verify it's the correct type
        self.assertIsInstance(auto_tokenizer, FastConformerTokenizer)
        
        # Test functionality
        ctc_output = [13, 1, 13, 2, 13]
        text = auto_tokenizer.decode(ctc_output, ctc_decode=True)
        self.assertEqual(text, "the to")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        tokenizer = FastConformerTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )
        
        # Empty input
        self.assertEqual(tokenizer.decode_ctc_tokens([]), "")
        self.assertEqual(tokenizer.decode([]), "")
        
        # Only blank tokens
        blank_only = [13, 13, 13]
        self.assertEqual(tokenizer.decode_ctc_tokens(blank_only), "")
        
        # Out of vocab tokens
        out_of_vocab = [999, 1000]
        decoded = tokenizer.decode_ctc_tokens(out_of_vocab)
        self.assertEqual(decoded, "")  # Filtered out


if __name__ == "__main__":
    unittest.main() 