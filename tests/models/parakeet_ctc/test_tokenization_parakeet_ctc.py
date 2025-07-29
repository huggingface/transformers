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
"""Testing suite for the ParakeetCTC tokenizer."""

import json
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer
from transformers.models.parakeet_ctc import ParakeetCTCTokenizer
from transformers.testing_utils import require_torch


class ParakeetCTCTokenizationTest(unittest.TestCase):
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

        with open(self.vocab_file, "w") as f:
            json.dump(self.test_vocab, f, indent=2)

    def tearDown(self):
        """Clean up temp files."""
        self.temp_dir.cleanup()

    def test_tokenizer_initialization(self):
        """Test that tokenizer initializes correctly."""
        tokenizer = ParakeetCTCTokenizer(
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
        tokenizer = ParakeetCTCTokenizer(
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

    def test_sentencepiece_processing(self):
        """Test SentencePiece-style token processing."""
        tokenizer = ParakeetCTCTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )

        # Test ▁ to space conversion
        tokens = ["▁hello", "▁world"]
        text = tokenizer.convert_tokens_to_string(tokens)
        expected = "hello world"
        self.assertEqual(text, expected)

        # Test with already CTC-decoded token IDs (as would come from model.generate())
        ctc_decoded_ids = [10, 11]  # "▁hello", "▁world" - already collapsed
        decoded_text = tokenizer.decode(ctc_decoded_ids)
        self.assertEqual(decoded_text, "hello world")

    def test_decode_methods(self):
        """Test decode methods expecting already CTC-decoded sequences."""
        tokenizer = ParakeetCTCTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )

        # These token sequences are already CTC-decoded (from model.generate())
        # So they should not contain blanks or consecutive duplicates
        already_decoded_ids = [1, 2]  # "▁the", "▁to"

        # Test standard decode (expects CTC-decoded input)
        text = tokenizer.decode(already_decoded_ids)
        self.assertEqual(text, "the to")

        # Test with single token
        single_text = tokenizer.decode(1)
        self.assertEqual(single_text, "the")

        # Test with empty sequence
        empty_text = tokenizer.decode([])
        self.assertEqual(empty_text, "")

    def test_batch_decode(self):
        """Test batch decoding functionality with already CTC-decoded sequences."""
        tokenizer = ParakeetCTCTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )

        # These are already CTC-decoded sequences (as would come from model.generate())
        batch_outputs = [
            [1, 2],  # "the to"
            [3, 4],  # "and a"
            [10, 11],  # "hello world"
        ]

        batch_decoded = tokenizer.batch_decode(batch_outputs)
        expected = ["the to", "and a", "hello world"]
        self.assertEqual(batch_decoded, expected)

    def test_save_and_load(self):
        """Test saving and loading tokenizer."""
        tokenizer = ParakeetCTCTokenizer(
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
        loaded_tokenizer = ParakeetCTCTokenizer.from_pretrained(save_dir)

        # Test that loaded tokenizer works the same with already CTC-decoded sequences
        already_decoded_ids = [1, 2]  # "▁the", "▁to"
        original_text = tokenizer.decode(already_decoded_ids)
        loaded_text = loaded_tokenizer.decode(already_decoded_ids)

        self.assertEqual(original_text, loaded_text)
        self.assertEqual(loaded_tokenizer.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded_tokenizer.blank_token_id, tokenizer.blank_token_id)

    @require_torch
    def test_auto_tokenizer_integration(self):
        """Test AutoTokenizer integration."""
        tokenizer = ParakeetCTCTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )

        # Save tokenizer
        save_dir = Path(self.temp_dir.name) / "auto_tokenizer_test"
        tokenizer.save_pretrained(save_dir)

        # Load via AutoTokenizer
        auto_tokenizer = AutoTokenizer.from_pretrained(save_dir)

        # Verify it's the correct type
        self.assertIsInstance(auto_tokenizer, ParakeetCTCTokenizer)

        # Test functionality with already CTC-decoded sequences
        already_decoded_ids = [1, 2]  # "▁the", "▁to"
        text = auto_tokenizer.decode(already_decoded_ids)
        self.assertEqual(text, "the to")

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        tokenizer = ParakeetCTCTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )

        # Empty input
        self.assertEqual(tokenizer.decode([]), "")

        # Out of vocab tokens (should be mapped to UNK token)
        out_of_vocab = [999]  # Should map to UNK token ID 0
        decoded = tokenizer.decode(out_of_vocab)
        # Since 999 maps to UNK token which is "<unk>", we expect that
        # But actually _convert_id_to_token returns the unk_token string for unknown IDs
        self.assertIn("<unk>", decoded)

        # Test various input types
        self.assertEqual(tokenizer.decode(1), "the")  # Single int
        self.assertEqual(tokenizer.decode([1]), "the")  # List with single int

    def test_tokenizer_expects_ctc_decoded_input(self):
        """Test that tokenizer is designed to work with already CTC-decoded sequences."""
        tokenizer = ParakeetCTCTokenizer(
            vocab_file=str(self.vocab_file),
            blank_token_id=13,
        )

        # The tokenizer should expect input from model.generate() which is already CTC-decoded
        # This means no blanks and no consecutive duplicates
        model_output = [1, 2, 3, 4]  # Already CTC-decoded: "▁the", "▁to", "▁and", "▁a"

        result = tokenizer.decode(model_output)
        expected = "the to and a"
        self.assertEqual(result, expected)

        # Verify the docstring mentions this expectation
        self.assertIn("already been CTC-decoded", tokenizer.__class__.__doc__)
        self.assertIn("model.generate", tokenizer.__class__.__doc__)


if __name__ == "__main__":
    unittest.main()
