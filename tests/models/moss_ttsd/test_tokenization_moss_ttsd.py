# Copyright 2025 OpenMOSS and The HuggingFace Inc. team. All rights reserved.
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

import unittest

from transformers import PreTrainedTokenizer
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


# Special tokens
PAD = 151643
S1 = 151844  # [S1] token
S2 = 151845  # [S2] token


class MossTTSDTokenizer(PreTrainedTokenizer):
    """
    Minimal tokenizer for testing purposes.
    MOSS-TTSD uses Qwen tokenizer under the hood through AutoTokenizer.
    This is a simplified byte-level tokenizer similar to DiaTokenizer.
    """

    def __init__(self, **kwargs):
        # Initialize vocabulary
        self.encoder = {}
        self.decoder = {}

        # Add special tokens
        self.encoder["<pad>"] = PAD
        self.encoder["[S1]"] = S1
        self.encoder["[S2]"] = S2

        # Add basic UTF-8 characters (byte-level tokenization)
        for i in range(256):
            if chr(i) not in self.encoder:
                self.encoder[chr(i)] = i

        # Build decoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Set special tokens - remove pad_token from kwargs if it exists
        if "pad_token" not in kwargs:
            kwargs["pad_token"] = "<pad>"

        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder)

    def _tokenize(self, text):
        """Tokenize a string into character/byte tokens."""
        # For special tokens, keep them as single tokens
        tokens = []
        i = 0
        text_str = text

        # Process the text string for special tokens first
        while i < len(text_str):
            # Check for special tokens
            if text_str[i : i + 4] == "[S1]":
                tokens.append("[S1]")
                i += 4
            elif text_str[i : i + 4] == "[S2]":
                tokens.append("[S2]")
                i += 4
            elif text_str[i : i + 5] == "<pad>":
                tokens.append("<pad>")
                i += 5
            else:
                # For regular text, use byte-level tokenization
                char = text_str[i]
                char_bytes = char.encode("utf-8")
                for byte in char_bytes:
                    # Convert byte to chr representation for vocab lookup
                    tokens.append(chr(byte))
                i += 1
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocabulary."""
        return self.encoder.get(token, self.encoder.get(self.unk_token, 0))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocabulary."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens to a single string."""
        # Handle special tokens directly
        result_bytes = []
        for token in tokens:
            if token in ["[S1]", "[S2]", "<pad>"]:
                # For special tokens, convert existing bytes to string first
                if result_bytes:
                    try:
                        partial = bytes(result_bytes).decode("utf-8", errors="ignore")
                    except (UnicodeDecodeError, ValueError):
                        partial = "".join(chr(b) for b in result_bytes)
                    result_bytes = []
                else:
                    partial = ""
                return partial + token + self.convert_tokens_to_string(tokens[tokens.index(token) + 1 :])
            else:
                # Regular tokens are single bytes represented as characters
                result_bytes.append(ord(token))

        # Convert accumulated bytes to string
        if result_bytes:
            try:
                return bytes(result_bytes).decode("utf-8", errors="ignore")
            except (UnicodeDecodeError, ValueError):
                return "".join(chr(b) for b in result_bytes)
        return ""

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the tokenizer vocabulary to a directory or file."""
        import json
        import os

        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
            )
        else:
            vocab_file = save_directory

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False)

        return (vocab_file,)


class MossTTSDTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MossTTSDTokenizer
    test_rust_tokenizer = False
    test_slow_tokenizer = True  # Enable slow tokenizer tests for our custom tokenizer

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tokenizer = MossTTSDTokenizer()
        tokenizer.save_pretrained(cls.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "i"
        token_id = 105  # ASCII code for 'i'

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab = self.get_tokenizer().get_vocab()

        self.assertEqual(vocab["<pad>"], PAD)
        self.assertEqual(vocab["[S1]"], S1)
        self.assertEqual(vocab["[S2]"], S2)
        # Should have at least the special tokens plus basic characters
        self.assertGreater(len(vocab), 256)

    def test_vocab_size(self):
        # Should have at least 256 UTF-8 characters + special tokens
        self.assertGreater(self.get_tokenizer().vocab_size, 256)

    def test_full_tokenizer(self):
        tokenizer = MossTTSDTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("Hello, world!")
        self.assertListEqual(tokens, ["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33])
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, ["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"])

        tokens = tokenizer.tokenize("[S1] Hello [S2] Hello<pad>")
        self.assertListEqual(
            tokens,
            ["[S1]", " ", "H", "e", "l", "l", "o", " ", "[S2]", " ", "H", "e", "l", "l", "o", "<pad>"],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [S1, 32, 72, 101, 108, 108, 111, 32, S2, 32, 72, 101, 108, 108, 111, PAD])
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens, ["[S1]", " ", "H", "e", "l", "l", "o", " ", "[S2]", " ", "H", "e", "l", "l", "o", "<pad>"]
        )

    def test_chinese_text_tokenization(self):
        """Test tokenization of Chinese text (MOSS-TTSD's primary use case)."""
        tokenizer = self.get_tokenizer()

        # Test Chinese text - note that our test tokenizer is byte-level
        chinese_text = "人工智能"
        tokens = tokenizer.tokenize(chinese_text)

        # Basic check that tokenization worked
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Test encoding/decoding
        encoded = tokenizer.encode(chinese_text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)

        # Byte-level tokenizer should preserve the original text
        self.assertEqual(decoded, chinese_text)

    def test_speaker_tags_tokenization(self):
        """Test tokenization of speaker tags used in MOSS-TTSD."""
        tokenizer = self.get_tokenizer()

        # Test speaker tags
        speaker_text = "[S1]你好世界[S2]Hello world"
        tokens = tokenizer.tokenize(speaker_text)

        # Basic validation
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Check that [S1] and [S2] are preserved as single tokens
        self.assertIn("[S1]", tokens)
        self.assertIn("[S2]", tokens)

        # Test that speaker tags are preserved in encoding/decoding
        encoded = tokenizer.encode(speaker_text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)

        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertIn("[S1]", decoded)
        self.assertIn("[S2]", decoded)

    def test_mixed_language_tokenization(self):
        """Test tokenization of mixed Chinese/English text."""
        tokenizer = self.get_tokenizer()

        # Test mixed language
        mixed_text = "MOSS-TTSD是一个text-to-speech模型"
        tokens = tokenizer.tokenize(mixed_text)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Test round-trip encoding/decoding
        encoded = tokenizer.encode(mixed_text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertEqual(decoded, mixed_text)

    def test_special_tokens_handling(self):
        """Test handling of special tokens in MOSS-TTSD context."""
        tokenizer = self.get_tokenizer()

        # Test that special tokens are properly handled
        text_with_special = "[S1] Test [S2] <pad>"

        # Test with skip_special_tokens=False
        encoded = tokenizer.encode(text_with_special, add_special_tokens=False)
        decoded_with_special = tokenizer.decode(encoded, skip_special_tokens=False)

        self.assertIn("[S1]", decoded_with_special)
        self.assertIn("[S2]", decoded_with_special)
        self.assertIn("<pad>", decoded_with_special)

        # Test with skip_special_tokens=True
        decoded_without_special = tokenizer.decode(encoded, skip_special_tokens=True)

        # Pad token should be removed
        self.assertNotIn("<pad>", decoded_without_special)

    def test_batch_tokenization(self):
        """Test batch tokenization functionality."""
        tokenizer = self.get_tokenizer()
        tokenizer.pad_token = "<pad>"  # Ensure pad token is set

        # Test batch of texts
        texts = ["这是第一个测试", "This is test", "[S1]混合[S2]Mixed"]

        # Test batch encoding
        encoded_batch = tokenizer(texts, padding=True, return_tensors="pt")

        self.assertIn("input_ids", encoded_batch)
        self.assertIn("attention_mask", encoded_batch)

        # Check shapes are consistent
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        self.assertEqual(input_ids.shape[0], len(texts))
        self.assertEqual(attention_mask.shape[0], len(texts))
        self.assertEqual(input_ids.shape, attention_mask.shape)

    def test_long_text_handling(self):
        """Test handling of long text sequences."""
        tokenizer = self.get_tokenizer()
        tokenizer.model_max_length = 512  # Set a reasonable max length

        # Create a long text
        base_text = "人工智能技术正在快速发展，"
        long_text = base_text * 20  # Repeat to make it long

        # Test tokenization with truncation
        encoded_truncated = tokenizer(long_text, max_length=128, truncation=True, return_tensors="pt")

        self.assertIn("input_ids", encoded_truncated)
        self.assertLessEqual(encoded_truncated["input_ids"].shape[1], 128)

    def test_tokenizer_consistency(self):
        """Test that tokenizer produces consistent results."""
        tokenizer = self.get_tokenizer()

        test_text = "一致性测试文本"

        # Tokenize the same text multiple times
        encoded_1 = tokenizer.encode(test_text)
        encoded_2 = tokenizer.encode(test_text)

        # Results should be identical
        self.assertEqual(encoded_1, encoded_2)

    @slow
    def test_tokenizer_integration(self):
        """Test tokenizer integration with actual MOSS-TTSD model."""
        # Test with actual MOSS-TTSD tokenizer from model hub
        from transformers import AutoTokenizer

        # Expected encoding for MOSS-TTSD tokenizer (Qwen-based) with comprehensive test sequences
        expected_encoding = {
            'input_ids': [
                [104455, 99361, 96555, 106389, 3837, 17714, 103952, 99424, 104923, 112303, 104126, 1773],
                [10531, 1220, 9285, 51, 5491, 374, 264, 1467, 4686, 1331, 39586, 1614, 6188, 369, 5810, 8806, 38875, 13],
                [42474, 16, 60, 108386, 99489, 42474, 17, 60, 9707, 1879]
            ],
            'attention_mask': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ]
        }  # fmt: skip

        sequences = [
            "人工智能技术正在快速发展，为我们的生活带来了深远的影响。",  # AI technology is developing rapidly
            "MOSS-TTSD is a text-to-speech model designed for natural speech synthesis.",  # English test
            "[S1]你好世界[S2]Hello world",  # Mixed with speaker tags
        ]

        tokenizer_classes = [AutoTokenizer]  # Use AutoTokenizer for MOSS-TTSD

        for tokenizer_class in tokenizer_classes:
            # Load actual MOSS-TTSD tokenizer from model hub
            tokenizer = tokenizer_class.from_pretrained("fnlp/MOSS-TTSD-v0.5")

            encoding = tokenizer(sequences)
            encoding_data = encoding.data

            # Check the encoding matches expected values
            self.assertDictEqual(encoding_data, expected_encoding)

            # Test decoding - ensure sequences round-trip correctly
            decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in encoding["input_ids"]]

            for expected, decoded in zip(sequences, decoded_sequences):
                # For MOSS-TTSD, the decoded text should match the original
                # Note: Speaker tags [S1], [S2] are special tokens in the tokenizer
                self.assertEqual(expected, decoded)

            # Additional test for Chinese text tokenization
            chinese_text = "人工智能浪潮正在席卷全球"
            chinese_encoded = tokenizer.encode(chinese_text)
            chinese_decoded = tokenizer.decode(chinese_encoded, skip_special_tokens=True)

            # The decoded text should match the original
            self.assertIsInstance(chinese_encoded, list)
            self.assertGreater(len(chinese_encoded), 0)
            self.assertIsInstance(chinese_decoded, str)
            self.assertEqual(chinese_decoded, chinese_text)

            # Test batch encoding/decoding
            batch_texts = [
                "这是第一个测试句子。",
                "This is the second test sentence.",
                "[S1]你好！[S2]Hello!",
            ]

            batch_encoding = tokenizer(batch_texts, padding=True, return_tensors="pt")

            # Verify batch encoding structure
            self.assertIn("input_ids", batch_encoding)
            self.assertIn("attention_mask", batch_encoding)
            self.assertEqual(batch_encoding["input_ids"].shape[0], len(batch_texts))

            # Test decoding of batch
            for i in range(len(batch_texts)):
                decoded = tokenizer.decode(batch_encoding["input_ids"][i], skip_special_tokens=True)
                self.assertIsInstance(decoded, str)
                # Check that essential content is preserved (may have added tokens)
                # Remove speaker tags for comparison if present
                clean_decoded = decoded.replace("[S1]", "").replace("[S2]", "")
                # Basic check that some content is preserved
                self.assertGreater(len(clean_decoded.strip()), 0)

    @unittest.skip(reason="MOSS-TTSD relies on byte-level tokenization similar to Dia.")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip("Not applicable for MOSS-TTSD tokenizer testing")
    def test_tokenizer_slow_store_full_signature(self):
        pass
