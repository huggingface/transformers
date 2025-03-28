import json
import os
import shutil
import tempfile
import unittest

from transformers import PreTrainedTokenizer
from transformers.models.arlow.tokenization_arlow import ArlowTokenizer
from transformers.models.arlow.tokenization_arlow_fast import ArlowTokenizerFast


# Minimal dummy vocab and merges for testing.
DUMMY_VOCAB = {
    "h": 0,
    "e": 1,
    "l": 2,
    "o": 3,
    "w": 4,
    "r": 5,
    "d": 6,
    "<|unk|>": 7,
    "<|pad|>": 8,
    "<|startoftext|>": 9,
    "<|endoftext|>": 10,
}
# We won't perform any merges, so merges is empty.
DUMMY_MERGES = ""

# Create a valid minimal dummy tokenizer.json.
DUMMY_TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": {
        "type": "BertNormalizer",
        "lowercase": True,
        "strip_accents": False,
        "clean_text": True,
        "handle_chinese_chars": True,
    },
    "pre_tokenizer": {"type": "BertPreTokenizer"},
    "post_processor": None,
    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "decode_special_tokens": True},
    "model": {
        "unk_token": "<|unk|>",
        "type": "BPE",
        "vocab": DUMMY_VOCAB,
        "merges": [],  # No merges
    },
}


class ArlowTokenizerTester(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to store dummy vocab, merges, and tokenizer.json.
        self.tmp_dir = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmp_dir, "vocab.json")
        self.merges_file = os.path.join(self.tmp_dir, "merges.txt")
        self.tokenizer_file = os.path.join(self.tmp_dir, "tokenizer.json")

        # Write the dummy vocab file.
        with open(self.vocab_file, "w", encoding="utf-8") as vf:
            json.dump(DUMMY_VOCAB, vf)

        # Write the dummy merges file.
        with open(self.merges_file, "w", encoding="utf-8") as mf:
            mf.write(DUMMY_MERGES)

        # Write the dummy tokenizer.json.
        with open(self.tokenizer_file, "w", encoding="utf-8") as tf:
            json.dump(DUMMY_TOKENIZER_JSON, tf)

        # Define a sample sentence.
        self.sample_text = "hello world"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_slow_tokenizer(self):
        # Instantiate the slow ArlowTokenizer.
        tokenizer: PreTrainedTokenizer = ArlowTokenizer(
            vocab_file=self.vocab_file,
            merges_file=self.merges_file,
        )
        # Test tokenization.
        tokens = tokenizer.tokenize(self.sample_text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Test conversion from tokens to IDs and back.
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))
        # Use the public API: convert_ids_to_tokens.
        recovered_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        self.assertIsInstance(recovered_tokens, list)

        # Test convert_tokens_to_string.
        recovered_text = tokenizer.convert_tokens_to_string(tokens)
        self.assertIsInstance(recovered_text, str)
        self.assertGreater(len(recovered_text), 0)

    def test_fast_tokenizer(self):
        # Instantiate the fast tokenizer from the temporary directory.
        tokenizer = ArlowTokenizerFast.from_pretrained(self.tmp_dir)
        # Test that it returns a BatchEncoding with input_ids.
        encoding = tokenizer(self.sample_text, padding=True)
        self.assertIn("input_ids", encoding)
        self.assertIn("attention_mask", encoding)
        self.assertIsInstance(encoding["input_ids"], list)
        self.assertGreater(len(encoding["input_ids"]), 0)

        # Test decoding.
        decoded = tokenizer.decode(encoding["input_ids"][0])
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), 0)


if __name__ == "__main__":
    unittest.main()
