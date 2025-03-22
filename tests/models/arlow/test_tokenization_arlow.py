# test_tokenization_arlow.py

import unittest

from transformers import TokenizerTesterMixin
from transformers.models.arlow import ArlowTokenizer
from transformers.testing_utils import CaptureStdout, require_tokenizers, slow


@require_tokenizers
class ArlowTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    """
    This test suite checks that your ArlowTokenizer (a fast tokenizer)
    behaves correctly. It inherits from TokenizerTesterMixin to run
    standard Hugging Face tokenizer tests.
    """

    # If your tokenizer is purely a fast tokenizer:
    tokenizer_class = ArlowTokenizer
    rust_tokenizer_class = ArlowTokenizer  # Re-affirming it's the same, if you only have a fast version
    test_slow_tokenizer = False  # There's no slow tokenizer, only fast

    def setUp(self):
        super().setUp()
        # Provide any additional setup, e.g. loading or creating a test-specific vocab.
        # If your ArlowTokenizer doesn't require explicit vocab/merges, you can skip.
        # If it does, create small test files or mock data here.

        # For example, if you need a special tokens map:
        self.special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }

        # Instantiate the tokenizer in a minimal way for testing
        # (You might pass vocab_file, merges_file if your tokenizer needs them.)
        self.tokenizer = ArlowTokenizer(
            **self.special_tokens_map
        )

    def test_full_tokenizer(self):
        """
        A basic encode/decode round-trip test.
        """
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)  # returns a list of token IDs
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(
            decoded, text,
            f"Decoded text '{decoded}' did not match the original: '{text}'"
        )

    def test_special_tokens(self):
        """
        Ensure that special tokens (bos, eos, pad, unk) work as expected.
        """
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        unk_id = self.tokenizer.unk_token_id

        self.assertIsNotNone(bos_id, "BOS token ID should be set.")
        self.assertIsNotNone(eos_id, "EOS token ID should be set.")
        self.assertIsNotNone(pad_id, "PAD token ID should be set.")
        self.assertIsNotNone(unk_id, "UNK token ID should be set.")

    def test_padding(self):
        """
        Check that the tokenizer can pad to a specific length or to the longest sequence.
        """
        inputs = ["Hello", "Hello, world!", "This is a test sentence"]
        batch_enc = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )
        # Now check shapes
        self.assertEqual(batch_enc["input_ids"].shape[1], 16)

    def test_add_tokens(self):
        """
        Test adding new tokens to the tokenizer vocabulary.
        """
        new_tokens = ["[NEW_TOKEN]", "[ANOTHER_TOKEN]"]
        added_num = self.tokenizer.add_tokens(new_tokens)
        self.assertEqual(added_num, len(new_tokens), "Mismatch in added tokens count.")

        # Now ensure they can be encoded
        test_str = " ".join(new_tokens)
        encoded = self.tokenizer.encode(test_str, add_special_tokens=False)
        self.assertTrue(all(tok_id != self.tokenizer.unk_token_id for tok_id in encoded))

    @slow
    def test_integration(self):
        """
        A 'slow' test for large or more complex usage, if needed. You can remove or adapt it.
        """
        text = "A longer piece of text to measure the integration performance."
        # Possibly do a "real" usage test or compare to some reference output.
        encoded = self.tokenizer(text, return_tensors="pt")
        with CaptureStdout() as cs:
            print(f"Tokens: {encoded['input_ids']}")
        # We won't assert anything specific here, but you can add real checks.
        self.assertIn("Tokens:", cs.out)

if __name__ == "__main__":
    unittest.main()
