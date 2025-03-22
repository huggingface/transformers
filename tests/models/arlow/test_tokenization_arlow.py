import unittest

from transformers.models.arlow import ArlowTokenizer
from transformers.testing_utils import CaptureStdout, require_tokenizers, slow


@require_tokenizers
class ArlowTokenizerTest(unittest.TestCase):
    """
    This test suite checks that your ArlowTokenizer (a fast tokenizer) behaves correctly.
    """

    def setUp(self):
        # Provide a minimal special tokens map for testing.
        self.special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        # Instantiate the tokenizer.
        self.tokenizer = ArlowTokenizer(**self.special_tokens_map)

    def test_encode_decode(self):
        """A basic encode/decode round-trip test."""
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text, f"Decoded text '{decoded}' did not match the original: '{text}'")

    def test_special_tokens(self):
        """Ensure that special tokens (bos, eos, pad, unk) are set."""
        self.assertIsNotNone(self.tokenizer.bos_token_id, "BOS token ID should be set.")
        self.assertIsNotNone(self.tokenizer.eos_token_id, "EOS token ID should be set.")
        self.assertIsNotNone(self.tokenizer.pad_token_id, "PAD token ID should be set.")
        self.assertIsNotNone(self.tokenizer.unk_token_id, "UNK token ID should be set.")

    def test_padding(self):
        """Check that the tokenizer pads sequences correctly."""
        inputs = ["Hello", "Hello, world!", "This is a test sentence"]
        batch_enc = self.tokenizer(inputs, padding=True, truncation=True, max_length=16, return_tensors="pt")
        # The resulting input_ids should have shape (batch_size, 16)
        self.assertEqual(batch_enc["input_ids"].shape[1], 16)

    def test_add_tokens(self):
        """Test adding new tokens to the vocabulary."""
        new_tokens = ["[NEW_TOKEN]", "[ANOTHER_TOKEN]"]
        added_num = self.tokenizer.add_tokens(new_tokens)
        self.assertEqual(added_num, len(new_tokens), "Mismatch in added tokens count.")

        # Check that the new tokens are encoded and not mapped to the unknown token.
        test_str = " ".join(new_tokens)
        encoded = self.tokenizer.encode(test_str, add_special_tokens=False)
        self.assertTrue(all(tok_id != self.tokenizer.unk_token_id for tok_id in encoded))

    @slow
    def test_integration(self):
        """A 'slow' integration test for a longer piece of text."""
        text = "A longer piece of text to measure the integration performance."
        encoded = self.tokenizer(text, return_tensors="pt")
        with CaptureStdout() as cs:
            print(f"Tokens: {encoded['input_ids']}")
        self.assertIn("Tokens:", cs.out)


if __name__ == "__main__":
    unittest.main()
