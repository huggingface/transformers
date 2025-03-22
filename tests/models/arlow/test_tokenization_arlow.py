import json
import os
import unittest

from transformers import ArlowTokenizer, ArlowTokenizerFast
from transformers.models.arlow.tokenization_arlow import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class ArlowTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "arlow/arlow-tokenizer"
    tokenizer_class = ArlowTokenizer
    rust_tokenizer_class = ArlowTokenizerFast
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def setUp(self):
        super().setUp()
        # Build a vocabulary using the byte_encoder values from ArlowTokenizer,
        # then extend with additional tokens for testing.
        vocab = list(ArlowTokenizer.byte_encoder.values())
        vocab.extend(["Hello", "world", "!", "H", "e", "l", "o", "w", "r", "d", " "])
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        # Define minimal merges for merging "Hello"
        merges = ["#version: 0.2", "H e", "He l", "Hel l", "Hell o"]

        self.special_tokens_map = {"eos_token": "<|endoftext|>"}
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return ArlowTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return ArlowTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        # For our test, tokenization of "Hello world!" should yield the same text.
        input_text = "Hello world!"
        output_text = "Hello world!"
        return input_text, output_text

    def test_python_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        sequence, _ = self.get_input_output_texts(tokenizer)
        # With our merge rules, "Hello" should be merged into one token.
        expected_bpe_tokens = ["Hello", " ", "world", "!"]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, expected_bpe_tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertTrue(all(isinstance(i, int) for i in token_ids))

    @unittest.skip(reason="Pretokenized inputs test not applicable")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip(reason="Clean up tokenization spaces test not applicable")
    def test_clean_up_tokenization_spaces(self):
        pass

    def test_nfc_normalization(self):
        # Using characters with different normalization forms.
        input_string = "\u03d2\u0301\u03d2\u0308\u017f\u0307"  # NFD form
        output_string = "\u03d3\u03d4\u1e9b"  # Expected NFC form
        if self.test_slow_tokenizer:
            tokenizer = self.get_tokenizer()
            normalized_text, _ = tokenizer.prepare_for_tokenization(input_string)
            self.assertEqual(normalized_text, output_string)
        if self.test_rust_tokenizer:
            tokenizer = self.get_rust_tokenizer()
            normalized_text = tokenizer.backend_tokenizer.normalizer.normalize_str(input_string)
            self.assertEqual(normalized_text, output_string)

    def test_slow_tokenizer_token_with_number_sign(self):
        if not self.test_slow_tokenizer:
            self.skipTest(reason="Slow tokenizer test disabled")
        sequence = " ###"
        # With our minimal vocab, assume that '#' is not merged; expect individual tokens.
        expected_tokens = [" ", "#", "#", "#"]
        tokenizer = self.get_tokenizer()
        self.assertListEqual(tokenizer.tokenize(sequence), expected_tokens)

    def test_slow_tokenizer_decode_spaces_between_special_tokens_default(self):
        if not self.test_slow_tokenizer:
            self.skipTest(reason="Slow tokenizer test disabled")
        # Decode tokens corresponding to "Hello world!"
        tokens = ["Hello", " ", "world", "!"]
        token_ids = self.get_tokenizer().convert_tokens_to_ids(tokens)
        expected_sequence = "Hello world!"
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.decode(token_ids), expected_sequence)

    @slow
    def test_tokenizer_integration(self):
        sequences = [
            "Transformers provides architectures for Natural Language Understanding and Generation.",
            "🤗 Transformers 提供了先进的预训练模型。",
            """```python\ntokenizer = AutoTokenizer.from_pretrained("arlow/arlow-tokenizer")\n"""
            """tokenizer("Hello, world!")```""",
        ]
        # For integration testing, we won't specify full expected encodings.
        expected_encoding = {"input_ids": [], "attention_mask": []}
        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="arlow/arlow-tokenizer",
            revision="dummy_revision",
            sequences=sequences,
        )


if __name__ == "__main__":
    unittest.main()
