# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import os
import tempfile
import unittest

from transformers.models.evo2.tokenization_evo2 import VOCAB_FILES_NAMES, Evo2Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Evo2TokenizationTest(unittest.TestCase):
    tokenizer_class = Evo2Tokenizer

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.tmpdirname = tempfile.mkdtemp()

        # Build a simple numeric vocab: "0" -> 0, "1" -> 1, ..., "255" -> 255
        vocab_size = 256
        vocab = {str(i): i for i in range(vocab_size)}

        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            json.dump(vocab, vocab_writer)

    def get_tokenizers(cls, **kwargs) -> list[PreTrainedTokenizerBase]:
        return [cls.get_tokenizer(**kwargs)]

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> PreTrainedTokenizer:
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def test_tokenizer_single_example(self):
        # Direct constructor
        tokenizer = self.tokenizer_class(self.vocab_file, vocab_size=256)

        text = "ABC"
        # ASCII codes: A=65, B=66, C=67
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ["65", "66", "67"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [65, 66, 67])

    def test_tokenizer_encode_single(self):
        tokenizer = self.tokenizer_class(self.vocab_file, vocab_size=256)

        text = "ABC"
        # encode() should NOT add BOS/EOS for this char-level tokenizer
        self.assertListEqual(tokenizer.encode(text), [65, 66, 67])

    def test_tokenizer_call_no_pad(self):
        tokenizer = self.tokenizer_class(self.vocab_file, vocab_size=256)

        seq_batch = ["AB", "XYZ"]
        encoded = tokenizer(seq_batch, padding=False)["input_ids"]

        # "AB" -> 65,66 ; "XYZ" -> 88,89,90
        self.assertListEqual(encoded, [[65, 66], [88, 89, 90]])

    def test_tokenizer_call_pad(self):
        tokenizer = self.tokenizer_class(self.vocab_file, vocab_size=256)

        seq_batch = ["AB", "XYZ"]
        encoded = tokenizer(seq_batch, padding=True)["input_ids"]

        # pad_token_id should be 1, so shorter seq gets padded with 1
        # max length = 3
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertListEqual(encoded, [[65, 66, 1], [88, 89, 90]])

    def test_detokenize_roundtrip(self):
        tokenizer = self.tokenizer_class(self.vocab_file, vocab_size=256)

        text = "Hello!"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)

        # Because of clamp, some low values could be bumped, but ASCII letters
        # should round-trip cleanly.
        self.assertEqual(decoded, text)

    def test_add_tokens(self):
        tokenizer = self.tokenizer_class(self.vocab_file, vocab_size=256)

        vocab_size = len(tokenizer)
        self.assertEqual(tokenizer.add_tokens(""), 0)
        self.assertEqual(tokenizer.add_tokens("testtoken"), 1)
        self.assertEqual(tokenizer.add_tokens(["testtoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer), vocab_size + 3)

        self.assertEqual(tokenizer.add_special_tokens({}), 0)
        self.assertEqual(tokenizer.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"}), 2)

        # additional_special_tokens logic
        self.assertRaises(AssertionError, tokenizer.add_special_tokens, {"additional_special_tokens": "<testtoken1>"})
        self.assertEqual(tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
        self.assertEqual(
            tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
        )
        self.assertIn("<testtoken3>", tokenizer.special_tokens_map["additional_special_tokens"])
        self.assertIsInstance(tokenizer.special_tokens_map["additional_special_tokens"], list)
        self.assertGreaterEqual(len(tokenizer.special_tokens_map["additional_special_tokens"]), 2)

        self.assertEqual(len(tokenizer), vocab_size + 8)
