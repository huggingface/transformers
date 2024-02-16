# coding=utf-8
# Copyright 2024 Huggingface
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

import os
import tempfile
import unittest

from transformers import (
    FLMRContextEncoderTokenizer,
    FLMRContextEncoderTokenizerFast,
    FLMRQueryEncoderTokenizer,
    FLMRQueryEncoderTokenizerFast,
    is_torch_available,
)
from transformers.testing_utils import require_tokenizers


if is_torch_available():
    pass

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer_config.json"}


@require_tokenizers
class FLMRContextEncoderTokenizationTest:
    tokenizer_class = FLMRContextEncoderTokenizer
    rust_tokenizer_class = FLMRContextEncoderTokenizerFast
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
            "[unused0]",  # Added for ColBERT Q Marker
            "[unused1]",  # Added for ColBERT D Marker
        ]
        # Create a temp folder to store the vocab using a package
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdirname = self.tmpdir.name
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def test_full_tokenizer(
        self,
        doc_maxlen=32,
    ):
        tokenizer = self.tokenizer_class(self.vocab_file, doc_maxlen=doc_maxlen)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [9, 6, 7, 12, 10, 11])

        test_sentences = ["UNwant\u00E9d,running", "UNwant\u00E9d,running"]
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        expected_encoding = {
            "input_ids": [
                [1, 16, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [1, 15, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        }

        # Test if one sentence can be converted to a list
        test_sentences = "UNwant\u00E9d,running"
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        expected_encoding = {
            "input_ids": [
                [1, 16, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
        }
        self.assertDictEqual(encoded, expected_encoding)

    def get_tokenizer(self, *args, **kwargs):
        return self.tokenizer_class(self.vocab_file, *args, **kwargs)

    def get_rust_tokenizer(self, *args, **kwargs):
        return self.rust_tokenizer_class(self.vocab_file, *args, **kwargs)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "UNwant\u00E9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        # With lower casing
        tokenizer = self.get_tokenizer(do_lower_case=True)
        rust_tokenizer = self.get_rust_tokenizer(do_lower_case=True)

        sequence = "UNwant\u00E9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    def test_doc_maxlen(self, doc_maxlen=16):
        tokenizer = self.tokenizer_class(self.vocab_file, doc_maxlen=doc_maxlen)
        test_sentences = "UNwant\u00E9d,running"
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        expected_encoding = {
            "input_ids": [[1, 16, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
        }
        self.assertDictEqual(encoded, expected_encoding)

    def test_doc_maxlen_truncated(self, doc_maxlen=4):
        tokenizer = self.tokenizer_class(self.vocab_file, doc_maxlen=doc_maxlen)
        test_sentences = "UNwant\u00E9d,running"
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        print(encoded)
        # Now [MASK] tokens are attended
        expected_encoding = {"input_ids": [[1, 16, 9, 2]], "attention_mask": [[1, 1, 1, 1]]}
        self.assertDictEqual(encoded, expected_encoding)


# Copied and modified from ..bert.test_tokenization_bert.BertTokenizationTest
@require_tokenizers
class FLMRQueryEncoderTokenizationTest(unittest.TestCase):
    tokenizer_class = FLMRQueryEncoderTokenizer
    rust_tokenizer_class = FLMRQueryEncoderTokenizerFast
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
            "[unused0]",  # Added for ColBERT Q Marker
            "[unused1]",  # Added for ColBERT D Marker
        ]
        # Create a temp folder to store the vocab using a package
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdirname = self.tmpdir.name
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def test_full_tokenizer(
        self,
        query_maxlen=32,
    ):
        tokenizer = self.tokenizer_class(self.vocab_file, query_maxlen=query_maxlen)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [9, 6, 7, 12, 10, 11])

        test_sentences = ["UNwant\u00E9d,running", "UNwant\u00E9d,running"]
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        expected_encoding = {
            "input_ids": [
                [1, 15, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [1, 15, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        }

        # Test if one sentence can be converted to a list
        test_sentences = "UNwant\u00E9d,running"
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        expected_encoding = {
            "input_ids": [
                [1, 15, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
        }
        self.assertDictEqual(encoded, expected_encoding)

    def get_tokenizer(self, *args, **kwargs):
        return self.tokenizer_class(self.vocab_file, *args, **kwargs)

    def get_rust_tokenizer(self, *args, **kwargs):
        return self.rust_tokenizer_class(self.vocab_file, *args, **kwargs)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "UNwant\u00E9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        # With lower casing
        tokenizer = self.get_tokenizer(do_lower_case=True)
        rust_tokenizer = self.get_rust_tokenizer(do_lower_case=True)

        sequence = "UNwant\u00E9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    def test_attend_to_mask_tokens(self):
        tokenizer = self.tokenizer_class(self.vocab_file, attend_to_mask_tokens=True)
        test_sentences = "UNwant\u00E9d,running"
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        # Now [MASK] tokens are attended
        expected_encoding = {
            "input_ids": [
                [1, 15, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ],
        }
        self.assertDictEqual(encoded, expected_encoding)

    def test_query_maxlen(self, query_maxlen=16):
        tokenizer = self.tokenizer_class(self.vocab_file, query_maxlen=query_maxlen)
        test_sentences = "UNwant\u00E9d,running"
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        expected_encoding = {
            "input_ids": [[1, 15, 9, 6, 7, 12, 10, 11, 2, 4, 4, 4, 4, 4, 4, 4]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
        }
        self.assertDictEqual(encoded, expected_encoding)

    def test_query_maxlen_truncated(self, query_maxlen=4):
        tokenizer = self.tokenizer_class(self.vocab_file, query_maxlen=query_maxlen)
        test_sentences = "UNwant\u00E9d,running"
        encoded = tokenizer(test_sentences)
        encoded = dict(**encoded)
        encoded["input_ids"] = encoded["input_ids"].tolist()
        encoded["attention_mask"] = encoded["attention_mask"].tolist()
        expected_encoding = {"input_ids": [[1, 15, 9, 2]], "attention_mask": [[1, 1, 1, 1]]}
        self.assertDictEqual(encoded, expected_encoding)
