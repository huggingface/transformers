# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Tests for the VITS tokenizer."""
import json
import os
import shutil
import tempfile
import unittest

from transformers import VitsTokenizer
from transformers.models.vits.tokenization_vits import VOCAB_FILES_NAMES
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


class VitsTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = VitsTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab = (
            "k ' z y u d h e s w – 3 c p - 1 j m i X f l o 0 b r a 4 2 n _ x v t q 5 6 g ț ţ < > | <pad> <unk>".split(
                " "
            )
        )
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        vocab_tokens[" "] = vocab_tokens["X"]
        del vocab_tokens["X"]

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>"}

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        kwargs["phonemize"] = False
        kwargs["normalize"] = False
        return VitsTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        txt = "beyonce lives in los angeles"
        ids = tokenizer.encode(txt, add_special_tokens=False)
        return txt, ids

    @unittest.skip("Adding multicharacter tokens does not work with the VITS tokenizer")
    def test_add_tokens_tokenizer(self):
        pass

    @unittest.skip("Adding multicharacter tokens does not work with the VITS tokenizer")
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip("The VITS tokenizer does not support `is_split_into_words`")
    def test_pretokenized_inputs(self):
        pass

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

    @unittest.skip("Adding multicharacter tokens does not work the VITS tokenizer")
    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        pass

    def test_ron_normalization(self):
        tokenizer = self.get_tokenizer()
        tokenizer.language = "ron"

        sequences = ["vițs"]
        normalized_sequences = ["viţs"]

        encoded_ids = tokenizer(sequences, normalize=True)["input_ids"]
        decoded_sequences = tokenizer.batch_decode(encoded_ids)
        self.assertEqual(normalized_sequences, decoded_sequences)

    def test_normalization(self):
        tokenizer = self.get_tokenizer()

        sequences = ["VITS; is a model for t-t-s!"]
        normalized_sequences = ["vits is a model for t-t-s"]
        unnormalized_sequences = [
            "<unk><unk><unk><unk><unk> is a model for t-t-s<unk>"
        ]  # can't handle upper-case or certain punctuations

        encoded_normalized_ids = tokenizer(sequences, normalize=True)
        encoded_unnormalized_ids = tokenizer(sequences, normalize=False)

        decoded_normalized_sequences = [
            tokenizer.decode(seq, skip_special_tokens=False) for seq in encoded_normalized_ids["input_ids"]
        ]
        decoded_unnormalized_sequences = [
            tokenizer.decode(seq, skip_special_tokens=False) for seq in encoded_unnormalized_ids["input_ids"]
        ]

        self.assertEqual(decoded_normalized_sequences, normalized_sequences)
        self.assertEqual(decoded_unnormalized_sequences, unnormalized_sequences)

    @slow
    def test_tokenizer_integration(self):
        sequences = [
            "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers.",
            "The quick brown fox! Jumps over the lazy dog...",
            "We use k as our padding token",
        ]

        normalized_sequences = [
            "bert is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers",
            "the quick brown fox jumps over the lazy dog",
            "we use k as our padding token",
        ]

        # fmt: off
        expected_encoding = {
            'input_ids': [
                [0, 24, 0, 7, 0, 25, 0, 33, 0, 19, 0, 18, 0, 8, 0, 19, 0, 5, 0, 7, 0, 8, 0, 18, 0, 37, 0, 29, 0, 7, 0, 5, 0, 19, 0, 33, 0, 22, 0, 19, 0, 13, 0, 25, 0, 7, 0, 14, 0, 33, 0, 25, 0, 26, 0, 18, 0, 29, 0, 19, 0, 5, 0, 7, 0, 7, 0, 13, 0, 19, 0, 24, 0, 18, 0, 5, 0, 18, 0, 25, 0, 7, 0, 12, 0, 33, 0, 18, 0, 22, 0, 29, 0, 26, 0, 21, 0, 19, 0, 25, 0, 7, 0, 13, 0, 25, 0, 7, 0, 8, 0, 7, 0, 29, 0, 33, 0, 26, 0, 33, 0, 18, 0, 22, 0, 29, 0, 8, 0, 19, 0, 20, 0, 25, 0, 22, 0, 17, 0, 19, 0, 4, 0, 29, 0, 21, 0, 26, 0, 24, 0, 7, 0, 21, 0, 7, 0, 5, 0, 19, 0, 33, 0, 7, 0, 31, 0, 33, 0, 19, 0, 24, 0, 3, 0, 19, 0, 16, 0, 22, 0, 18, 0, 29, 0, 33, 0, 21, 0, 3, 0, 19, 0, 12, 0, 22, 0, 29, 0, 5, 0, 18, 0, 33, 0, 18, 0, 22, 0, 29, 0, 18, 0, 29, 0, 37, 0, 19, 0, 22, 0, 29, 0, 19, 0, 24, 0, 22, 0, 33, 0, 6, 0, 19, 0, 21, 0, 7, 0, 20, 0, 33, 0, 19, 0, 26, 0, 29, 0, 5, 0, 19, 0, 25, 0, 18, 0, 37, 0, 6, 0, 33, 0, 19, 0, 12, 0, 22, 0, 29, 0, 33, 0, 7, 0, 31, 0, 33, 0, 19, 0, 18, 0, 29, 0, 19, 0, 26, 0, 21, 0, 21, 0, 19, 0, 21, 0, 26, 0, 3, 0, 7, 0, 25, 0, 8, 0],
                [0, 33, 0, 6, 0, 7, 0, 19, 0, 34, 0, 4, 0, 18, 0, 12, 0, 0, 0, 19, 0, 24, 0, 25, 0, 22, 0, 9, 0, 29, 0, 19, 0, 20, 0, 22, 0, 31, 0, 19, 0, 16, 0, 4, 0, 17, 0, 13, 0, 8, 0, 19, 0, 22, 0, 32, 0, 7, 0, 25, 0, 19, 0, 33, 0, 6, 0, 7, 0, 19, 0, 21, 0, 26, 0, 2, 0, 3, 0, 19, 0, 5, 0, 22, 0, 37, 0, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
                [0, 9, 0, 7, 0, 19, 0, 4, 0, 8, 0, 7, 0, 19, 0, 0, 0, 19, 0, 26, 0, 8, 0, 19, 0, 22, 0, 4, 0, 25, 0, 19, 0, 13, 0, 26, 0, 5, 0, 5, 0, 18, 0, 29, 0, 37, 0, 19, 0, 33, 0, 22, 0, 0, 0, 7, 0, 29, 0, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            ],
            'attention_mask': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        }
        # fmt: on
        tokenizer_classes = [self.tokenizer_class]
        if self.test_rust_tokenizer:
            tokenizer_classes.append(self.rust_tokenizer_class)
        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained(
                "facebook/mms-tts-eng",
                revision="28cedf176aa99de5023a4344fd8a2cc477126fb8",  # to pin the tokenizer version
                pad_token="<pad>",
            )

            encoding = tokenizer(sequences, padding=True, normalize=True)
            decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in encoding["input_ids"]]

            encoding_data = encoding.data
            self.assertDictEqual(encoding_data, expected_encoding)

            for expected, decoded in zip(normalized_sequences, decoded_sequences):
                self.assertEqual(expected, decoded)
