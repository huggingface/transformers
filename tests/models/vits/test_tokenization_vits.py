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

from transformers import VitsMmsTokenizer
from transformers.models.vits.tokenization_vits import VOCAB_FILES_NAMES
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


class VitsTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = VitsMmsTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab = "k ' z y u d h e s w – 3 c p - 1 j m i X f l o 0 b r a 4 2 n _ x v t q 5 6 g < > | <pad> <unk>".split(
            " "
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
        return VitsMmsTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        txt = "beyonce lives in los angeles"
        ids = tokenizer.encode(txt, add_special_tokens=False)
        return txt, ids

    @unittest.skip("adding multicharacter tokens does not work this tokenizer")
    def test_add_tokens_tokenizer(self):
        pass

    @unittest.skip("adding multicharacter tokens does not work this tokenizer")
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip("this tokenizer does not support is_split_into_words")
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

    @unittest.skip("adding multicharacter tokens does not work this tokenizer")
    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        pass

    @slow
    def test_tokenizer_integration(self):
        # TODO: add this
        pass
