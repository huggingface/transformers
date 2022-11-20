# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
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

from transformers import RemBertTokenizer, RemBertTokenizerFast
from transformers.testing_utils import require_sentencepiece, require_tokenizers
from transformers.utils import is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin

FRAMEWORK = "pt" if is_torch_available() else "tf"


@require_sentencepiece
@require_tokenizers
class RemBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    
    tokenizer_class = RemBertTokenizer
    rust_tokenizer_class = RemBertTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        tokenizer = RemBertTokenizer.from_pretrained("google/rembert")
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "[PAD]"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "[PAD]")
        self.assertEqual(vocab_keys[1], "</s>")
        self.assertEqual(vocab_keys[-1], "ઞ")
        self.assertEqual(len(vocab_keys), 250300)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 250300)

    def test_rust_and_python_bpe_tokenizers(self):
        tokenizer = RemBertTokenizer.from_pretrained("google/rembert")
        tokenizer.save_pretrained(self.tmpdirname)
        rust_tokenizer = RemBertTokenizerFast.from_pretrained(self.tmpdirname)

        sequence = "Hi I am new at huggingface."

        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        tokens = tokenizer.convert_ids_to_tokens(ids)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "Hi I am new at huggingface."

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

    # can't train new_tokenizer via Tokenizers lib
    def test_training_new_tokenizer(self):
        pass

    # can't train new_tokenizer via Tokenizers lib
    def test_training_new_tokenizer_with_special_tokens_change(self):
        pass

    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        pass