# coding=utf-8
# Copyright 2021 HuggingFace Inc. team.
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

from transformers import BartphoTokenizer, BartphoTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe.model")


@require_sentencepiece
@require_tokenizers
class BartphoTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BartphoTokenizer
    rust_tokenizer_class = BartphoTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        self.special_tokens_map = {"unk_token": "<unk>"}

        tokenizer = BartphoTokenizer(SAMPLE_VOCAB, **self.special_tokens_map)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BartphoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "This is a là test"
        output_text = "This is a l<unk> test"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = BartphoTokenizer(SAMPLE_VOCAB, **self.special_tokens_map)
        text = "This is a là test"
        bpe_tokens = "▁This ▁is ▁a ▁l à ▁t est".split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [475, 98, 6, 41, 3, 4, 264, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "I was born in 2000."

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
