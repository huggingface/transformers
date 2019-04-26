# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from io import open
import shutil
import pytest

from pytorch_pretrained_bert.tokenization_transfo_xl import TransfoXLTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP


class TransfoXLTokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        vocab_tokens = [
            "<unk>", "[CLS]", "[SEP]", "want", "unwanted", "wa", "un", "running", ","
        ]
        with open("/tmp/transfo_xl_tokenizer_test.txt", "w", encoding='utf-8') as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
            vocab_file = vocab_writer.name

        tokenizer = TransfoXLTokenizer(vocab_file=vocab_file, lower_case=True)
        tokenizer.build_vocab()
        os.remove(vocab_file)

        tokens = tokenizer.tokenize(u"<unk> UNwanted , running")
        self.assertListEqual(tokens, ["<unk>", "unwanted", ",", "running"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [0, 4, 8, 7])

        vocab_file = tokenizer.save_vocabulary(vocab_path="/tmp/")
        tokenizer.from_pretrained(vocab_file)
        os.remove(vocab_file)

        tokens = tokenizer.tokenize(u"<unk> UNwanted , running")
        self.assertListEqual(tokens, ["<unk>", "unwanted", ",", "running"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [0, 4, 8, 7])


    def test_full_tokenizer_lower(self):
        tokenizer = TransfoXLTokenizer(lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(u" \tHeLLo ! how  \n Are yoU ?  "),
            ["hello", "!", "how", "are", "you", "?"])

    def test_full_tokenizer_no_lower(self):
        tokenizer = TransfoXLTokenizer(lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(u" \tHeLLo ! how  \n Are yoU ?  "),
            ["HeLLo", "!", "how", "Are", "yoU", "?"])

    @pytest.mark.slow
    def test_tokenizer_from_pretrained(self):
        cache_dir = "/tmp/pytorch_pretrained_bert_test/"
        for model_name in list(PRETRAINED_VOCAB_ARCHIVE_MAP.keys())[:1]:
            tokenizer = TransfoXLTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(tokenizer)

if __name__ == '__main__':
    unittest.main()
