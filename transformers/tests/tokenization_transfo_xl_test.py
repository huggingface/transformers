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
import pytest
from io import open

from transformers import is_torch_available

if is_torch_available():
    import torch
    from transformers.tokenization_transfo_xl import TransfoXLTokenizer, VOCAB_FILES_NAMES
else:
    pytestmark = pytest.mark.skip("Require Torch")  # TODO: untangle Transfo-XL tokenizer from torch.load and torch.save

from .tokenization_tests_commons import CommonTestCases

class TransfoXLTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = TransfoXLTokenizer if is_torch_available() else None

    def setUp(self):
        super(TransfoXLTokenizationTest, self).setUp()

        vocab_tokens = [
            "<unk>", "[CLS]", "[SEP]", "want", "unwanted", "wa", "un",
            "running", ",", "low", "l",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, "w", encoding='utf-8') as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        kwargs['lower_case'] = True
        return TransfoXLTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"<unk> UNwanted , running"
        output_text = u"<unk> unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = TransfoXLTokenizer(vocab_file=self.vocab_file, lower_case=True)

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


if __name__ == '__main__':
    unittest.main()
