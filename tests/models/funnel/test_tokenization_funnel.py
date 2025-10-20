# Copyright 2020 HuggingFace Inc. team.
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
import unittest
import collections

from transformers import FunnelTokenizer, AutoTokenizer
from transformers.models.funnel.tokenization_funnel import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin

# Master input string of combined test cases
input_string = """This is a test
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello



 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""

expected_tokens = ['this', 'is', 'a', 'test', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '<unk>', '的', '真', '<unk>', '<unk>', 'hi', 'hello', 'hi', 'hello', 'hello', '<s>', 'hi', '<s>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '<unk>', 'ir', '##d', '<unk>', 'hey', 'how', 'are', 'you', 'doing']
expected_token_ids = [101, 2023, 2003, 1037, 3231, 1045, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 6270, 1012, 1910, 100, 1916, 1921, 100, 100, 7632, 7592, 7632, 7592, 7592, 96, 7632, 96, 2045, 1996, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 7592, 1012, 2021, 20868, 2094, 1998, 100, 20868, 2094, 100, 4931, 2129, 2024, 2017, 2725, 102]


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


@require_tokenizers
class FunnelTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "funnel-transformer/small"
    tokenizer_class = FunnelTokenizer
    rust_tokenizer_class = FunnelTokenizer
    test_rust_tokenizer = False
    space_between_special_tokens = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from_pretrained_id = "funnel-transformer/small"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        cls.vocab = load_vocab(tok_auto.vocab_file)

        tok_auto.save_pretrained(cls.tmpdirname)
        tok_from_vocab = FunnelTokenizer(vocab=cls.vocab)

        cls.tokenizers = [tok_auto, tok_from_vocab]

    def test_integration_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_integration_expected_token_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)
