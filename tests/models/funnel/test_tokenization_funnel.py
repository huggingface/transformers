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
    space_between_special_tokens = True


    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['this', 'is', 'a', 'test', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '<unk>', '的', '真', '<unk>', '<unk>', 'hi', 'hello', 'hi', 'hello', 'hello', '<s>', 'hi', '<s>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '<unk>', 'ir', '##d', '<unk>', 'hey', 'how', 'are', 'you', 'doing']
    integration_expected_token_ids = [101, 2023, 2003, 1037, 3231, 1045, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 6270, 1012, 1910, 100, 1916, 1921, 100, 100, 7632, 7592, 7632, 7592, 7592, 96, 7632, 96, 2045, 1996, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 7592, 1012, 2021, 20868, 2094, 1998, 100, 20868, 2094, 100, 4931, 2129, 2024, 2017, 2725, 102]
    integration_expected_decoded_text = '<unk> is a test <unk> <unk> was born in 92000, and this is <unk>. 生 <unk> 的 真 <unk> <unk> <unk> <unk> <unk> <unk> <unk> <s> hi <s> there <unk> following string should be properly encoded : <unk>. <unk> ird and <unk> ird <unk> <unk> how are you doing'
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from_pretrained_id = "funnel-transformer/small"

        tokenizer = FunnelTokenizer.from_pretrained(from_pretrained_id)
        cls.vocab = load_vocab(tokenizer.vocab_file)

        tokenizer.save_pretrained(cls.tmpdirname)
        tokenizer_from_vocab = FunnelTokenizer(vocab=cls.vocab)

        cls.tokenizers = [tokenizer, tokenizer_from_vocab]
