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


import os
import unittest

from transformers import SPIECE_UNDERLINE, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers.file_utils import cached_property
from transformers.testing_utils import require_sentencepiece, require_tokenizers, slow

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class XLMRobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = XLMRobertaTokenizer
    rust_tokenizer_class = XLMRobertaTokenizerFast
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = XLMRobertaTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = XLMRobertaTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 2, 4]
                #                                       ^ unk: 2 + 1 = 3                  unk: 2 + 1 = 3 ^
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    @cached_property
    def big_tokenizer(self):
        return XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "I was born in 92000, and this is falsé."

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

    @slow
    def test_tokenization_base_easy_symbols(self):
        symbols = "Hello World!"
        original_tokenizer_encodings = [0, 35378, 6661, 38, 2]
        # xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base')  # xlmr.large has same tokenizer
        # xlmr.eval()
        # xlmr.encode(symbols)

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @slow
    def test_tokenization_base_hard_symbols(self):
        symbols = 'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : - . Also we will add words that should not exsist and be tokenized to <unk>, such as saoneuhaoesuth'
        original_tokenizer_encodings = [
            0,
            3293,
            83,
            10,
            4552,
            4989,
            7986,
            678,
            10,
            5915,
            111,
            179459,
            124850,
            4,
            6044,
            237,
            12,
            6,
            5,
            6,
            4,
            6780,
            705,
            15,
            1388,
            44,
            378,
            10114,
            711,
            152,
            20,
            6,
            5,
            22376,
            642,
            1221,
            15190,
            34153,
            450,
            5608,
            959,
            1119,
            57702,
            136,
            186,
            47,
            1098,
            29367,
            47,
            # 4426, # What fairseq tokenizes from "<unk>": "_<"
            # 3678, # What fairseq tokenizes from "<unk>": "unk"
            # 2740, # What fairseq tokenizes from "<unk>": ">"
            3,  # What we tokenize from "<unk>": "<unk>"
            6,  # Residue from the tokenization: an extra sentencepiece underline
            4,
            6044,
            237,
            6284,
            50901,
            528,
            31,
            90,
            34,
            927,
            2,
        ]
        # xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base')  # xlmr.large has same tokenizer
        # xlmr.eval()
        # xlmr.encode(symbols)

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))
