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


import unittest

from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer

from .utils import slow


class XLMRobertaTokenizationIntegrationTest(unittest.TestCase):
    @slow
    def test_tokenization_base_easy_symbols(self):
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        symbols = "Hello World!"
        original_tokenizer_encodings = [0, 35378, 6661, 38, 2]
        # xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base')  # xlmr.large has same tokenizer
        # xlmr.eval()
        # xlmr.encode(symbols)

        self.assertListEqual(original_tokenizer_encodings, tokenizer.encode(symbols))

    @slow
    def test_tokenization_base_hard_symbols(self):
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

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
            4426,
            3678,
            2740,
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

        self.assertListEqual(original_tokenizer_encodings, tokenizer.encode(symbols))
