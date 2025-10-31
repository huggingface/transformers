# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import XLMRobertaTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

# import cached_property
from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class XLMRobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "FacebookAI/xlm-roberta-base"
    tokenizer_class = XLMRobertaTokenizer
    rust_tokenizer_class = XLMRobertaTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = [
        "▁This",
        "▁is",
        "▁a",
        "▁test",
        "▁",
        "😊",
        "▁I",
        "▁was",
        "▁born",
        "▁in",
        "▁9",
        "2000",
        ",",
        "▁and",
        "▁this",
        "▁is",
        "▁fals",
        "é",
        ".",
        "▁",
        "生活的",
        "真",
        "谛",
        "是",
        "▁Hi",
        "▁Hello",
        "▁Hi",
        "▁Hello",
        "▁Hello",
        "<s>",
        "▁hi",
        "<s>",
        "▁there",
        "▁The",
        "▁following",
        "▁string",
        "▁should",
        "▁be",
        "▁properly",
        "▁en",
        "code",
        "d",
        ":",
        "▁Hello",
        ".",
        "▁But",
        "▁ir",
        "d",
        "▁and",
        "▁ปี",
        "▁ir",
        "d",
        "▁ด",
        "▁Hey",
        "▁how",
        "▁are",
        "▁you",
        "▁doing",
    ]
    integration_expected_token_ids = [
        3293,
        83,
        10,
        3034,
        6,
        82803,
        87,
        509,
        103122,
        23,
        483,
        13821,
        4,
        136,
        903,
        83,
        84047,
        446,
        5,
        6,
        62668,
        5364,
        245875,
        354,
        2673,
        35378,
        2673,
        35378,
        35378,
        0,
        1274,
        0,
        2685,
        581,
        25632,
        79315,
        5608,
        186,
        155965,
        22,
        40899,
        71,
        12,
        35378,
        5,
        4966,
        193,
        71,
        136,
        10249,
        193,
        71,
        48229,
        28240,
        3642,
        621,
        398,
        20594,
    ]
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "FacebookAI/xlm-roberta-base"

        tokenizer = XLMRobertaTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.save_pretrained(cls.tmpdirname)

    # def test_full_tokenizer(self):
    #     extractor = SentencePieceExtractor(SAMPLE_VOCAB)
    #     vocab, merges = extractor.extract()
    #     tokenizer = XLMRobertaTokenizer(vocab=vocab, merges=merges, keep_accents=True)

    #     tokens = tokenizer.tokenize("This is a test")
    #     self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

    #     self.assertListEqual(
    #         tokenizer.convert_tokens_to_ids(tokens),
    #         [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
    #     )

    #     tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
    #     self.assertListEqual(
    #         tokens,
    #         [
    #             SPIECE_UNDERLINE + "I",
    #             SPIECE_UNDERLINE + "was",
    #             SPIECE_UNDERLINE + "b",
    #             "or",
    #             "n",
    #             SPIECE_UNDERLINE + "in",
    #             SPIECE_UNDERLINE + "",
    #             "9",
    #             "2",
    #             "0",
    #             "0",
    #             "0",
    #             ",",
    #             SPIECE_UNDERLINE + "and",
    #             SPIECE_UNDERLINE + "this",
    #             SPIECE_UNDERLINE + "is",
    #             SPIECE_UNDERLINE + "f",
    #             "al",
    #             "s",
    #             "é",
    #             ".",
    #         ],
    #     )
    #     ids = tokenizer.convert_tokens_to_ids(tokens)
    #     self.assertListEqual(
    #         ids,
    #         [
    #             value + tokenizer.fairseq_offset
    #             for value in [8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 2, 4]
    #             #                                       ^ unk: 2 + 1 = 3                  unk: 2 + 1 = 3 ^
    #         ],
    #     )

    #     back_tokens = tokenizer.convert_ids_to_tokens(ids)
    #     self.assertListEqual(
    #         back_tokens,
    #         [
    #             SPIECE_UNDERLINE + "I",
    #             SPIECE_UNDERLINE + "was",
    #             SPIECE_UNDERLINE + "b",
    #             "or",
    #             "n",
    #             SPIECE_UNDERLINE + "in",
    #             SPIECE_UNDERLINE + "",
    #             "<unk>",
    #             "2",
    #             "0",
    #             "0",
    #             "0",
    #             ",",
    #             SPIECE_UNDERLINE + "and",
    #             SPIECE_UNDERLINE + "this",
    #             SPIECE_UNDERLINE + "is",
    #             SPIECE_UNDERLINE + "f",
    #             "al",
    #             "s",
    #             "<unk>",
    #             ".",
    #         ],
    #     )

    def test_tokenization_base_easy_symbols(self):
        symbols = "Hello World!"
        original_tokenizer_encodings = [0, 35378, 6661, 38, 2]
        # xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base')  # xlmr.large has same tokenizer
        # xlmr.eval()
        # xlmr.encode(symbols)

        tokenizer = self.get_tokenizer()

        self.assertListEqual(original_tokenizer_encodings, tokenizer.encode(symbols))

    def test_tokenization_base_hard_symbols(self):
        symbols = (
            'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : - . Also we will'
            " add words that should not exist and be tokenized to <unk>, such as saoneuhaoesuth"
        )
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

        tokenizer = self.get_tokenizer()
        self.assertListEqual(original_tokenizer_encodings, tokenizer.encode(symbols))

    # def test_tokenizer_integration(self):
    #     expected_encoding = {'input_ids': [[0, 11062, 82772, 7, 15, 82772, 538, 51529, 237, 17198, 1290, 206, 9, 215175, 1314, 136, 17198, 1290, 206, 9, 56359, 42, 122009, 9, 16466, 16, 87344, 4537, 9, 4717, 78381, 6, 159958, 7, 15, 24480, 618, 4, 527, 22693, 5428, 4, 2777, 24480, 9874, 4, 43523, 594, 4, 803, 18392, 33189, 18, 4, 43523, 24447, 12399, 100, 24955, 83658, 9626, 144057, 15, 839, 22335, 16, 136, 24955, 83658, 83479, 15, 39102, 724, 16, 678, 645, 2789, 1328, 4589, 42, 122009, 115774, 23, 805, 1328, 46876, 7, 136, 53894, 1940, 42227, 41159, 17721, 823, 425, 4, 27512, 98722, 206, 136, 5531, 4970, 919, 17336, 5, 2], [0, 20080, 618, 83, 82775, 47, 479, 9, 1517, 73, 53894, 333, 80581, 110117, 18811, 5256, 1295, 51, 152526, 297, 7986, 390, 124416, 538, 35431, 214, 98, 15044, 25737, 136, 7108, 43701, 23, 756, 135355, 7, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 581, 63773, 119455, 6, 147797, 88203, 7, 645, 70, 21, 3285, 10269, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # fmt: skip

    #     self.tokenizer_integration_test_util(
    #         expected_encoding=expected_encoding,
    #         model_name="FacebookAI/xlm-roberta-base",
    #         revision="d9d8a8ea5eb94b1c6654ae9249df7793cd2933d3",
    #     )
