# Copyright 2019 Hugging Face inc.
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

from transformers import DebertaTokenizer

from ...test_tokenization_common import TokenizerTesterMixin


class DebertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["microsoft/deberta-base"]
    tokenizer_class = DebertaTokenizer
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'Ġ', 'Ġ', 'ĠHello', 'ĊĊ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [713, 16, 10, 1296, 17841, 27969, 50118, 100, 21, 2421, 11, 8403, 151, 6, 8, 42, 16, 22461, 1140, 4, 50118, 48998, 37127, 20024, 2023, 44574, 49122, 4333, 36484, 7487, 3726, 48569, 50118, 30086, 1437, 20920, 50118, 30086, 1437, 1437, 20920, 50140, 1437, 50118, 1437, 1437, 50118, 20920, 50118, 41552, 29, 15698, 50118, 3592, 41552, 29, 15698, 8585, 50118, 133, 511, 6755, 197, 28, 5083, 45320, 35, 20920, 4, 50118, 1708, 1437, 8602, 8, 1437, 24107, 3726, 24107, 8906, 1437, 1437, 1437, 8602, 1437, 1437, 1437, 24107, 10674, 50118, 13368, 141, 32, 47, 608]  # fmt: skip
    expected_tokens_from_ids = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'Ġ', 'Ġ', 'ĠHello', 'ĊĊ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    # @classmethod
    # def setUpClass(cls):
    #     super().setUpClass()

    #     # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
    #     vocab = [
    #         "l",
    #         "o",
    #         "w",
    #         "e",
    #         "r",
    #         "s",
    #         "t",
    #         "i",
    #         "d",
    #         "n",
    #         "\u0120",
    #         "\u0120l",
    #         "\u0120n",
    #         "\u0120lo",
    #         "\u0120low",
    #         "er",
    #         "\u0120lowest",
    #         "\u0120newer",
    #         "\u0120wider",
    #         "[UNK]",
    #     ]
    #     vocab_tokens = dict(zip(vocab, range(len(vocab))))
    #     # merges as list of tuples, matching what load_merges returns
    #     merges = [("\u0120", "l"), ("\u0120l", "o"), ("\u0120lo", "w"), ("e", "r")]
    #     cls.special_tokens_map = {"unk_token": "[UNK]"}

    #     cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
    #     cls.merges_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
    #     with open(cls.vocab_file, "w", encoding="utf-8") as fp:
    #         fp.write(json.dumps(vocab_tokens) + "\n")
    #     with open(cls.merges_file, "w", encoding="utf-8") as fp:
    #         # Write merges file in the standard format
    #         fp.write("#version: 0.2\n")
    #         fp.write("\n".join([f"{a} {b}" for a, b in merges]))

    #     tokenizer = DebertaTokenizer(vocab=vocab_tokens, merges=merges)
    #     tokenizer.save_pretrained(cls.tmpdirname)

    #     cls.tokenizers = [tokenizer]

    # @classmethod
    # def get_tokenizer(cls, pretrained_name=None, **kwargs):
    #     kwargs.update(cls.special_tokens_map)
    #     pretrained_name = pretrained_name or cls.tmpdirname
    #     return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    # def get_input_output_texts(self, tokenizer):
    #     input_text = "lower newer"
    #     output_text = "lower newer"
    #     return input_text, output_text

    # def test_full_tokenizer(self):
    #     tokenizer = self.get_tokenizer()
    #     text = "lower newer"
    #     bpe_tokens = ["l", "o", "w", "er", "\u0120", "n", "e", "w", "er"]
    #     tokens = tokenizer.tokenize(text)
    #     self.assertListEqual(tokens, bpe_tokens)

    #     input_tokens = tokens + [tokenizer.unk_token]
    #     input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
    #     self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    # def test_tokenizer_integration(self):
    #     tokenizer_classes = [self.tokenizer_class]
    #     if self.test_rust_tokenizer:
    #         tokenizer_classes.append(self.rust_tokenizer_class)

    #     for tokenizer_class in tokenizer_classes:
    #         tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    #         sequences = [
    #             "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
    #             "ALBERT incorporates two parameter reduction techniques",
    #             "The first one is a factorized embedding parameterization. By decomposing the large vocabulary"
    #             " embedding matrix into two small matrices, we separate the size of the hidden layers from the size of"
    #             " vocabulary embedding.",
    #         ]
    #         encoding = tokenizer(sequences, padding=True)
    #         decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in encoding["input_ids"]]

    #         # fmt: off
    #         expected_encoding = {
    #             'input_ids': [
    #                 [1, 2118, 11126, 565, 35, 83, 25191, 163, 18854, 13, 12156, 12, 16101, 25376, 13807, 9, 22205, 27893, 1635, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 2118, 11126, 565, 24536, 80, 43797, 4878, 7373, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 133, 78, 65, 16, 10, 3724, 1538, 33183, 11303, 43797, 1938, 4, 870, 24165, 29105, 5, 739, 32644, 33183, 11303, 36173, 88, 80, 650, 7821, 45940, 6, 52, 2559, 5, 1836, 9, 5, 7397, 13171, 31, 5, 1836, 9, 32644, 33183, 11303, 4, 2]
    #             ],
    #             'token_type_ids': [
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #             ],
    #             'attention_mask': [
    #                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #             ]
    #         }
    #         # fmt: on

    #         expected_decoded_sequence = [
    #             "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
    #             "ALBERT incorporates two parameter reduction techniques",
    #             "The first one is a factorized embedding parameterization. By decomposing the large vocabulary"
    #             " embedding matrix into two small matrices, we separate the size of the hidden layers from the size of"
    #             " vocabulary embedding.",
    #         ]

    #         #  self.assertDictEqual(encoding.data, expected_encoding)

    #         for expected, decoded in zip(expected_decoded_sequence, decoded_sequences):
    #             self.assertEqual(expected, decoded)
