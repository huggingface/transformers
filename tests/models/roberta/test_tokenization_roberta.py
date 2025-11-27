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


import json
import unittest

from transformers import AutoTokenizer, RobertaTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    with open(vocab_file, "r", encoding="utf-8") as reader:
        return json.load(reader)


def load_merges(merges_file):
    """Loads a merges file into a list."""
    merges = []
    with open(merges_file, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line and not line.startswith("#"):
                merges.append(tuple(line.split()))
    return merges


@require_tokenizers
class RobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "FacebookAI/roberta-base"
    tokenizer_class = RobertaTokenizer
    rust_tokenizer_class = RobertaTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"cls_token": "<s>"}

    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'Ġ', 'Ġ', 'ĠHello', 'ĊĊ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<s>', 'Ċ', 'hi', '<s>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [713, 16, 10, 1296, 17841, 27969, 50118, 100, 21, 2421, 11, 8403, 151, 6, 8, 42, 16, 22461, 1140, 4, 50118, 48998, 37127, 20024, 2023, 44574, 49122, 4333, 36484, 7487, 3726, 48569, 50118, 30086, 1437, 20920, 50118, 30086, 1437, 1437, 20920, 50140, 1437, 50118, 1437, 1437, 50118, 20920, 50118, 0, 50118, 3592, 0, 8585, 50118, 133, 511, 6755, 197, 28, 5083, 45320, 35, 20920, 4, 50118, 1708, 1437, 8602, 8, 1437, 24107, 3726, 24107, 8906, 1437, 1437, 1437, 8602, 1437, 1437, 1437, 24107, 10674, 50118, 13368, 141, 32, 47, 608]  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "FacebookAI/roberta-base"

        # Create tokenizer from AutoTokenizer
        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        # Create tokenizer from vocab and merges
        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        cls.vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges_raw = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        cls.merges = []
        for line in merges_raw:
            line = line.strip()
            if line and not line.startswith("#"):
                cls.merges.append(tuple(line.split()))

        tok_from_vocab = RobertaTokenizer(vocab=cls.vocab_tokens, merges=cls.merges, unk_token="<unk>")

        cls.tokenizers = [tok_auto, tok_from_vocab]
        cls.special_tokens_map = {"unk_token": "<unk>"}

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(vocab=self.vocab_tokens, merges=self.merges, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text)  # , add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    # def roberta_dict_integration_testing(self):
    #     tokenizer = self.get_tokenizer()

    #     self.assertListEqual(tokenizer.encode("Hello world!", add_special_tokens=False), [0, 31414, 232, 328, 2])
    #     self.assertListEqual(
    #         tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
    #         [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
    #      )
    # def test_space_encoding(self):
    #     tokenizer = self.get_tokenizer()

    #     sequence = "Encode this sequence."
    #     space_encoding = tokenizer.byte_encoder[b" "[0]]

    #     # Testing encoder arguments
    #     encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=False)
    #     first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
    #     self.assertNotEqual(first_char, space_encoding)

    #     encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)
    #     first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
    #     self.assertEqual(first_char, space_encoding)

    #     tokenizer.add_special_tokens({"bos_token": "<s>"})
    #     encoded = tokenizer.encode(sequence, add_special_tokens=True)
    #     first_char = tokenizer.convert_ids_to_tokens(encoded[1])[0]
    #     self.assertNotEqual(first_char, space_encoding)

    #     # Testing spaces after special tokens
    #     mask = "<mask>"
    #     tokenizer.add_special_tokens(
    #         {"mask_token": AddedToken(mask, lstrip=True, rstrip=False)}
    #     )  # mask token has a left space
    #     mask_ind = tokenizer.convert_tokens_to_ids(mask)

    #     sequence = "Encode <mask> sequence"
    #     sequence_nospace = "Encode <mask>sequence"

    #     encoded = tokenizer.encode(sequence)
    #     mask_loc = encoded.index(mask_ind)
    #     first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
    #     self.assertEqual(first_char, space_encoding)

    #     encoded = tokenizer.encode(sequence_nospace)
    #     mask_loc = encoded.index(mask_ind)
    #     first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
    #     self.assertNotEqual(first_char, space_encoding)

    # def test_change_add_prefix_space_and_trim_offsets_args(self):
    #     for trim_offsets, add_prefix_space in itertools.product([True, False], repeat=2):
    #         tokenizer_r = self.get_rust_tokenizer(
    #             self.tmpdirname, use_fast=True, add_prefix_space=add_prefix_space, trim_offsets=trim_offsets
    #         )

    #         pre_tokenizer_state = json.loads(tokenizer_r.backend_tokenizer.pre_tokenizer.__getstate__())
    #         post_processor_state = json.loads(tokenizer_r.backend_tokenizer.post_processor.__getstate__())

    #         self.assertEqual(pre_tokenizer_state["add_prefix_space"], add_prefix_space)

    #         self.assertEqual(post_processor_state["add_prefix_space"], add_prefix_space)
    #         self.assertEqual(post_processor_state["trim_offsets"], trim_offsets)

    # def test_offsets_mapping_with_different_add_prefix_space_and_trim_space_arguments(self):
    #     # Test which aims to verify that the offsets are well adapted to the argument `add_prefix_space` and
    #     # `trim_offsets`
    #     for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
    #         with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
    #             text_of_1_token = "hello"  # `hello` is a token in the vocabulary of `pretrained_name`
    #             text = f"{text_of_1_token} {text_of_1_token}"

    #             tokenizer_r = self.get_rust_tokenizer(
    #                 pretrained_name, use_fast=True, add_prefix_space=True, trim_offsets=True
    #             )
    #             encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
    #             self.assertEqual(
    #                 encoding.offset_mapping[1],
    #                 (len(text_of_1_token) + 1, len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             )

    #             tokenizer_r = self.get_rust_tokenizer(
    #                 pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=True
    #             )
    #             encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
    #             self.assertEqual(
    #                 encoding.offset_mapping[1],
    #                 (len(text_of_1_token) + 1, len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             )

    #             tokenizer_r = self.get_rust_tokenizer(
    #                 pretrained_name, use_fast=True, add_prefix_space=True, trim_offsets=False
    #             )
    #             encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
    #             self.assertEqual(
    #                 encoding.offset_mapping[1],
    #                 (len(text_of_1_token), len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             )

    #             tokenizer_r = self.get_rust_tokenizer(
    #                 pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=False
    #             )
    #             encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
    #             self.assertEqual(
    #                 encoding.offset_mapping[1],
    #                 (len(text_of_1_token), len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             )

    #             text = f" {text}"

    #             # tokenizer_r = self.rust_tokenizer_class.from_pretrained(
    #             #     pretrained_name, use_fast=True, add_prefix_space=True, trim_offsets=True
    #             # )
    #             # encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             # self.assertEqual(encoding.offset_mapping[0], (1, 1 + len(text_of_1_token)))
    #             # self.assertEqual(
    #             #     encoding.offset_mapping[1],
    #             #     (1 + len(text_of_1_token) + 1, 1 + len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             # )

    #             tokenizer_r = self.get_rust_tokenizer(
    #                 pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=True
    #             )
    #             encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             self.assertEqual(encoding.offset_mapping[0], (1, 1 + len(text_of_1_token)))
    #             self.assertEqual(
    #                 encoding.offset_mapping[1],
    #                 (1 + len(text_of_1_token) + 1, 1 + len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             )

    #             tokenizer_r = self.get_rust_tokenizer(
    #                 pretrained_name, use_fast=True, add_prefix_space=True, trim_offsets=False
    #             )
    #             encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             self.assertEqual(encoding.offset_mapping[0], (0, 1 + len(text_of_1_token)))
    #             self.assertEqual(
    #                 encoding.offset_mapping[1],
    #                 (1 + len(text_of_1_token), 1 + len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             )

    #             tokenizer_r = self.get_rust_tokenizer(
    #                 pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=False
    #             )
    #             encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
    #             self.assertEqual(encoding.offset_mapping[0], (0, 1 + len(text_of_1_token)))
    #             self.assertEqual(
    #                 encoding.offset_mapping[1],
    #                 (1 + len(text_of_1_token), 1 + len(text_of_1_token) + 1 + len(text_of_1_token)),
    #             )
