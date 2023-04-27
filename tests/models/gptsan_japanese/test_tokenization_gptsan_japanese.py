# coding=utf-8
# Copyright 2023 Toshiyuki Sakamoto(tanreinama) and HuggingFace Inc. team.
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
import os
import unittest

from transformers.models.gptsan_japanese.tokenization_gptsan_japanese import (
    VOCAB_FILES_NAMES,
    GPTSanJapaneseTokenizer,
)
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class GPTSanJapaneseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = GPTSanJapaneseTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"do_clean_text": False, "add_prefix_space": False}

    def setUp(self):
        super().setUp()

        # fmt: off
        vocab_tokens = ["こん", "こんに", "にちは", "ばんは", "世界,㔺界", "、", "。", "<BR>", "<SP>", "<TAB>", "<URL>", "<EMAIL>", "<TEL>", "<DATE>", "<PRICE>", "<BLOCK>", "<KIGOU>", "<U2000U2BFF>", "<|emoji1|>", "<unk>", "<|bagoftoken|>", "<|endoftext|>"]
        # fmt: on
        emoji_tokens = {"emoji": {"\ud83d\ude00": "<|emoji1|>"}, "emoji_inv": {"<|emoji1|>": "\ud83d\ude00"}}  # 😀
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.emoji_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["emoji_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        with open(self.emoji_file, "w") as emoji_writer:
            emoji_writer.write(json.dumps(emoji_tokens))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return GPTSanJapaneseTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.get_input_output_texts
    def get_input_output_texts(self, tokenizer):
        input_text = "こんにちは、世界。 \nこんばんは、㔺界。😀"
        output_text = "こんにちは、世界。 \nこんばんは、世界。😀"
        return input_text, output_text

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.get_clean_sequence
    def get_clean_sequence(self, tokenizer):
        input_text, output_text = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        return text, ids

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.test_pretokenized_inputs
    def test_pretokenized_inputs(self):
        pass  # TODO add if relevant

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.test_maximum_encoding_length_pair_input
    def test_maximum_encoding_length_pair_input(self):
        pass  # TODO add if relevant

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.test_maximum_encoding_length_single_input
    def test_maximum_encoding_length_single_input(self):
        pass  # TODO add if relevant

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.test_full_tokenizer
    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        # Testing tokenization
        input_text = "こんにちは、世界。　こんばんは、㔺界。"
        expected_token = ["こん", "にちは", "、", "世界", "。", "<SP>", "こん", "ばんは", "、", "㔺界", "。"]
        tokens = tokenizer.tokenize(input_text)
        self.assertListEqual(tokens, expected_token)

        # Testing conversion to ids without special tokens
        expected_ids = [0, 2, 5, 4, 6, 8, 0, 3, 5, 4, 6]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(input_ids, expected_ids)

        # Testing conversion to ids with special tokens
        input_tokens = tokens + [tokenizer.unk_token]
        expected_ids = [0, 2, 5, 4, 6, 8, 0, 3, 5, 4, 6, 19]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        self.assertListEqual(input_ids, expected_ids)

    def test_token_bagging(self):
        tokenizer = self.get_tokenizer()

        # Testing tokenization
        input_text = "こんにちは、<|bagoftoken|>世界。こんばんは、<|bagoftoken|>㔺界。"
        expected_text = "こんにちは、、、、世界。こんばんは、、、、世界。"
        tokens = tokenizer.encode(input_text)
        output_text = tokenizer.decode(tokens)
        self.assertEqual(output_text, expected_text)

    @slow
    def test_prefix_input(self):
        tokenizer = self.tokenizer_class.from_pretrained("Tanrei/GPTSAN-japanese")

        # Testing tokenization
        prefix_text = "こんにちは、世界。"
        input_text = "こんばんは、㔺界。😀"
        expected_text = "こんにちは、世界。こんばんは、世界。😀"
        tokens_1 = tokenizer.encode(prefix_text + input_text)
        tokens_2 = tokenizer.encode("", prefix_text=prefix_text + input_text)
        tokens_3 = tokenizer.encode(input_text, prefix_text=prefix_text)
        output_text_1 = tokenizer.decode(tokens_1)
        output_text_2 = tokenizer.decode(tokens_2)
        output_text_3 = tokenizer.decode(tokens_3)
        self.assertEqual(output_text_1, expected_text)
        self.assertEqual(output_text_2, expected_text)
        self.assertEqual(output_text_3, expected_text)

    @slow
    def test_token_type_ids(self):
        tokenizer = self.tokenizer_class.from_pretrained("Tanrei/GPTSAN-japanese")

        # Testing tokenization
        prefix_text = "こんにちは、世界。"
        input_text = "こんばんは、㔺界。😀"

        len_prefix = len(tokenizer.encode(prefix_text)) - 2
        len_text = len(tokenizer.encode(input_text)) - 2

        expected_mask_1 = [1] + [0] * (len_prefix + len_text + 1)
        expected_mask_2 = [1] * (len_prefix + len_text + 1) + [0]
        expected_mask_3 = [1] + [1] * (len_prefix) + [0] * (len_text + 1)

        type_id_1 = tokenizer(prefix_text + input_text).token_type_ids
        type_id_2 = tokenizer("", prefix_text=prefix_text + input_text).token_type_ids
        type_id_3 = tokenizer(input_text, prefix_text=prefix_text).token_type_ids
        self.assertListEqual(type_id_1, expected_mask_1)
        self.assertListEqual(type_id_2, expected_mask_2)
        self.assertListEqual(type_id_3, expected_mask_3)

    @slow
    def test_prefix_tokens(self):
        tokenizer = self.tokenizer_class.from_pretrained("Tanrei/GPTSAN-japanese")

        x_token_1 = tokenizer.encode("あンいワ")
        x_token_2 = tokenizer.encode("", prefix_text="あンいワ")
        x_token_3 = tokenizer.encode("いワ", prefix_text="あン")

        self.assertEqual(tokenizer.decode(x_token_1), tokenizer.decode(x_token_2))
        self.assertEqual(tokenizer.decode(x_token_1), tokenizer.decode(x_token_3))
        self.assertNotEqual(x_token_1, x_token_2)
        self.assertNotEqual(x_token_1, x_token_3)
        self.assertEqual(x_token_1[1], x_token_2[-1])  # SEG token
        self.assertEqual(x_token_1[1], x_token_3[3])  # SEG token

    @slow
    def test_batch_encode(self):
        tokenizer = self.tokenizer_class.from_pretrained("Tanrei/GPTSAN-japanese")

        input_pairs = [["武田信玄", "は、"], ["織田信長", "の配下の、"]]
        x_token = tokenizer(input_pairs, padding=True)
        x_token_2 = tokenizer.batch_encode_plus(input_pairs, padding=True)

        # fmt: off
        expected_outputs = [[35993, 8640, 25948, 35998, 30647, 35675, 35999, 35999], [35993, 10382, 9868, 35998, 30646, 9459, 30646, 35675]]
        expected_typeids = [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]
        expected_attmask = [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
        # fmt: on
        self.assertListEqual(x_token.input_ids, expected_outputs)
        self.assertListEqual(x_token.token_type_ids, expected_typeids)
        self.assertListEqual(x_token.attention_mask, expected_attmask)
        self.assertListEqual(x_token_2.input_ids, expected_outputs)
        self.assertListEqual(x_token_2.token_type_ids, expected_typeids)
        self.assertListEqual(x_token_2.attention_mask, expected_attmask)

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.test_conversion_reversible
    def test_conversion_reversible(self):
        # Intentionally convert some words to accommodate character fluctuations unique to Japanese
        pass

    # Copied from tests.models.gpt_neox_japanese.test_tokenization_gpt_neox_japanese.GPTNeoXJapaneseTokenizationTest.test_padding_different_model_input_name
    def test_padding_different_model_input_name(self):
        # tokenizer has no padding token
        pass
