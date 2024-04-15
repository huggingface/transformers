# coding=utf-8
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
import os
import unittest

from transformers.models.gpt_neox_japanese.tokenization_gpt_neox_japanese import (
    VOCAB_FILES_NAMES,
    GPTNeoXJapaneseTokenizer,
)
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class GPTNeoXJapaneseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "abeja/gpt-neox-japanese-2.7b"
    tokenizer_class = GPTNeoXJapaneseTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"do_clean_text": False, "add_prefix_space": False}

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "こん",
            "こんに",
            "にちは",
            "ばんは",
            "世界,㔺界",
            "、",
            "。",
            "<BR>",
            "<SP>",
            "<TAB>",
            "<URL>",
            "<EMAIL>",
            "<TEL>",
            "<DATE>",
            "<PRICE>",
            "<BLOCK>",
            "<KIGOU>",
            "<U2000U2BFF>",
            "<|emoji1|>",
            "<unk>",
            "<|startoftext|>",
            "<|endoftext|>",
        ]
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
        return GPTNeoXJapaneseTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "こんにちは、世界。 \nこんばんは、㔺界。😀"
        output_text = "こんにちは、世界。 \nこんばんは、世界。😀"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer):
        input_text, output_text = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        return text, ids

    def test_pretokenized_inputs(self):
        pass  # TODO add if relevant

    def test_maximum_encoding_length_pair_input(self):
        pass  # TODO add if relevant

    def test_maximum_encoding_length_single_input(self):
        pass  # TODO add if relevant

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

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("abeja/gpt-neox-japanese-2.7b")

        ids_1 = tokenizer.encode("ありがとう。", add_special_tokens=False)
        ids_2 = tokenizer.encode("どういたしまして。", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(ids_1)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(ids_1, ids_2)

        assert encoded_sentence == ids_1
        assert encoded_pair == ids_1 + ids_2

    def test_conversion_reversible(self):
        # Intentionally convert some words to accommodate character fluctuations unique to Japanese
        pass

    def test_padding_different_model_input_name(self):
        # tokenizer has no padding token
        pass
