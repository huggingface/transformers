# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from tempfile import TemporaryDirectory

from transformers import CharacterBertTokenizer
from transformers.testing_utils import slow


class CharacterBertTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterBertTokenizer(max_characters_per_token=12)

    def test_single_text_encoding(self):
        encoding = self.tokenizer("CharacterBERT handles any token.")

        self.assertIn("input_ids", encoding)
        self.assertIn("attention_mask", encoding)
        self.assertIn("token_type_ids", encoding)

        token_sequence = encoding["input_ids"]
        self.assertGreater(len(token_sequence), 2)
        self.assertEqual(len(token_sequence[0]), 12)

    def test_pair_encoding_and_token_types(self):
        encoding = self.tokenizer("Hello world", "Second sequence")

        self.assertEqual(len(encoding["input_ids"]), len(encoding["token_type_ids"]))
        self.assertIn(1, encoding["token_type_ids"])

    def test_batch_padding(self):
        encoding = self.tokenizer(["short", "a bit longer input"], padding=True)

        self.assertEqual(len(encoding["input_ids"]), 2)
        self.assertEqual(len(encoding["input_ids"][0]), len(encoding["input_ids"][1]))

    def test_roundtrip_to_tokens(self):
        encoding = self.tokenizer("Token roundtrip check")
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        self.assertEqual(tokens[0], "[CLS]")
        self.assertEqual(tokens[-1], "[SEP]")

    def test_added_special_token_encoding(self):
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENTITY]"]})
        encoding = self.tokenizer("[ENTITY] appears", add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        self.assertEqual(tokens[0], "[ENTITY]")

    def test_mask_token_with_no_special_tokens(self):
        encoding = self.tokenizer("hello [MASK]", add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        self.assertEqual(tokens, ["hello", "[MASK]"])
        self.assertTrue(all(isinstance(token_ids, list) for token_ids in encoding["input_ids"]))

    def test_save_and_load_tokenizer(self):
        with TemporaryDirectory() as tmp_dir:
            self.tokenizer.save_pretrained(tmp_dir)
            reloaded_tokenizer = CharacterBertTokenizer.from_pretrained(tmp_dir)

            expected = self.tokenizer("Saving and loading should be stable.")["input_ids"]
            actual = reloaded_tokenizer("Saving and loading should be stable.")["input_ids"]
            self.assertEqual(actual, expected)


@slow
class CharacterBertTokenizerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_id = os.environ.get("CHARACTER_BERT_INTEGRATION_MODEL", "helboukkouri/character-bert-base-uncased")
        cls.tokenizer = CharacterBertTokenizer.from_pretrained(model_id)

    def _expected_char_ids(self, token: str) -> list[int]:
        max_characters_per_token = self.tokenizer.max_characters_per_token
        encoded = token.encode("utf-8", "ignore")[: max_characters_per_token - 2]
        expected = [261] * max_characters_per_token
        expected[0] = 259
        for index, byte in enumerate(encoded, start=1):
            expected[index] = byte + 1
        expected[len(encoded) + 1] = 260
        return expected

    def test_pretrained_tokenizer_uses_utf8_character_ids(self):
        encoding = self.tokenizer("Hello 你 [MASK]", add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        self.assertEqual(tokens, ["hello", "你", "[MASK]"])

        self.assertEqual(encoding["input_ids"][0], self._expected_char_ids("hello"))
        self.assertEqual(encoding["input_ids"][1], self._expected_char_ids("你"))
        self.assertEqual(encoding["input_ids"][2][:3], [259, 262, 260])
        self.assertEqual(encoding["input_ids"][2][3:], [261] * (self.tokenizer.max_characters_per_token - 3))

    def test_pretrained_tokenizer_encodes_mask_token_as_character_ids(self):
        encoding = self.tokenizer("Hello [MASK]")
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        mask_index = tokens.index(self.tokenizer.mask_token)
        mask_char_ids = encoding["input_ids"][mask_index]

        self.assertEqual(
            tokens,
            [self.tokenizer.cls_token, "hello", self.tokenizer.mask_token, self.tokenizer.sep_token],
        )
        self.assertEqual(mask_char_ids[:3], [259, 262, 260])
        self.assertEqual(mask_char_ids[3:], [261] * (self.tokenizer.max_characters_per_token - 3))

    def test_pretrained_tokenizer_truncates_long_tokens_to_max_length(self):
        max_characters_per_token = self.tokenizer.max_characters_per_token
        token = "a" * (max_characters_per_token * 2)
        encoding = self.tokenizer(token, add_special_tokens=False)
        char_ids = encoding["input_ids"][0]

        self.assertEqual(len(char_ids), max_characters_per_token)
        self.assertEqual(char_ids, self._expected_char_ids("a" * (max_characters_per_token - 2)))
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(char_ids), "a" * (max_characters_per_token - 2))
