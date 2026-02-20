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

import unittest
from tempfile import TemporaryDirectory

from transformers import CharacterBertTokenizer


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

    def test_save_and_load_tokenizer(self):
        with TemporaryDirectory() as tmp_dir:
            self.tokenizer.save_pretrained(tmp_dir)
            reloaded_tokenizer = CharacterBertTokenizer.from_pretrained(tmp_dir)

            expected = self.tokenizer("Saving and loading should be stable.")["input_ids"]
            actual = reloaded_tokenizer("Saving and loading should be stable.")["input_ids"]
            self.assertEqual(actual, expected)
