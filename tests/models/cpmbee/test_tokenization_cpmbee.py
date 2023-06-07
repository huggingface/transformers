# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch CpmBee tokenizer. """

import os
import unittest

from transformers.models.cpmbee.tokenization_cpmbee import VOCAB_FILES_NAMES, CpmBeeTokenizer
from transformers.tokenization_utils import AddedToken

from ...test_tokenization_common import TokenizerTesterMixin


class CPMBeeTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = CpmBeeTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "<d>",
            "</d>",
            "<s>",
            "</s>",
            "</_>",
            "<unk>",
            "<pad>",
            "<mask>",
            "</n>",
            "我",
            "是",
            "C",
            "P",
            "M",
            "B",
            "e",
            "e",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        vocab_tokens = list(set(vocab_tokens))
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
    
    # override test_add_tokens_tokenizer because <...> is special token in CpmBeeTokenizer.
    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||;;;||;"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||;;;||; l", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_added_tokens_do_lower_case(self):
        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if not hasattr(tokenizer, "do_lower_case") or not tokenizer.do_lower_case:
                    continue

                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                toks_before_adding = tokenizer.tokenize(text)  # toks before adding new_toks

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB", "CCCCCCCCCDDDDDDDD"]
                added = tokenizer.add_tokens([AddedToken(tok, lstrip=True, rstrip=True) for tok in new_toks])

                toks_after_adding = tokenizer.tokenize(text)
                toks_after_adding2 = tokenizer.tokenize(text2)

                # Rust tokenizers dont't lowercase added tokens at the time calling `tokenizer.add_tokens`,
                # while python tokenizers do, so new_toks 0 and 2 would be treated as the same, so do new_toks 1 and 3.
                self.assertIn(added, [2, 4])

                self.assertListEqual(toks_after_adding, toks_after_adding2)
                self.assertTrue(
                    len(toks_before_adding) > len(toks_after_adding),  # toks_before_adding should be longer
                )

                # Check that none of the special tokens are lowercased
                sequence_with_special_tokens = "A " + " yEs ".join(tokenizer.all_special_tokens) + " B"
                # Convert the tokenized list to str as some special tokens are tokenized like normal tokens
                # which have a prefix spacee e.g. the mask token of Albert, and cannot match the original
                # special tokens exactly.
                tokenized_sequence = "".join(tokenizer.tokenize(sequence_with_special_tokens))

                for special_token in tokenizer.all_special_tokens:
                    self.assertTrue(special_token in tokenized_sequence)

        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if hasattr(tokenizer, "do_lower_case") and tokenizer.do_lower_case:
                    continue

                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                toks_before_adding = tokenizer.tokenize(text)  # toks before adding new_toks

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB", "CCCCCCCCCDDDDDDDD"]
                added = tokenizer.add_tokens([AddedToken(tok, lstrip=True, rstrip=True) for tok in new_toks])
                self.assertIn(added, [2, 4])

                toks_after_adding = tokenizer.tokenize(text)
                toks_after_adding2 = tokenizer.tokenize(text2)

                self.assertEqual(len(toks_after_adding), len(toks_after_adding2))  # Length should still be the same
                self.assertNotEqual(
                    toks_after_adding[1], toks_after_adding2[1]
                )  # But at least the first non-special tokens should differ
                self.assertTrue(
                    len(toks_before_adding) > len(toks_after_adding),  # toks_before_adding should be longer
                )
    
    def test_pre_tokenization(self):
        tokenizer = CpmBeeTokenizer.from_pretrained("openbmb/cpm-bee-10b")
        texts = {"input": "你好，", "<ans>": ""}
        tokens = tokenizer(texts)
        tokens = tokens["input_ids"][0]

        input_tokens = [6, 8, 7, 6, 65678, 7, 6, 10273, 246, 7, 6, 9, 7]
        self.assertListEqual(tokens, input_tokens)

        normalized_text = "<s><root></s><s>input</s><s>你好，</s><s><ans></s>"
        reconstructed_text = tokenizer.decode(tokens)
        self.assertEqual(reconstructed_text, normalized_text)
