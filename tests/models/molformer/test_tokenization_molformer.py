# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import inspect
import json
import os
import unittest

from transformers import MolformerTokenizer, MolformerTokenizerFast, SpecialTokensMixin
from transformers.models.molformer.tokenization_molformer import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers
from transformers.tokenization_utils import AddedToken

from ...test_tokenization_common import TokenizerTesterMixin


SMALL_TRAINING_CORPUS = [
    ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"],
    ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CCO"],
]


@require_tokenizers
class MolformerTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MolformerTokenizer
    rust_tokenizer_class = MolformerTokenizerFast
    space_between_special_tokens = False
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        vocab = [
            "(",
            ")",
            "=",
            "1",
            "2",
            "B",
            "Br",
            "C",
            "Cl",
            "F",
            "I",
            "N",
            "O",
            "P",
            "S",
            "b",
            "c",
            "n",
            "o",
            "p",
            "s",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    def get_input_output_texts(self, tokenizer):
        input_text = "CC(=O)Oc1ccccc1C(=O)O"
        output_text = "CC(=O)Oc1ccccc1C(=O)O"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        input_text, output_text = self.get_input_output_texts(None)
        tokens = tokenizer.tokenize(input_text)

        self.assertListEqual(tokens, list(output_text))

        input_tokens = tokens + [tokenizer.unk_token]
        exp_tokens = [7, 7, 0, 2, 12, 1, 12, 16, 3, 16, 16, 16, 16, 16, 3, 7, 0, 2, 12, 1, 12, 21]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), exp_tokens)

    # "l" not in vocab
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

                tokens = tokenizer.encode("aaaaa bbbbbb cow cccccccccdddddddd c", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb cow cccccccccdddddddd <<<<<|||>|>>>>|> c", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    # SMILES regex assumes anything inside [] is a single token so use <> instead
    def test_split_special_tokens(self):
        if not self.test_slow_tokenizer:
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            special_token = "<SPECIAL_TOKEN>"
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                if not tokenizer.is_fast:
                    # bloom, gptneox etc only have a fast
                    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
                    encoded_special_token = tokenizer.encode(special_token, add_special_tokens=False)
                    self.assertEqual(len(encoded_special_token), 1)

                    encoded_split_special_token = tokenizer.encode(
                        special_token, add_special_tokens=False, split_special_tokens=True
                    )
                    if len(encoded_split_special_token) == 1:
                        # if we have subword tokenization or special vocab
                        self.assertTrue(
                            encoded_split_special_token[0] != tokenizer.convert_tokens_to_ids(special_token)
                        )
                    else:
                        self.assertTrue(len(encoded_split_special_token) > 1)

    # MolformerTokenizerFast uses WordLevel tokenization with custom regex
    def test_alignement_methods(self):
        pass

    # can't train new tokenizer on English corpus
    def test_training_new_tokenizer(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_rust_tokenizer()
        new_tokenizer = tokenizer.train_new_from_iterator(SMALL_TRAINING_CORPUS, 100)

        # Test we can use the new tokenizer with something not seen during training
        inputs = new_tokenizer(["CC(=O)Oc1ccccc1C(=O)O", "O.O=C1O[Bi]Oc2ccccc21"])
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "CC(=O)Oc1ccccc1C(=O)O"

        if tokenizer.backend_tokenizer.normalizer is not None:
            expected_result = tokenizer.backend_tokenizer.normalizer.normalize_str(expected_result)
        self.assertEqual(expected_result, decoded_input)

        # We check that the parameters of the tokenizer remained the same
        # Check we have the same number of added_tokens for both pair and non-pair inputs.
        self.assertEqual(tokenizer.num_special_tokens_to_add(False), new_tokenizer.num_special_tokens_to_add(False))
        self.assertEqual(tokenizer.num_special_tokens_to_add(True), new_tokenizer.num_special_tokens_to_add(True))

        # Check we have the correct max_length for both pair and non-pair inputs.
        self.assertEqual(tokenizer.max_len_single_sentence, new_tokenizer.max_len_single_sentence)
        self.assertEqual(tokenizer.max_len_sentences_pair, new_tokenizer.max_len_sentences_pair)

        # Assert the set of special tokens match as we didn't ask to change them
        self.assertSequenceEqual(
            tokenizer.all_special_tokens_extended,
            new_tokenizer.all_special_tokens_extended,
        )

        self.assertDictEqual(tokenizer.special_tokens_map, new_tokenizer.special_tokens_map)

    # can't train new tokenizer on English corpus
    def test_training_new_tokenizer_with_special_tokens_change(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_rust_tokenizer()
        # Test with a special tokens map
        class_signature = inspect.signature(tokenizer.__class__)
        if "cls_token" in class_signature.parameters:
            new_tokenizer = tokenizer.train_new_from_iterator(
                SMALL_TRAINING_CORPUS, 100, special_tokens_map={tokenizer.cls_token: "<cls>"}
            )
            cls_id = new_tokenizer.get_vocab()["<cls>"]
            self.assertEqual(new_tokenizer.cls_token, "<cls>")
            self.assertEqual(new_tokenizer.cls_token_id, cls_id)

        # Create a new mapping from the special tokens defined in the original tokenizer
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove("additional_special_tokens")
        special_tokens_map = {}
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, f"_{token}") is not None:
                special_token = getattr(tokenizer, token)
                special_tokens_map[special_token] = f"{special_token}a"

        # Train new tokenizer
        new_tokenizer = tokenizer.train_new_from_iterator(
            SMALL_TRAINING_CORPUS, 100, special_tokens_map=special_tokens_map
        )

        # Check the changes
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, f"_{token}") is None:
                continue
            special_token = getattr(tokenizer, token)
            if special_token in special_tokens_map:
                new_special_token = getattr(new_tokenizer, token)
                self.assertEqual(special_tokens_map[special_token], new_special_token)

                new_id = new_tokenizer.get_vocab()[new_special_token]
                self.assertEqual(getattr(new_tokenizer, f"{token}_id"), new_id)

        # Check if the AddedToken / string format has been kept
        for special_token in tokenizer.all_special_tokens_extended:
            if isinstance(special_token, AddedToken) and special_token.content not in special_tokens_map:
                # The special token must appear identically in the list of the new tokenizer.
                self.assertTrue(
                    special_token in new_tokenizer.all_special_tokens_extended,
                    f"'{special_token}' should be in {new_tokenizer.all_special_tokens_extended}",
                )
            elif isinstance(special_token, AddedToken):
                # The special token must appear in the list of the new tokenizer as an object of type AddedToken with
                # the same parameters as the old AddedToken except the content that the user has requested to change.
                special_token_str = special_token.content
                new_special_token_str = special_tokens_map[special_token_str]

                find = False
                for candidate in new_tokenizer.all_special_tokens_extended:
                    if (
                        isinstance(candidate, AddedToken)
                        and candidate.content == new_special_token_str
                        and candidate.lstrip == special_token.lstrip
                        and candidate.rstrip == special_token.rstrip
                        and candidate.normalized == special_token.normalized
                        and candidate.single_word == special_token.single_word
                    ):
                        find = True
                        break
                self.assertTrue(
                    find,
                    f"'{new_special_token_str}' doesn't appear in the list "
                    f"'{new_tokenizer.all_special_tokens_extended}' as an AddedToken with the same parameters as "
                    f"'{special_token}' in the list {tokenizer.all_special_tokens_extended}",
                )
            elif special_token not in special_tokens_map:
                # The special token must appear identically in the list of the new tokenizer.
                self.assertTrue(
                    special_token in new_tokenizer.all_special_tokens_extended,
                    f"'{special_token}' should be in {new_tokenizer.all_special_tokens_extended}",
                )

            else:
                # The special token must appear in the list of the new tokenizer as an object of type string.
                self.assertTrue(special_tokens_map[special_token] in new_tokenizer.all_special_tokens_extended)

        # Test we can use the new tokenizer with something not seen during training
        inputs = new_tokenizer(["CC(=O)Oc1ccccc1C(=O)O", "O.O=C1O[Bi]Oc2ccccc21"])
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "CC(=O)Oc1ccccc1C(=O)O"

        if tokenizer.backend_tokenizer.normalizer is not None:
            expected_result = tokenizer.backend_tokenizer.normalizer.normalize_str(expected_result)
        self.assertEqual(expected_result, decoded_input)
