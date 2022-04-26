# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest
from typing import List

import pandas as pd

from transformers import AddedToken, TapexTokenizer
from transformers.models.tapex.tokenization_tapex import VOCAB_FILES_NAMES
from transformers.testing_utils import is_pt_tf_cross_test, require_pandas, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_pandas
class TapexTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = TapexTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"cls_token": "<s>"}
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        # fmt: off
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "\u0120", "\u0120l", "\u0120n", "\u0120lo", "\u0120low", "er", "\u0120lowest", "\u0120newer", "\u0120wider", "<unk>"]  # noqa: E231
        # fmt: on
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_table(self, tokenizer, length=5):
        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]

        if length == 0:
            data = {}
        else:
            data = {toks[0]: [toks[tok] for tok in range(1, length)]}

        table = pd.DataFrame.from_dict(data)

        return table

    def get_table_and_query(self, tokenizer, length=5):
        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]
        table = self.get_table(tokenizer, length=length - 3)
        query = " ".join(toks[:3])

        return table, query

    def get_clean_sequence(
        self,
        tokenizer,
        with_prefix_space=False,
        max_length=20,
        min_length=5,
        empty_table: bool = False,
        add_special_tokens: bool = True,
        return_table_and_query: bool = False,
    ):

        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]

        if empty_table:
            table = pd.DataFrame.from_dict({})
            query = " ".join(toks[:min_length])
        else:
            data = {toks[0]: [toks[tok] for tok in range(1, min_length - 3)]}
            table = pd.DataFrame.from_dict(data)
            query = " ".join(toks[:3])

        output_ids = tokenizer.encode(table, query, add_special_tokens=add_special_tokens)
        output_txt = tokenizer.decode(output_ids)

        if len(output_ids) < min_length:
            raise ValueError("Update the code to generate the sequences so that they are larger")
        if len(output_ids) > max_length:
            raise ValueError("Update the code to generate the sequences so that they are smaller")

        if return_table_and_query:
            return output_txt, output_ids, table, query

        return output_txt, output_ids

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer_roberta(self):
        tokenizer = self.tokenizer_class(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def roberta_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(tokenizer.encode("Hello world!", add_special_tokens=False), [0, 31414, 232, 328, 2])
        self.assertListEqual(
            tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
        )

    def test_add_tokens_tokenizer(self):
        tokenizers: List[TapexTokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
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

                tokens = tokenizer.encode(table, "aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

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
                    table,
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                    add_special_tokens=False,
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_token_type_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                empty_table = self.get_table(tokenizer, length=0)
                seq_0 = "Test this method."

                # We want to have sequence 0 and sequence 1 are tagged
                # respectively with 0 and 1 token_ids
                # (regardless of whether the model use token type ids)
                # We use this assumption in the QA pipeline among other place
                output = tokenizer(empty_table, seq_0, return_token_type_ids=True)

                # Assert that the token type IDs have the same length as the input IDs
                self.assertEqual(len(output["token_type_ids"]), len(output["input_ids"]))
                self.assertIn(0, output["token_type_ids"])

    def test_add_special_tokens(self):
        tokenizers: List[TapexTokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_table = self.get_table(tokenizer, length=0)

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(input_table, special_token, add_special_tokens=False)
                self.assertEqual(len(encoded_special_token), 1)

                decoded = tokenizer.decode(encoded_special_token, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_batch_encode_plus_overflowing_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            table = self.get_table(tokenizer, length=10)
            string_sequences = ["Testing the prepare_for_model method.", "Test"]

            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            tokenizer.batch_encode_plus(
                table, string_sequences, return_overflowing_tokens=True, truncation=True, padding=True, max_length=3
            )

    @is_pt_tf_cross_test
    def test_batch_encode_plus_tensors(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                table = self.get_table(tokenizer, length=0)

                # A Tensor cannot be build by sequences which are not the same size
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, table, sequences, return_tensors="pt")
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, table, sequences, return_tensors="tf")

                if tokenizer.pad_token_id is None:
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        table,
                        sequences,
                        padding=True,
                        return_tensors="pt",
                    )
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        table,
                        sequences,
                        padding="longest",
                        return_tensors="tf",
                    )
                else:
                    pytorch_tensor = tokenizer.batch_encode_plus(table, sequences, padding=True, return_tensors="pt")
                    tensorflow_tensor = tokenizer.batch_encode_plus(
                        table, sequences, padding="longest", return_tensors="tf"
                    )
                    encoded_sequences = tokenizer.batch_encode_plus(table, sequences, padding=True)

                    for key in encoded_sequences.keys():
                        pytorch_value = pytorch_tensor[key].tolist()
                        tensorflow_value = tensorflow_tensor[key].numpy().tolist()
                        encoded_value = encoded_sequences[key]

                        self.assertEqual(pytorch_value, tensorflow_value, encoded_value)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                # Test not batched
                table = self.get_table(tokenizer, length=0)
                encoded_sequences_1 = tokenizer.encode_plus(table, sequences[0])
                encoded_sequences_2 = tokenizer(table, sequences[0])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                table = self.get_table(tokenizer, length=10)
                encoded_sequences_1 = tokenizer.encode_plus(table, sequences[1])
                encoded_sequences_2 = tokenizer(table, sequences[1])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                table = self.get_table(tokenizer, length=0)
                encoded_sequences_1 = tokenizer.batch_encode_plus(table, sequences)
                encoded_sequences_2 = tokenizer(table, sequences)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(table, input_text, add_special_tokens=False)
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                self.assertEqual(text_2, output_text)

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                table = self.get_table(tokenizer, length=0)
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(table, sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(table, sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

    def test_number_of_added_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                table, query = self.get_table_and_query(tokenizer)

                sequences = tokenizer.encode(table, query, add_special_tokens=False)
                attached_sequences = tokenizer.encode(table, query, add_special_tokens=True)

                self.assertEqual(2, len(attached_sequences) - len(sequences))

    @unittest.skip("TAPEX cannot handle `prepare_for_model` without passing by `encode_plus` or `batch_encode_plus`")
    def test_prepare_for_model(self):
        pass

    @unittest.skip("TAPEX tokenizer does not support pairs.")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip("TAPEX tokenizer does not support pairs.")
    def test_maximum_encoding_length_single_input(self):
        pass

    @unittest.skip("Not implemented")
    def test_right_and_left_truncation(self):
        pass

    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)

                new_toks = [AddedToken("[ABC]", normalized=False), AddedToken("[DEF]", normalized=False)]
                tokenizer.add_tokens(new_toks)
                input = "[ABC][DEF][ABC][DEF]"
                if self.space_between_special_tokens:
                    output = "[ABC] [DEF] [ABC] [DEF]"
                else:
                    output = input
                encoded = tokenizer.encode(table, input, add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    def test_tokenize_special_tokens(self):
        """Test `tokenize` with special tokens."""
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
                SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"

                # TODO:
                # Can we combine `unique_no_split_tokens` and `all_special_tokens`(and properties related to it)
                # with one variable(property) for a better maintainability?

                # `add_tokens` method stores special tokens only in `tokenizer.unique_no_split_tokens`. (in tokenization_utils.py)
                tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
                # `add_special_tokens` method stores special tokens in `tokenizer.additional_special_tokens`,
                # which also occur in `tokenizer.all_special_tokens`. (in tokenization_utils_base.py)
                tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN_2]})

                token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
                token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

                self.assertEqual(len(token_1), 1)
                self.assertEqual(len(token_2), 1)
                self.assertEqual(token_1[0], SPECIAL_TOKEN_1)
                self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequence_0 = "Encode this."
                # Testing single inputs
                encoded_sequence = tokenizer.encode(table, sequence_0, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    table, sequence_0, add_special_tokens=True, return_special_tokens_mask=True
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [x for i, x in enumerate(encoded_sequence_w_special) if not special_tokens_mask[i]]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_padding_to_max_length(self):
        """We keep this test for backward compatibility but it should be removed when `pad_to_max_length` will be deprecated"""
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer)
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    table,
                    sequence,
                    max_length=sequence_length + padding_size,
                    pad_to_max_length=True,
                )
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertListEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(table, sequence, pad_to_max_length=True)
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertListEqual(encoded_sequence, padded_sequence_right)

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                else:
                    empty_tokens = tokenizer(table, padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer(table, "This is a sample input", padding=True, pad_to_multiple_of=8)
                    for key, value in empty_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    normal_tokens = tokenizer(table, "This", pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # Should also work with truncation
                    normal_tokens = tokenizer(table, "This", padding=True, truncation=True, pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

    def test_right_and_left_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    table, sequence, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertListEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    table, sequence, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertListEqual([padding_idx] * padding_size + encoded_sequence, padded_sequence)

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
                encoded_sequence = tokenizer.encode(table, sequence)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(table, sequence, padding=True)
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertListEqual(encoded_sequence, padded_sequence_right)

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(table, sequence, padding="longest")
                padded_sequence_left_length = len(padded_sequence_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertListEqual(encoded_sequence, padded_sequence_left)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(table, sequence)
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertListEqual(encoded_sequence, padded_sequence_right)

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(table, sequence, padding=False)
                padded_sequence_left_length = len(padded_sequence_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertListEqual(encoded_sequence, padded_sequence_left)

    def test_encode_plus_with_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequence = "Sequence"

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_size = 10
                padding_idx = tokenizer.pad_token_id
                token_type_padding_idx = tokenizer.pad_token_type_id

                encoded_sequence = tokenizer.encode_plus(table, sequence, return_special_tokens_mask=True)
                input_ids = encoded_sequence["input_ids"]
                special_tokens_mask = encoded_sequence["special_tokens_mask"]
                sequence_length = len(input_ids)

                # Test 'longest' and 'no_padding' don't do anything
                tokenizer.padding_side = "right"

                not_padded_sequence = tokenizer.encode_plus(
                    table,
                    sequence,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertListEqual(input_ids, not_padded_input_ids)
                self.assertListEqual(special_tokens_mask, not_padded_special_tokens_mask)

                not_padded_sequence = tokenizer.encode_plus(
                    table,
                    sequence,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertListEqual(input_ids, not_padded_input_ids)
                self.assertListEqual(special_tokens_mask, not_padded_special_tokens_mask)

                # Test right padding
                tokenizer.padding_side = "right"

                right_padded_sequence = tokenizer.encode_plus(
                    table,
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                right_padded_input_ids = right_padded_sequence["input_ids"]

                right_padded_special_tokens_mask = right_padded_sequence["special_tokens_mask"]
                right_padded_sequence_length = len(right_padded_input_ids)

                self.assertEqual(sequence_length + padding_size, right_padded_sequence_length)
                self.assertListEqual(input_ids + [padding_idx] * padding_size, right_padded_input_ids)
                self.assertListEqual(special_tokens_mask + [1] * padding_size, right_padded_special_tokens_mask)

                # Test left padding
                tokenizer.padding_side = "left"
                left_padded_sequence = tokenizer.encode_plus(
                    table,
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                self.assertEqual(sequence_length + padding_size, left_padded_sequence_length)
                self.assertListEqual([padding_idx] * padding_size + input_ids, left_padded_input_ids)
                self.assertListEqual([1] * padding_size + special_tokens_mask, left_padded_special_tokens_mask)

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence["token_type_ids"]
                    right_padded_token_type_ids = right_padded_sequence["token_type_ids"]

                    self.assertListEqual(
                        (token_type_ids + [[token_type_padding_idx] * 7] * padding_size, right_padded_token_type_ids)
                    )
                    self.assertListEqual(
                        [[token_type_padding_idx] * 7] * padding_size + token_type_ids, left_padded_token_type_ids
                    )

                if "attention_mask" in tokenizer.model_input_names:
                    attention_mask = encoded_sequence["attention_mask"]
                    right_padded_attention_mask = right_padded_sequence["attention_mask"]
                    left_padded_attention_mask = left_padded_sequence["attention_mask"]

                    self.assertListEqual(attention_mask + [0] * padding_size, right_padded_attention_mask)
                    self.assertListEqual([0] * padding_size + attention_mask, left_padded_attention_mask)

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus

        # Right padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode_plus(table, sequence, max_length=max_length, padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    table, sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

        # Left padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.padding_side = "left"
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode_plus(table, sequence, max_length=max_length, padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    table, sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                encoded_sequences = [tokenizer.encode_plus(table, sequence) for sequence in sequences]
                encoded_sequences_batch = tokenizer.batch_encode_plus(table, sequences, padding=False)
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

                maximum_length = len(
                    max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len)
                )

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences_padded = [
                    tokenizer.encode_plus(table, sequence, max_length=maximum_length, padding="max_length")
                    for sequence in sequences
                ]

                encoded_sequences_batch_padded = tokenizer.batch_encode_plus(table, sequences, padding=True)
                self.assertListEqual(
                    encoded_sequences_padded,
                    self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(table, sequences, padding=True)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    table, sequences, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(table, sequences, padding=False)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    table, sequences, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                empty_table = self.get_table(tokenizer, length=0)
                table = self.get_table(tokenizer, length=10)
                encoded_sequence = tokenizer.encode(empty_table, sequence_0, add_special_tokens=False)
                number_of_tokens = len(encoded_sequence)
                encoded_sequence += tokenizer.encode(table, "", add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    table,
                    sequence_0,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                # NOTE: as TAPEX adds a space between a table and a sequence, we need to remove it
                # in order to have equivalent results with encoding an empty table or empty sequence
                del filtered_sequence[number_of_tokens + 1]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                print("Encoded sequence:", encoded_sequence)
                print("Filtered sequence:", filtered_sequence)
                self.assertEqual(encoded_sequence, filtered_sequence)

    @slow
    def test_full_tokenizer(self):
        question = "Greece held its last Summer Olympics in 2004"
        table_dict = {
            "header": ["Year", "City", "Country", "Nations"],
            "rows": [
                [1896, "Athens", "Greece", 14],
                [1900, "Paris", "France", 24],
                [1904, "St. Louis", "USA", 12],
                [2004, "Athens", "Greece", 201],
                [2008, "Beijing", "China", 204],
                [2012, "London", "UK", 204],
            ],
        }
        table = pd.DataFrame.from_dict(table_dict["rows"])
        table.columns = table_dict["header"]

        tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
        encoding = tokenizer(table, question)

        # fmt: off
        expected_results = {'input_ids': [0, 821, 5314, 1755, 547, 63, 94, 1035, 1021, 31434, 2857, 11, 4482, 11311, 4832, 76, 1721, 343, 1721, 247, 1721, 3949, 3236, 112, 4832, 42773, 1721, 23, 27859, 1721, 821, 5314, 1755, 1721, 501, 3236, 132, 4832, 23137, 1721, 2242, 354, 1721, 6664, 2389, 1721, 706, 3236, 155, 4832, 42224, 1721, 1690, 4, 26120, 354, 1721, 201, 102, 1721, 316, 3236, 204, 4832, 4482, 1721, 23, 27859, 1721, 821, 5314, 1755, 1721, 21458, 3236, 195, 4832, 2266, 1721, 28, 40049, 1721, 1855, 1243, 1721, 28325, 3236, 231, 4832, 1125, 1721, 784, 24639, 1721, 1717, 330, 1721, 28325, 2]}
        # fmt: on

        self.assertListEqual(encoding.input_ids, expected_results["input_ids"])

    def test_tokenizer_as_target(self):
        # by default the tokenizer do_lower_case
        tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
        answer_text = "tapex is a good model!"
        expected_src_tokens = [0, 90, 5776, 1178, 16, 10, 205, 1421, 328, 2]
        with tokenizer.as_target_tokenizer():
            answer_encoding = tokenizer(answer=answer_text)
            self.assertListEqual(answer_encoding.input_ids, expected_src_tokens)

    @slow
    def test_tokenizer_lower_case(self):
        cased_tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base", do_lower_case=False)
        uncased_tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base", do_lower_case=True)
        answer_text = "Beijing, London, Paris"
        answer_text_lower = "beijing, london, paris"

        with cased_tokenizer.as_target_tokenizer():
            with uncased_tokenizer.as_target_tokenizer():
                self.assertNotEqual(
                    cased_tokenizer(answer=answer_text).input_ids, uncased_tokenizer(answer=answer_text).input_ids
                )
                self.assertEqual(
                    cased_tokenizer(answer=answer_text_lower).input_ids,
                    uncased_tokenizer(answer=answer_text).input_ids,
                )
                # batched encoding assert
                self.assertNotEqual(
                    cased_tokenizer(answer=[answer_text]).input_ids, uncased_tokenizer(answer=[answer_text]).input_ids
                )
                self.assertEqual(
                    cased_tokenizer(answer=[answer_text_lower]).input_ids,
                    uncased_tokenizer(answer=[answer_text]).input_ids,
                )
        # test input encoding lowercase
        question = "Greece held its last Summer Olympics in 2004"
        table_dict = {
            "header": ["Year", "City", "Country", "Nations"],
            "rows": [
                [1896, "Athens", "Greece", 14],
                [1900, "Paris", "France", 24],
                [1904, "St. Louis", "USA", 12],
                [2004, "Athens", "Greece", 201],
                [2008, "Beijing", "China", 204],
                [2012, "London", "UK", 204],
            ],
        }
        table = pd.DataFrame.from_dict(table_dict["rows"])
        table.columns = table_dict["header"]

        self.assertNotEqual(
            cased_tokenizer(table=table, query=question).input_ids,
            uncased_tokenizer(table=table, query=question).input_ids,
        )
