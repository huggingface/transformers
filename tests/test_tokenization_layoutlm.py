# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors, The Hugging Face Team.
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
import re
import shutil
import tempfile
import unittest
from typing import Tuple

from tokenizers import AddedToken
from transformers import LayoutLMTokenizer, PreTrainedTokenizerFast
from transformers.models.layoutlm.tokenization_layoutlm import VOCAB_FILES_NAMES
from transformers.testing_utils import is_pt_tf_cross_test, require_tf, require_tokenizers, require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin, merge_model_tokenizer_mappings


@require_tokenizers
class LayoutLMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = LayoutLMTokenizer
    rust_tokenizer_class = None
    test_rust_tokenizer = False
    space_between_special_tokens = True
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        return LayoutLMTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> Tuple[str, list]:
        toks = [(i, tokenizer.decode([i], clean_up_tokenization_spaces=False)) for i in range(len(tokenizer))]
        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(
            filter(lambda t: [t[0]] == tokenizer.encode(t[1], bbox=(1, 2, 3, 4), add_special_tokens=False)[0], toks)
        )
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids, _ = tokenizer.encode(output_txt, bbox=(1, 2, 3, 4), add_special_tokens=False)
        return output_txt, output_ids

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = self.get_clean_sequence(tokenizer)

                special_token = "[SPECIAL_TOKEN]"
                special_bbox = (1, 2, 3, 4)

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token, _ = tokenizer.encode(special_token, bbox=special_bbox, add_special_tokens=False)
                self.assertEqual(len(encoded_special_token), 1)

                text = tokenizer.decode(ids + encoded_special_token, clean_up_tokenization_spaces=False)
                encoded, _ = tokenizer.encode(text, bbox=(1, 2, 3, 4), add_special_tokens=False)

                input_encoded, _ = tokenizer.encode(input_text, special_bbox, add_special_tokens=False)
                special_token_id, _ = tokenizer.encode(special_token, special_bbox, add_special_tokens=False)
                self.assertEqual(encoded, input_encoded + special_token_id)

                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

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

                tokens, _ = tokenizer.encode(
                    "aaaaa bbbbbb low cccccccccdddddddd l", bbox=(1, 2, 3, 4), add_special_tokens=False
                )

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

                tokens, _ = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                    bbox=(1, 2, 3, 4),
                    add_special_tokens=False,
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]
                bboxes = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 12, 13)]

                encoded_sequences = [
                    tokenizer.encode_plus(sequence, bbox=bbox) for sequence, bbox in zip(sequences, bboxes)
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    sequences, batch_bbox_or_bbox_pairs=bboxes, padding=False
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

                maximum_length = len(
                    max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len)
                )

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences_padded = [
                    tokenizer.encode_plus(sequence, bbox=bbox, max_length=maximum_length, padding="max_length")
                    for sequence, bbox in zip(sequences, bboxes)
                ]

                encoded_sequences_batch_padded = tokenizer.batch_encode_plus(
                    sequences, batch_bbox_or_bbox_pairs=bboxes, padding=True
                )
                self.assertListEqual(
                    encoded_sequences_padded,
                    self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(
                    sequences, batch_bbox_or_bbox_pairs=bboxes, padding=True
                )
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    sequences, batch_bbox_or_bbox_pairs=bboxes, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(
                    sequences, batch_bbox_or_bbox_pairs=bboxes, padding=False
                )
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    sequences, batch_bbox_or_bbox_pairs=bboxes, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

    def test_batch_encode_plus_overflowing_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            string_sequences = ["Testing the prepare_for_model method.", "Test"]
            bboxes = [(1, 2, 3, 4), (5, 6, 7, 8)]

            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            tokenizer.batch_encode_plus(
                string_sequences,
                batch_bbox_or_bbox_pairs=bboxes,
                return_overflowing_tokens=True,
                truncation=True,
                padding=True,
                max_length=3,
            )

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                for padding_side in ["right", "left"]:
                    with self.subTest(f"padding side {padding_side}"):
                        tokenizer.padding_side = padding_side
                        sequences = [
                            "Testing batch encode plus",
                            "Testing batch encode plus with different sequence lengths",
                            "Testing batch encode plus with different sequence lengths correctly pads",
                        ]
                        bboxes = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 12, 13)]

                        max_length = 100

                        # check correct behaviour if no pad_token_id exists and add it eventually
                        self._check_no_pad_token_padding(tokenizer, sequences)

                        encoded_sequences = [
                            tokenizer.encode_plus(sequence, bbox=bbox, max_length=max_length, padding="max_length")
                            for sequence, bbox in zip(sequences, bboxes)
                        ]
                        encoded_sequences_batch = tokenizer.batch_encode_plus(
                            sequences, batch_bbox_or_bbox_pairs=bboxes, max_length=max_length, padding="max_length"
                        )
                        self.assertListEqual(
                            encoded_sequences,
                            self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch),
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
                bbox = [(1, 2, 3, 4)] * len(sequences)

                # A Tensor cannot be build by sequences which are not the same size
                self.assertRaises(
                    ValueError,
                    tokenizer.batch_encode_plus,
                    sequences,
                    batch_bbox_or_bbox_pairs=bbox,
                    return_tensors="pt",
                )
                self.assertRaises(
                    ValueError,
                    tokenizer.batch_encode_plus,
                    sequences,
                    batch_bbox_or_bbox_pairs=bbox,
                    return_tensors="tf",
                )

                if tokenizer.pad_token_id is None:
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        sequences,
                        bbox=bbox,
                        padding=True,
                        return_tensors="pt",
                    )
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        sequences,
                        bbox=bbox,
                        padding="longest",
                        return_tensors="tf",
                    )
                else:
                    pytorch_tensor = tokenizer.batch_encode_plus(
                        sequences, batch_bbox_or_bbox_pairs=bbox, padding=True, return_tensors="pt"
                    )
                    tensorflow_tensor = tokenizer.batch_encode_plus(
                        sequences, batch_bbox_or_bbox_pairs=bbox, padding="longest", return_tensors="tf"
                    )
                    encoded_sequences = tokenizer.batch_encode_plus(
                        sequences, batch_bbox_or_bbox_pairs=bbox, padding=True
                    )

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
                bboxes = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 12, 13)]

                # Test not batched
                encoded_sequences_1 = tokenizer.encode_plus(sequences[0], bbox=bboxes[0])
                encoded_sequences_2 = tokenizer(sequences[0], bbox=bboxes[0])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                encoded_sequences_1 = tokenizer.encode_plus(sequences[0], bboxes[0], sequences[1], bboxes[1])
                encoded_sequences_2 = tokenizer(sequences[0], bboxes[0], sequences[1], bboxes[1])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                encoded_sequences_1 = tokenizer.batch_encode_plus(sequences, bboxes)
                encoded_sequences_2 = tokenizer(sequences, bboxes)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched pairs
                encoded_sequences_1 = tokenizer.batch_encode_plus(
                    list(zip(sequences, sequences)), batch_bbox_or_bbox_pairs=list(zip(bboxes, bboxes))
                )
                encoded_sequences_2 = tokenizer(sequences, bbox=bboxes, text_pair=sequences, bbox_pair=bboxes)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

    @require_tokenizers
    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                # new_toks = ["[ABC]", "[DEF]"]  # TODO(thom) add this one back when Rust toks are ready: , "GHI IHG"]
                new_toks = [AddedToken("[ABC]", normalized=False), AddedToken("[DEF]", normalized=False)]
                tokenizer.add_tokens(new_toks)
                input = "[ABC][DEF][ABC][DEF]"  # TODO(thom) add back cf above: "[ABC] [DEF] [ABC] GHI IHG [DEF]"
                if self.space_between_special_tokens:
                    output = "[ABC] [DEF] [ABC] [DEF]"
                else:
                    output = input
                encoded, _ = tokenizer.encode(input, bbox=(1, 2, 3, 4), add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    def test_encode_plus_with_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                bbox = (1, 2, 3, 4)

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_size = 10
                padding_idx = tokenizer.pad_token_id
                token_type_padding_idx = tokenizer.pad_token_type_id

                encoded_sequence = tokenizer.encode_plus(sequence, bbox=bbox, return_special_tokens_mask=True)
                input_ids = encoded_sequence["input_ids"]
                special_tokens_mask = encoded_sequence["special_tokens_mask"]
                sequence_length = len(input_ids)

                # Test 'longest' and 'no_padding' don't do anything
                tokenizer.padding_side = "right"

                not_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    bbox=bbox,
                    padding=True,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertEqual(input_ids, not_padded_input_ids)
                self.assertEqual(special_tokens_mask, not_padded_special_tokens_mask)

                not_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    bbox=bbox,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertEqual(input_ids, not_padded_input_ids)
                self.assertEqual(special_tokens_mask, not_padded_special_tokens_mask)

                # Test right padding
                tokenizer.padding_side = "right"

                right_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    bbox=bbox,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                right_padded_input_ids = right_padded_sequence["input_ids"]

                right_padded_special_tokens_mask = right_padded_sequence["special_tokens_mask"]
                right_padded_sequence_length = len(right_padded_input_ids)

                self.assertEqual(sequence_length + padding_size, right_padded_sequence_length)
                self.assertEqual(input_ids + [padding_idx] * padding_size, right_padded_input_ids)
                self.assertEqual(special_tokens_mask + [1] * padding_size, right_padded_special_tokens_mask)

                # Test left padding
                tokenizer.padding_side = "left"
                left_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    bbox=bbox,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                self.assertEqual(sequence_length + padding_size, left_padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size + input_ids, left_padded_input_ids)
                self.assertEqual([1] * padding_size + special_tokens_mask, left_padded_special_tokens_mask)

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence["token_type_ids"]
                    right_padded_token_type_ids = right_padded_sequence["token_type_ids"]

                    self.assertEqual(
                        token_type_ids + [token_type_padding_idx] * padding_size, right_padded_token_type_ids
                    )
                    self.assertEqual(
                        [token_type_padding_idx] * padding_size + token_type_ids, left_padded_token_type_ids
                    )

                if "attention_mask" in tokenizer.model_input_names:
                    attention_mask = encoded_sequence["attention_mask"]
                    right_padded_attention_mask = right_padded_sequence["attention_mask"]
                    left_padded_attention_mask = left_padded_sequence["attention_mask"]

                    self.assertEqual(attention_mask + [0] * padding_size, right_padded_attention_mask)
                    self.assertEqual([0] * padding_size + attention_mask, left_padded_attention_mask)

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, output_text = self.get_input_output_texts(tokenizer)
                input_bbox = (1, 2, 3, 4)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2, out_bbox = tokenizer.encode(input_text, bbox=input_bbox, add_special_tokens=False)
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                self.assertEqual(text_2, output_text)

    def test_maximum_encoding_length_pair_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Build a sequence from our model's vocabulary
                stride = 2
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                if len(ids) <= 2 + stride:
                    seq_0 = (seq_0 + " ") * (2 + stride)
                    ids = None

                bbox = (1, 2, 3, 4)

                seq0_tokens, seq0_boxes = tokenizer.encode(seq_0, bbox=bbox, add_special_tokens=False)
                self.assertGreater(len(seq0_tokens), 2 + stride)

                seq_1 = "This is another sentence to be encoded."
                seq1_tokens, _ = tokenizer.encode(seq_1, bbox=bbox, add_special_tokens=False)
                if abs(len(seq0_tokens) - len(seq1_tokens)) <= 2:
                    seq1_tokens = seq1_tokens + seq1_tokens
                    seq_1 = tokenizer.decode(seq1_tokens, clean_up_tokenization_spaces=False)
                seq1_tokens, seq1_boxes = tokenizer.encode(seq_1, bbox=bbox, add_special_tokens=False)

                self.assertGreater(len(seq1_tokens), 2 + stride)

                smallest = seq1_tokens if len(seq0_tokens) > len(seq1_tokens) else seq0_tokens

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence, box_sequence = tokenizer.encode(
                    seq_0, bbox=bbox, text_pair=seq_1, bbox_pair=bbox, add_special_tokens=False
                )  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                self.assertGreater(len(seq_2), model_max_length)

                sequence1 = tokenizer(seq_1, bbox=bbox, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer(seq_2, bbox=bbox, text_pair=seq_1, bbox_pair=bbox, add_special_tokens=False)
                total_length2 = len(sequence2["input_ids"])
                assert total_length1 < model_max_length - 10, "Issue with the testing sequence, please update it."
                assert total_length2 > model_max_length, "Issue with the testing sequence, please update it."

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"{tokenizer.__class__.__name__} Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"{tokenizer.__class__.__name__} Truncation: {truncation_state}"):
                                output = tokenizer(
                                    seq_2,
                                    bbox=bbox,
                                    text_pair=seq_1,
                                    bbox_pair=bbox,
                                    padding=padding_state,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"]), model_max_length)

                                output = tokenizer(
                                    [seq_2],
                                    bbox=[bbox],
                                    text_pair=[seq_1],
                                    bbox_pair=[bbox],
                                    padding=padding_state,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple
                        output = tokenizer(
                            seq_1,
                            bbox=bbox,
                            text_pair=seq_2,
                            bbox_pair=bbox,
                            padding=padding_state,
                            truncation="only_second",
                        )
                        self.assertEqual(len(output["input_ids"]), model_max_length)

                        output = tokenizer(
                            [seq_1],
                            bbox=[bbox],
                            text_pair=[seq_2],
                            bbox_pair=[bbox],
                            padding=padding_state,
                            truncation="only_second",
                        )
                        self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(
                                seq_1,
                                bbox=bbox,
                                text_pair=seq_2,
                                bbox_pair=bbox,
                                padding=padding_state,
                                truncation=False,
                            )
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(
                                [seq_1],
                                bbox=[bbox],
                                text_pair=[seq_2],
                                bbox_pair=[bbox],
                                padding=padding_state,
                                truncation=False,
                            )
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                first_seq, first_box_seq = tokenizer.encode(seq_0, bbox=bbox, add_special_tokens=False)
                second_seq, second_box_seq = tokenizer.encode(seq_1, bbox=bbox, add_special_tokens=False)
                truncated_first_sequence = first_seq[:-2] + second_seq
                truncated_first_box_sequence = first_box_seq[:-2] + second_box_seq
                truncated_second_sequence = first_seq + second_seq[:-2]
                truncated_second_box_sequence = first_box_seq + second_box_seq[:-2]

                truncated_longest_sequence = (
                    truncated_first_sequence if len(seq0_tokens) > len(seq1_tokens) else truncated_second_sequence
                )
                truncated_longest_box_sequence = (
                    truncated_first_box_sequence
                    if len(seq0_tokens) > len(seq1_tokens)
                    else truncated_second_box_sequence
                )

                first_overflow_seq, first_overflow_box_seq = tokenizer.encode(
                    seq_0, bbox=bbox, add_special_tokens=False
                )
                second_overflow_seq, second_overflow_box_seq = tokenizer.encode(
                    seq_1, bbox=bbox, add_special_tokens=False
                )
                overflow_first_sequence = first_overflow_seq[-(2 + stride) :] + second_overflow_seq
                overflow_second_sequence = first_overflow_seq + second_overflow_seq[-(2 + stride) :]

                overflow_longest_sequence = (
                    overflow_first_sequence if len(seq0_tokens) > len(seq1_tokens) else overflow_second_sequence
                )

                information = tokenizer.encode_plus(
                    seq_0,
                    bbox=bbox,
                    text_pair=seq_1,
                    bbox_pair=bbox,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="longest_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                else:
                    truncated_sequence = information["input_ids"]
                    truncated_box_sequence = information["bbox"]
                    overflowing_tokens = information["overflowing_tokens"]
                    overflowing_boxes = information["overflowing_boxes"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(truncated_box_sequence), len(box_sequence) - 2)
                    self.assertEqual(truncated_box_sequence, truncated_longest_box_sequence)

                    self.assertEqual(
                        len(overflowing_tokens), 2 + stride
                    )  # No overflowing tokens when using 'longest' in python tokenizers
                    self.assertEqual(len(overflowing_boxes), 2 + stride)

                information = tokenizer.encode_plus(
                    seq_0,
                    bbox=bbox,
                    text_pair=seq_1,
                    bbox_pair=bbox,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation=True,
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                else:
                    truncated_sequence = information["input_ids"]
                    truncated_box_sequence = information["bbox"]
                    overflowing_tokens = information["overflowing_tokens"]
                    overflowing_boxes = information["overflowing_boxes"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(truncated_box_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_box_sequence, truncated_longest_box_sequence)

                    self.assertEqual(
                        len(overflowing_tokens), 2 + stride
                    )  # No overflowing tokens when using 'longest' in python tokenizers
                    self.assertEqual(len(overflowing_boxes), 2 + stride)

                information_first_truncated = tokenizer.encode_plus(
                    seq_0,
                    bbox=bbox,
                    text_pair=seq_1,
                    bbox_pair=bbox,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information_first_truncated["input_ids"][0]
                    overflowing_tokens = information_first_truncated["input_ids"][1]
                    self.assertEqual(len(information_first_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq1_tokens))
                    self.assertEqual(overflowing_tokens, overflow_first_sequence)
                else:
                    truncated_sequence = information_first_truncated["input_ids"]
                    truncated_box_sequence = information_first_truncated["bbox"]
                    overflowing_tokens = information_first_truncated["overflowing_tokens"]
                    overflowing_boxes = information_first_truncated["overflowing_boxes"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(truncated_box_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_box_sequence, truncated_first_box_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq0_tokens[-(2 + stride) :])

                    self.assertEqual(len(overflowing_boxes), 2 + stride)
                    self.assertEqual(overflowing_boxes, seq0_boxes[-(2 + stride) :])

                information_second_truncated = tokenizer.encode_plus(
                    seq_0,
                    bbox=bbox,
                    text_pair=seq_1,
                    bbox_pair=bbox,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information_second_truncated["input_ids"][0]
                    overflowing_tokens = information_second_truncated["input_ids"][1]
                    self.assertEqual(len(information_second_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq0_tokens))
                    self.assertEqual(overflowing_tokens, overflow_second_sequence)
                else:
                    truncated_sequence = information_second_truncated["input_ids"]
                    truncated_box_sequence = information_second_truncated["bbox"]
                    overflowing_tokens = information_second_truncated["overflowing_tokens"]
                    overflowing_boxes = information_second_truncated["overflowing_boxes"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(truncated_box_sequence), len(box_sequence) - 2)
                    self.assertEqual(truncated_box_sequence, truncated_second_box_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq1_tokens[-(2 + stride) :])

                    self.assertEqual(len(overflowing_boxes), 2 + stride)
                    self.assertEqual(overflowing_boxes, seq1_boxes[-(2 + stride) :])

    def test_padding(self, max_length=50):

        token_input = ["This", "is", "a", "tokenized", "input"]
        token_bbox_input = [[1.0, 2.0, 3.0, 4.0]] * len(token_input)
        pair_token_input = ["This", "is", "a", "pair"]
        pair_bbox_input = [[2.0, 3.0, 4.0, 5.0]] * len(pair_token_input)

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest("{} ({})".format(tokenizer.__class__.__name__, pretrained_name)):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                self.assertEqual(tokenizer_p.pad_token_id, tokenizer_r.pad_token_id)
                self.assertEqual(tokenizer_p.pad_box, tokenizer_r.pad_box)
                pad_token_id = tokenizer_p.pad_token_id
                pad_box = tokenizer_p.pad_box

                # Encode - tokenized input
                input_r = tokenizer_r.encode(
                    token_input, bbox=token_bbox_input, max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode(
                    token_input, bbox=token_bbox_input, max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(
                    token_input, bbox=token_bbox_input, max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode(
                    token_input, bbox=token_bbox_input, max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.encode(token_input, bbox=token_bbox_input, padding="longest")
                input_p = tokenizer_p.encode(token_input, bbox=token_bbox_input, padding=True)
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode - Pair input
                input_r = tokenizer_r.encode(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                input_p = tokenizer_p.encode(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    padding="max_length",
                )
                input_p = tokenizer_p.encode(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    padding="max_length",
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(
                    token_input, token_bbox_input, pair_token_input, pair_bbox_input, padding=True
                )
                input_p = tokenizer_p.encode(
                    token_input, token_bbox_input, pair_token_input, pair_bbox_input, padding="longest"
                )
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode_plus - Simple input
                input_r = tokenizer_r.encode_plus(
                    token_input, token_bbox_input, max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode_plus(
                    token_input, token_bbox_input, max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(
                    token_input, token_bbox_input, max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode_plus(
                    token_input, token_bbox_input, max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assert_padded_input_match(input_r["bbox"], input_p["bbox"], max_length, pad_box)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                input_r = tokenizer_r.encode_plus(token_input, token_bbox_input, padding="longest")
                input_p = tokenizer_p.encode_plus(token_input, token_bbox_input, padding=True)
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )
                self.assert_padded_input_match(input_r["bbox"], input_p["bbox"], len(input_r["bbox"]), pad_box)

                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                # Encode_plus - Pair input
                input_r = tokenizer_r.encode_plus(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                input_p = tokenizer_p.encode_plus(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assert_padded_input_match(input_r["bbox"], input_p["bbox"], max_length, pad_box)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    padding="max_length",
                )
                input_p = tokenizer_p.encode_plus(
                    token_input,
                    token_bbox_input,
                    pair_token_input,
                    pair_bbox_input,
                    max_length=max_length,
                    padding="max_length",
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assert_padded_input_match(input_r["bbox"], input_p["bbox"], max_length, pad_box)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(
                    token_input, token_bbox_input, pair_token_input, pair_bbox_input, padding="longest"
                )
                input_p = tokenizer_p.encode_plus(
                    token_input, token_bbox_input, pair_token_input, pair_bbox_input, padding=True
                )
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )
                self.assert_padded_input_match(input_r["bbox"], input_p["bbox"], len(input_r["bbox"]), pad_box)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                # Batch_encode_plus - Simple input
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    max_length=max_length,
                    padding="max_length",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    max_length=max_length,
                    padding="max_length",
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    max_length=max_length,
                    padding="longest",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    max_length=max_length,
                    padding=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    padding="longest",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                    padding=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Batch_encode_plus - Pair input
                input_r = tokenizer_r.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    batch_bbox_or_bbox_pairs=[[(1, 2, 3, 4), (1, 2, 3, 4)], [(1, 2, 3, 4), (1, 2, 3, 4)]],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    batch_bbox_or_bbox_pairs=[[(1, 2, 3, 4), (1, 2, 3, 4)], [(1, 2, 3, 4), (1, 2, 3, 4)]],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    batch_bbox_or_bbox_pairs=[[(1, 2, 3, 4), (1, 2, 3, 4)], [(1, 2, 3, 4), (1, 2, 3, 4)]],
                    padding=True,
                )
                input_p = tokenizer_p.batch_encode_plus(
                    [
                        ("This is a simple input 1", "This is a simple input 2"),
                        ("This is a simple pair 1", "This is a simple pair 2"),
                    ],
                    batch_bbox_or_bbox_pairs=[[(1, 2, 3, 4), (1, 2, 3, 4)], [(1, 2, 3, 4), (1, 2, 3, 4)]],
                    padding="longest",
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad on single examples after tokenization
                input_r = tokenizer_r.encode_plus("This is a input 1", bbox=(1, 2, 3, 4))
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.encode_plus("This is a input 1", bbox=(1, 2, 3, 4))
                input_p = tokenizer_r.pad(input_p)

                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                # Using pad on single examples after tokenization
                input_r = tokenizer_r.encode_plus("This is a input 1", bbox=(1, 2, 3, 4))
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.encode_plus("This is a input 1", bbox=(1, 2, 3, 4))
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                # Using pad after tokenization
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input which should be padded"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                )
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input which should be padded"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                )
                input_p = tokenizer_r.pad(input_p)

                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad after tokenization
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input which should be padded"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                )
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input which should be padded"],
                    batch_bbox_or_bbox_pairs=[(1, 2, 3, 4), (1, 2, 3, 4)],
                )
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

    def test_padding_to_max_length(self):
        """We keep this test for backward compatibility but it should be remove when `pad_to_max_length` will be deprecated"""
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                bbox_seq = (1, 2, 3, 4)
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id
                padding_box = tokenizer.pad_box

                # Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence, encoded_bbox_sequence = tokenizer.encode(sequence, bbox=bbox_seq)
                sequence_length = len(encoded_sequence)
                box_sequence_length = len(encoded_bbox_sequence)
                # FIXME: the next line should be padding(max_length) to avoid warning
                padded_sequence, padded_box_sequence = tokenizer.encode(
                    sequence, bbox=bbox_seq, max_length=sequence_length + padding_size, pad_to_max_length=True
                )
                self.assertEqual(sequence_length + padding_size, len(padded_sequence))
                self.assertEqual(sequence_length + padding_size, len(padded_box_sequence))
                self.assertEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)
                self.assertEqual(encoded_bbox_sequence + [padding_box] * padding_size, padded_box_sequence)

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence, encoded_bbox_sequence = tokenizer.encode(sequence, bbox=bbox_seq)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right, padded_box_sequence_right = tokenizer.encode(
                    sequence, bbox=bbox_seq, pad_to_max_length=True
                )
                padded_sequence_right_length = len(padded_sequence_right)
                padded_box_sequence_right_length = len(padded_box_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(box_sequence_length, padded_box_sequence_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)
                self.assertEqual(encoded_bbox_sequence, padded_box_sequence_right)

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                else:
                    empty_tokens = tokenizer("", bbox=(), padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer(
                        "This is a sample input", bbox=(1, 2, 3, 4), padding=True, pad_to_multiple_of=8
                    )
                    for key, value in empty_tokens.items():
                        self.assertEqual(len(value) % 8, 0, "BatchEncoding.{} is not multiple of 8".format(key))
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, "BatchEncoding.{} is not multiple of 8".format(key))

                    normal_tokens = tokenizer("This", bbox=(1, 2, 3, 4), pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, "BatchEncoding.{} is not multiple of 8".format(key))

                    # Should also work with truncation
                    normal_tokens = tokenizer(
                        "This", bbox=(1, 2, 3, 4), padding=True, truncation=True, pad_to_multiple_of=8
                    )
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, "BatchEncoding.{} is not multiple of 8".format(key))

                    # truncation to something which is not a multiple of pad_to_multiple_of raises an error
                    self.assertRaises(
                        ValueError,
                        tokenizer.__call__,
                        "This",
                        bbox=(1, 2, 3, 4),
                        padding=True,
                        truncation=True,
                        max_length=12,
                        pad_to_multiple_of=8,
                    )

    def test_maximum_encoding_length_single_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                bbox = (1, 2, 3, 4)

                sequence, box_sequence = tokenizer.encode(seq_0, bbox=bbox, add_special_tokens=False)
                total_length = len(sequence)

                assert total_length > 4, "Issue with the testing sequence, please update it it's too short"

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length

                sequence1 = tokenizer(seq_1, bbox=bbox, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                assert (
                    total_length1 > model_max_length
                ), "Issue with the testing sequence, please update it it's too short"

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"Truncation: {truncation_state}"):
                                output = tokenizer(
                                    seq_1, bbox=bbox, padding=padding_state, truncation=truncation_state
                                )
                                self.assertEqual(len(output["input_ids"]), model_max_length)
                                self.assertEqual(len(output["bbox"]), model_max_length)

                                output = tokenizer(
                                    [seq_1], bbox=[bbox], padding=padding_state, truncation=truncation_state
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)
                                self.assertEqual(len(output["bbox"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(seq_1, bbox=bbox, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                            self.assertNotEqual(len(output["bbox"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer([seq_1], bbox=[bbox], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                            self.assertNotEqual(len(output["bbox"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                # Overflowing tokens
                stride = 2
                information = tokenizer(
                    seq_0,
                    bbox=bbox,
                    max_length=total_length - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="longest_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence[-(2 + stride) :])
                else:
                    truncated_sequence = information["input_ids"]
                    truncated_box_sequence = information["bbox"]
                    overflowing_tokens = information["overflowing_tokens"]
                    overflowing_boxes = information["overflowing_boxes"]

                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])

                    self.assertEqual(len(truncated_box_sequence), total_length - 2)
                    self.assertEqual(truncated_box_sequence, box_sequence[:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)

                    self.assertEqual(len(overflowing_boxes), 2 + stride)

    def test_prepare_for_model(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                string_sequence = "Testing the prepare_for_model method."
                bbox = (1, 2, 3, 4)
                ids, boxes = tokenizer.encode(string_sequence, bbox=bbox, add_special_tokens=False)
                prepared_input_dict = tokenizer.prepare_for_model(ids, boxes, add_special_tokens=True)

                input_dict = tokenizer.encode_plus(string_sequence, bbox=bbox, add_special_tokens=True)

                self.assertEqual(input_dict, prepared_input_dict)

    def test_pretokenized_inputs(self):
        # Test when inputs are pretokenized

        tokenizers = self.get_tokenizers(do_lower_case=False)  # , add_prefix_space=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if hasattr(tokenizer, "add_prefix_space") and not tokenizer.add_prefix_space:
                    continue

                # Prepare a sequence from our tokenizer vocabulary
                sequence, ids = self.get_clean_sequence(tokenizer, with_prefix_space=True, max_length=20)
                # sequence = " " + sequence  # To be sure the byte-level tokenizers are feeling good
                token_sequence = sequence.split()
                # sequence_no_prefix_space = sequence.strip()
                bbox = (1, 2, 3, 4)
                token_sequence_bbox = [bbox] * len(token_sequence)

                # Test encode for pretokenized inputs
                output, output_bbox = tokenizer.encode(
                    token_sequence, bbox=token_sequence_bbox, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence, output_sequence_bbox = tokenizer.encode(sequence, bbox=bbox, add_special_tokens=False)
                self.assertEqual(output, output_sequence)
                self.assertEqual(output_bbox, output_sequence_bbox)

                output, output_bbox = tokenizer.encode(
                    token_sequence, bbox=token_sequence_bbox, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence, output_sequence_bbox = tokenizer.encode(sequence, bbox=bbox, add_special_tokens=True)
                self.assertEqual(output, output_sequence)
                self.assertEqual(output_bbox, output_sequence_bbox)

                # Test encode_plus for pretokenized inputs
                output = tokenizer.encode_plus(
                    token_sequence, bbox=token_sequence_bbox, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer.encode_plus(sequence, bbox=bbox, add_special_tokens=False)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.encode_plus(
                    token_sequence, bbox=token_sequence_bbox, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer.encode_plus(sequence, bbox=bbox, add_special_tokens=True)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test batch_encode_plus for pretokenized inputs
                sequence_batch = [sequence.strip()] * 2 + [sequence.strip() + " " + sequence.strip()]
                token_sequence_batch = [s.split() for s in sequence_batch]
                bbox_sequence_batch = [[(1, 2, 3, 4)] * len(seq) for seq in token_sequence_batch]
                sequence_batch_cleaned_up_spaces = [" " + " ".join(s) for s in token_sequence_batch]
                bbox_sequence_batch_cleaned_up_spaces = [(1, 2, 3, 4) for _ in sequence_batch_cleaned_up_spaces]

                output = tokenizer.batch_encode_plus(
                    token_sequence_batch,
                    batch_bbox_or_bbox_pairs=bbox_sequence_batch,
                    is_split_into_words=True,
                    add_special_tokens=False,
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_batch_cleaned_up_spaces,
                    batch_bbox_or_bbox_pairs=bbox_sequence_batch_cleaned_up_spaces,
                    add_special_tokens=False,
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.batch_encode_plus(
                    token_sequence_batch,
                    batch_bbox_or_bbox_pairs=bbox_sequence_batch,
                    is_split_into_words=True,
                    add_special_tokens=True,
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_batch_cleaned_up_spaces,
                    batch_bbox_or_bbox_pairs=bbox_sequence_batch_cleaned_up_spaces,
                    add_special_tokens=True,
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test encode for pretokenized inputs pairs
                output, output_bbox = tokenizer.encode(
                    token_sequence,
                    bbox=token_sequence_bbox,
                    text_pair=token_sequence,
                    bbox_pair=token_sequence_bbox,
                    is_split_into_words=True,
                    add_special_tokens=False,
                )
                output_sequence, output_sequence_bbox = tokenizer.encode(
                    sequence, bbox=bbox, text_pair=sequence, bbox_pair=bbox, add_special_tokens=False
                )
                self.assertEqual(output, output_sequence)
                self.assertEqual(output_bbox, output_sequence_bbox)

                output, output_bbox = tokenizer.encode(
                    token_sequence,
                    bbox=token_sequence_bbox,
                    text_pair=token_sequence,
                    bbox_pair=token_sequence_bbox,
                    is_split_into_words=True,
                    add_special_tokens=True,
                )
                output_sequence, output_sequence_bbox = tokenizer.encode(
                    sequence, bbox=bbox, text_pair=sequence, bbox_pair=bbox, add_special_tokens=True
                )
                self.assertEqual(output, output_sequence)
                self.assertEqual(output_bbox, output_sequence_bbox)

                # Test encode_plus for pretokenized inputs pairs
                output = tokenizer.encode_plus(
                    token_sequence,
                    bbox=token_sequence_bbox,
                    text_pair=token_sequence,
                    bbox_pair=token_sequence_bbox,
                    is_split_into_words=True,
                    add_special_tokens=False,
                )
                output_sequence = tokenizer.encode_plus(
                    sequence, bbox=bbox, text_pair=sequence, bbox_pair=bbox, add_special_tokens=False
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                output = tokenizer.encode_plus(
                    token_sequence,
                    bbox=token_sequence_bbox,
                    text_pair=token_sequence,
                    bbox_pair=token_sequence_bbox,
                    is_split_into_words=True,
                    add_special_tokens=True,
                )
                output_sequence = tokenizer.encode_plus(
                    sequence, bbox=bbox, text_pair=sequence, bbox_pair=bbox, add_special_tokens=True
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test batch_encode_plus for pretokenized inputs pairs
                sequence_pair_batch = [(sequence.strip(), sequence.strip())] * 2 + [
                    (sequence.strip() + " " + sequence.strip(), sequence.strip())
                ]
                token_sequence_pair_batch = [tuple(s.split() for s in pair) for pair in sequence_pair_batch]
                bbox_sequence_pair_batch = [
                    ([(1, 2, 3, 4)] * len(tok1), [(1, 2, 3, 4)] * len(tok2))
                    for tok1, tok2 in token_sequence_pair_batch
                ]
                sequence_pair_batch_cleaned_up_spaces = [
                    tuple(" " + " ".join(s) for s in pair) for pair in token_sequence_pair_batch
                ]
                bbox_sequence_pair_batch_cleaned_up_spaces = [
                    ((1, 2, 3, 4), (1, 2, 3, 4)) for _ in sequence_pair_batch_cleaned_up_spaces
                ]

                output = tokenizer.batch_encode_plus(
                    token_sequence_pair_batch,
                    batch_bbox_or_bbox_pairs=bbox_sequence_pair_batch,
                    is_split_into_words=True,
                    add_special_tokens=False,
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_pair_batch_cleaned_up_spaces,
                    batch_bbox_or_bbox_pairs=bbox_sequence_pair_batch_cleaned_up_spaces,
                    add_special_tokens=False,
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.batch_encode_plus(
                    token_sequence_pair_batch,
                    batch_bbox_or_bbox_pairs=bbox_sequence_pair_batch,
                    is_split_into_words=True,
                    add_special_tokens=True,
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_pair_batch_cleaned_up_spaces,
                    batch_bbox_or_bbox_pairs=bbox_sequence_pair_batch_cleaned_up_spaces,
                    add_special_tokens=True,
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

    def test_right_and_left_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                bbox = (1, 2, 3, 4)
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id
                padding_box = tokenizer.pad_box

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence, encoded_bbox = tokenizer.encode(sequence, bbox=bbox)
                sequence_length = len(encoded_sequence)
                padded_sequence, padded_bbox = tokenizer.encode(
                    sequence, bbox=bbox, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                padded_bbox_length = len(padded_bbox)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual(sequence_length + padding_size, padded_bbox_length)
                self.assertEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)
                self.assertEqual(encoded_bbox + [padding_box] * padding_size, padded_bbox)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence, encoded_bbox = tokenizer.encode(sequence, bbox=bbox)
                sequence_length = len(encoded_sequence)
                padded_sequence, padded_bbox = tokenizer.encode(
                    sequence, bbox=bbox, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                padded_bbox_length = len(padded_bbox)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual(sequence_length + padding_size, padded_bbox_length)
                self.assertEqual([padding_idx] * padding_size + encoded_sequence, padded_sequence)
                self.assertEqual([padding_box] * padding_size + encoded_bbox, padded_bbox)

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
                encoded_sequence, encoded_bbox = tokenizer.encode(sequence, bbox=bbox)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right, padded_bbox_right = tokenizer.encode(sequence, bbox=bbox, padding=True)
                padded_sequence_right_length = len(padded_sequence_right)
                padded_bbox_right_length = len(padded_bbox_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(sequence_length, padded_bbox_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)
                self.assertEqual(encoded_bbox, padded_bbox_right)

                tokenizer.padding_side = "left"
                padded_sequence_left, padded_bbox_left = tokenizer.encode(sequence, bbox=bbox, padding="longest")
                padded_sequence_left_length = len(padded_sequence_left)
                padded_bbox_left_length = len(padded_bbox_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertEqual(sequence_length, padded_bbox_left_length)
                self.assertEqual(encoded_sequence, padded_sequence_left)
                self.assertEqual(encoded_bbox, padded_bbox_left)

                tokenizer.padding_side = "right"
                padded_sequence_right, padded_bbox_right = tokenizer.encode(sequence, bbox=bbox)
                padded_sequence_right_length = len(padded_sequence_right)
                padded_bbox_right_length = len(padded_bbox_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(sequence_length, padded_bbox_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)
                self.assertEqual(encoded_bbox, padded_bbox_right)

                tokenizer.padding_side = "left"
                padded_sequence_left, padded_bbox_left = tokenizer.encode(sequence, bbox=bbox, padding=False)
                padded_sequence_left_length = len(padded_sequence_left)
                padded_bbox_left_length = len(padded_bbox_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertEqual(sequence_length, padded_bbox_left_length)
                self.assertEqual(encoded_sequence, padded_sequence_left)
                self.assertEqual(encoded_bbox, padded_bbox_left)

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
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                sample_box = (1, 2, 3, 4)
                before_tokens, before_boxes = tokenizer.encode(sample_text, bbox=sample_box, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens, after_boxes = after_tokenizer.encode(
                    sample_text, bbox=sample_box, add_special_tokens=False
                )
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertListEqual(before_boxes, after_boxes)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                sample_bbox = (1, 2, 3, 4)
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
                before_tokens, before_boxes = tokenizer.encode(sample_text, bbox=sample_bbox, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens, after_boxes = after_tokenizer.encode(
                    sample_text, bbox=sample_bbox, add_special_tokens=False
                )
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertListEqual(before_boxes, after_boxes)
                self.assertDictEqual(before_vocab, after_vocab)
                self.assertIn("bim", after_vocab)
                self.assertIn("bambam", after_vocab)
                self.assertIn("new_additional_special_token", after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

        # Test that we can also use the non-legacy saving format for fast tokenizers
        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            if not tokenizer.is_fast:
                continue
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                sample_bbox = (1, 2, 3, 4)
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
                before_tokens, before_boxes = tokenizer.encode(sample_text, bbox=sample_bbox, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens, after_boxes = after_tokenizer.encode(
                    sample_text, bbox=sample_bbox, add_special_tokens=False
                )
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertListEqual(before_boxes, after_boxes)
                self.assertDictEqual(before_vocab, after_vocab)
                self.assertIn("bim", after_vocab)
                self.assertIn("bambam", after_vocab)
                self.assertIn("new_additional_special_token", after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                bbox = (1, 2, 3, 4)
                # Testing single inputs
                encoded_sequence, encoded_bbox = tokenizer.encode(sequence_0, bbox=bbox, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    sequence_0, bbox=bbox, add_special_tokens=True, return_special_tokens_mask=True
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                encoded_bbox_w_special = encoded_sequence_dict["bbox"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
                self.assertEqual(len(special_tokens_mask), len(encoded_bbox_w_special))

                filtered_sequence = [x for i, x in enumerate(encoded_sequence_w_special) if not special_tokens_mask[i]]
                self.assertEqual(encoded_sequence, filtered_sequence)
                filtered_bbox_sequence = [
                    x for i, x in enumerate(encoded_bbox_w_special) if not special_tokens_mask[i]
                ]
                self.assertEqual(encoded_bbox, filtered_bbox_sequence)

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                bbox_0 = (1, 2, 3, 4)
                sequence_1 = "This one too please."
                bbox_1 = (1, 2, 3, 4)
                encoded_sequence, encoded_bbox = tokenizer.encode(sequence_0, bbox=bbox_0, add_special_tokens=False)
                encoded_sequence_1, encoded_bbox_1 = tokenizer.encode(
                    sequence_1, bbox=bbox_1, add_special_tokens=False
                )
                encoded_sequence += encoded_sequence_1
                encoded_bbox += encoded_bbox_1
                encoded_sequence_dict = tokenizer.encode_plus(
                    sequence_0,
                    bbox=bbox_0,
                    text_pair=sequence_1,
                    bbox_pair=bbox_1,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    # add_prefix_space=False,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                encoded_bbox_w_special = encoded_sequence_dict["bbox"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
                self.assertEqual(len(special_tokens_mask), len(encoded_bbox_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)
                filtered_bbox_sequence = [
                    x for i, x in enumerate(encoded_bbox_w_special) if not special_tokens_mask[i]
                ]
                self.assertEqual(encoded_bbox, filtered_bbox_sequence)

    def test_mask_output(self):
        tokenizers = self.get_tokenizers(fast=False, do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if (
                    tokenizer.build_inputs_with_special_tokens.__qualname__.split(".")[0] != "PreTrainedTokenizer"
                    and "token_type_ids" in tokenizer.model_input_names
                ):
                    seq_0 = "Test this method."
                    seq_1 = "With these inputs."
                    bbox = (1, 2, 3, 4)
                    information = tokenizer.encode_plus(
                        seq_0, bbox=bbox, text_pair=seq_1, bbox_pair=bbox, add_special_tokens=True
                    )
                    sequences, mask = information["input_ids"], information["token_type_ids"]
                    self.assertEqual(len(sequences), len(mask))

    def test_token_type_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0 = "Test this method."
                bbox = (1, 2, 3, 4)

                # We want to have sequence 0 and sequence 1 are tagged
                # respectively with 0 and 1 token_ids
                # (regardless of whether the model use token type ids)
                # We use this assumption in the QA pipeline among other place
                output = tokenizer(seq_0, bbox=bbox, return_token_type_ids=True)
                self.assertIn(0, output["token_type_ids"])

    def test_number_of_added_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                seq_0 = "Test this method."
                seq_1 = "With these inputs."
                bbox = (1, 2, 3, 4)

                sequences, bbox_sequences = tokenizer.encode(
                    seq_0, bbox=bbox, text_pair=seq_1, bbox_pair=bbox, add_special_tokens=False
                )
                attached_sequences, attached_bbox_sequences = tokenizer.encode(
                    seq_0, bbox=bbox, text_pair=seq_1, bbox_pair=bbox, add_special_tokens=True
                )

                # Method is implemented (e.g. not GPT-2)
                if len(attached_sequences) != 2:
                    self.assertEqual(
                        tokenizer.num_special_tokens_to_add(pair=True), len(attached_sequences) - len(sequences)
                    )

    @require_torch
    @slow
    def test_torch_encode_plus_sent_to_model(self):
        import torch

        from transformers import MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    return

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    return

                model = model_class(config)

                # Make sure the model contains at least the full vocabulary size in its embedding matrix
                is_using_common_embeddings = hasattr(model.get_input_embeddings(), "weight")
                assert (
                    (model.get_input_embeddings().weight.shape[0] >= len(tokenizer))
                    if is_using_common_embeddings
                    else True
                )

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                bbox = (1, 2, 3, 4)
                encoded_sequence = tokenizer.encode_plus(sequence, bbox=bbox, return_tensors="pt")

                # Ensure that the BatchEncoding.to() method works.
                encoded_sequence.to(model.device)

                batch_encoded_sequence = tokenizer.batch_encode_plus(
                    [sequence, sequence], batch_bbox_or_bbox_pairs=[bbox, bbox], return_tensors="pt"
                )  # This should not fail

                with torch.no_grad():  # saves some time
                    model(**encoded_sequence)
                    model(**batch_encoded_sequence)

    @require_tf
    @slow
    def test_tf_encode_plus_sent_to_model(self):
        from transformers import TF_MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(TF_MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    return

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    return

                model = model_class(config)

                # Make sure the model contains at least the full vocabulary size in its embedding matrix
                self.assertGreaterEqual(model.config.vocab_size, len(tokenizer))

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                bbox = [(1, 2, 3, 4)] * len(sequence)
                encoded_sequence = tokenizer.encode_plus(sequence, bbox=bbox, return_tensors="tf")
                batch_encoded_sequence = tokenizer.batch_encode_plus(
                    [sequence, sequence], batch_bbox_or_bbox_pairs=[bbox, bbox], return_tensors="tf"
                )

                # This should not fail
                model(encoded_sequence)
                model(batch_encoded_sequence)

    # TODO: Check if require_torch is the best to test for numpy here ... Maybe move to require_flax when available
    @require_torch
    @slow
    def test_np_encode_plus_sent_to_model(self):
        from transformers import MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizer = self.get_tokenizer()
        if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
            return

        config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
        config = config_class()

        if config.is_encoder_decoder or config.pad_token_id is None:
            return

        # Build sequence
        first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
        sequence = " ".join(first_ten_tokens)
        bbox = (1, 2, 3, 4)
        encoded_sequence = tokenizer.encode_plus(sequence, bbox=bbox, return_tensors="np")
        batch_encoded_sequence = tokenizer.batch_encode_plus(
            [sequence, sequence], batch_bbox_or_bbox_pairs=[bbox, bbox], return_tensors="np"
        )

        # TODO: add forward through JAX/Flax when PR is merged
        # This is currently here to make flake8 happy !
        if encoded_sequence is None:
            raise ValueError("Cannot convert list to numpy tensor on  encode_plus()")

        if batch_encoded_sequence is None:
            raise ValueError("Cannot convert list to numpy tensor on  batch_encode_plus()")

        if self.test_rust_tokenizer:
            fast_tokenizer = self.get_rust_tokenizer()
            encoded_sequence_fast = fast_tokenizer.encode_plus(sequence, bbox=bbox, return_tensors="np")
            batch_encoded_sequence_fast = fast_tokenizer.batch_encode_plus(
                [sequence, sequence], batch_bbox_or_bbox_pairs=[bbox, bbox], return_tensors="np"
            )

            # TODO: add forward through JAX/Flax when PR is merged
            # This is currently here to make flake8 happy !
            if encoded_sequence_fast is None:
                raise ValueError("Cannot convert list to numpy tensor on  encode_plus() (fast)")

            if batch_encoded_sequence_fast is None:
                raise ValueError("Cannot convert list to numpy tensor on  batch_encode_plus() (fast)")

    def test_bbox_normalization(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                bbox = (1.2, 2.5, 3.6, 4.9)
                encoded_sequence, encoded_bbox_seq = tokenizer.encode(sequence, bbox=bbox, add_special_tokens=False)
                rounded_bbox_seq = [(1, 2, 3, 4)]
                self.assertEqual(encoded_bbox_seq, rounded_bbox_seq)
                encoded_sequence, encoded_bbox_seq = tokenizer.encode(
                    sequence, bbox=bbox, orig_width_and_height=(100, 100), add_special_tokens=False
                )
                # default target size is (1000, 1000) so the scaling from (100, 100) is equal to multiplying with 10
                scaled_bbox_seq = [(12, 25, 36, 49)]
                self.assertEqual(encoded_bbox_seq, scaled_bbox_seq)

    def test_non_matching_bbox(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                one_sequence = "Sequence"
                two_sequence = ["Another", "Sequence"]
                batch_sequence = [["Sequence"], ["Another Sequence"]]
                one_bbox = (1, 2, 3, 4)
                two_bbox = [(1, 2, 3, 4), (1, 2, 3, 4)]
                batch_bbox = [[(1, 2, 3, 4)], [(1, 2, 3, 4), (1, 2, 3, 4)]]
                # both, __call__ and encode should make sure the number of samples matches
                for mthd in (tokenizer, tokenizer.encode):
                    self.assertRaises(AssertionError, mthd, one_sequence, two_bbox)
                    self.assertRaises(AssertionError, mthd, one_sequence, batch_bbox)
                    self.assertRaises(AssertionError, mthd, two_sequence, one_bbox)
                    self.assertRaises(AssertionError, mthd, two_sequence, batch_bbox)
                    self.assertRaises(AssertionError, mthd, batch_sequence, one_bbox)
                    self.assertRaises(AssertionError, mthd, batch_sequence, two_sequence)
