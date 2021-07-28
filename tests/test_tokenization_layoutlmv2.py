# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
import os
import shutil
import tempfile
import unittest
from typing import List

from transformers import AddedToken
from transformers.models.layoutlmv2.tokenization_layoutlmv2 import (
    VOCAB_FILES_NAMES,
    BasicTokenizer,
    LayoutLMv2Tokenizer,
    WordpieceTokenizer,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)
from transformers.testing_utils import (
    is_pt_tf_cross_test,
    require_pandas,
    require_scatter,
    require_tokenizers,
    require_torch,
    slow,
)

from .test_tokenization_common import TokenizerTesterMixin, filter_non_english, merge_model_tokenizer_mappings


@require_tokenizers
@require_pandas
class LayoutLMv2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LayoutLMv2Tokenizer
    test_rust_tokenizer = False
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = False

    def get_words_and_boxes(self):
        words = ["a", "weirdly", "test"]
        boxes = [[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]]

        return words, boxes

    def get_words_and_boxes_batch(self):
        words = [["a", "weirdly", "test"], ["hello", "my", "name", "is", "bob"]]
        boxes = [
            [[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
            [[961, 885, 992, 912], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [34, 42, 66, 69]],
        ]

        return words, boxes

    def get_question_words_and_boxes(self):
        question = "what's his name?"
        words = ["a", "weirdly", "test"]
        boxes = [[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]]

        return question, words, boxes

    def get_question_words_and_boxes_batch(self):
        questions = ["what's his name?", "how is he called?"]
        words = [["a", "weirdly", "test"], ["what", "a", "laif", "gastn"]]
        boxes = [
            [[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
            [[256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [34, 42, 66, 69]],
        ]

        return questions, words, boxes

    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "what",
            "s",
            "his",
            "name",
            "?",
            "a",
            "weird",
            "##ly",
            "test",
            "lowest",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "UNwant\u00E9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        # With lower casing
        tokenizer = self.get_tokenizer(do_lower_case=True)
        rust_tokenizer = self.get_rust_tokenizer(do_lower_case=True)

        sequence = "UNwant\u00E9d,running"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"), ["ah", "\u535A", "\u63A8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["hello", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hällo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["h\u00E9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["HeLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HäLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HaLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"), ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"]
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("unwanted running"), ["un", "##want", "##ed", "runn", "##ing"])

        self.assertListEqual(tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_is_whitespace(self):
        self.assertTrue(_is_whitespace(" "))
        self.assertTrue(_is_whitespace("\t"))
        self.assertTrue(_is_whitespace("\r"))
        self.assertTrue(_is_whitespace("\n"))
        self.assertTrue(_is_whitespace("\u00A0"))

        self.assertFalse(_is_whitespace("A"))
        self.assertFalse(_is_whitespace("-"))

    def test_is_control(self):
        self.assertTrue(_is_control("\u0005"))

        self.assertFalse(_is_control("A"))
        self.assertFalse(_is_control(" "))
        self.assertFalse(_is_control("\t"))
        self.assertFalse(_is_control("\r"))

    def test_is_punctuation(self):
        self.assertTrue(_is_punctuation("-"))
        self.assertTrue(_is_punctuation("$"))
        self.assertTrue(_is_punctuation("`"))
        self.assertTrue(_is_punctuation("."))

        self.assertFalse(_is_punctuation("A"))
        self.assertFalse(_is_punctuation(" "))

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["Hello", "\xad", "hello"]], [["[UNK]"], [], ["[UNK]"]])

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("microsoft/layoutlmv2-base-uncased")

        question, words, boxes = self.get_question_words_and_boxes()

        text = tokenizer.encode(
            question.split(),
            boxes=[tokenizer.pad_token_box for _ in range(len(question.split()))],
            add_special_tokens=False,
        )
        text_2 = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)

        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_pair == [101] + text + [102] + text_2 + [102]

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                sentence = f"A, naïve {tokenizer_r.mask_token} AllenNLP sentence."
                tokens = tokenizer_r.encode_plus(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                do_lower_case = tokenizer_r.do_lower_case if hasattr(tokenizer_r, "do_lower_case") else False
                expected_results = (
                    [
                        ((0, 0), tokenizer_r.cls_token),
                        ((0, 1), "A"),
                        ((1, 2), ","),
                        ((3, 5), "na"),
                        ((5, 6), "##ï"),
                        ((6, 8), "##ve"),
                        ((9, 15), tokenizer_r.mask_token),
                        ((16, 21), "Allen"),
                        ((21, 23), "##NL"),
                        ((23, 24), "##P"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer_r.sep_token),
                    ]
                    if not do_lower_case
                    else [
                        ((0, 0), tokenizer_r.cls_token),
                        ((0, 1), "a"),
                        ((1, 2), ","),
                        ((3, 8), "naive"),
                        ((9, 15), tokenizer_r.mask_token),
                        ((16, 21), "allen"),
                        ((21, 23), "##nl"),
                        ((23, 24), "##p"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer_r.sep_token),
                    ]
                )

                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer_r.convert_ids_to_tokens(tokens["input_ids"])
                )
                self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])

    def test_add_special_tokens(self):
        tokenizers: List[LayoutLMv2Tokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                special_token = "[SPECIAL_TOKEN]"
                special_token_box = [1000, 1000, 1000, 1000]

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(
                    [special_token], boxes=[special_token_box], add_special_tokens=False
                )
                self.assertEqual(len(encoded_special_token), 1)

                decoded = tokenizer.decode(encoded_special_token, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_add_tokens_tokenizer(self):
        tokenizers: List[LayoutLMv2Tokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa", "bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                words = "aaaaa bbbbbb low cccccccccdddddddd l".split()
                boxes = ([1000, 1000, 1000, 1000] for _ in range(len(words)))

                tokens = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)

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

                words = ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l".split()
                boxes = ([1000, 1000, 1000, 1000] for _ in range(len(words)))

                tokens = tokenizer.encode(
                    words,
                    boxes=boxes,
                    add_special_tokens=False,
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    @require_tokenizers
    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()

                new_toks = [AddedToken("[ABC]", normalized=False), AddedToken("[DEF]", normalized=False)]
                tokenizer.add_tokens(new_toks)
                input = "[ABC][DEF][ABC][DEF]"
                if self.space_between_special_tokens:
                    output = "[ABC] [DEF] [ABC] [DEF]"
                else:
                    output = input
                encoded = tokenizer.encode(input.split(), boxes=boxes, add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    def test_encode_plus_with_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, words)

                padding_size = 10
                padding_idx = tokenizer.pad_token_id

                encoded_sequence = tokenizer.encode_plus(words, boxes=boxes, return_special_tokens_mask=True)
                input_ids = encoded_sequence["input_ids"]
                special_tokens_mask = encoded_sequence["special_tokens_mask"]
                sequence_length = len(input_ids)

                # Test 'longest' and 'no_padding' don't do anything
                tokenizer.padding_side = "right"

                not_padded_sequence = tokenizer.encode_plus(
                    words,
                    boxes=boxes,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                assert sequence_length == not_padded_sequence_length
                assert input_ids == not_padded_input_ids
                assert special_tokens_mask == not_padded_special_tokens_mask

                not_padded_sequence = tokenizer.encode_plus(
                    words,
                    boxes=boxes,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                assert sequence_length == not_padded_sequence_length
                assert input_ids == not_padded_input_ids
                assert special_tokens_mask == not_padded_special_tokens_mask

                # Test right padding
                tokenizer.padding_side = "right"

                right_padded_sequence = tokenizer.encode_plus(
                    words,
                    boxes=boxes,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                right_padded_input_ids = right_padded_sequence["input_ids"]

                right_padded_special_tokens_mask = right_padded_sequence["special_tokens_mask"]
                right_padded_sequence_length = len(right_padded_input_ids)

                assert sequence_length + padding_size == right_padded_sequence_length
                assert input_ids + [padding_idx] * padding_size == right_padded_input_ids
                assert special_tokens_mask + [1] * padding_size == right_padded_special_tokens_mask

                # Test left padding
                tokenizer.padding_side = "left"
                left_padded_sequence = tokenizer.encode_plus(
                    words,
                    boxes=boxes,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                assert sequence_length + padding_size == left_padded_sequence_length
                assert [padding_idx] * padding_size + input_ids == left_padded_input_ids
                assert [1] * padding_size + special_tokens_mask == left_padded_special_tokens_mask

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence["token_type_ids"]
                    right_padded_token_type_ids = right_padded_sequence["token_type_ids"]

                    assert token_type_ids + [0] * padding_size == right_padded_token_type_ids
                    assert [0] * padding_size + token_type_ids == left_padded_token_type_ids

                if "attention_mask" in tokenizer.model_input_names:
                    attention_mask = encoded_sequence["attention_mask"]
                    right_padded_attention_mask = right_padded_sequence["attention_mask"]
                    left_padded_attention_mask = left_padded_sequence["attention_mask"]

                    assert attention_mask + [0] * padding_size == right_padded_attention_mask
                    assert [0] * padding_size + attention_mask == left_padded_attention_mask

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()

                tokens = []
                for word in words:
                    tokens.extend(tokenizer.tokenize(word))
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                output_text = "a weirdly test"
                self.assertEqual(text_2, output_text)

    def test_mask_output(self):
        tokenizers = self.get_tokenizers(fast=False, do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()

                if (
                    tokenizer.build_inputs_with_special_tokens.__qualname__.split(".")[0] != "PreTrainedTokenizer"
                    and "token_type_ids" in tokenizer.model_input_names
                ):
                    information = tokenizer.encode_plus(words, boxes=boxes, add_special_tokens=True)
                    sequences, mask = information["input_ids"], information["token_type_ids"]
                    self.assertEqual(len(sequences), len(mask))

    def test_number_of_added_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                # test 1: single sequence
                words, boxes = self.get_words_and_boxes()

                sequences = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
                attached_sequences = tokenizer.encode(words, boxes=boxes, add_special_tokens=True)

                # Method is implemented (e.g. not GPT-2)
                if len(attached_sequences) != 2:
                    self.assertEqual(
                        tokenizer.num_special_tokens_to_add(pair=False), len(attached_sequences) - len(sequences)
                    )

                # test 2: two sequences
                question, words, boxes = self.get_question_words_and_boxes()

                sequences = tokenizer.encode(question, words, boxes, add_special_tokens=False)
                attached_sequences = tokenizer.encode(question, words, boxes, add_special_tokens=True)

                # Method is implemented (e.g. not GPT-2)
                if len(attached_sequences) != 2:
                    self.assertEqual(
                        tokenizer.num_special_tokens_to_add(pair=True), len(attached_sequences) - len(sequences)
                    )

    def test_padding_to_max_length(self):
        """We keep this test for backward compatibility but it should be removed when `pad_to_max_length` will be deprecated"""
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, words)

                padding_idx = tokenizer.pad_token_id

                # Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(words, boxes=boxes)
                sequence_length = len(encoded_sequence)
                # FIXME: the next line should be padding(max_length) to avoid warning
                padded_sequence = tokenizer.encode(
                    words, boxes=boxes, max_length=sequence_length + padding_size, pad_to_max_length=True
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence = tokenizer.encode(words, boxes=boxes)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(words, boxes=boxes, pad_to_max_length=True)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Test not batched
                words, boxes = self.get_words_and_boxes()
                encoded_sequences_1 = tokenizer.encode_plus(words, boxes=boxes)
                encoded_sequences_2 = tokenizer(words, boxes=boxes)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                question, words, boxes = self.get_question_words_and_boxes()
                encoded_sequences_1 = tokenizer.encode_plus(words, boxes=boxes)
                encoded_sequences_2 = tokenizer(words, boxes=boxes)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                words, boxes = self.get_words_and_boxes_batch()
                encoded_sequences_1 = tokenizer.batch_encode_plus(words, is_pair=False, boxes=boxes)
                encoded_sequences_2 = tokenizer(words, boxes=boxes)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes_batch()

                encoded_sequences = [
                    tokenizer.encode_plus(words_example, boxes=boxes_example)
                    for words_example, boxes_example in zip(words, boxes)
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(words, is_pair=False, boxes=boxes, padding=False)
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

                maximum_length = len(
                    max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len)
                )

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, words)

                encoded_sequences_padded = [
                    tokenizer.encode_plus(
                        words_example, boxes=boxes_example, max_length=maximum_length, padding="max_length"
                    )
                    for words_example, boxes_example in zip(words, boxes)
                ]

                encoded_sequences_batch_padded = tokenizer.batch_encode_plus(
                    words, is_pair=False, boxes=boxes, padding=True
                )
                self.assertListEqual(
                    encoded_sequences_padded,
                    self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(
                    words, is_pair=False, boxes=boxes, padding=True
                )
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    words, is_pair=False, boxes=boxes, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(
                    words, is_pair=False, boxes=boxes, padding=False
                )
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    words, is_pair=False, boxes=boxes, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

    @unittest.skip("batch_encode_plus does not handle overflowing tokens.")
    def test_batch_encode_plus_overflowing_tokens(self):
        pass

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus

        # Right padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes_batch()

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, words)

                encoded_sequences = [
                    tokenizer.encode_plus(
                        words_example, boxes=boxes_example, max_length=max_length, padding="max_length"
                    )
                    for words_example, boxes_example in zip(words, boxes)
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    words, is_pair=False, boxes=boxes, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

        # Left padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.padding_side = "left"
                words, boxes = self.get_words_and_boxes_batch()

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, words)

                encoded_sequences = [
                    tokenizer.encode_plus(
                        words_example, boxes=boxes_example, max_length=max_length, padding="max_length"
                    )
                    for words_example, boxes_example in zip(words, boxes)
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    words, is_pair=False, boxes=boxes, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                else:
                    words, boxes = self.get_words_and_boxes()

                    # empty_tokens = tokenizer([""], [[]], padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer(words, boxes=boxes, padding=True, pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        print(key, len(value))
                    # for key, value in empty_tokens.items():
                    #     self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    normal_tokens = tokenizer(words, boxes=boxes, pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # Should also work with truncation
                    normal_tokens = tokenizer(words, boxes=boxes, padding=True, truncation=True, pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # truncation to something which is not a multiple of pad_to_multiple_of raises an error
                    self.assertRaises(
                        ValueError,
                        tokenizer.__call__,
                        words,
                        boxes=boxes,
                        padding=True,
                        truncation=True,
                        max_length=12,
                        pad_to_multiple_of=8,
                    )

    def test_tokenizer_slow_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()
                encoded_sequence = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    words,
                    boxes=boxes,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    # add_prefix_space=False,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()
                # Testing single inputs
                encoded_sequence = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    words, boxes=boxes, add_special_tokens=True, return_special_tokens_mask=True
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [x for i, x in enumerate(encoded_sequence_w_special) if not special_tokens_mask[i]]
                self.assertEqual(encoded_sequence, filtered_sequence)

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
                words, boxes = self.get_words_and_boxes()
                tmpdirname = tempfile.mkdtemp()

                before_tokens = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

    def test_right_and_left_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(words, boxes=boxes)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    words, boxes=boxes, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(words, boxes=boxes)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    words, boxes=boxes, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert [padding_idx] * padding_size + encoded_sequence == padded_sequence

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
                encoded_sequence = tokenizer.encode(words, boxes=boxes)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(words, boxes=boxes, padding=True)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(words, boxes=boxes, padding="longest")
                padded_sequence_left_length = len(padded_sequence_left)
                assert sequence_length == padded_sequence_left_length
                assert encoded_sequence == padded_sequence_left

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(words, boxes=boxes)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(words, boxes=boxes, padding=False)
                padded_sequence_left_length = len(padded_sequence_left)
                assert sequence_length == padded_sequence_left_length
                assert encoded_sequence == padded_sequence_left

    def test_token_type_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                # test 1: single sequence
                words, boxes = self.get_words_and_boxes()

                output = tokenizer(words, boxes=boxes, return_token_type_ids=True)

                # Assert that the token type IDs have the same length as the input IDs
                self.assertEqual(len(output["token_type_ids"]), len(output["input_ids"]))

                # Assert that the token type IDs have the same length as the attention mask
                self.assertEqual(len(output["token_type_ids"]), len(output["attention_mask"]))

                self.assertIn(0, output["token_type_ids"])
                self.assertNotIn(1, output["token_type_ids"])

                # test 2: two sequences (question + words)
                question, words, boxes = self.get_question_words_and_boxes()

                output = tokenizer(question, words, boxes, return_token_type_ids=True)

                # Assert that the token type IDs have the same length as the input IDs
                self.assertEqual(len(output["token_type_ids"]), len(output["input_ids"]))

                # Assert that the token type IDs have the same length as the attention mask
                self.assertEqual(len(output["token_type_ids"]), len(output["attention_mask"]))

                self.assertIn(0, output["token_type_ids"])
                self.assertIn(1, output["token_type_ids"])

    @require_torch
    @slow
    @require_scatter
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
                words, boxes = self.get_words_and_boxes()
                encoded_sequence = tokenizer.encode_plus(words, boxes=boxes, return_tensors="pt")
                batch_encoded_sequence = tokenizer.batch_encode_plus(
                    [words, words], [boxes, boxes], return_tensors="pt"
                )
                # This should not fail

                with torch.no_grad():  # saves some time
                    model(**encoded_sequence)
                    model(**batch_encoded_sequence)

    @slow
    def test_layoutlmv2_truncation_integration_test(self):
        words, boxes = self.get_words_and_boxes()

        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased", model_max_length=512)

        for i in range(12, 512):
            new_encoded_inputs = tokenizer.encode(words, boxes=boxes, max_length=i, truncation=True)

            # Ensure that the input IDs are less than the max length defined.
            self.assertLessEqual(len(new_encoded_inputs), i)

        tokenizer.model_max_length = 20
        new_encoded_inputs = tokenizer.encode(words, boxes=boxes, truncation=True)
        dropped_encoded_inputs = tokenizer.encode(words, boxes=boxes, truncation=True)

        # Ensure that the input IDs are still truncated when no max_length is specified
        self.assertListEqual(new_encoded_inputs, dropped_encoded_inputs)
        self.assertLessEqual(len(new_encoded_inputs), 20)

    @is_pt_tf_cross_test
    def test_batch_encode_plus_tensors(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()

                # A Tensor cannot be build by sequences which are not the same size
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, words, boxes=boxes, return_tensors="pt")
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, words, boxes=boxes, return_tensors="tf")

                if tokenizer.pad_token_id is None:
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        words,
                        boxes=boxes,
                        padding=True,
                        return_tensors="pt",
                    )
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        words,
                        boxes=boxes,
                        padding="longest",
                        return_tensors="tf",
                    )
                else:
                    pytorch_tensor = tokenizer.batch_encode_plus(words, boxes=boxes, padding=True, return_tensors="pt")
                    tensorflow_tensor = tokenizer.batch_encode_plus(
                        words, boxes=boxes, padding="longest", return_tensors="tf"
                    )
                    encoded_sequences = tokenizer.batch_encode_plus(words, boxes=boxes, padding=True)

                    for key in encoded_sequences.keys():
                        pytorch_value = pytorch_tensor[key].tolist()
                        tensorflow_value = tensorflow_tensor[key].numpy().tolist()
                        encoded_value = encoded_sequences[key]

                        self.assertEqual(pytorch_value, tensorflow_value, encoded_value)

    def test_prepare_for_model(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()
                prepared_input_dict = tokenizer.prepare_for_model(words, boxes=boxes, add_special_tokens=True)

                input_dict = tokenizer.encode_plus(words, boxes=boxes, add_special_tokens=True)

                self.assertEqual(input_dict, prepared_input_dict)

    @unittest.skip("LayoutLMv2 tokenizer requires boxes besides sequences.")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip("LayoutLMv2 tokenizer requires boxes besides sequences.")
    def test_maximum_encoding_length_single_input(self):
        pass

    @unittest.skip("LayoutLMv2 tokenizer requires boxes besides sequences.")
    def test_pretokenized_inputs(self):
        pass

    @slow
    def test_layoutlmv2_integration_test(self):

        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased", model_max_length=512)

        # There are 3 cases:
        # CASE 1: document image classification (training + inference), document image token classification (inference),
        # in which case only words and normalized bounding boxes are provided to the tokenizer
        # CASE 2: document image token classification (training),
        # in which case one also provides word labels to the tokenizer
        # CASE 3: document image visual question answering (inference),
        # in which case one also provides a question to the tokenizer

        # We need to test all 3 cases both on batched and non-batched inputs.

        # CASE 1: not batched
        words, boxes = self.get_words_and_boxes()

        # fmt: off
        expected_results = {'input_ids': [101, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bbox': [[0, 0, 0, 0], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}  # noqa: E231
        # fmt: on

        encoding = tokenizer(words, boxes=boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding), expected_results)

        # CASE 1: batched
        words, boxes = self.get_words_and_boxes_batch()

        # fmt: off
        expected_results = {'input_ids': [[101, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 7592, 2026, 2171, 2003, 3960, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'bbox': [[[0, 0, 0, 0], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [961, 885, 992, 912], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [34, 42, 66, 69], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E231
        # fmt: on

        encoding = tokenizer(words, boxes=boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding), expected_results)

        # CASE 2: not batched
        words, boxes = self.get_words_and_boxes()
        word_labels = [1, 2, 3]

        # fmt: off
        expected_results = {'input_ids': [101, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bbox': [[0, 0, 0, 0], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [-100, 1, 2, -100, 3, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100], 'attention_mask': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}  # noqa: E231
        # fmt: on

        encoding = tokenizer(words, boxes=boxes, word_labels=word_labels, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding), expected_results)

        # CASE 2: batched
        words, boxes = self.get_words_and_boxes_batch()
        word_labels = [[1, 2, 3], [46, 17, 22, 3]]

        # fmt: off
        expected_results = {'input_ids': [[101, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 7592, 2026, 2171, 2003, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'bbox': [[[0, 0, 0, 0], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [961, 885, 992, 912], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'labels': [[-100, 1, 2, -100, 3, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100], [-100, 46, 17, 22, 3, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E231
        # fmt: on

        encoding = tokenizer(words, boxes=boxes, word_labels=word_labels, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding), expected_results)

        # CASE 3: not batched
        question, words, boxes = self.get_question_words_and_boxes()

        # fmt: off
        expected_results = {'input_ids': [101, 2054, 1005, 1055, 2010, 2171, 1029, 102, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0], 'bbox': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}  # noqa: E231
        # fmt: on

        encoding = tokenizer(question, words, boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding), expected_results)

        # CASE 3: batched
        questions, words, boxes = self.get_question_words_and_boxes_batch()

        # fmt: off
        expected_results = {'input_ids': [[101, 2054, 1005, 1055, 2010, 2171, 1029, 102, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 2003, 2002, 2170, 1029, 102, 2054, 1037, 21110, 2546, 3806, 2102, 2078, 102, 0, 0, 0, 0, 0]], 'bbox': [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [336, 42, 353, 57], [34, 42, 66, 69], [34, 42, 66, 69], [34, 42, 66, 69], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]}  # noqa: E231
        # fmt: on

        encoding = tokenizer(questions, words, boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding), expected_results)

    @unittest.skip("Doesn't support another framework than PyTorch")
    def test_np_encode_plus_sent_to_model(self):
        pass
