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
import re
import shutil
import tempfile
import unittest
from typing import List

from transformers import AddedToken, LayoutLMv2TokenizerFast, SpecialTokensMixin, is_tf_available, is_torch_available
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

from .test_tokenization_common import (
    SMALL_TRAINING_CORPUS,
    TokenizerTesterMixin,
    filter_non_english,
    merge_model_tokenizer_mappings,
)


@require_tokenizers
@require_pandas
class LayoutLMv2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LayoutLMv2Tokenizer
    rust_tokenizer_class = LayoutLMv2TokenizerFast
    test_rust_tokenizer = True
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

                words, boxes = self.get_words_and_boxes()
                words[1] = tokenizer_r.mask_token
                tokens = tokenizer_r.encode_plus(
                    words,
                    boxes=boxes,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                expected_results = [
                    ((0, 0), tokenizer_r.cls_token),
                    ((0, 1), "a"),
                    ((0, 6), tokenizer_r.mask_token),
                    ((0, 4), "test"),
                    ((0, 0), tokenizer_r.sep_token),
                ]

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
                boxes = [[1000, 1000, 1000, 1000] for _ in range(len(words))]

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
                boxes = [[1000, 1000, 1000, 1000] for _ in range(len(words))]

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

    @unittest.skip("Not implemented")
    def test_right_and_left_truncation(self):
        pass

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

                self.assertTrue(sequence_length == not_padded_sequence_length)
                self.assertTrue(input_ids == not_padded_input_ids)
                self.assertTrue(special_tokens_mask == not_padded_special_tokens_mask)

                not_padded_sequence = tokenizer.encode_plus(
                    words,
                    boxes=boxes,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertTrue(sequence_length == not_padded_sequence_length)
                self.assertTrue(input_ids == not_padded_input_ids)
                self.assertTrue(special_tokens_mask == not_padded_special_tokens_mask)

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

                self.assertTrue(sequence_length + padding_size == right_padded_sequence_length)
                self.assertTrue(input_ids + [padding_idx] * padding_size == right_padded_input_ids)
                self.assertTrue(special_tokens_mask + [1] * padding_size == right_padded_special_tokens_mask)

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

                self.assertTrue(sequence_length + padding_size == left_padded_sequence_length)
                self.assertTrue([padding_idx] * padding_size + input_ids == left_padded_input_ids)
                self.assertTrue([1] * padding_size + special_tokens_mask == left_padded_special_tokens_mask)

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

                    self.assertTrue(attention_mask + [0] * padding_size == right_padded_attention_mask)
                    self.assertTrue([0] * padding_size + attention_mask == left_padded_attention_mask)

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

                sequences = tokenizer.encode(question, words, boxes=boxes, add_special_tokens=False)
                attached_sequences = tokenizer.encode(question, words, boxes=boxes, add_special_tokens=True)

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

    def test_padding(self, max_length=50):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                self.assertEqual(tokenizer_p.pad_token_id, tokenizer_r.pad_token_id)
                pad_token_id = tokenizer_p.pad_token_id

                # Encode - Simple input
                words, boxes = self.get_words_and_boxes()
                input_r = tokenizer_r.encode(words, boxes=boxes, max_length=max_length, pad_to_max_length=True)
                input_p = tokenizer_p.encode(words, boxes=boxes, max_length=max_length, pad_to_max_length=True)
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(words, boxes=boxes, max_length=max_length, padding="max_length")
                input_p = tokenizer_p.encode(words, boxes=boxes, max_length=max_length, padding="max_length")
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.encode(words, boxes=boxes, padding="longest")
                input_p = tokenizer_p.encode(words, boxes=boxes, padding=True)
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode - Pair input
                question, words, boxes = self.get_question_words_and_boxes()
                input_r = tokenizer_r.encode(
                    question, words, boxes=boxes, max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode(
                    question, words, boxes=boxes, max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(question, words, boxes=boxes, max_length=max_length, padding="max_length")
                input_p = tokenizer_p.encode(question, words, boxes=boxes, max_length=max_length, padding="max_length")
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(question, words, boxes=boxes, padding=True)
                input_p = tokenizer_p.encode(question, words, boxes=boxes, padding="longest")
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode_plus - Simple input
                words, boxes = self.get_words_and_boxes()
                input_r = tokenizer_r.encode_plus(words, boxes=boxes, max_length=max_length, pad_to_max_length=True)
                input_p = tokenizer_p.encode_plus(words, boxes=boxes, max_length=max_length, pad_to_max_length=True)
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(words, boxes=boxes, max_length=max_length, padding="max_length")
                input_p = tokenizer_p.encode_plus(words, boxes=boxes, max_length=max_length, padding="max_length")
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                input_r = tokenizer_r.encode_plus(words, boxes=boxes, padding="longest")
                input_p = tokenizer_p.encode_plus(words, boxes=boxes, padding=True)
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                # Encode_plus - Pair input
                question, words, boxes = self.get_question_words_and_boxes()
                input_r = tokenizer_r.encode_plus(
                    question, words, boxes=boxes, max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode_plus(
                    question, words, boxes=boxes, max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(
                    question, words, boxes=boxes, max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode_plus(
                    question, words, boxes=boxes, max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(question, words, boxes=boxes, padding="longest")
                input_p = tokenizer_p.encode_plus(question, words, boxes=boxes, padding=True)
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                # Batch_encode_plus - Simple input
                words, boxes = self.get_words_and_boxes_batch()

                input_r = tokenizer_r.batch_encode_plus(
                    words,
                    boxes=boxes,
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                input_p = tokenizer_p.batch_encode_plus(
                    words,
                    boxes=boxes,
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    words,
                    boxes=boxes,
                    max_length=max_length,
                    padding="max_length",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    words,
                    boxes=boxes,
                    max_length=max_length,
                    padding="max_length",
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    words,
                    boxes=boxes,
                    max_length=max_length,
                    padding="longest",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    words,
                    boxes=boxes,
                    max_length=max_length,
                    padding=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(words, boxes=boxes, padding="longest")
                input_p = tokenizer_p.batch_encode_plus(words, boxes=boxes, padding=True)
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Batch_encode_plus - Pair input
                questions, words, boxes = self.get_question_words_and_boxes_batch()

                input_r = tokenizer_r.batch_encode_plus(
                    list(zip(questions, words)),
                    is_pair=True,
                    boxes=boxes,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    list(zip(questions, words)),
                    is_pair=True,
                    boxes=boxes,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    list(zip(questions, words)),
                    is_pair=True,
                    boxes=boxes,
                    padding=True,
                )
                input_p = tokenizer_p.batch_encode_plus(
                    list(zip(questions, words)),
                    is_pair=True,
                    boxes=boxes,
                    padding="longest",
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad on single examples after tokenization
                words, boxes = self.get_words_and_boxes()
                input_r = tokenizer_r.encode_plus(words, boxes=boxes)
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.encode_plus(words, boxes=boxes)
                input_p = tokenizer_r.pad(input_p)

                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                # Using pad on single examples after tokenization
                input_r = tokenizer_r.encode_plus(words, boxes=boxes)
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.encode_plus(words, boxes=boxes)
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                # Using pad after tokenization
                words, boxes = self.get_words_and_boxes_batch()
                input_r = tokenizer_r.batch_encode_plus(
                    words,
                    boxes=boxes,
                )
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.batch_encode_plus(
                    words,
                    boxes=boxes,
                )
                input_p = tokenizer_r.pad(input_p)

                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad after tokenization
                words, boxes = self.get_words_and_boxes_batch()
                input_r = tokenizer_r.batch_encode_plus(
                    words,
                    boxes=boxes,
                )
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.batch_encode_plus(
                    words,
                    boxes=boxes,
                )
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

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

    def test_build_inputs_with_special_tokens(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                # Input tokens id
                words, boxes = self.get_words_and_boxes()
                input_simple = tokenizer_p.encode(words, boxes=boxes, add_special_tokens=False)
                input_pair = tokenizer_p.encode(words, boxes=boxes, add_special_tokens=False)

                # Generate output
                output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple)
                output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple)
                self.assertEqual(output_p, output_r)

                # Generate pair output
                output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple, input_pair)
                output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple, input_pair)
                self.assertEqual(output_p, output_r)

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

    def test_offsets_mapping(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                text = ["a", "wonderful", "test"]
                boxes = [[1, 8, 12, 20] for _ in range(len(text))]

                # No pair
                tokens_with_offsets = tokenizer_r.encode_plus(
                    text,
                    boxes=boxes,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )
                added_tokens = tokenizer_r.num_special_tokens_to_add(False)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

                # Pairs
                text = "what's his name"
                pair = ["a", "wonderful", "test"]
                boxes = [[1, 8, 12, 20] for _ in range(len(pair))]
                tokens_with_offsets = tokenizer_r.encode_plus(
                    text,
                    pair,
                    boxes=boxes,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )
                added_tokens = tokenizer_r.num_special_tokens_to_add(True)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

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

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        words, boxes = self.get_words_and_boxes()

        ids = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(words, boxes=boxes, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        ids = tokenizer.encode(words, boxes=boxes, add_special_tokens=True)
        rust_ids = rust_tokenizer.encode(words, boxes=boxes, add_special_tokens=True)
        self.assertListEqual(ids, rust_ids)

    def test_tokenization_python_rust_equals(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                words, boxes = self.get_words_and_boxes()

                # Ensure basic input match
                input_p = tokenizer_p.encode_plus(words, boxes=boxes)
                input_r = tokenizer_r.encode_plus(words, boxes=boxes)

                for key in filter(
                    lambda x: x in ["input_ids", "token_type_ids", "attention_mask", "bbox"], input_p.keys()
                ):
                    self.assertSequenceEqual(input_p[key], input_r[key])

                input_pairs_p = tokenizer_p.encode_plus(words, boxes=boxes)
                input_pairs_r = tokenizer_r.encode_plus(words, boxes=boxes)

                for key in filter(
                    lambda x: x in ["input_ids", "token_type_ids", "attention_mask", "bbox"], input_p.keys()
                ):
                    self.assertSequenceEqual(input_pairs_p[key], input_pairs_r[key])

                words = ["hello" for _ in range(1000)]
                boxes = [[1000, 1000, 1000, 1000] for _ in range(1000)]

                # Ensure truncation match
                input_p = tokenizer_p.encode_plus(words, boxes=boxes, max_length=512, truncation=True)
                input_r = tokenizer_r.encode_plus(words, boxes=boxes, max_length=512, truncation=True)

                for key in filter(
                    lambda x: x in ["input_ids", "token_type_ids", "attention_mask", "bbox"], input_p.keys()
                ):
                    self.assertSequenceEqual(input_p[key], input_r[key])

                # Ensure truncation with stride match
                input_p = tokenizer_p.encode_plus(
                    words, boxes=boxes, max_length=512, truncation=True, stride=3, return_overflowing_tokens=True
                )
                input_r = tokenizer_r.encode_plus(
                    words, boxes=boxes, max_length=512, truncation=True, stride=3, return_overflowing_tokens=True
                )

                for key in filter(
                    lambda x: x in ["input_ids", "token_type_ids", "attention_mask", "bbox"], input_p.keys()
                ):
                    self.assertSequenceEqual(input_p[key], input_r[key][0])

    def test_embeded_special_tokens(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                words, boxes = self.get_words_and_boxes()
                tokens_r = tokenizer_r.encode_plus(
                    words,
                    boxes=boxes,
                    add_special_tokens=True,
                )
                tokens_p = tokenizer_p.encode_plus(
                    words,
                    boxes=boxes,
                    add_special_tokens=True,
                )

                for key in tokens_p.keys():
                    self.assertEqual(tokens_r[key], tokens_p[key])

                if "token_type_ids" in tokens_r:
                    self.assertEqual(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

                tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
                tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
                self.assertSequenceEqual(tokens_r, tokens_p)

    def test_compare_add_special_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                simple_num_special_tokens_to_add = tokenizer_r.num_special_tokens_to_add(pair=False)

                words, boxes = self.get_words_and_boxes()
                # tokenize()
                no_special_tokens = tokenizer_r.tokenize(" ".join(words), add_special_tokens=False)
                with_special_tokens = tokenizer_r.tokenize(" ".join(words), add_special_tokens=True)
                self.assertEqual(len(no_special_tokens), len(with_special_tokens) - simple_num_special_tokens_to_add)

                # encode()
                no_special_tokens = tokenizer_r.encode(words, boxes=boxes, add_special_tokens=False)
                with_special_tokens = tokenizer_r.encode(words, boxes=boxes, add_special_tokens=True)
                self.assertEqual(len(no_special_tokens), len(with_special_tokens) - simple_num_special_tokens_to_add)

                # encode_plus()
                no_special_tokens = tokenizer_r.encode_plus(words, boxes=boxes, add_special_tokens=False)
                with_special_tokens = tokenizer_r.encode_plus(words, boxes=boxes, add_special_tokens=True)
                for key in no_special_tokens.keys():
                    self.assertEqual(
                        len(no_special_tokens[key]),
                        len(with_special_tokens[key]) - simple_num_special_tokens_to_add,
                    )

                # # batch_encode_plus
                words, boxes = self.get_words_and_boxes_batch()

                no_special_tokens = tokenizer_r.batch_encode_plus(words, boxes=boxes, add_special_tokens=False)
                with_special_tokens = tokenizer_r.batch_encode_plus(words, boxes=boxes, add_special_tokens=True)
                for key in no_special_tokens.keys():
                    for i_no, i_with in zip(no_special_tokens[key], with_special_tokens[key]):
                        self.assertEqual(len(i_no), len(i_with) - simple_num_special_tokens_to_add)

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
                words, boxes = self.get_words_and_boxes_batch()

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

    def test_sequence_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            if not tokenizer.is_fast:
                continue
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0 = "Test this method."
                seq_1 = ["With", "these", "inputs."]
                boxes = [[1000, 1000, 1000, 1000] for _ in range(len(seq_1))]

                # We want to have sequence 0 and sequence 1 are tagged
                # respectively with 0 and 1 token_ids
                # (regardless of whether the model use token type ids)
                # We use this assumption in the QA pipeline among other place
                output = tokenizer(seq_0.split(), boxes=boxes)
                self.assertIn(0, output.sequence_ids())

                output = tokenizer(seq_0, seq_1, boxes=boxes)
                self.assertIn(0, output.sequence_ids())
                self.assertIn(1, output.sequence_ids())

                if tokenizer.num_special_tokens_to_add(pair=True):
                    self.assertIn(None, output.sequence_ids())

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                words = "Hey this is a <special> token".split()
                boxes = [[1000, 1000, 1000, 1000] for _ in range(len(words))]
                r_output = tokenizer_r.encode(words, boxes=boxes)

                special_token_id = tokenizer_r.encode(
                    ["<special>"], boxes=[1000, 1000, 1000, 1000], add_special_tokens=False
                )[0]

                self.assertTrue(special_token_id in r_output)

                if self.test_slow_tokenizer:
                    tokenizer_cr = self.rust_tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs, from_slow=True
                    )
                    tokenizer_p = self.tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs
                    )

                    words = "Hey this is a <special> token".split()
                    boxes = [[1000, 1000, 1000, 1000] for _ in range(len(words))]

                    p_output = tokenizer_p.encode(words, boxes=boxes)
                    cr_output = tokenizer_cr.encode(words, boxes=boxes)

                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)

    def test_training_new_tokenizer(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_rust_tokenizer()
        new_tokenizer = tokenizer.train_new_from_iterator(SMALL_TRAINING_CORPUS, 100)

        # Test we can use the new tokenizer with something not seen during training
        text = [["this", "is", "the"], ["how", "are", "you"]]
        boxes = [[[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 4, 8]], [[5, 6, 7, 8], [4, 5, 6, 7], [3, 9, 2, 7]]]
        inputs = new_tokenizer(text, boxes=boxes)
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "this is the"

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
                    (
                        f"'{new_special_token_str}' doesn't appear in the list "
                        f"'{new_tokenizer.all_special_tokens_extended}' as an AddedToken with the same parameters as "
                        f"'{special_token}' in the list {tokenizer.all_special_tokens_extended}"
                    ),
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
        words = [["this", "is"], ["hello", "🤗"]]
        boxes = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]]
        inputs = new_tokenizer(words, boxes=boxes)
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "this is"

        if tokenizer.backend_tokenizer.normalizer is not None:
            expected_result = tokenizer.backend_tokenizer.normalizer.normalize_str(expected_result)
        self.assertEqual(expected_result, decoded_input)

    def test_prepare_for_model(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            # only test prepare_for_model for the slow tokenizer
            if tokenizer.__class__.__name__ == "LayoutLMv2TokenizerFast":
                continue
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                words, boxes = self.get_words_and_boxes()
                prepared_input_dict = tokenizer.prepare_for_model(words, boxes=boxes, add_special_tokens=True)

                input_dict = tokenizer.encode_plus(words, boxes=boxes, add_special_tokens=True)

                self.assertEqual(input_dict, prepared_input_dict)

    def test_padding_different_model_input_name(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                self.assertEqual(tokenizer_p.pad_token_id, tokenizer_r.pad_token_id)
                pad_token_id = tokenizer_p.pad_token_id

                words, boxes = self.get_words_and_boxes_batch()

                input_r = tokenizer_r.batch_encode_plus(words, boxes=boxes)
                input_p = tokenizer_r.batch_encode_plus(words, boxes=boxes)

                # rename encoded batch to "inputs"
                input_r["inputs"] = input_r[tokenizer_r.model_input_names[0]]
                del input_r[tokenizer_r.model_input_names[0]]

                input_p["inputs"] = input_p[tokenizer_p.model_input_names[0]]
                del input_p[tokenizer_p.model_input_names[0]]

                # Renaming `input_ids` to `inputs`
                tokenizer_r.model_input_names = ["inputs"] + tokenizer_r.model_input_names[1:]
                tokenizer_p.model_input_names = ["inputs"] + tokenizer_p.model_input_names[1:]

                input_r = tokenizer_r.pad(input_r, padding="longest")
                input_p = tokenizer_r.pad(input_p, padding="longest")

                max_length = len(input_p["inputs"][0])
                self.assert_batch_padded_input_match(
                    input_r, input_p, max_length, pad_token_id, model_main_input_name="inputs"
                )

    def test_batch_encode_dynamic_overflowing(self):
        """
        When calling batch_encode with multiple sequences, it can return different number of
        overflowing encoding for each sequence:
        [
          Sequence 1: [Encoding 1, Encoding 2],
          Sequence 2: [Encoding 1],
          Sequence 3: [Encoding 1, Encoding 2, ... Encoding N]
        ]
        This needs to be padded so that it can represented as a tensor
        """
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name}, {tokenizer.__class__.__name__})"):

                if is_torch_available():
                    returned_tensor = "pt"
                elif is_tf_available():
                    returned_tensor = "tf"
                else:
                    returned_tensor = "jax"

                # Single example
                words, boxes = self.get_words_and_boxes()
                tokens = tokenizer.encode_plus(
                    words,
                    boxes=boxes,
                    max_length=6,
                    padding=True,
                    truncation=True,
                    return_tensors=returned_tensor,
                    return_overflowing_tokens=True,
                )

                for key in filter(lambda x: "overflow_to_sample_mapping" not in x, tokens.keys()):
                    if key != "bbox":
                        self.assertEqual(len(tokens[key].shape), 2)
                    else:
                        self.assertEqual(len(tokens[key].shape), 3)

                # Batch of examples
                # For these 2 examples, 3 training examples will be created
                words, boxes = self.get_words_and_boxes_batch()
                tokens = tokenizer.batch_encode_plus(
                    words,
                    boxes=boxes,
                    max_length=6,
                    padding=True,
                    truncation="only_first",
                    return_tensors=returned_tensor,
                    return_overflowing_tokens=True,
                )

                for key in filter(lambda x: "overflow_to_sample_mapping" not in x, tokens.keys()):
                    if key != "bbox":
                        self.assertEqual(len(tokens[key].shape), 2)
                        self.assertEqual(tokens[key].shape[-1], 6)
                    else:
                        self.assertEqual(len(tokens[key].shape), 3)
                        self.assertEqual(tokens[key].shape[-1], 4)

    @unittest.skip("TO DO: overwrite this very extensive test.")
    def test_alignement_methods(self):
        pass

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        toks = [(i, tokenizer.decode([i], clean_up_tokenization_spaces=False)) for i in range(len(tokenizer))]
        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(
            filter(
                lambda t: [t[0]]
                == tokenizer.encode(t[1].split(" "), boxes=len(t[1]) * [[1, 1, 1, 1]], add_special_tokens=False),
                toks,
            )
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
        words = output_txt.split(" ")
        boxes = [[i, i, i, i] for i in range(len(words))]
        output_ids = tokenizer.encode(words, boxes=boxes, add_special_tokens=False)

        return words, boxes, output_ids

    # @unittest.skip("LayoutLMv2 tokenizer requires boxes besides sequences.")
    def test_maximum_encoding_length_pair_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Build a sequence from our model's vocabulary
                stride = 2
                seq_0, boxes_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                question_0 = " ".join(map(str, seq_0))
                if len(ids) <= 2 + stride:
                    seq_0 = (seq_0 + " ") * (2 + stride)
                    ids = None

                seq0_tokens = tokenizer(seq_0, boxes=boxes_0, add_special_tokens=False)
                self.assertGreater(len(seq0_tokens["input_ids"]), 2 + stride)
                question_1 = "This is another sentence to be encoded."
                seq_1 = ["what", "a", "weird", "test", "weirdly", "weird"]
                boxes_1 = [[i, i, i, i] for i in range(len(seq_1))]
                seq1_tokens = tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)
                if abs(len(seq0_tokens["input_ids"]) - len(seq1_tokens["input_ids"])) <= 2:
                    seq1_tokens_input_ids = seq1_tokens["input_ids"] + seq1_tokens["input_ids"]
                    seq_1 = tokenizer.decode(seq1_tokens_input_ids, clean_up_tokenization_spaces=False)
                    seq_1 = seq_1.split(" ")
                    boxes_1 = [[i, i, i, i] for i in range(len(seq_1))]
                seq1_tokens = tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)

                self.assertGreater(len(seq1_tokens["input_ids"]), 2 + stride)

                smallest = (
                    seq1_tokens["input_ids"]
                    if len(seq0_tokens["input_ids"]) > len(seq1_tokens["input_ids"])
                    else seq0_tokens["input_ids"]
                )

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence = tokenizer(
                    question_0, seq_1, boxes=boxes_1, add_special_tokens=False
                )  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                question_2 = " ".join(map(str, seq_2))
                boxes_2 = boxes_0 * model_max_length
                self.assertGreater(len(seq_2), model_max_length)

                sequence1 = tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer(question_2, seq_1, boxes=boxes_1, add_special_tokens=False)
                total_length2 = len(sequence2["input_ids"])
                self.assertLess(total_length1, model_max_length, "Issue with the testing sequence, please update it.")
                self.assertGreater(
                    total_length2, model_max_length, "Issue with the testing sequence, please update it."
                )

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"{tokenizer.__class__.__name__} Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"{tokenizer.__class__.__name__} Truncation: {truncation_state}"):
                                output = tokenizer(
                                    question_2,
                                    seq_1,
                                    boxes=boxes_1,
                                    padding=padding_state,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"]), model_max_length)
                                self.assertEqual(len(output["bbox"]), model_max_length)

                                output = tokenizer(
                                    [question_2],
                                    [seq_1],
                                    boxes=[boxes_1],
                                    padding=padding_state,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)
                                self.assertEqual(len(output["bbox"][0]), model_max_length)

                        # Simple
                        output = tokenizer(
                            question_1, seq_2, boxes=boxes_2, padding=padding_state, truncation="only_second"
                        )
                        self.assertEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(output["bbox"]), model_max_length)

                        output = tokenizer(
                            [question_1], [seq_2], boxes=[boxes_2], padding=padding_state, truncation="only_second"
                        )
                        self.assertEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(output["bbox"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(
                                question_1, seq_2, boxes=boxes_2, padding=padding_state, truncation=False
                            )
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
                            output = tokenizer(
                                [question_1], [seq_2], boxes=[boxes_2], padding=padding_state, truncation=False
                            )
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                            self.assertNotEqual(len(output["bbox"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )
                # Check the order of Sequence of input ids, overflowing tokens and bbox sequence with truncation
                truncated_first_sequence = (
                    tokenizer(seq_0, boxes=boxes_0, add_special_tokens=False)["input_ids"][:-2]
                    + tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)["input_ids"]
                )
                truncated_second_sequence = (
                    tokenizer(seq_0, boxes=boxes_0, add_special_tokens=False)["input_ids"]
                    + tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)["input_ids"][:-2]
                )
                truncated_longest_sequence = (
                    truncated_first_sequence if len(seq0_tokens) > len(seq1_tokens) else truncated_second_sequence
                )

                overflow_first_sequence = (
                    tokenizer(seq_0, boxes=boxes_0, add_special_tokens=False)["input_ids"][-(2 + stride) :]
                    + tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)["input_ids"]
                )
                overflow_second_sequence = (
                    tokenizer(seq_0, boxes=boxes_0, add_special_tokens=False)["input_ids"]
                    + tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)["input_ids"][-(2 + stride) :]
                )
                overflow_longest_sequence = (
                    overflow_first_sequence if len(seq0_tokens) > len(seq1_tokens) else overflow_second_sequence
                )

                bbox_first = [[0, 0, 0, 0]] * (len(seq_0) - 2)
                bbox_first_sequence = bbox_first + tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)["bbox"]
                overflowing_token_bbox_first_sequence_slow = [[0, 0, 0, 0]] * (2 + stride)
                overflowing_token_bbox_first_sequence_fast = [[0, 0, 0, 0]] * (2 + stride) + tokenizer(
                    seq_1, boxes=boxes_1, add_special_tokens=False
                )["bbox"]

                bbox_second = [[0, 0, 0, 0]] * len(seq_0)
                bbox_second_sequence = (
                    bbox_second + tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)["bbox"][:-2]
                )
                overflowing_token_bbox_second_sequence_slow = tokenizer(
                    seq_1, boxes=boxes_1, add_special_tokens=False
                )["bbox"][-(2 + stride) :]
                overflowing_token_bbox_second_sequence_fast = [[0, 0, 0, 0]] * len(seq_0) + tokenizer(
                    seq_1, boxes=boxes_1, add_special_tokens=False
                )["bbox"][-(2 + stride) :]

                bbox_longest_sequence = (
                    bbox_first_sequence if len(seq0_tokens) > len(seq1_tokens) else bbox_second_sequence
                )
                overflowing_token_bbox_longest_sequence_fast = (
                    overflowing_token_bbox_first_sequence_fast
                    if len(seq0_tokens) > len(seq1_tokens)
                    else overflowing_token_bbox_second_sequence_fast
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, LayoutLMv2TokenizerFast):
                    information = tokenizer(
                        question_0,
                        seq_1,
                        boxes=boxes_1,
                        max_length=len(sequence["input_ids"]) - 2,
                        add_special_tokens=False,
                        stride=stride,
                        truncation="longest_first",
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    bbox = information["bbox"][0]
                    overflowing_bbox = information["bbox"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence["input_ids"]) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                    self.assertEqual(bbox, bbox_longest_sequence)

                    self.assertEqual(len(overflowing_bbox), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_bbox, overflowing_token_bbox_longest_sequence_fast)
                else:
                    # No overflowing tokens when using 'longest' in python tokenizers
                    with self.assertRaises(ValueError) as context:
                        information = tokenizer(
                            question_0,
                            seq_1,
                            boxes=boxes_1,
                            max_length=len(sequence["input_ids"]) - 2,
                            add_special_tokens=False,
                            stride=stride,
                            truncation="longest_first",
                            return_overflowing_tokens=True,
                            # add_prefix_space=False,
                        )

                    self.assertTrue(
                        context.exception.args[0].startswith(
                            "Not possible to return overflowing tokens for pair of sequences with the "
                            "`longest_first`. Please select another truncation strategy than `longest_first`, "
                            "for instance `only_second` or `only_first`."
                        )
                    )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, LayoutLMv2TokenizerFast):
                    information = tokenizer(
                        question_0,
                        seq_1,
                        boxes=boxes_1,
                        max_length=len(sequence["input_ids"]) - 2,
                        add_special_tokens=False,
                        stride=stride,
                        truncation=True,
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    bbox = information["bbox"][0]
                    overflowing_bbox = information["bbox"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence["input_ids"]) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                    self.assertEqual(bbox, bbox_longest_sequence)
                    self.assertEqual(overflowing_bbox, overflowing_token_bbox_longest_sequence_fast)
                else:
                    # No overflowing tokens when using 'longest' in python tokenizers
                    with self.assertRaises(ValueError) as context:
                        information = tokenizer(
                            question_0,
                            seq_1,
                            boxes=boxes_1,
                            max_length=len(sequence["input_ids"]) - 2,
                            add_special_tokens=False,
                            stride=stride,
                            truncation=True,
                            return_overflowing_tokens=True,
                            # add_prefix_space=False,
                        )

                    self.assertTrue(
                        context.exception.args[0].startswith(
                            "Not possible to return overflowing tokens for pair of sequences with the "
                            "`longest_first`. Please select another truncation strategy than `longest_first`, "
                            "for instance `only_second` or `only_first`."
                        )
                    )

                information_first_truncated = tokenizer(
                    question_0,
                    seq_1,
                    boxes=boxes_1,
                    max_length=len(sequence["input_ids"]) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, LayoutLMv2TokenizerFast):
                    truncated_sequence = information_first_truncated["input_ids"][0]
                    overflowing_tokens = information_first_truncated["input_ids"][1]
                    bbox = information_first_truncated["bbox"][0]
                    overflowing_bbox = information_first_truncated["bbox"][1]
                    self.assertEqual(len(information_first_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence["input_ids"]) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq1_tokens["input_ids"]))
                    self.assertEqual(overflowing_tokens, overflow_first_sequence)
                    self.assertEqual(bbox, bbox_first_sequence)
                    self.assertEqual(overflowing_bbox, overflowing_token_bbox_first_sequence_fast)
                else:
                    truncated_sequence = information_first_truncated["input_ids"]
                    overflowing_tokens = information_first_truncated["overflowing_tokens"]
                    overflowing_bbox = information_first_truncated["overflowing_token_boxes"]
                    bbox = information_first_truncated["bbox"]

                    self.assertEqual(len(truncated_sequence), len(sequence["input_ids"]) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq0_tokens["input_ids"][-(2 + stride) :])
                    self.assertEqual(bbox, bbox_first_sequence)
                    self.assertEqual(overflowing_bbox, overflowing_token_bbox_first_sequence_slow)

                information_second_truncated = tokenizer(
                    question_0,
                    seq_1,
                    boxes=boxes_1,
                    max_length=len(sequence["input_ids"]) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, LayoutLMv2TokenizerFast):
                    truncated_sequence = information_second_truncated["input_ids"][0]
                    overflowing_tokens = information_second_truncated["input_ids"][1]
                    bbox = information_second_truncated["bbox"][0]
                    overflowing_bbox = information_second_truncated["bbox"][1]

                    self.assertEqual(len(information_second_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence["input_ids"]) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq0_tokens["input_ids"]))
                    self.assertEqual(overflowing_tokens, overflow_second_sequence)
                    self.assertEqual(bbox, bbox_second_sequence)
                    self.assertEqual(overflowing_bbox, overflowing_token_bbox_second_sequence_fast)
                else:
                    truncated_sequence = information_second_truncated["input_ids"]
                    overflowing_tokens = information_second_truncated["overflowing_tokens"]
                    bbox = information_second_truncated["bbox"]
                    overflowing_bbox = information_second_truncated["overflowing_token_boxes"]

                    self.assertEqual(len(truncated_sequence), len(sequence["input_ids"]) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq1_tokens["input_ids"][-(2 + stride) :])
                    self.assertEqual(bbox, bbox_second_sequence)
                    self.assertEqual(overflowing_bbox, overflowing_token_bbox_second_sequence_slow)

    # @unittest.skip("LayoutLMv2 tokenizer requires boxes besides sequences.")
    def test_maximum_encoding_length_single_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0, boxes_0, ids = self.get_clean_sequence(tokenizer, max_length=20)

                sequence = tokenizer(seq_0, boxes=boxes_0, add_special_tokens=False)
                total_length = len(sequence["input_ids"])

                self.assertGreater(total_length, 4, "Issue with the testing sequence, please update it it's too short")

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length
                boxes_1 = boxes_0 * model_max_length
                sequence1 = tokenizer(seq_1, boxes=boxes_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                self.assertGreater(
                    total_length1, model_max_length, "Issue with the testing sequence, please update it it's too short"
                )

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"Truncation: {truncation_state}"):
                                output = tokenizer(
                                    seq_1,
                                    boxes=boxes_1,
                                    padding=padding_state,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"]), model_max_length)
                                self.assertEqual(len(output["bbox"]), model_max_length)

                                output = tokenizer(
                                    [seq_1],
                                    boxes=[boxes_1],
                                    padding=padding_state,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)
                                self.assertEqual(len(output["bbox"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(seq_1, boxes=boxes_1, padding=padding_state, truncation=False)
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
                            output = tokenizer([seq_1], boxes=[boxes_1], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                            self.assertNotEqual(len(output["bbox"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )
                # Check the order of Sequence of input ids, overflowing tokens and bbox sequence with truncation
                stride = 2
                information = tokenizer(
                    seq_0,
                    boxes=boxes_0,
                    max_length=total_length - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation=True,
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, LayoutLMv2TokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    bbox = information["bbox"][0]
                    overflowing_bbox = information["bbox"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence["input_ids"][:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence["input_ids"][-(2 + stride) :])

                    self.assertEqual(bbox, sequence["bbox"][:-2])
                    self.assertEqual(overflowing_bbox, sequence["bbox"][-(2 + stride) :])
                else:
                    truncated_sequence = information["input_ids"]
                    overflowing_tokens = information["overflowing_tokens"]
                    bbox = information["bbox"]
                    overflowing_bbox = information["overflowing_token_boxes"]
                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence["input_ids"][:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence["input_ids"][-(2 + stride) :])
                    self.assertEqual(bbox, sequence["bbox"][:-2])
                    self.assertEqual(overflowing_bbox, sequence["bbox"][-(2 + stride) :])

    @unittest.skip("LayoutLMv2 tokenizer requires boxes besides sequences.")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip("LayoutLMv2 tokenizer always expects pretokenized inputs.")
    def test_compare_pretokenized_inputs(self):
        pass

    @unittest.skip("LayoutLMv2 fast tokenizer does not support prepare_for_model")
    def test_compare_prepare_for_model(self):
        pass

    @slow
    def test_only_label_first_subword(self):
        words = ["hello", "niels"]
        boxes = [[1000, 1000, 1000, 1000] for _ in range(len(words))]
        word_labels = [0, 1]

        # test slow tokenizer
        tokenizer_p = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
        encoding = tokenizer_p(words, boxes=boxes, word_labels=word_labels)
        self.assertListEqual(encoding.labels, [-100, 0, 1, -100, -100])

        tokenizer_p = LayoutLMv2Tokenizer.from_pretrained(
            "microsoft/layoutlmv2-base-uncased", only_label_first_subword=False
        )
        encoding = tokenizer_p(words, boxes=boxes, word_labels=word_labels)
        self.assertListEqual(encoding.labels, [-100, 0, 1, 1, -100])

        # test fast tokenizer
        tokenizer_r = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
        encoding = tokenizer_r(words, boxes=boxes, word_labels=word_labels)
        self.assertListEqual(encoding.labels, [-100, 0, 1, -100, -100])

        tokenizer_r = LayoutLMv2Tokenizer.from_pretrained(
            "microsoft/layoutlmv2-base-uncased", only_label_first_subword=False
        )
        encoding = tokenizer_r(words, boxes=boxes, word_labels=word_labels)
        self.assertListEqual(encoding.labels, [-100, 0, 1, 1, -100])

    @slow
    def test_layoutlmv2_integration_test(self):

        tokenizer_p = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
        tokenizer_r = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")

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

        encoding_p = tokenizer_p(words, boxes=boxes, padding="max_length", max_length=20)
        encoding_r = tokenizer_r(words, boxes=boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding_p), expected_results)
        self.assertDictEqual(dict(encoding_r), expected_results)

        # CASE 1: batched
        words, boxes = self.get_words_and_boxes_batch()

        # fmt: off
        expected_results = {'input_ids': [[101, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 7592, 2026, 2171, 2003, 3960, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'bbox': [[[0, 0, 0, 0], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [961, 885, 992, 912], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [34, 42, 66, 69], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E231
        # fmt: on

        encoding_p = tokenizer_p(words, boxes=boxes, padding="max_length", max_length=20)
        encoding_r = tokenizer_r(words, boxes=boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding_p), expected_results)
        self.assertDictEqual(dict(encoding_r), expected_results)

        # CASE 2: not batched
        words, boxes = self.get_words_and_boxes()
        word_labels = [1, 2, 3]

        # fmt: off
        expected_results = {'input_ids': [101, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bbox': [[0, 0, 0, 0], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [-100, 1, 2, -100, 3, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100], 'attention_mask': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}  # noqa: E231
        # fmt: on

        encoding_p = tokenizer_p(words, boxes=boxes, word_labels=word_labels, padding="max_length", max_length=20)
        encoding_r = tokenizer_r(words, boxes=boxes, word_labels=word_labels, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding_p), expected_results)
        self.assertDictEqual(dict(encoding_r), expected_results)

        # CASE 2: batched
        words, boxes = self.get_words_and_boxes_batch()
        word_labels = [[1, 2, 3], [2, 46, 17, 22, 3]]

        # fmt: off
        expected_results = {'input_ids': [[101, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 7592, 2026, 2171, 2003, 3960, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'bbox': [[[0, 0, 0, 0], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [961, 885, 992, 912], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [34, 42, 66, 69], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'labels': [[-100, 1, 2, -100, 3, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100], [-100, 2, 46, 17, 22, 3, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E231
        # fmt: on

        encoding_p = tokenizer_p(words, boxes=boxes, word_labels=word_labels, padding="max_length", max_length=20)
        encoding_r = tokenizer_r(words, boxes=boxes, word_labels=word_labels, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding_p), expected_results)
        self.assertDictEqual(dict(encoding_r), expected_results)

        # CASE 3: not batched
        question, words, boxes = self.get_question_words_and_boxes()

        # fmt: off
        expected_results = {'input_ids': [101, 2054, 1005, 1055, 2010, 2171, 1029, 102, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0], 'bbox': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}  # noqa: E231
        # fmt: on

        encoding_p = tokenizer_p(question, words, boxes, padding="max_length", max_length=20)
        encoding_r = tokenizer_r(question, words, boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding_p), expected_results)
        self.assertDictEqual(dict(encoding_r), expected_results)

        # CASE 3: batched
        questions, words, boxes = self.get_question_words_and_boxes_batch()

        # fmt: off
        expected_results = {'input_ids': [[101, 2054, 1005, 1055, 2010, 2171, 1029, 102, 1037, 6881, 2135, 3231, 102, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 2003, 2002, 2170, 1029, 102, 2054, 1037, 21110, 2546, 3806, 2102, 2078, 102, 0, 0, 0, 0, 0]], 'bbox': [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [423, 237, 440, 251], [427, 272, 441, 287], [427, 272, 441, 287], [419, 115, 437, 129], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [336, 42, 353, 57], [34, 42, 66, 69], [34, 42, 66, 69], [34, 42, 66, 69], [1000, 1000, 1000, 1000], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]}  # noqa: E231
        # fmt: on

        encoding_p = tokenizer_p(questions, words, boxes, padding="max_length", max_length=20)
        encoding_r = tokenizer_r(questions, words, boxes, padding="max_length", max_length=20)
        self.assertDictEqual(dict(encoding_p), expected_results)
        self.assertDictEqual(dict(encoding_r), expected_results)

    @unittest.skip("Doesn't support another framework than PyTorch")
    def test_np_encode_plus_sent_to_model(self):
        pass
